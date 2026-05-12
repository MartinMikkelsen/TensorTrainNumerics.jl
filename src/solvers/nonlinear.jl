# Nonlinear ALS (SCF-ALS) for 1D NLS / GPE ground-state problems
#
# Method: Self-Consistent Field + Alternating Linear Scheme (SCF-ALS)
# The nonlinear density |ψ|² is frozen once per outer sweep, converting
# the nonlinear eigenvalue problem into a sequence of linear subproblems
# solved site-by-site with the standard ALS micro-step machinery.
#
# Effective operator at site i:  K_eff = K_lin + K_nl
#   K_lin  assembled from environments G_lin, H_lin  (linear part)
#   K_nl   assembled from environments G_nl,  H_nl   (frozen diagonal part)
#
# Note on canonical form: this is a first step toward the interpolative
# ALS-X framework (Ye & Yang 2025, arXiv:2512.15703), where CUR
# canonicalization would make K_nl evaluation pointwise-local without
# environment arrays.  The SCF approximation here plays the same role as
# the outer DLR-G loop: we freeze |ψ|² and solve a linear problem per sweep.

using LinearAlgebra

_density_tt(ψ::TTvector) = hadamard(conj(ψ), ψ)

function _nls_chemical_potential(ψ::TTvector{T}, H_lin::TToperator{T}, g::Real) where {T <: Number}
    ρ = _density_tt(ψ)
    return real(dot(ψ, H_lin * ψ) + convert(T, g) * dot(ρ, ρ))
end

"""
    nonlinear_als_eigsolve(H_lin, g, tt_start; sweep_count, verbose)

Ground-state solver for the 1D NLS / GPE eigenvalue problem

    (H_lin + g · diag(|ψ|²)) ψ = μ ψ ,   ‖ψ‖ = 1 ,

via SCF-ALS: alternating site-by-site linear eigenvalue micro-steps with
a frozen-density nonlinear potential rebuilt once per full sweep.

# Arguments
- `H_lin    :: TToperator{T}` — linear Hamiltonian (kinetic + trap potential)
- `g        :: Real`          — nonlinear coupling (g > 0 repulsive, g < 0 attractive)
- `tt_start :: TTvector{T}`   — initial guess (need not be normalized)

# Keyword arguments
- `sweep_count :: Int  = 8`   — number of full (forward + backward) sweeps
- `verbose     :: Bool = true` — print the global μ and pre-normalization norm after each sweep

# Returns
`(μ_hist, ψ)` where `μ_hist :: Vector{Float64}` collects the global
chemical potential of the normalized iterate after each full sweep and
`ψ :: TTvector{T}` is the approximate ground state with ‖ψ‖ = 1.

# Notes
- TT ranks are fixed to those of `tt_start` (no rank adaptation).
- Normalization is ‖ψ‖² = 1 in the discrete l² sense; multiply by the
  grid spacing h to get the physical integral norm.
- For large |g| convergence may require more sweeps or a better initial
  guess (e.g., the g=0 ground state from `als_eigsolve`).
"""
function nonlinear_als_eigsolve(
        H_lin   :: TToperator{T},
        g       :: Real,
        tt_start:: TTvector{T};
        sweep_count :: Int  = 8,
        verbose     :: Bool = true
    ) where {T <: Number}

    d    = H_lin.N
    dims = tt_start.ttv_dims

    # Canonicalize and normalize initial guess
    tt_opt = orthogonalize(tt_start)
    tt_opt = (one(T) / norm(tt_opt)) * tt_opt

    rks = tt_opt.ttv_rks          # rank vector — fixed throughout
    gT  = convert(T, g)
    μ_hist = Float64[]

    for sweep in 1:sweep_count

        # ── SCF step: rebuild frozen-density operator ─────────────────────
        # |ψ|² in TT format; ranks = rks .* rks (entry-wise)
        ρ    = _density_tt(tt_opt)
        D_nl = gT * ttv_to_diag_tto(ρ)    # g · diag(|ψ|²), same rank as ρ

        # ── Initialize left environments (G) ─────────────────────────────
        G_lin = Array{Array{T}}(undef, d)
        G_nl  = Array{Array{T}}(undef, d)
        for i in 1:d
            G_lin[i] = zeros(T, dims[i], rks[i], dims[i], rks[i], H_lin.tto_rks[i+1])
            G_nl[i]  = zeros(T, dims[i], rks[i], dims[i], rks[i], D_nl.tto_rks[i+1])
        end
        # Left boundary: operator core at site 1 contracted with ψ_0 = 1
        G_lin[1] = reshape(H_lin.tto_vec[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)
        G_nl[1]  = reshape(D_nl.tto_vec[1][:, :, 1, :],  dims[1], 1, dims[1], 1, :)

        # ── Initialize right environments (H) from current ψ ─────────────
        H_lin_env = init_H(tt_opt, H_lin)
        H_nl_env  = init_H(tt_opt, D_nl)

        # ── Forward half-sweep: sites 1 → d-1 ────────────────────────────
        for i in 1:(d-1)
            K_dims = (dims[i], rks[i], rks[i+1])
            K_eff  = K_full(G_lin[i], H_lin_env[i], K_dims) .+
                     K_full(G_nl[i],  H_nl_env[i],  K_dims)
            F  = eigen(Hermitian(K_eff), 1:1)
            V  = reshape(F.vectors[:, 1], K_dims)

            tt_opt = right_core_move(tt_opt, V, i, rks)
            update_G!(tt_opt.ttv_vec[i], H_lin.tto_vec[i+1], G_lin[i], G_lin[i+1])
            update_G!(tt_opt.ttv_vec[i], D_nl.tto_vec[i+1],  G_nl[i],  G_nl[i+1])
        end

        # ── Backward half-sweep: sites d → 2 ─────────────────────────────
        for i in d:(-1):2
            K_dims = (dims[i], rks[i], rks[i+1])
            K_eff  = K_full(G_lin[i], H_lin_env[i], K_dims) .+
                     K_full(G_nl[i],  H_nl_env[i],  K_dims)
            F  = eigen(Hermitian(K_eff), 1:1)
            V  = reshape(F.vectors[:, 1], K_dims)

            tt_opt = left_core_move(tt_opt, V, i, rks)
            update_H!(tt_opt.ttv_vec[i], H_lin.tto_vec[i],  H_lin_env[i], H_lin_env[i-1])
            update_H!(tt_opt.ttv_vec[i], D_nl.tto_vec[i],   H_nl_env[i],  H_nl_env[i-1])
        end

        # Explicit normalization — SCF does not conserve ‖ψ‖
        n_prev = norm(tt_opt)
        tt_opt = (one(T) / n_prev) * tt_opt
        μ = _nls_chemical_potential(tt_opt, H_lin, g)
        push!(μ_hist, μ)

        if verbose
            println("  sweep $sweep  μ ≈ $(round(μ, digits=8))" *
                    "  ‖ψ‖_before_norm = $(round(n_prev, sigdigits=6))")
        end
    end

    return μ_hist, tt_opt
end

"""
    nonlinear_tdvp_imagtime(H_lin, g, tt_start; dτ, n_steps, verbose)

Ground-state solver for the 1D NLS / GPE equation using imaginary-time ALS:
at each step the SCF density is frozen, then a full forward+backward ALS sweep
is performed where each site update applies the local imaginary-time propagator

    v_new = exp(−dτ · K_eff) · v_old

via exact diagonalisation of the small local matrix K_eff.  This is equivalent
to single-site imaginary-time TDVP (without the backward bond step), and is
unconditionally stable for positive-definite Hamiltonians of any spectral range.

Note: SCF-ALS (`nonlinear_als_eigsolve`) corresponds to the limit dτ → ∞ of
this method, where exp(-dτ K_eff)*v projects directly onto the ground state of
K_eff.  Smaller dτ requires more sweeps to converge (convergence rate ∝ dτ·Δ
where Δ is the local spectral gap), making this a natural comparison partner for
SCF-ALS convergence benchmarks.

# Arguments
- `H_lin    :: TToperator{T}` — linear Hamiltonian
- `g        :: Real`          — nonlinear coupling
- `tt_start :: TTvector{T}`   — initial guess (need not be normalized)

# Keyword arguments
- `dτ      :: Float64 = 0.02` — imaginary-time step per site update
- `n_steps :: Int     = 100`  — number of full (forward+backward) sweeps
- `verbose :: Bool    = false` — print μ after each sweep

# Returns
`(μ_hist, ψ)` where `μ_hist[k] = ⟨ψ_k|H_lin + g·diag(|ψ_k|²)|ψ_k⟩`
is measured after sweep k, and ψ is the final normalized state.
"""
function nonlinear_tdvp_imagtime(
        H_lin   :: TToperator{T},
        g       :: Real,
        tt_start:: TTvector{T};
        dτ      :: Float64 = 0.02,
        n_steps :: Int     = 100,
        verbose :: Bool    = false
    ) where {T <: Number}

    d    = H_lin.N
    dims = tt_start.ttv_dims

    ψ   = orthogonalize(tt_start)
    ψ   = (one(T) / norm(ψ)) * ψ
    gT  = convert(T, g)
    μ_hist = Float64[]

    for step in 1:n_steps
        rks = ψ.ttv_rks

        # ── SCF step: rebuild frozen-density operator ─────────────────────
        ρ    = _density_tt(ψ)
        D_nl = gT * ttv_to_diag_tto(ρ)

        # ── Initialize left environments (G) ─────────────────────────────
        G_lin = Array{Array{T}}(undef, d)
        G_nl  = Array{Array{T}}(undef, d)
        for i in 1:d
            G_lin[i] = zeros(T, dims[i], rks[i], dims[i], rks[i], H_lin.tto_rks[i+1])
            G_nl[i]  = zeros(T, dims[i], rks[i], dims[i], rks[i], D_nl.tto_rks[i+1])
        end
        G_lin[1] = reshape(H_lin.tto_vec[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)
        G_nl[1]  = reshape(D_nl.tto_vec[1][:, :, 1, :],  dims[1], 1, dims[1], 1, :)

        # ── Initialize right environments (H) from current ψ ─────────────
        H_lin_env = init_H(ψ, H_lin)
        H_nl_env  = init_H(ψ, D_nl)

        # ── Forward half-sweep: sites 1 → d-1 ────────────────────────────
        for i in 1:(d-1)
            K_dims = (dims[i], rks[i], rks[i+1])
            K_eff  = K_full(G_lin[i], H_lin_env[i], K_dims) .+
                     K_full(G_nl[i],  H_nl_env[i],  K_dims)
            F = eigen(Hermitian(K_eff))
            v = vec(ψ.ttv_vec[i])
            # Shifted exp(-dτ (K_eff - λ_min I)) v: preserves ground-state component
            # exactly (exp(0)=1) regardless of absolute energy scale, preventing
            # underflow when all eigenvalues are large (e.g. cold start from random ψ).
            λ = real.(F.values)
            v_new = F.vectors * (exp.(-dτ .* (λ .- λ[1])) .* (F.vectors' * v))
            normalize!(v_new)
            V = reshape(v_new, K_dims)

            ψ = right_core_move(ψ, convert(Array{T,3}, V), i, rks)
            update_G!(ψ.ttv_vec[i], H_lin.tto_vec[i+1], G_lin[i], G_lin[i+1])
            update_G!(ψ.ttv_vec[i], D_nl.tto_vec[i+1],  G_nl[i],  G_nl[i+1])
        end

        # ── Backward half-sweep: sites d → 2 ─────────────────────────────
        for i in d:(-1):2
            K_dims = (dims[i], rks[i], rks[i+1])
            K_eff  = K_full(G_lin[i], H_lin_env[i], K_dims) .+
                     K_full(G_nl[i],  H_nl_env[i],  K_dims)
            F = eigen(Hermitian(K_eff))
            v = vec(ψ.ttv_vec[i])
            λ = real.(F.values)
            v_new = F.vectors * (exp.(-dτ .* (λ .- λ[1])) .* (F.vectors' * v))
            normalize!(v_new)
            V = reshape(v_new, K_dims)

            ψ = left_core_move(ψ, convert(Array{T,3}, V), i, rks)
            update_H!(ψ.ttv_vec[i], H_lin.tto_vec[i],  H_lin_env[i], H_lin_env[i-1])
            update_H!(ψ.ttv_vec[i], D_nl.tto_vec[i],   H_nl_env[i],  H_nl_env[i-1])
        end

        # Normalize; measure global μ after the sweep
        ψ = (one(T) / norm(ψ)) * ψ
        μ = _nls_chemical_potential(ψ, H_lin, g)
        push!(μ_hist, μ)

        verbose && println("  step $step  μ ≈ $(round(μ, digits=8))")
    end

    return μ_hist, ψ
end

"""
    nls_energy(ψ, H_lin, g)

NLS / GPE energy functional

    E[ψ] = ⟨ψ|H_lin|ψ⟩ + (g/2) ⟨|ψ|², |ψ|²⟩

Inner product is the discrete l² dot product (multiply by h to get the
physical integral). State ψ need not be normalized.
"""
function nls_energy(ψ::TTvector{T}, H_lin::TToperator{T}, g::Real) where {T <: Number}
    ρ    = _density_tt(ψ)                                # |ψ|²
    E_lin = real(dot(ψ, H_lin * ψ))                      # ⟨ψ|H_lin|ψ⟩
    E_nl  = real(convert(T, g) / 2 * dot(ρ, ρ))         # (g/2)‖ψ‖₄⁴
    return E_lin + E_nl
end
