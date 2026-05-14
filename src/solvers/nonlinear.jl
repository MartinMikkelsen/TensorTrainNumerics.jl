# Nonlinear ALS / MALS (SCF-ALS / SCF-MALS) for 1D NLS / GPE ground-state problems
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
_nls_diag_operator(ψ::TTvector{T}, g::Number) where {T <: Number} = convert(T, g) * ttv_to_diag_tto(_density_tt(ψ))

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
        # Frozen-density operator g · diag(|ψ|²)
        D_nl = _nls_diag_operator(tt_opt, gT)

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
    nonlinear_mals_eigsolve(H_lin, g, tt_start; tol, sweep_schedule, rmax_schedule, it_solver, linsolv_maxiter, linsolv_tol, itslv_thresh, verbose)

Ground-state solver for the 1D NLS / GPE eigenvalue problem

    (H_lin + g · diag(|ψ|²)) ψ = μ ψ ,   ‖ψ‖ = 1 ,

via SCF-MALS: at the start of each sweep the density is frozen, producing a
linearized effective operator that is minimized with one rank-adaptive
two-site MALS sweep.

# Arguments
- `H_lin    :: TToperator{T}` — linear Hamiltonian (kinetic + trap potential)
- `g        :: Real`          — nonlinear coupling (g > 0 repulsive, g < 0 attractive)
- `tt_start :: TTvector{T}`   — initial guess (need not be normalized)

# Keyword arguments
- `tol::Float64=1e-12` — relative SVD truncation threshold for rank adaptation.
- `sweep_schedule::Vector{Int}=[2]` — SCF-MALS sweep schedule, using the same stage semantics as `mals_eigsolve`.
- `rmax_schedule::Vector{Int}` — maximum bond dimension at each stage.
- `it_solver::Bool=false` — use an iterative eigensolver for local two-site subproblems.
- `linsolv_maxiter::Int=200` — maximum iterations for the iterative eigensolver.
- `linsolv_tol::Float64=max(sqrt(tol), 1e-8)` — tolerance for the iterative eigensolver.
- `itslv_thresh::Int=256` — local problem size above which iterative solve activates.
- `verbose::Bool=true` — print the global μ, pre-normalization norm, and max rank after each sweep.

# Returns
`(μ_hist, ψ, r_hist)` where `μ_hist :: Vector{Float64}` collects the global
chemical potential of the normalized iterate after each full SCF-MALS sweep,
`ψ :: TTvector{T}` is the approximate ground state with ‖ψ‖ = 1, and
`r_hist :: Vector{Int}` stores the maximum bond dimension after each local
two-site update.

# Notes
- Rank adaptation follows the same truncation and sweep-schedule conventions as `mals_eigsolve`.
- Normalization is ‖ψ‖² = 1 in the discrete l² sense; multiply by the
  grid spacing h to get the physical integral norm.
"""
function nonlinear_mals_eigsolve(
        H_lin   :: TToperator{T},
        g       :: Real,
        tt_start:: TTvector{T};
        tol::Float64 = 1.0e-12,
        sweep_schedule::Vector{Int} = [2],
        rmax_schedule::Vector{Int} = [round(Int, sqrt(prod(tt_start.ttv_dims)::Int))],
        it_solver::Bool = false,
        linsolv_maxiter::Int = 200,
        linsolv_tol::Float64 = max(sqrt(tol), 1.0e-8),
        itslv_thresh::Int = 256,
        verbose::Bool = true
    ) where {T <: Number}

    @assert(length(rmax_schedule) == length(sweep_schedule), "Sweep schedule error")

    tt_opt = orthogonalize(tt_start)
    tt_opt = (one(T) / norm(tt_opt)) * tt_opt

    μ_hist = Float64[]
    r_hist = Int[]
    nsweeps = 0
    i_schedule = 1

    while i_schedule <= length(sweep_schedule)
        nsweeps += 1

        if nsweeps == sweep_schedule[i_schedule]
            i_schedule += 1
            if i_schedule > length(sweep_schedule)
                return μ_hist, tt_opt, r_hist
            end
        end

        D_nl = _nls_diag_operator(tt_opt, g)
        H_eff = H_lin + D_nl

        _, tt_opt, r_hist_sweep = mals_eigsolve(
            H_eff, tt_opt;
            tol = tol,
            sweep_schedule = [2],
            rmax_schedule = [rmax_schedule[i_schedule]],
            it_solver = it_solver,
            linsolv_maxiter = linsolv_maxiter,
            linsolv_tol = linsolv_tol,
            itslv_thresh = itslv_thresh
        )
        append!(r_hist, r_hist_sweep)

        n_prev = norm(tt_opt)
        tt_opt = (one(T) / n_prev) * tt_opt
        μ = _nls_chemical_potential(tt_opt, H_lin, g)
        push!(μ_hist, μ)

        if verbose
            println("  sweep $nsweeps  μ ≈ $(round(μ, digits=8))" *
                    "  ‖ψ‖_before_norm = $(round(n_prev, sigdigits=6))" *
                    "  max_rank = $(maximum(tt_opt.ttv_rks))")
        end
    end

    return μ_hist, tt_opt, r_hist
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
        D_nl = _nls_diag_operator(ψ, gT)

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

"""
    burgers_scf_als_step(u_prev, D_x, D_xx, ν, dt; ...)

One implicit-Euler time step for the viscous Burgers equation

    ∂_t u + u · ∂_x u = ν · ∂_xx u

via Picard (SCF) linearization: freeze the advection coefficient at the
current iterate, solve the resulting linear system with ALS, repeat until
convergence.  The linear system at each Picard iteration is

    [(1/dt)·I + diag(u^k)·D_x + ν·D_xx] · u^{k+1} = u^{prev}/dt

Note: `D_xx` here is the negative Laplacian (`Δ_DN`, positive eigenvalues), so
`+ν·D_xx` corresponds to the stabilising `−ν·∂²/∂x²` term in the PDE.

MALS is intentionally not supported: `mals_linsolve` symmetrises the local
matrix via `Hermitian(K)`, which discards the antisymmetric advection
contribution and produces wrong results for non-Hermitian operators.

# Arguments
- `u_prev :: TTvector{T}`   — solution at the previous time step
- `D_x    :: TToperator{T}` — first-derivative operator (already scaled by 1/dx)
- `D_xx   :: TToperator{T}` — second-derivative operator (already scaled by 1/dx²),
                               expected to be the negative Laplacian (positive eigenvalues)
- `ν      :: Real`          — viscosity
- `dt     :: Real`          — time step

# Keyword arguments
- `max_scf   :: Int  = 10`   — maximum Picard iterations per step
- `scf_tol   :: Real = 1e-8` — relative convergence: ‖u^{k+1} − u^k‖ / ‖u^{k+1}‖
- `max_bond  :: Int  = 20`   — TT bond dimension cap after compression
- `als_sweeps:: Int  = 5`    — number of ALS sweeps for the inner linear solve
- `verbose   :: Bool = false` — print per-Picard-iteration residual

# Returns
`u_new :: TTvector{T}` — solution at the new time level
"""
function burgers_scf_als_step(
        u_prev    :: TTvector{T},
        D_x       :: TToperator{T},
        D_xx      :: TToperator{T},
        ν         :: Real,
        dt        :: Real;
        max_scf   :: Int  = 10,
        scf_tol   :: Real = 1e-8,
        max_bond  :: Int  = 20,
        als_sweeps:: Int  = 5,
        verbose   :: Bool = false
    ) where {T <: Number}

    d     = u_prev.N
    I_tto = id_tto(T, d; n_dim = 2)
    rhs   = convert(T, inv(dt)) * u_prev
    νT    = convert(T, ν)
    invdt = convert(T, inv(dt))

    u = u_prev
    for iter in 1:max_scf
        u_old = u
        A_adv = ttv_to_diag_tto(u) * D_x
        A_eff = invdt * I_tto + A_adv + νT * D_xx
        u = als_linsolve(A_eff, rhs, u_old; sweep_count = als_sweeps)
        u = tt_compress!(u, max_bond)
        rel_diff = norm(u - u_old) / (norm(u) + eps(real(T)))
        verbose && println("    Picard $iter  rel_diff = $(round(rel_diff, sigdigits=4))")
        rel_diff < scf_tol && break
    end
    return u
end

"""
    burgers_scf_als(u₀, D_x, D_xx, ν, dt, n_steps; ...)

Time-integrate the viscous Burgers equation for `n_steps` implicit-Euler steps
using `burgers_scf_als_step` at each step.

# Arguments
- `u₀      :: TTvector{T}`   — initial condition
- `D_x     :: TToperator{T}` — first-derivative operator (scaled by 1/dx)
- `D_xx    :: TToperator{T}` — second-derivative operator (scaled by 1/dx²)
- `ν       :: Real`          — viscosity
- `dt      :: Real`          — time step size
- `n_steps :: Int`           — number of time steps

# Keyword arguments
Forwarded to `burgers_scf_als_step`: `max_scf`, `scf_tol`, `max_bond`, `als_sweeps`, `verbose`.
- `verbose_steps :: Bool = false` — print max rank after each time step

# Returns
`u :: TTvector{T}` — solution at time `n_steps * dt`
"""
function burgers_scf_als(
        u₀           :: TTvector{T},
        D_x          :: TToperator{T},
        D_xx         :: TToperator{T},
        ν            :: Real,
        dt           :: Real,
        n_steps      :: Int;
        max_scf      :: Int  = 10,
        scf_tol      :: Real = 1e-8,
        max_bond     :: Int  = 20,
        als_sweeps   :: Int  = 5,
        verbose      :: Bool = false,
        verbose_steps:: Bool = false
    ) where {T <: Number}

    u = u₀
    for step in 1:n_steps
        u = burgers_scf_als_step(u, D_x, D_xx, ν, dt;
                max_scf = max_scf, scf_tol = scf_tol,
                max_bond = max_bond, als_sweeps = als_sweeps,
                verbose = verbose)
        verbose_steps &&
            println("  step $step / $n_steps  max_rank = $(maximum(u.ttv_rks))")
    end
    return u
end

"""
    burgers_scf_mals_step(u_prev, D_x, D_xx, ν, dt; ...)

One implicit-Euler time step for the viscous Burgers equation via Picard (SCF)
linearization with a **rank-adaptive two-site ALS sweep**.

Identical in spirit to `burgers_scf_als_step` but uses two-site (MALS-style)
local solves that allow bond dimension to grow up to `max_bond`.  The local
2-site matrix is assembled without the `Hermitian()` wrapping used by
`mals_linsolve`, which is required because the advection term `diag(u)·D_x`
makes `A_eff` non-Hermitian.

# Keyword arguments
- `max_scf   :: Int  = 10`   — maximum Picard iterations per step
- `scf_tol   :: Real = 1e-8` — relative convergence: ‖u^{k+1} − u^k‖ / ‖u^{k+1}‖
- `max_bond  :: Int  = 20`   — maximum TT bond dimension (hard cap on SVD rank)
- `verbose   :: Bool = false` — print per-Picard-iteration residual
"""
function burgers_scf_mals_step(
        u_prev   :: TTvector{T},
        D_x      :: TToperator{T},
        D_xx     :: TToperator{T},
        ν        :: Real,
        dt       :: Real;
        max_scf  :: Int  = 10,
        scf_tol  :: Real = 1e-8,
        max_bond :: Int  = 20,
        verbose  :: Bool = false
    ) where {T <: Number}

    d     = u_prev.N
    dims  = u_prev.ttv_dims
    I_tto = id_tto(T, d; n_dim = 2)
    rhs   = convert(T, inv(dt)) * u_prev
    νT    = convert(T, ν)
    invdt = convert(T, inv(dt))

    u = orthogonalize(u_prev)

    for iter in 1:max_scf
        u_old = u

        A_adv = ttv_to_diag_tto(u) * D_x
        A_eff = invdt * I_tto + A_adv + νT * D_xx
        A_rks = A_eff.tto_rks
        b_rks = rhs.ttv_rks

        # Allocate left environments G (for A_eff) and G_b (for rhs)
        G   = Array{Array{T, 5}}(undef, d)
        G_b = Array{Array{T, 3}}(undef, d)
        for i in 1:d
            rmax_i = min(max_bond, prod(dims[1:(i - 1)]), prod(dims[i:end]))
            G[i]   = zeros(T, dims[i], rmax_i, dims[i], rmax_i, A_rks[i + 1])
            G_b[i] = zeros(T, dims[i], rmax_i, b_rks[i + 1])
        end
        G[1][:, 1:1, :, 1:1, :] = reshape(A_eff.tto_vec[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)
        G_b[1] = reshape(rhs.ttv_vec[1], dims[1], 1, :)

        H   = init_H_mals(u, A_eff, max_bond)
        H_b = init_Hb_mals(u, rhs, max_bond)

        # Forward half-sweep: 2-site updates at pairs (1,2), (2,3), ..., (d-1,d)
        for i in 1:(d - 1)
            rks  = u.ttv_rks
            Gi   = @view G[i][:, 1:rks[i],     :, 1:rks[i],     :]
            Hi   = @view H[i][:, :, 1:rks[i+2], :, 1:rks[i+2]]
            G_bi = @view G_b[i][:, 1:rks[i],   :]
            H_bi = @view H_b[i][:, :, 1:rks[i+2]]

            K_dims = (dims[i], rks[i], dims[i+1], rks[i+2])
            K  = zeros(T, prod(K_dims), prod(K_dims))
            Kr = reshape(K, (K_dims..., K_dims...))
            @tensor Kr[a, b, c, q, e, f, g, h] = Gi[a, b, e, f, z] * Hi[z, c, q, g, h]
            Pb = zeros(T, K_dims)
            @tensor Pb[a, b, c, q] = G_bi[a, b, z] * H_bi[z, c, q]

            V = reshape(K \ Pb[:], K_dims)
            u = right_core_move_mals(u, i, V, 0.0, max_bond)

            rks   = u.ttv_rks
            Gip   = @view G[i+1][:, 1:rks[i+1], :, 1:rks[i+1], :]
            G_bip = @view G_b[i+1][:, 1:rks[i+1], :]
            update_G!(u.ttv_vec[i], A_eff.tto_vec[i+1], Gi, Gip)
            update_Gb!(u.ttv_vec[i], rhs.ttv_vec[i+1], G_bi, G_bip)
        end

        # Backward half-sweep: 2-site updates at pairs (d-1,d), ..., (1,2)
        for i in (d - 1):-1:1
            rks  = u.ttv_rks
            Gi   = @view G[i][:, 1:rks[i],     :, 1:rks[i],     :]
            Hi   = @view H[i][:, :, 1:rks[i+2], :, 1:rks[i+2]]
            G_bi = @view G_b[i][:, 1:rks[i],   :]
            H_bi = @view H_b[i][:, :, 1:rks[i+2]]

            K_dims = (dims[i], rks[i], dims[i+1], rks[i+2])
            K  = zeros(T, prod(K_dims), prod(K_dims))
            Kr = reshape(K, (K_dims..., K_dims...))
            @tensor Kr[a, b, c, q, e, f, g, h] = Gi[a, b, e, f, z] * Hi[z, c, q, g, h]
            Pb = zeros(T, K_dims)
            @tensor Pb[a, b, c, q] = G_bi[a, b, z] * H_bi[z, c, q]

            V = reshape(K \ Pb[:], K_dims)
            u = left_core_move_mals(u, i, V, 0.0, max_bond)

            if i > 1
                rks   = u.ttv_rks
                Him   = @view H[i-1][:, :, 1:rks[i+1], :, 1:rks[i+1]]
                H_bim = @view H_b[i-1][:, :, 1:rks[i+1]]
                updateH_mals!(u.ttv_vec[i+1], A_eff.tto_vec[i], Hi, Him)
                updateHb_mals!(u.ttv_vec[i+1], rhs.ttv_vec[i], H_bi, H_bim)
            end
        end

        rel_diff = norm(u - u_old) / (norm(u) + eps(real(T)))
        verbose && println("    Picard $iter  rel_diff = $(round(rel_diff, sigdigits=4))")
        rel_diff < scf_tol && break
    end
    return u
end

"""
    burgers_scf_mals(u₀, D_x, D_xx, ν, dt, n_steps; ...)

Time-integrate the viscous Burgers equation for `n_steps` implicit-Euler steps
using `burgers_scf_mals_step` (rank-adaptive two-site ALS) at each step.

# Arguments
- `u₀      :: TTvector{T}`   — initial condition
- `D_x     :: TToperator{T}` — first-derivative operator (scaled by 1/dx)
- `D_xx    :: TToperator{T}` — second-derivative operator (scaled by 1/dx²)
- `ν       :: Real`          — viscosity
- `dt      :: Real`          — time step size
- `n_steps :: Int`           — number of time steps

# Keyword arguments
Forwarded to `burgers_scf_mals_step`: `max_scf`, `scf_tol`, `max_bond`, `verbose`.
- `verbose_steps :: Bool = false` — print max rank after each time step
"""
function burgers_scf_mals(
        u₀           :: TTvector{T},
        D_x          :: TToperator{T},
        D_xx         :: TToperator{T},
        ν            :: Real,
        dt           :: Real,
        n_steps      :: Int;
        max_scf      :: Int  = 10,
        scf_tol      :: Real = 1e-8,
        max_bond     :: Int  = 20,
        verbose      :: Bool = false,
        verbose_steps:: Bool = false
    ) where {T <: Number}

    u = u₀
    for step in 1:n_steps
        u = burgers_scf_mals_step(u, D_x, D_xx, ν, dt;
                max_scf = max_scf, scf_tol = scf_tol,
                max_bond = max_bond, verbose = verbose)
        verbose_steps &&
            println("  step $step / $n_steps  max_rank = $(maximum(u.ttv_rks))")
    end
    return u
end

# ─── Allen-Cahn ──────────────────────────────────────────────────────────────
#
# PDE:  ∂_t u = ε²·∂_xx u + u - u³
#
# Implicit Euler + Picard linearisation (freeze u² in the cubic term):
#
#   [(1/dt - 1)·I + ε²·D_xx + diag((u^k)²)] · u^{k+1} = u^{prev}/dt
#
# where D_xx = (1/dx²)·Δ_DN is the discrete negative Laplacian.
# The effective operator is symmetric positive semi-definite for dt < 1,
# so standard `als_linsolve` / `mals_linsolve` (which uses Hermitian(K))
# are both correct here — no special non-symmetric treatment needed.

"""
    allen_cahn_als_step(u_prev, D_xx, ε, dt; ...)

One implicit-Euler time step for the Allen-Cahn equation

    ∂_t u = ε²·∂_xx u + u - u³

via Picard (SCF) linearization: freeze u² in the cubic term, giving

    [(1/dt - 1)·I + ε²·D_xx + diag((u^k)²)] · u^{k+1} = u^{prev}/dt

`D_xx` should be the negative Laplacian (positive eigenvalues, e.g.
`(1/dx²) * Δ_DN(d)`).  The effective operator is symmetric, so 1-site ALS
solves it correctly at fixed rank.

# Arguments
- `u_prev :: TTvector{T}`   — solution at the previous time step
- `D_xx   :: TToperator{T}` — second-derivative operator (negative Laplacian, scaled by 1/dx²)
- `ε      :: Real`          — interface width parameter
- `dt     :: Real`          — time step

# Keyword arguments
- `max_scf   :: Int  = 10`   — maximum Picard iterations per step
- `scf_tol   :: Real = 1e-8` — relative convergence: ‖u^{k+1} − u^k‖ / ‖u^{k+1}‖
- `max_bond  :: Int  = 20`   — TT bond dimension cap after compression
- `als_sweeps:: Int  = 5`    — ALS sweeps for the inner linear solve
- `verbose   :: Bool = false` — print per-Picard-iteration residual
"""
function allen_cahn_als_step(
        u_prev    :: TTvector{T},
        D_xx      :: TToperator{T},
        ε         :: Real,
        dt        :: Real;
        max_scf   :: Int  = 10,
        scf_tol   :: Real = 1e-8,
        max_bond  :: Int  = 20,
        als_sweeps:: Int  = 5,
        verbose   :: Bool = false
    ) where {T <: Number}

    d     = u_prev.N
    I_tto = id_tto(T, d; n_dim = 2)
    invdt = convert(T, inv(dt))
    εT    = convert(T, ε)
    rhs   = invdt * u_prev

    u = u_prev
    for iter in 1:max_scf
        u_old    = u
        A_react  = ttv_to_diag_tto(hadamard(u, u))         # diag(u²)
        A_eff    = (invdt - one(T)) * I_tto + εT^2 * D_xx + A_react
        u        = als_linsolve(A_eff, rhs, u_old; sweep_count = als_sweeps)
        u        = tt_compress!(u, max_bond)
        rel_diff = norm(u - u_old) / (norm(u) + eps(real(T)))
        verbose && println("    Picard $iter  rel_diff = $(round(rel_diff, sigdigits=4))")
        rel_diff < scf_tol && break
    end
    return u
end

"""
    allen_cahn_als(u₀, D_xx, ε, dt, n_steps; ...)

Time-integrate the Allen-Cahn equation for `n_steps` implicit-Euler steps
using `allen_cahn_als_step` (1-site ALS, fixed rank) at each step.

# Arguments
- `u₀      :: TTvector{T}`   — initial condition
- `D_xx    :: TToperator{T}` — negative Laplacian operator (scaled by 1/dx²)
- `ε       :: Real`          — interface width parameter
- `dt      :: Real`          — time step size
- `n_steps :: Int`           — number of time steps

# Keyword arguments
Forwarded to `allen_cahn_als_step`: `max_scf`, `scf_tol`, `max_bond`, `als_sweeps`, `verbose`.
- `verbose_steps :: Bool = false` — print max rank after each time step

# Returns
`Vector{TTvector{T}}` of length `n_steps + 1`: entry `k` is the solution at
time `(k-1)*dt`, with `sol[1] == u₀` and `sol[end]` the solution at `T = n_steps*dt`.
"""
function allen_cahn_als(
        u₀           :: TTvector{T},
        D_xx         :: TToperator{T},
        ε            :: Real,
        dt           :: Real,
        n_steps      :: Int;
        max_scf      :: Int  = 10,
        scf_tol      :: Real = 1e-8,
        max_bond     :: Int  = 20,
        als_sweeps   :: Int  = 5,
        verbose      :: Bool = false,
        verbose_steps:: Bool = false
    ) where {T <: Number}

    u         = u₀
    snapshots = TTvector{T}[u₀]
    for step in 1:n_steps
        u = allen_cahn_als_step(u, D_xx, ε, dt;
                max_scf = max_scf, scf_tol = scf_tol,
                max_bond = max_bond, als_sweeps = als_sweeps,
                verbose = verbose)
        push!(snapshots, u)
        verbose_steps &&
            println("  step $step / $n_steps  max_rank = $(maximum(u.ttv_rks))")
    end
    return snapshots
end

"""
    allen_cahn_mals_step(u_prev, D_xx, ε, dt; ...)

One implicit-Euler time step for the Allen-Cahn equation via Picard (SCF)
linearization with a rank-adaptive two-site ALS sweep.

Identical to `allen_cahn_als_step` but uses `mals_linsolve` for rank-adaptive
2-site local solves.  Because the Allen-Cahn effective operator is symmetric
(diffusion + diagonal reaction), `mals_linsolve` (which wraps the local
matrix in `Hermitian`) is correct here — unlike the non-symmetric Burgers case.

# Keyword arguments
- `max_scf  :: Int  = 10`
- `scf_tol  :: Real = 1e-8`
- `max_bond :: Int  = 20`  — passed as `rmax` to `mals_linsolve`
- `verbose  :: Bool = false`
"""
function allen_cahn_mals_step(
        u_prev   :: TTvector{T},
        D_xx     :: TToperator{T},
        ε        :: Real,
        dt       :: Real;
        max_scf  :: Int  = 10,
        scf_tol  :: Real = 1e-8,
        max_bond :: Int  = 20,
        verbose  :: Bool = false
    ) where {T <: Number}

    d     = u_prev.N
    I_tto = id_tto(T, d; n_dim = 2)
    invdt = convert(T, inv(dt))
    εT    = convert(T, ε)
    rhs   = invdt * u_prev

    u = orthogonalize(u_prev)
    for iter in 1:max_scf
        u_old    = u
        A_react  = ttv_to_diag_tto(hadamard(u, u))
        A_eff    = (invdt - one(T)) * I_tto + εT^2 * D_xx + A_react
        u        = mals_linsolve(A_eff, rhs, u_old; rmax = max_bond)
        rel_diff = norm(u - u_old) / (norm(u) + eps(real(T)))
        verbose && println("    Picard $iter  rel_diff = $(round(rel_diff, sigdigits=4))")
        rel_diff < scf_tol && break
    end
    return u
end

"""
    allen_cahn_mals(u₀, D_xx, ε, dt, n_steps; ...)

Time-integrate the Allen-Cahn equation for `n_steps` implicit-Euler steps
using `allen_cahn_mals_step` (rank-adaptive two-site ALS) at each step.

# Arguments
- `u₀      :: TTvector{T}`   — initial condition
- `D_xx    :: TToperator{T}` — negative Laplacian operator (scaled by 1/dx²)
- `ε       :: Real`          — interface width parameter
- `dt      :: Real`          — time step size
- `n_steps :: Int`           — number of time steps

# Keyword arguments
Forwarded to `allen_cahn_mals_step`: `max_scf`, `scf_tol`, `max_bond`, `verbose`.
- `verbose_steps :: Bool = false` — print max rank after each time step

# Returns
`Vector{TTvector{T}}` of length `n_steps + 1`: entry `k` is the solution at
time `(k-1)*dt`, with `sol[1] == u₀` and `sol[end]` the solution at `T = n_steps*dt`.
"""
function allen_cahn_mals(
        u₀           :: TTvector{T},
        D_xx         :: TToperator{T},
        ε            :: Real,
        dt           :: Real,
        n_steps      :: Int;
        max_scf      :: Int  = 10,
        scf_tol      :: Real = 1e-8,
        max_bond     :: Int  = 20,
        verbose      :: Bool = false,
        verbose_steps:: Bool = false
    ) where {T <: Number}

    u         = u₀
    snapshots = TTvector{T}[u₀]
    for step in 1:n_steps
        u = allen_cahn_mals_step(u, D_xx, ε, dt;
                max_scf = max_scf, scf_tol = scf_tol,
                max_bond = max_bond, verbose = verbose)
        push!(snapshots, u)
        verbose_steps &&
            println("  step $step / $n_steps  max_rank = $(maximum(u.ttv_rks))")
    end
    return snapshots
end
