# Eikonal equation solver via viscous regularization + Newton-SCF-ALS/MALS
#
# PDE:  |∇u|² = 1   (unit wave speed, c = 1)
# Viscous form:  -ε²∆u + |∇u|² = 1,   u = 0 on ∂Ω
#
# Exact (ε→0) solutions (viscosity solutions):
#   1D: u(x)   = min(x, 1-x)            (tent, max 0.5)
#   2D: u(x,y) = min(x,y,1-x,1-y)      (pyramid, max 0.5)
#
# Newton SCF linearization:
#   Given u^k, define F(u) = (ε²/h²)Δu + |∇u|² - 1.
#   The Newton step is F'(u^k) u^{k+1} = ones + |∇u^k|²
#   where F'(u^k) = (ε²/h²)Δ + 2·diag(∇u^k)·∇
#
# Stability constraint: requires h < ε (grid must resolve the viscous layer).
# ε-continuation: warm-starts from a large ε_start and decreases in n_cont
# log-spaced steps to the target ε, circumventing large-step Newton divergence.
#
# Operator conventions (d = QTT bits per dimension, N = 2^d, h = 1/(N+1)):
#   Δ(d)    — negative Laplacian stencil [2,-1,-1], NO h² scaling
#   ∇_c(d)  — centered difference S₊₁ - S₋₁, NO 1/(2h) scaling

function _nonsymmetric_two_site_linsolve(
        Gi::AbstractArray{T, 5}, Hi::AbstractArray{T, 5},
        G_bi::AbstractArray{T, 3}, H_bi::AbstractArray{T, 3}
    ) where {T <: Number}
    K_dims = (size(Gi, 1), size(Gi, 2), size(Hi, 2), size(Hi, 3))
    K = zeros(T, prod(K_dims), prod(K_dims))
    Kr = reshape(K, (K_dims..., K_dims...))
    @tensor Kr[a, b, c, q, e, f, g, h] = Gi[a, b, e, f, z] * Hi[z, c, q, g, h]
    Pb = zeros(T, K_dims)
    @tensor Pb[a, b, c, q] = G_bi[a, b, z] * H_bi[z, c, q]
    return reshape(K \ Pb[:], K_dims)
end

function _nonsymmetric_mals_linsolve(
        A::AbstractTToperator,
        b::AbstractTTvector,
        tt_start::AbstractTTvector;
        max_bond::Int = 20,
        tol::Float64 = 0.0
    )
    d = b.N
    @assert d >= 2 "MALS linear solve requires at least two TT cores"

    T = eltype(tt_start)
    tt_opt = orthogonalize(tt_start)
    dims = tt_start.ttv_dims
    A_rks = A.tto_rks
    b_rks = b.ttv_rks

    G = Array{Array{T, 5}}(undef, d)
    G_b = Array{Array{T, 3}}(undef, d)
    for i in 1:d
        rmax_i = min(max_bond, prod(dims[1:(i - 1)]), prod(dims[i:end]))
        G[i] = zeros(T, dims[i], rmax_i, dims[i], rmax_i, A_rks[i + 1])
        G_b[i] = zeros(T, dims[i], rmax_i, b_rks[i + 1])
    end
    G[1][:, 1:1, :, 1:1, :] = reshape(A.tto_vec[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)
    G_b[1] = reshape(b.ttv_vec[1], dims[1], 1, :)

    H = init_H_mals(tt_opt, A, max_bond)
    H_b = init_Hb_mals(tt_opt, b, max_bond)

    for i in 1:(d - 1)
        rks = tt_opt.ttv_rks
        Gi = @view G[i][:, 1:rks[i], :, 1:rks[i], :]
        Hi = @view H[i][:, :, 1:rks[i + 2], :, 1:rks[i + 2]]
        G_bi = @view G_b[i][:, 1:rks[i], :]
        H_bi = @view H_b[i][:, :, 1:rks[i + 2]]

        V = _nonsymmetric_two_site_linsolve(Gi, Hi, G_bi, H_bi)
        tt_opt = right_core_move_mals(tt_opt, i, V, tol, max_bond)

        rks = tt_opt.ttv_rks
        Gip = @view G[i + 1][:, 1:rks[i + 1], :, 1:rks[i + 1], :]
        G_bip = @view G_b[i + 1][:, 1:rks[i + 1], :]
        update_G!(tt_opt.ttv_vec[i], A.tto_vec[i + 1], Gi, Gip)
        update_Gb!(tt_opt.ttv_vec[i], b.ttv_vec[i + 1], G_bi, G_bip)
    end

    for i in (d - 1):-1:1
        rks = tt_opt.ttv_rks
        Gi = @view G[i][:, 1:rks[i], :, 1:rks[i], :]
        Hi = @view H[i][:, :, 1:rks[i + 2], :, 1:rks[i + 2]]
        G_bi = @view G_b[i][:, 1:rks[i], :]
        H_bi = @view H_b[i][:, :, 1:rks[i + 2]]

        V = _nonsymmetric_two_site_linsolve(Gi, Hi, G_bi, H_bi)
        tt_opt = left_core_move_mals(tt_opt, i, V, tol, max_bond)

        if i > 1
            rks = tt_opt.ttv_rks
            Him = @view H[i - 1][:, :, 1:rks[i + 1], :, 1:rks[i + 1]]
            H_bim = @view H_b[i - 1][:, :, 1:rks[i + 1]]
            updateH_mals!(tt_opt.ttv_vec[i + 1], A.tto_vec[i], Hi, Him)
            updateHb_mals!(tt_opt.ttv_vec[i + 1], b.ttv_vec[i], H_bi, H_bim)
        end
    end

    return tt_opt
end

function _eikonal_inner_linsolve(
        A::AbstractTToperator,
        f::AbstractTTvector,
        u_old::AbstractTTvector;
        inner_solver::Symbol,
        max_bond::Int,
        als_sweeps::Int,
        mals_tol::Real
    )
    if inner_solver == :als
        u = als_linsolve(A, f, u_old; sweep_count = als_sweeps)
        return tt_compress!(u, max_bond)
    elseif inner_solver == :mals
        return _nonsymmetric_mals_linsolve(A, f, u_old;
            max_bond = max_bond,
            tol = Float64(mals_tol)
        )
    else
        throw(ArgumentError("unknown Eikonal inner solver $(repr(inner_solver)); use :als or :mals"))
    end
end

"""
    _eikonal_scf_1d(d; inner_solver, ε, ε_start, n_cont, max_scf, scf_tol, max_bond, als_sweeps, mals_tol, verbose)

Solve the 1D viscous Eikonal equation on [0,1]:

    -ε² u'' + (u')² = 1,   u(0) = u(1) = 0

via Newton-SCF with ε-continuation. The frozen Newton systems are solved by
the selected `inner_solver` (`:als` or `:mals`). The ε→0 solution is
`u(x) = min(x, 1-x)`.

# Arguments
- `d :: Int` — grid size N = 2^d (interior points), h = 1/(N+1)

# Keyword arguments
- `ε          :: Float64 = 5e-3` — target regularization; O(ε) accuracy
- `ε_start    :: Float64 = 0.1`  — initial (large) ε for continuation warm-start
- `n_cont     :: Int     = 5`    — number of continuation steps from ε_start to ε
- `max_scf    :: Int     = 20`   — maximum Newton iterations per continuation level
- `scf_tol    :: Real    = 1e-7` — relative convergence: ‖u^{k+1} - u^k‖/‖u^k‖
- `max_bond   :: Int     = 20`   — bond dimension cap after each inner solve
- `als_sweeps :: Int     = 3`    — ALS sweeps per inner linear solve
- `verbose    :: Bool    = false`

# Returns
`(u, resid_hist)` where `u :: TTvector` approximates `min(x,1-x)`
and `resid_hist[k]` is the final relative change at each continuation level.

# Stability note
Requires h = 1/(2^d+1) < ε. For ε = 5e-3 use d ≤ 8; for ε = 0.01 use d ≤ 7.
"""
function _eikonal_scf_1d(
        d :: Int;
        inner_solver::Symbol = :als,
        ε          :: Float64 = 5e-3,
        ε_start    :: Float64 = 0.1,
        n_cont     :: Int     = 5,
        max_scf    :: Int     = 20,
        scf_tol    :: Real    = 1e-7,
        max_bond   :: Int     = 20,
        als_sweeps :: Int     = 3,
        mals_tol   :: Real    = 0.0,
        verbose    :: Bool    = false
    )
    h  = 1.0 / (2^d + 1)
    Dx = (1.0 / (2h)) * ∇_c(d)

    ε_vals = exp.(range(log(ε_start), log(ε); length = max(2, n_cont)))

    # Initial guess: sin(πx)/2 approximates the tent, bounded gradient ≤ π/2
    u = function_to_qtt(x -> sin(π * x) / 2, d)

    resid_hist = Float64[]
    for ε_k in ε_vals
        A_lap = (ε_k^2 / h^2) * Δ(d)
        verbose && println("  ε = $ε_k:")
        for iter in 1:max_scf
            u_old  = u
            grad_u = Dx * u
            J_nl   = 2 * ttv_to_diag_tto(grad_u) * Dx
            A      = A_lap + J_nl
            f      = ones_tt(2, d) + hadamard(grad_u, grad_u)
            u      = _eikonal_inner_linsolve(A, f, u_old;
                inner_solver = inner_solver,
                max_bond = max_bond,
                als_sweeps = als_sweeps,
                mals_tol = mals_tol)
            rel    = norm(u - u_old) / (norm(u) + eps(Float64))
            verbose && println("    iter $iter  rel = $(round(rel, sigdigits = 4))")
            push!(resid_hist, rel)
            rel < scf_tol && break
        end
    end
    return u, resid_hist
end

"""
    eikonal_als_1d(d; ε, ε_start, n_cont, max_scf, scf_tol, max_bond, als_sweeps, verbose)

Fixed-rank ALS variant of the 1D Newton-SCF viscous Eikonal solver.
"""
function eikonal_als_1d(d::Int; kwargs...)
    return _eikonal_scf_1d(d; inner_solver = :als, kwargs...)
end

"""
    eikonal_mals_1d(d; ε, ε_start, n_cont, max_scf, scf_tol, max_bond, mals_tol, verbose)

Rank-adaptive two-site MALS variant of `eikonal_als_1d`. The nonlinear Newton
linearization is unchanged, but each frozen linear system is solved with a
non-Hermitian two-site MALS sweep capped by `max_bond`.
"""
function eikonal_mals_1d(d::Int; kwargs...)
    return _eikonal_scf_1d(d; inner_solver = :mals, kwargs...)
end

"""
    _eikonal_scf_2d(d; inner_solver, ε, ε_start, n_cont, max_scf, scf_tol, max_bond, als_sweeps, mals_tol, verbose)

Solve the 2D viscous Eikonal equation on [0,1]²:

    -ε² ∆u + |∇u|² = 1,   u = 0 on ∂[0,1]²

via Newton-SCF with ε-continuation on a (2^d)×(2^d)-point grid stored as a
2d-mode serial QTT (first d modes = x, last d modes = y). The frozen Newton
systems are solved by the selected `inner_solver` (`:als` or `:mals`).

The ε→0 solution is the pyramid `u(x,y) = min(x,y,1-x,1-y)`.

# Arguments
- `d :: Int` — grid size N×N with N = 2^d, h = 1/(N+1)

# Keyword arguments
- `ε          :: Float64 = 0.04` — target regularization; O(ε) accuracy
- `ε_start    :: Float64 = 0.1`
- `n_cont     :: Int     = 4`    — continuation steps from ε_start to ε
- `max_scf    :: Int     = 20`
- `scf_tol    :: Real    = 1e-7`
- `max_bond   :: Int     = 24`
- `als_sweeps :: Int     = 3`
- `verbose    :: Bool    = false`

# Returns
`(u, resid_hist)` where `u :: TTvector` (2d modes) approximates the pyramid.
Reshape: `permutedims(reshape(real.(qtt_to_function(u)), N, N))` → matrix [y,x].

# Stability note
Requires h = 1/(2^d+1) < ε. For ε = 0.04 use d ≤ 5; for ε = 0.02 use d ≤ 5 (marginal).
"""
function _eikonal_scf_2d(
        d :: Int;
        inner_solver::Symbol = :als,
        ε          :: Float64 = 0.04,
        ε_start    :: Float64 = 0.1,
        n_cont     :: Int     = 4,
        max_scf    :: Int     = 20,
        scf_tol    :: Real    = 1e-7,
        max_bond   :: Int     = 24,
        als_sweeps :: Int     = 3,
        mals_tol   :: Real    = 0.0,
        verbose    :: Bool    = false
    )
    h   = 1.0 / (2^d + 1)
    I_d = id_tto(d)
    Dx  = (1.0 / (2h)) * (∇_c(d) ⊗ I_d)
    Dy  = (1.0 / (2h)) * (I_d ⊗ ∇_c(d))

    ε_vals = exp.(range(log(ε_start), log(ε); length = max(2, n_cont)))

    # Rank-1 separable initial guess: sin(πx)·sin(πy)/2
    u = function_to_qtt(x -> sin(π * x), d) ⊗ function_to_qtt(y -> sin(π * y) / 2, d)

    resid_hist = Float64[]
    for ε_k in ε_vals
        A_lap = (ε_k^2 / h^2) * (Δ(d) ⊗ I_d + I_d ⊗ Δ(d))
        verbose && println("  ε = $ε_k:")
        for iter in 1:max_scf
            u_old  = u
            gx     = Dx * u
            gy     = Dy * u
            J_nl   = 2 * ttv_to_diag_tto(gx) * Dx + 2 * ttv_to_diag_tto(gy) * Dy
            A      = A_lap + J_nl
            f      = ones_tt(2, 2d) + hadamard(gx, gx) + hadamard(gy, gy)
            u      = _eikonal_inner_linsolve(A, f, u_old;
                inner_solver = inner_solver,
                max_bond = max_bond,
                als_sweeps = als_sweeps,
                mals_tol = mals_tol)
            rel    = norm(u - u_old) / (norm(u) + eps(Float64))
            verbose && println("    iter $iter  rel = $(round(rel, sigdigits = 4))")
            push!(resid_hist, rel)
            rel < scf_tol && break
        end
    end
    return u, resid_hist
end

"""
    eikonal_als_2d(d; ε, ε_start, n_cont, max_scf, scf_tol, max_bond, als_sweeps, verbose)

Fixed-rank ALS variant of the 2D serial-QTT Newton-SCF viscous Eikonal solver.
"""
function eikonal_als_2d(d::Int; kwargs...)
    return _eikonal_scf_2d(d; inner_solver = :als, kwargs...)
end

"""
    eikonal_mals_2d(d; ε, ε_start, n_cont, max_scf, scf_tol, max_bond, mals_tol, verbose)

Rank-adaptive two-site MALS variant of `eikonal_als_2d` for the serial-QTT
2D viscous Eikonal problem.
"""
function eikonal_mals_2d(d::Int; kwargs...)
    return _eikonal_scf_2d(d; inner_solver = :mals, kwargs...)
end
