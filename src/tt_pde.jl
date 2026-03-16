# src/tt_pde.jl
# High-level PDE interface for TT/QTT — three layers:
#   1. QTTGrid  — discretization
#   2. Operator builders — diffusion, advection, reaction
#   3. Problem types + solve + PDESolution

# ─── Layer 1: QTTGrid ────────────────────────────────────────────────────────

struct QTTGrid{N}
    levels   :: Int
    domain   :: NTuple{N, Tuple{Float64, Float64}}
    bc       :: NTuple{N, Symbol}
    ordering :: Symbol
end

const _VALID_BC = (:dn, :nd, :nn, :periodic)

function QTTGrid(
        N        :: Int,
        levels   :: Int;
        domain          = ntuple(_ -> (0.0, 1.0), N),
        bc              = ntuple(_ -> :dn, N),
        ordering :: Symbol = :serial,
)
    N ≥ 1      || throw(ArgumentError("N must be ≥ 1, got $N"))
    levels ≥ 4 || throw(ArgumentError("levels must be ≥ 4, got $levels (required by Δ_DN/Δ_ND/Δ_NN/Δ_P)"))
    ordering == :serial || throw(ArgumentError("ordering=:$ordering is not yet supported; only :serial is valid"))
    for b in bc
        b ∈ _VALID_BC || throw(ArgumentError("bc entry :$b is invalid; must be one of $_VALID_BC"))
    end
    domain_t = ntuple(N) do i
        lo, hi = Float64(domain[i][1]), Float64(domain[i][2])
        lo < hi || throw(ArgumentError(
            "domain[$i]: lo must be strictly less than hi, got ($lo, $hi)"))
        (lo, hi)
    end
    bc_t     = ntuple(i -> bc[i], N)
    return QTTGrid{N}(levels, domain_t, bc_t, ordering)
end

npoints(g::QTTGrid)          = 2^g.levels
gridstep(g::QTTGrid, d::Int) = (g.domain[d][2] - g.domain[d][1]) / npoints(g)
nodes(g::QTTGrid, d::Int)    = collect(g.domain[d][1] .+ (0:npoints(g) - 1) .* gridstep(g, d))

# ─── Layer 2: Operator builders ──────────────────────────────────────────────

# Returns (1/h²) * Δ_XX(grid.levels) where h = gridstep(grid, d).
# Δ_DN/Δ_ND/Δ_NN/Δ_P all have diagonal +2 (positive-definite, discretising -∂²/∂x²).
function _laplacian_1d(grid::QTTGrid, d::Int)
    h  = gridstep(grid, d)
    L1 = if grid.bc[d] == :dn
        Δ_DN(grid.levels)
    elseif grid.bc[d] == :nd
        Δ_ND(grid.levels)
    elseif grid.bc[d] == :nn
        Δ_NN(grid.levels)
    elseif grid.bc[d] == :periodic
        Δ_P(grid.levels)
    else
        error("unreachable: unhandled BC symbol $(grid.bc[d])")
    end
    return (1.0 / h^2) * L1
end

# κ · Σ_d  (I ⊗ … ⊗ Δ_d ⊗ … ⊗ I)
function diffusion_operator(grid::QTTGrid{N}; κ::Real = 1.0) where {N}
    result = nothing
    for d in 1:N
        L_d     = _laplacian_1d(grid, d)
        factors = [k == d ? L_d : id_tto(grid.levels) for k in 1:N]
        op_d    = foldl(⊗, factors)
        result  = isnothing(result) ? op_d : result + op_d
    end
    return isone(κ) ? result : κ * result
end

laplacian(grid::QTTGrid) = diffusion_operator(grid; κ = 1.0)

# Σ_d  (v[d]/h_d) · (I ⊗ … ⊗ ∇(levels) ⊗ … ⊗ I)
function advection_operator(grid::QTTGrid{N}; v::AbstractVector{<:Real}) where {N}
    length(v) == N || throw(ArgumentError("v must have length $N, got $(length(v))"))
    result = nothing
    for d in 1:N
        iszero(v[d]) && continue
        h_d     = gridstep(grid, d)
        grad_d  = (v[d] / h_d) * ∇(grid.levels)
        factors = [k == d ? grad_d : id_tto(grid.levels) for k in 1:N]
        op_d    = foldl(⊗, factors)
        result  = isnothing(result) ? op_d : result + op_d
    end
    # all-zero v → zero operator expressed as 0·(I⊗…⊗I)
    if isnothing(result)
        result = 0 * foldl(⊗, [id_tto(grid.levels) for _ in 1:N])
    end
    return result
end

# σ · ⊗_{d=1}^{N} id_tto(levels)
function reaction_operator(grid::QTTGrid{N}; σ::Real = 0.0) where {N}
    return σ * foldl(⊗, [id_tto(grid.levels) for _ in 1:N])
end

# ─── Layer 3: Problem types, source, solve, PDESolution ──────────────────────

# ── Solver algorithm structs ──────────────────────────────────────────────────

struct ALSLinsolve
    maxiter :: Int     # → sweep_count in als_linsolve
end
ALSLinsolve(; maxiter = 100) = ALSLinsolve(maxiter)

struct MALSLinsolve
    tol  :: Float64    # → tol in mals_linsolve
    rmax :: Int        # → rmax in mals_linsolve
end
MALSLinsolve(; tol = 1e-10, rmax = 100) = MALSLinsolve(tol, rmax)

struct DMRGLinsolve
    maxiter :: Int     # → sweep_schedule = [maxiter]
    rmax    :: Int     # → rmax_schedule  = [rmax]
end
DMRGLinsolve(; maxiter = 100, rmax = 100) = DMRGLinsolve(maxiter, rmax)

struct TDVP1Solver
    imaginary_time :: Bool   # → imaginary_time in tdvp
    normalize      :: Bool   # → normalize in tdvp (false = preserve norm decay)
end
TDVP1Solver(; imaginary_time = true, normalize = false) = TDVP1Solver(imaginary_time, normalize)

struct TDVP2Solver
    truncerr       :: Float64   # → truncerr in tdvp2
    rmax           :: Int       # → max_bond in tdvp2
    imaginary_time :: Bool      # → imaginary_time in tdvp2
    normalize      :: Bool      # → normalize in tdvp2 (false = preserve norm decay)
end
TDVP2Solver(; truncerr = 1e-10, rmax = 100, imaginary_time = true, normalize = false) =
    TDVP2Solver(truncerr, rmax, imaginary_time, normalize)

# ── Problem types ─────────────────────────────────────────────────────────────

struct EllipticPDE
    operator :: TToperator   # A in Au = b  (positive-definite for Poisson with laplacian)
    rhs      :: TTvector     # b
    grid     :: QTTGrid
end

struct ParabolicPDE
    operator :: TToperator   # L in ∂ₜu = Lu  (solved via imaginary-time TDVP)
    u0       :: TTvector     # initial condition
    grid     :: QTTGrid
    tspan    :: Tuple{Float64, Float64}
    dt       :: Float64
end

# Keyword constructor for ergonomic use: ParabolicPDE(L, u0, grid; tspan=..., dt=...)
function ParabolicPDE(operator, u0, grid; tspan::Tuple{Float64, Float64}, dt::Float64)
    tspan[2] > tspan[1] || throw(ArgumentError("tspan must satisfy tspan[1] < tspan[2], got $tspan"))
    dt > 0              || throw(ArgumentError("dt must be positive, got $dt"))
    return ParabolicPDE(operator, u0, grid, tspan, dt)
end

# ── source ────────────────────────────────────────────────────────────────────

# Pointwise form: f(x::Vector{Float64}) → scalar.
# Uses a binary QTT domain ({0,1}^(N*levels)) so the result has N*levels
# binary sites — the same structure as all QTT operators from diffusion_operator etc.
#
# Binary domain {0,1}^(N*levels) is used (not nodes) so that the output
# has N*levels sites of dim 2 — compatible with QTT operators from
# diffusion_operator/laplacian which have the same structure.
#
# Encoding convention (must match qtt_to_function and reshape in to_array):
#   - qtt_to_function gives flat output m (0-based) where site l (1-indexed) contributes
#     bit 2^(N*L - l) to m (MSB-first: site 1 = most significant bit).
#   - dim d occupies sites (N-d)*L+1 : (N-d+1)*L so that dim 1 is in the
#     least-significant L bits.  This ensures column-major reshape of the flat vector
#     to (npoints,...,npoints) gives arr[i1,...,iN] = f(x1[i1], ..., xN[iN]).
#   - Within the L sites for dim d (at offset (N-d)*L), site l has weight 2^(L-l)
#     (MSB-first within the block), giving k_d = (m >> (N-d)*L) ... is simply
#     the index for dim d when read from flat in column-major order.
function source(grid::QTTGrid{N}, f::Function; alg::CrossAlgorithm = MaxVol()) where {N}
    L      = grid.levels
    lo_vec = [grid.domain[d][1] for d in 1:N]
    h_vec  = [gridstep(grid, d) for d in 1:N]
    binary_domain = [Float64[0, 1] for _ in 1:N * L]
    function f_batch(coords::AbstractMatrix)
        nrows = size(coords, 1)
        X = Matrix{Float64}(undef, nrows, N)
        for d in 1:N
            site_offset = (N - d) * L   # dim d occupies sites site_offset+1 : site_offset+L
            for i in 1:nrows
                k = 0
                for l in 1:L
                    k += round(Int, coords[i, site_offset + l]) * (1 << (L - l))
                end
                X[i, d] = lo_vec[d] + k * h_vec[d]
            end
        end
        return [f(X[i, :]) for i in 1:nrows]
    end
    return tt_cross(f_batch, binary_domain; alg = alg)
end

# Passthrough: return b unchanged (reference equality, no copy).
source(::QTTGrid, b::TTvector) = b

# ── PDESolution ───────────────────────────────────────────────────────────────

struct PDESolution{N}
    u    :: TTvector
    grid :: QTTGrid{N}
end

# Reconstruct solution as an N-dimensional array on the QTT grid.
# Uses real.() to handle internal complex promotion in tdvp/tdvp2.
#
# source() encodes dim d in sites (N-d)*L+1 : (N-d+1)*L so that dim 1 is in the
# least-significant L bits of the flat QTT index.  Column-major reshape therefore
# gives arr[i1,...,iN] = u(x_1[i1], ..., x_N[iN]) directly, with no extra permutation.
function to_array(sol::PDESolution{N}) where {N}
    n    = npoints(sol.grid)
    flat = real.(qtt_to_function(sol.u))
    return reshape(flat, ntuple(_ -> n, N))
end

nodes(sol::PDESolution, d::Int) = nodes(sol.grid, d)

# ── solve — Elliptic ──────────────────────────────────────────────────────────

function solve(prob::EllipticPDE, alg::ALSLinsolve)
    x0     = rand_tt(prob.rhs.ttv_dims, 2)
    result = als_linsolve(prob.operator, prob.rhs, x0; sweep_count = alg.maxiter)
    return PDESolution(result, prob.grid)
end

function solve(prob::EllipticPDE, alg::MALSLinsolve)
    x0     = rand_tt(prob.rhs.ttv_dims, 2)
    result = mals_linsolve(prob.operator, prob.rhs, x0; tol = alg.tol, rmax = alg.rmax)
    return PDESolution(result, prob.grid)
end

function solve(prob::EllipticPDE, alg::DMRGLinsolve)
    x0     = rand_tt(prob.rhs.ttv_dims, 2)
    result = dmrg_linsolve(
        prob.operator, prob.rhs, x0;
        sweep_schedule = [alg.maxiter],
        rmax_schedule  = [alg.rmax],
    )
    return PDESolution(result, prob.grid)
end

# ── solve — Parabolic ─────────────────────────────────────────────────────────

function solve(prob::ParabolicPDE, alg::TDVP1Solver)
    nsteps_exact = (prob.tspan[2] - prob.tspan[1]) / prob.dt
    nsteps = round(Int, nsteps_exact)
    isapprox(nsteps_exact, nsteps; rtol = 1e-6) ||
        @warn "tspan length is not an exact multiple of dt; integrating to $(nsteps * prob.dt) instead of $(prob.tspan[2] - prob.tspan[1])"
    steps  = fill(prob.dt, nsteps)
    # tdvp imaginary-time convention: dt_eff = +im*h gives propagator exp(+h*H).
    # For decay (∂ₜu = Lu, L > 0), we need exp(-h*L), so pass H = -L.
    H = alg.imaginary_time ? (-1) * prob.operator : prob.operator
    result = tdvp(
        H, prob.u0, steps;
        imaginary_time = alg.imaginary_time,
        normalize      = alg.normalize,
    )
    return PDESolution(result, prob.grid)
end

function solve(prob::ParabolicPDE, alg::TDVP2Solver)
    nsteps_exact = (prob.tspan[2] - prob.tspan[1]) / prob.dt
    nsteps = round(Int, nsteps_exact)
    isapprox(nsteps_exact, nsteps; rtol = 1e-6) ||
        @warn "tspan length is not an exact multiple of dt; integrating to $(nsteps * prob.dt) instead of $(prob.tspan[2] - prob.tspan[1])"
    steps  = fill(prob.dt, nsteps)
    # tdvp imaginary-time convention: dt_eff = +im*h gives propagator exp(+h*H).
    # For decay (∂ₜu = Lu, L > 0), we need exp(-h*L), so pass H = -L.
    H = alg.imaginary_time ? (-1) * prob.operator : prob.operator
    result = tdvp2(
        H, prob.u0, steps;
        truncerr       = alg.truncerr,
        max_bond       = alg.rmax,
        imaginary_time = alg.imaginary_time,
        normalize      = alg.normalize,
    )
    return PDESolution(result, prob.grid)
end
