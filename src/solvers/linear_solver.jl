using KrylovKit
using ProgressMeter

struct _RankBoundedTTvector{V <: AbstractTTvector}
    tt::V
    max_bond::Int
end

"""
    LinearSolverAlgorithm

Supertype of algorithm objects accepted by [`linear_solve`](@ref) and by the
implicit time steppers: [`ALS`](@ref), [`MALS`](@ref), [`DMRG`](@ref), and
[`Krylov`](@ref).
"""
abstract type LinearSolverAlgorithm end

"""
    EigenSolverAlgorithm <: LinearSolverAlgorithm

Supertype of algorithm objects that [`eigen_solve`](@ref) also accepts:
[`ALS`](@ref), [`MALS`](@ref), and [`DMRG`](@ref).
"""
abstract type EigenSolverAlgorithm <: LinearSolverAlgorithm end

"""
    ALS(; kwargs...)

Alternating Linear Scheme (Holtz, Rohwedder & Schneider 2012). Each micro-step
optimizes a single core with all other cores fixed, so the TT ranks stay equal
to those of the initial guess during a linear solve.

Pass to [`linear_solve`](@ref) to solve `A x = b`, or to [`eigen_solve`](@ref)
to find the smallest eigenpair by minimizing the Rayleigh quotient. The two
problems read different fields; setting a field that the chosen problem does not
read throws an `ArgumentError`.

# Keyword arguments used by `linear_solve`
- `sweep_count::Int=2`: number of half-sweeps, alternating left-to-right and
  right-to-left; the default is one full back-and-forth sweep.
- `it_solver::Bool=false`: solve local systems iteratively instead of densely.
- `r_itsolver::Int=5000`: local system size above which `it_solver` takes effect.
- `return_info::Bool=false`: return `(x, (; residual))`, where `residual` is
  `‖A x − b‖ / ‖b‖`, instead of `x`.

# Keyword arguments used by `eigen_solve`
- `sweep_schedule::Vector{Int}=[2]`: sweep numbers at which each rank stage ends;
  the solve stops after `sweep_schedule[end]` sweeps.
- `rmax_schedule::Vector{Int}`: maximum bond dimension for each stage (defaults
  to the largest rank of the initial guess).
- `noise_schedule::Vector{Float64}`: noise amplitude used to pad new rank
  directions at each stage (defaults to zero).
- `it_solver::Bool=false`: solve local eigenproblems iteratively.
- `itslv_thresh::Int=1024`: local problem size above which `it_solver` takes effect.

`eigen_solve` returns `(E, x)`, where `E` is the history of eigenvalue estimates
(one entry per micro-step) and `x` is the eigenvector approximation.

# Keyword arguments used by both
- `maxiter::Int=200`: maximum iterations of the local iterative solver.
- `linsolv_tol::Float64=1e-8`: tolerance of the local iterative solver.
- `show_progress::Bool=false`: display a progress bar.
"""
struct ALS <: EigenSolverAlgorithm
    sweep_count::Int
    it_solver::Bool
    r_itsolver::Int
    return_info::Bool
    sweep_schedule::Union{Nothing, Vector{Int}}
    rmax_schedule::Union{Nothing, Vector{Int}}
    noise_schedule::Union{Nothing, Vector{Float64}}
    itslv_thresh::Int
    maxiter::Int
    linsolv_tol::Float64
    show_progress::Bool
end

function ALS(;
        sweep_count::Int = 2,
        it_solver::Bool = false,
        r_itsolver::Int = 5000,
        return_info::Bool = false,
        sweep_schedule::Union{Nothing, Vector{Int}} = nothing,
        rmax_schedule::Union{Nothing, Vector{Int}} = nothing,
        noise_schedule::Union{Nothing, Vector{Float64}} = nothing,
        itslv_thresh::Int = 1024,
        maxiter::Int = 200,
        linsolv_tol::Float64 = 1.0e-8,
        show_progress::Bool = false
    )
    return ALS(
        sweep_count, it_solver, r_itsolver, return_info,
        sweep_schedule, rmax_schedule, noise_schedule,
        itslv_thresh, maxiter, linsolv_tol, show_progress
    )
end

"""
    MALS(; kwargs...)

Modified Alternating Linear Scheme (Holtz, Rohwedder & Schneider 2012). Each
micro-step optimizes two neighboring cores jointly and splits them with a
truncated SVD, so the TT ranks adapt during the solve.

Pass to [`linear_solve`](@ref) or [`eigen_solve`](@ref). Setting a field that
the chosen problem does not read throws an `ArgumentError`.

# Keyword arguments used by `linear_solve`
A linear solve performs one left-to-right and one right-to-left sweep.
- `tol::Float64=1e-12`: relative SVD truncation threshold for rank adaptation.
- `rmax::Union{Nothing,Int}=nothing`: maximum bond dimension; `nothing` means
  `round(Int, √prod(dims))`.
- `return_info::Bool=false`: return `(x, (; residual))`, where `residual` is
  `‖A x − b‖ / ‖b‖`, instead of `x`.

# Keyword arguments used by `eigen_solve`
- `tol::Float64=1e-12`: relative SVD truncation threshold for rank adaptation.
- `sweep_schedule::Vector{Int}=[2]`: sweep numbers at which each rank stage ends.
- `rmax_schedule::Vector{Int}`: maximum bond dimension for each stage. If it is
  not given, every stage uses `rmax`, or `round(Int, √prod(dims))` when `rmax`
  is also not given; setting both is an error.
- `it_solver::Bool=false`: solve local eigenproblems iteratively.
- `linsolv_maxiter::Int=200`: maximum iterations of the local iterative solver.
- `linsolv_tol`: tolerance of the local iterative solver; `nothing` means
  `max(√tol, 1e-8)`.
- `itslv_thresh::Int=256`: local problem size above which `it_solver` takes effect.

`eigen_solve` returns `(E, x, r_hist)`: the eigenvalue history, the eigenvector
approximation, and the maximum bond dimension after each micro-step.

`show_progress::Bool=false` displays a progress bar for either problem.
"""
struct MALS <: EigenSolverAlgorithm
    tol::Float64
    rmax::Union{Nothing, Int}
    return_info::Bool
    sweep_schedule::Union{Nothing, Vector{Int}}
    rmax_schedule::Union{Nothing, Vector{Int}}
    it_solver::Bool
    linsolv_maxiter::Int
    linsolv_tol::Union{Nothing, Float64}
    itslv_thresh::Int
    show_progress::Bool
end

function MALS(;
        tol::Float64 = 1.0e-12,
        rmax::Union{Nothing, Int} = nothing,
        return_info::Bool = false,
        sweep_schedule::Union{Nothing, Vector{Int}} = nothing,
        rmax_schedule::Union{Nothing, Vector{Int}} = nothing,
        it_solver::Bool = false,
        linsolv_maxiter::Int = 200,
        linsolv_tol::Union{Nothing, Real} = nothing,
        itslv_thresh::Int = 256,
        show_progress::Bool = false
    )
    linsolv_tol_value = isnothing(linsolv_tol) ? nothing : Float64(linsolv_tol)
    return MALS(tol, rmax, return_info, sweep_schedule, rmax_schedule, it_solver, linsolv_maxiter, linsolv_tol_value, itslv_thresh, show_progress)
end

"""
    DMRG(; kwargs...)

DMRG-style alternating scheme (White 1992; Oseledets & Dolgov 2012) that
optimizes `N` neighboring cores jointly at each micro-step. With `N ≥ 2` the
ranks adapt through truncated SVDs.

Pass to [`linear_solve`](@ref) or [`eigen_solve`](@ref). Both read the same
fields, except that `eigen_solve` rejects `return_info = true`:

# Keyword arguments
- `N::Int=2`: number of cores optimized per micro-step.
- `tol::Float64=1e-12`: relative SVD truncation threshold (used when `N ≥ 2`).
- `sweep_schedule::Vector{Int}=[sweep_count]`: sweep numbers at which each rank
  stage ends; the solve stops after `sweep_schedule[end]` sweeps.
- `sweep_count::Int=2`: shorthand for the one-stage schedule `[sweep_count]`;
  setting both `sweep_count` and `sweep_schedule` is an error.
- `rmax_schedule::Vector{Int}`: maximum bond dimension for each stage (defaults
  to `isqrt(prod(dims))`).
- `it_solver::Bool=true`: solve local problems iteratively.
- `linsolv_maxiter::Int=200`: maximum iterations of the local iterative solver.
- `linsolv_tol`: tolerance of the local iterative solver; `nothing` means
  `max(√tol, 1e-8)`.
- `itslv_thresh::Int=256`: local problem size above which `it_solver` takes effect.
- `return_info::Bool=false`: for `linear_solve`, return `(x, (; residual))`
  instead of `x`.
- `verbose::Bool=false`: log ranks and discarded weight at every core move.
- `show_progress::Bool=false`: display a progress bar.

`eigen_solve` returns `(E, x, r_hist)`: the eigenvalue history, the eigenvector
approximation, and the maximum bond dimension after each micro-step.
"""
struct DMRG <: EigenSolverAlgorithm
    sweep_count::Int
    N::Int
    tol::Float64
    sweep_schedule::Union{Nothing, Vector{Int}}
    rmax_schedule::Union{Nothing, Vector{Int}}
    it_solver::Bool
    linsolv_maxiter::Int
    linsolv_tol::Union{Nothing, Float64}
    itslv_thresh::Int
    return_info::Bool
    verbose::Bool
    show_progress::Bool
end

function DMRG(;
        sweep_count::Int = 2,
        N::Int = 2,
        tol::Float64 = 1.0e-12,
        sweep_schedule::Union{Nothing, Vector{Int}} = nothing,
        rmax_schedule::Union{Nothing, Vector{Int}} = nothing,
        it_solver::Bool = true,
        linsolv_maxiter::Int = 200,
        linsolv_tol::Union{Nothing, Real} = nothing,
        itslv_thresh::Int = 256,
        return_info::Bool = false,
        verbose::Bool = false,
        show_progress::Bool = false
    )
    linsolv_tol_value = isnothing(linsolv_tol) ? nothing : Float64(linsolv_tol)
    return DMRG(sweep_count, N, tol, sweep_schedule, rmax_schedule, it_solver, linsolv_maxiter, linsolv_tol_value, itslv_thresh, return_info, verbose, show_progress)
end

"""
    Krylov(; kwargs...)

Solve `A x = b` with a KrylovKit.jl Krylov method whose vectors are
`TTvector`s. Pass to [`linear_solve`](@ref); not supported by `eigen_solve`.

With `max_bond > 0`, every operator application is followed by
[`tt_compress!`](@ref) to bond dimension `max_bond`, and the result is compressed
the same way. With `max_bond = 0` no truncation is applied, so ranks grow with
every iteration.

# Keyword arguments
- `max_bond::Int=0`: bond dimension cap applied after each operator application.
- `krylov_solver::Symbol=:auto`: `:gmres`, `:bicgstab`, `:cg`, or `:auto`. `:auto`
  selects `:cg` when `isposdef` and (`issymmetric` or `ishermitian`) are set,
  otherwise `:bicgstab` when `max_bond > 0` and `:gmres` when it is not.
- `krylovdim::Int=8`: Krylov subspace dimension (GMRES restart length).
- `maxiter::Int=20`: maximum number of iterations (restarts for GMRES; CG uses
  `krylovdim * maxiter` iterations).
- `rtol::Float64=1e-8`, `atol::Float64=1e-12`: the stopping tolerance is
  `max(atol, rtol * norm(b))` unless `tol` is given.
- `tol=nothing`: absolute stopping tolerance, overriding `rtol` and `atol`.
- `orth`: orthogonalization method passed to KrylovKit's GMRES.
- `issymmetric`, `ishermitian`, `isposdef`: properties of `A` used by `:auto`.
- `verbosity::Int=1`: KrylovKit verbosity level. At the default level KrylovKit
  warns when the solve stops without reaching the tolerance; `0` silences it.
- `return_info::Bool=false`: return `(x, info)` instead of `x`, where
  `info = (; converged, residual, numiter)` holds whether the tolerance was
  reached, the relative residual `‖A x − b‖ / ‖b‖` reported by KrylovKit (for the
  rank-truncated vectors when `max_bond > 0`), and the number of iterations.
- `show_progress::Bool=false`: display a progress bar.
"""
struct Krylov <: LinearSolverAlgorithm
    max_bond::Int
    krylov_solver::Symbol
    krylovdim::Int
    maxiter::Int
    rtol::Float64
    atol::Float64
    tol::Union{Nothing, Float64}
    orth::Any
    issymmetric::Bool
    ishermitian::Bool
    isposdef::Bool
    verbosity::Int
    return_info::Bool
    show_progress::Bool
end

function Krylov(;
        max_bond::Int = 0,
        krylov_solver::Symbol = :auto,
        krylovdim::Int = 8,
        maxiter::Int = 20,
        rtol::Float64 = 1.0e-8,
        atol::Float64 = 1.0e-12,
        tol::Union{Nothing, Real} = nothing,
        orth = KrylovKit.KrylovDefaults.orth,
        issymmetric::Bool = false,
        ishermitian::Union{Nothing, Bool} = nothing,
        isposdef::Bool = false,
        verbosity::Int = 1,
        return_info::Bool = false,
        show_progress::Bool = false
    )
    tol_value = isnothing(tol) ? nothing : Float64(tol)
    hermitian_value = isnothing(ishermitian) ? issymmetric : ishermitian
    return Krylov(max_bond, krylov_solver, krylovdim, maxiter, Float64(rtol), Float64(atol), tol_value, orth, issymmetric, hermitian_value, isposdef, verbosity, return_info, show_progress)
end

"Alias for [`LinearSolverAlgorithm`](@ref)."
const TTLinearSolver = LinearSolverAlgorithm
"Alias for [`ALS`](@ref)."
const ALSSolver = ALS
"Alias for [`MALS`](@ref)."
const MALSSolver = MALS
"Alias for [`DMRG`](@ref)."
const DMRGSolver = DMRG
"Alias for [`Krylov`](@ref)."
const KrylovSolver = Krylov

# Throw if any of `fields` of `alg` differs from its default: `problem` does not
# read those options, and `used` lists the ones it does.
function _reject_unused(alg, problem::AbstractString, fields, used::AbstractString)
    default = typeof(alg)()
    for f in fields
        isequal(getfield(alg, f), getfield(default, f)) && continue
        throw(ArgumentError("`$f` is not used by $problem with $(nameof(typeof(alg))); it uses $used"))
    end
    return nothing
end

_check_linear_options(::LinearSolverAlgorithm) = nothing
_check_linear_options(alg::ALS) = _reject_unused(
    alg, "linear_solve", (:sweep_schedule, :rmax_schedule, :noise_schedule, :itslv_thresh),
    "`sweep_count`, `it_solver`, `r_itsolver`, `maxiter`, `linsolv_tol`, `return_info`, and `show_progress`"
)
_check_linear_options(alg::MALS) = _reject_unused(
    alg, "linear_solve", (:sweep_schedule, :rmax_schedule, :it_solver, :linsolv_maxiter, :linsolv_tol, :itslv_thresh),
    "`tol`, `rmax`, `return_info`, and `show_progress`"
)
_check_linear_options(alg::DMRG) = (_dmrg_sweep_schedule(alg); nothing)

# `sweep_count` is shorthand for the one-stage schedule `[sweep_count]`.
function _dmrg_sweep_schedule(alg::DMRG)
    isnothing(alg.sweep_schedule) && return [alg.sweep_count]
    alg.sweep_count == DMRG().sweep_count ||
        throw(ArgumentError("DMRG: give either `sweep_count` or `sweep_schedule`, not both"))
    return alg.sweep_schedule
end

function _linear_solver_algorithm(name::AbstractString)
    name == "als" && return ALS()
    name == "mals" && return MALS()
    name == "dmrg" && return DMRG()
    name == "krylov" && return Krylov()
    throw(ArgumentError("Unknown TT solver: $name. Use \"als\", \"mals\", \"dmrg\", \"krylov\", or a LinearSolverAlgorithm instance."))
end

_solver_progress(total::Integer, show_progress::Bool; desc::AbstractString = "") = Progress(max(total, 1); desc = desc, enabled = show_progress)

"""
    linear_solve(A, b, guess; alg=MALS())
    linear_solve(A, b, guess, alg)

Solve `A * x = b` in TT format with the algorithm object `alg`.
"""
linear_solve(A, b, guess; alg::LinearSolverAlgorithm = MALS()) = linear_solve(A, b, guess, alg)

function linear_solve(A, b, guess, alg::Krylov)
    return krylov_linsolve(
        A, b, guess;
        max_bond = alg.max_bond,
        krylov_solver = alg.krylov_solver,
        krylovdim = alg.krylovdim,
        maxiter = alg.maxiter,
        rtol = alg.rtol,
        atol = alg.atol,
        tol = alg.tol,
        orth = alg.orth,
        issymmetric = alg.issymmetric,
        ishermitian = alg.ishermitian,
        isposdef = alg.isposdef,
        verbosity = alg.verbosity,
        return_info = alg.return_info,
        show_progress = alg.show_progress
    )
end

function linear_solve(A, b, guess, alg::ALS)
    _check_linear_options(alg)
    return _als_linsolve_impl(
        A, b, guess;
        sweep_count = alg.sweep_count,
        it_solver = alg.it_solver,
        r_itsolver = alg.r_itsolver,
        maxiter = alg.maxiter,
        linsolv_tol = alg.linsolv_tol,
        return_info = alg.return_info,
        show_progress = alg.show_progress
    )
end

"""
    als_linsolve(A, b, guess; kwargs...)

Compatibility wrapper for `linear_solve(A, b, guess, ALS(; kwargs...))`.
"""
function als_linsolve(A, b, guess; kwargs...)
    return linear_solve(A, b, guess, ALS(; kwargs...))
end

function linear_solve(A, b, guess, alg::MALS)
    _check_linear_options(alg)
    rmax = isnothing(alg.rmax) ? round(Int, sqrt(prod(guess.ttv_dims)::Int)) : alg.rmax
    return _mals_linsolve_impl(A, b, guess; tol = alg.tol, rmax = rmax, return_info = alg.return_info, show_progress = alg.show_progress)
end

"""
    mals_linsolve(A, b, guess; kwargs...)

Compatibility wrapper for `linear_solve(A, b, guess, MALS(; kwargs...))`.
"""
function mals_linsolve(A, b, guess; kwargs...)
    return linear_solve(A, b, guess, MALS(; kwargs...))
end

function linear_solve(A, b, guess, alg::DMRG)
    sweep_schedule = _dmrg_sweep_schedule(alg)
    rmax_schedule = isnothing(alg.rmax_schedule) ? [isqrt(prod(guess.ttv_dims)::Int)] : alg.rmax_schedule
    linsolv_tol = isnothing(alg.linsolv_tol) ? max(sqrt(alg.tol), 1.0e-8) : alg.linsolv_tol
    return _dmrg_linsolve_impl(
        A, b, guess;
        N = alg.N,
        tol = alg.tol,
        sweep_schedule = sweep_schedule,
        rmax_schedule = rmax_schedule,
        it_solver = alg.it_solver,
        linsolv_maxiter = alg.linsolv_maxiter,
        linsolv_tol = linsolv_tol,
        itslv_thresh = alg.itslv_thresh,
        return_info = alg.return_info,
        verbose = alg.verbose,
        show_progress = alg.show_progress
    )
end

"""
    dmrg_linsolve(A, b, guess; kwargs...)

Compatibility wrapper for `linear_solve(A, b, guess, DMRG(; kwargs...))`.
"""
function dmrg_linsolve(A, b, guess; kwargs...)
    return linear_solve(A, b, guess, DMRG(; kwargs...))
end

function _stepper_algorithm(
        alg::ALS;
        sweep_count::Int = alg.sweep_count,
        it_solver::Bool = alg.it_solver,
        r_itsolver::Int = alg.r_itsolver,
        maxiter::Int = alg.maxiter,
        linsolv_tol::Float64 = alg.linsolv_tol,
        return_info::Bool = alg.return_info
    )
    return ALS(;
        sweep_count = sweep_count,
        it_solver = it_solver,
        r_itsolver = r_itsolver,
        maxiter = maxiter,
        linsolv_tol = linsolv_tol,
        return_info = return_info,
        show_progress = false
    )
end

function _stepper_algorithm(
        alg::MALS;
        tol::Float64 = alg.tol,
        rmax::Union{Nothing, Int} = alg.rmax,
        return_info::Bool = alg.return_info
    )
    return MALS(;
        tol = tol,
        rmax = rmax,
        return_info = return_info,
        show_progress = false
    )
end

function _stepper_algorithm(
        alg::DMRG;
        sweep_count::Int = alg.sweep_count,
        N::Int = alg.N,
        tol::Float64 = alg.tol,
        sweep_schedule::Union{Nothing, Vector{Int}} = alg.sweep_schedule,
        rmax_schedule::Union{Nothing, Vector{Int}} = alg.rmax_schedule,
        it_solver::Bool = alg.it_solver,
        linsolv_maxiter::Int = alg.linsolv_maxiter,
        linsolv_tol::Union{Nothing, Real} = alg.linsolv_tol,
        itslv_thresh::Int = alg.itslv_thresh,
        return_info::Bool = alg.return_info,
        verbose::Bool = alg.verbose
    )
    return DMRG(;
        sweep_count = sweep_count,
        N = N,
        tol = tol,
        sweep_schedule = sweep_schedule,
        rmax_schedule = rmax_schedule,
        it_solver = it_solver,
        linsolv_maxiter = linsolv_maxiter,
        linsolv_tol = linsolv_tol,
        itslv_thresh = itslv_thresh,
        return_info = return_info,
        verbose = verbose,
        show_progress = false
    )
end

function _stepper_algorithm(
        alg::Krylov;
        max_bond::Int,
        krylov_solver::Symbol = alg.krylov_solver,
        krylovdim::Int = alg.krylovdim,
        maxiter::Int = alg.maxiter,
        rtol::Float64 = alg.rtol,
        atol::Float64 = alg.atol,
        tol::Union{Nothing, Real} = alg.tol,
        orth = alg.orth,
        issymmetric::Bool = alg.issymmetric,
        ishermitian::Union{Nothing, Bool} = alg.ishermitian,
        isposdef::Bool = alg.isposdef,
        verbosity::Int = alg.verbosity
    )
    return Krylov(;
        max_bond = max_bond,
        krylov_solver = krylov_solver,
        krylovdim = krylovdim,
        maxiter = maxiter,
        rtol = rtol,
        atol = atol,
        tol = tol,
        orth = orth,
        issymmetric = issymmetric,
        ishermitian = ishermitian,
        isposdef = isposdef,
        verbosity = verbosity,
        show_progress = false
    )
end

# The steppers rebuild `solver` without the options that `linear_solve` does not read,
# so those options are checked on the caller's object first.
function _stepper_linear_solve(A, b, guess, solver::Krylov; max_bond::Int, kwargs...)
    return linear_solve(A, b, guess, _stepper_algorithm(solver; max_bond = max_bond, kwargs...))
end
function _stepper_linear_solve(A, b, guess, solver::LinearSolverAlgorithm; max_bond::Int, kwargs...)
    _check_linear_options(solver)
    return linear_solve(A, b, guess, _stepper_algorithm(solver; kwargs...))
end

function _krylov_algorithm(
        krylov_solver::Symbol, max_bond::Int;
        krylovdim::Int,
        maxiter::Int,
        tol::Real,
        orth,
        verbosity::Int
    )
    solver = krylov_solver == :auto ? (max_bond > 0 ? :bicgstab : :gmres) : krylov_solver
    if solver == :bicgstab
        return KrylovKit.BiCGStab(; maxiter = maxiter, tol = tol, verbosity = verbosity)
    elseif solver == :gmres
        return KrylovKit.GMRES(;
            krylovdim = krylovdim,
            maxiter = maxiter,
            tol = tol,
            orth = orth,
            verbosity = verbosity
        )
    elseif solver == :cg
        return KrylovKit.CG(; maxiter = krylovdim * maxiter, tol = tol, verbosity = verbosity)
    end
    throw(ArgumentError("Unknown Krylov solver: $krylov_solver. Use :auto, :bicgstab, :cg, or :gmres."))
end

function krylov_linsolve(
        A::AbstractTToperator, b::AbstractTTvector, guess::AbstractTTvector;
        max_bond::Int = 0,
        krylov_solver::Symbol = :auto,
        krylovdim::Int = 8,
        maxiter::Int = 20,
        rtol::Real = 1.0e-8,
        atol::Real = 1.0e-12,
        tol::Union{Nothing, Real} = nothing,
        orth = KrylovKit.KrylovDefaults.orth,
        issymmetric::Bool = false,
        ishermitian::Bool = issymmetric,
        isposdef::Bool = false,
        verbosity::Int = 1,
        return_info::Bool = false,
        show_progress::Bool = false,
        kwargs...
    )
    solver = krylov_solver == :auto && isposdef && (issymmetric || ishermitian) ? :cg : krylov_solver
    tol_value = isnothing(tol) ? max(atol, rtol * norm(b)) : tol
    alg = _krylov_algorithm(
        solver, max_bond;
        krylovdim = krylovdim,
        maxiter = maxiter,
        tol = tol_value,
        orth = orth,
        verbosity = verbosity
    )
    progress = _solver_progress(1, show_progress; desc = "Krylov linear solve")
    if max_bond > 0
        op = function (x::_RankBoundedTTvector)
            y = tt_compress!(A * x.tt, x.max_bond)
            return _RankBoundedTTvector(y, x.max_bond)
        end
        xb, info = linsolve(
            op,
            _RankBoundedTTvector(b, max_bond),
            _RankBoundedTTvector(guess, max_bond),
            alg;
            kwargs...
        )
        x = tt_compress!(xb.tt, max_bond)
    else
        x, info = linsolve(x -> A * x, b, guess, alg; kwargs...)
    end
    next!(progress)
    return_info || return x
    return x, (; converged = info.converged > 0, residual = info.normres / norm(b), numiter = info.numiter)
end
