using KrylovKit
using ProgressMeter

struct _RankBoundedTTVector{V <: AbstractTTVector}
    tt::V
    max_bond::Int
end

# `x` as a `Vector{T}` if it is a vector of stage values, otherwise as a `T`.
_as_stage(x::AbstractVector, ::Type{T}) where {T} = collect(T, x)
_as_stage(x, ::Type{T}) where {T} = convert(T, x)

# Per-stage solver settings expanded to vectors of equal length; a scalar
# applies to every stage.
function _stages(; kwargs...)
    lens = [k => length(v) for (k, v) in pairs(kwargs) if v isa AbstractVector]
    n = isempty(lens) ? 1 : last(first(lens))
    n ≥ 1 || throw(ArgumentError("per-stage vectors must not be empty"))
    if !all(p -> last(p) == n, lens)
        throw(ArgumentError(join(("`$k` has length $l" for (k, l) in lens), " and ")))
    end
    stages = map(v -> v isa AbstractVector ? collect(v) : fill(v, n), values(kwargs))
    if haskey(stages, :max_sweeps) && any(<(1), stages.max_sweeps)
        throw(ArgumentError("`max_sweeps` entries must be ≥ 1; got $(stages.max_sweeps)"))
    end
    return stages
end

# Whether a local problem with `n` unknowns is solved iteratively.
_use_iterative(local_solver::Symbol, n::Integer, local_threshold::Integer) =
    local_solver === :iterative || (local_solver === :auto && n > local_threshold)

function _check_local_solver(local_solver::Symbol)
    local_solver in (:auto, :direct, :iterative) ||
        throw(ArgumentError("`local_solver` must be :auto, :direct, or :iterative; got :$local_solver"))
    return local_solver
end

"""
    LinearSolverAlgorithm

Supertype of algorithm objects accepted by [`linear_solve`](@ref) and by the
implicit time steppers: [`ALS`](@ref), [`MALS`](@ref), [`AMEn`](@ref),
[`DMRG`](@ref), and [`Krylov`](@ref).
"""
abstract type LinearSolverAlgorithm end

"""
    EigenSolverAlgorithm <: LinearSolverAlgorithm

Supertype of algorithm objects that [`eigen_solve`](@ref) also accepts:
[`ALS`](@ref), [`MALS`](@ref), [`AMEn`](@ref), and [`DMRG`](@ref).
"""
abstract type EigenSolverAlgorithm <: LinearSolverAlgorithm end

"""
    ALS(; kwargs...)

Alternating Linear Scheme (Holtz, Rohwedder & Schneider 2012). Each micro-step
optimizes a single core with all other cores fixed. One sweep optimizes the
cores left to right and then right to left.

Pass to [`linear_solve`](@ref) to solve `A x = b`, or to [`eigen_solve`](@ref)
to find the smallest eigenpair by minimizing the Rayleigh quotient. Setting a
field that the chosen problem does not read throws an `ArgumentError`.

# Keyword arguments
- `max_sweeps=1`: number of sweeps; ALS has no convergence test and runs all of
  them. For `eigen_solve`, a vector gives the sweeps of each rank stage.
- `max_bond=nothing` (`eigen_solve` only): largest bond dimension of each stage,
  an integer or a vector with one entry per stage; `nothing` keeps the largest
  rank of the initial guess. When a stage's `max_bond` exceeds the current
  largest rank, the ranks are raised with `increase_ranks`; a smaller
  value is an error because ALS cannot lower ranks. In `linear_solve` the ranks
  of the initial guess are kept.
- `noise=0.0` (`eigen_solve` only): noise amplitude for the new rank directions
  of each stage, a number or a per-stage vector.
- `local_solver=:auto`: `:direct`, `:iterative`, or `:auto` (direct up to
  `local_threshold` unknowns, iterative above).
- `local_threshold=nothing`: `nothing` means never iterative under `:auto`.
- `local_maxiter::Int=200`, `local_tol=1e-8`: limits of the local iterative solver.
- `return_info::Bool=false` (`linear_solve` only): return `(x, (; residual))`,
  where `residual` is `‖A x − b‖ / ‖b‖`, instead of `x`.
- `verbosity::Int=1`: `2` logs one line per sweep.
- `show_progress::Bool=true`: display a progress bar over the sweeps.

`eigen_solve` returns `(E, x)`, where `E` is the history of eigenvalue estimates
(one entry per micro-step) and `x` is the eigenvector approximation.
"""
struct ALS <: EigenSolverAlgorithm
    max_sweeps::Union{Int, Vector{Int}}
    max_bond::Union{Nothing, Int, Vector{Int}}
    noise::Union{Float64, Vector{Float64}}
    local_solver::Symbol
    local_threshold::Union{Nothing, Int}
    local_maxiter::Int
    local_tol::Float64
    return_info::Bool
    verbosity::Int
    show_progress::Bool
end

function ALS(;
        max_sweeps = 1,
        max_bond = nothing,
        noise = 0.0,
        local_solver::Symbol = :auto,
        local_threshold = nothing,
        local_maxiter::Int = 200,
        local_tol::Real = 1.0e-8,
        return_info::Bool = false,
        verbosity::Int = 1,
        show_progress::Bool = true
    )
    return ALS(
        _as_stage(max_sweeps, Int), isnothing(max_bond) ? nothing : _as_stage(max_bond, Int),
        _as_stage(noise, Float64), _check_local_solver(local_solver), local_threshold,
        local_maxiter, local_tol, return_info, verbosity, show_progress
    )
end

"""
    MALS(; kwargs...)

Modified Alternating Linear Scheme (Holtz, Rohwedder & Schneider 2012). Each
micro-step optimizes two neighboring cores jointly and splits them with a
truncated SVD, so the TT ranks adapt during the solve. One sweep moves left to
right and then right to left.

Pass to [`linear_solve`](@ref) or [`eigen_solve`](@ref). Setting a field that
the chosen problem does not read throws an `ArgumentError`.

# Keyword arguments
- `max_sweeps=1`: number of sweeps; MALS has no convergence test and runs all of
  them. For `eigen_solve`, a vector gives the sweeps of each rank stage.
- `max_bond=nothing`: largest bond dimension, an integer or (for `eigen_solve`)
  a per-stage vector; `nothing` means `round(Int, √prod(dims))`.
- `trunc_tol=1e-6`: relative truncation tolerance of the two-core SVDs, with
  the rule of [`tt_round!`](@ref).
- `local_solver=:auto`, `local_threshold=nothing` (256 unknowns),
  `local_maxiter=200`, `local_tol=1e-6` (`eigen_solve` only): local
  eigensolver settings; see [`ALS`](@ref). `linear_solve` solves local systems
  with a dense direct solve.
- `return_info::Bool=false` (`linear_solve` only): return `(x, (; residual))`,
  where `residual` is `‖A x − b‖ / ‖b‖`, instead of `x`.
- `verbosity::Int=1`: `2` logs one line per sweep.
- `show_progress::Bool=true`: display a progress bar over the sweeps.

`eigen_solve` returns `(E, x, r_hist)`: the eigenvalue history (one entry per
micro-step), the eigenvector approximation, and the largest bond dimension
after each micro-step.
"""
struct MALS <: EigenSolverAlgorithm
    max_sweeps::Union{Int, Vector{Int}}
    max_bond::Union{Nothing, Int, Vector{Int}}
    trunc_tol::Float64
    local_solver::Symbol
    local_threshold::Union{Nothing, Int}
    local_maxiter::Int
    local_tol::Float64
    return_info::Bool
    verbosity::Int
    show_progress::Bool
end

function MALS(;
        max_sweeps = 1,
        max_bond = nothing,
        trunc_tol::Real = 1.0e-6,
        local_solver::Symbol = :auto,
        local_threshold = nothing,
        local_maxiter::Int = 200,
        local_tol::Real = 1.0e-6,
        return_info::Bool = false,
        verbosity::Int = 1,
        show_progress::Bool = true
    )
    return MALS(
        _as_stage(max_sweeps, Int), isnothing(max_bond) ? nothing : _as_stage(max_bond, Int),
        trunc_tol, _check_local_solver(local_solver), local_threshold, local_maxiter,
        local_tol, return_info, verbosity, show_progress
    )
end

"""
    AMEn(; kwargs...)

Alternating minimal energy method (Dolgov & Savostyanov 2014). Each micro-step
optimizes a single core, as [`ALS`](@ref) does, and then enlarges the basis of
that core with an approximation of the residual. The approximation is a second
tensor train of rank `kickrank`. A truncated SVD after each local solve removes
directions that are not needed, so the TT ranks adapt during the solve without
the two-core local problems of [`MALS`](@ref) and [`DMRG`](@ref).

Pass to [`linear_solve`](@ref) to solve `A x = b`, or to [`eigen_solve`](@ref)
to find the smallest eigenpair of a Hermitian `A` (the eigenvalue variant
follows Kressner, Steinlechner & Uschmajew 2014).

Convergence of the linear solver is proven for Hermitian positive definite `A`.
For other operators the local systems are the Galerkin projections of `A`
itself, with no symmetrization; this works for many non-symmetric problems but
is not covered by the proof.

One sweep orthogonalizes the cores from right to left and then solves from left
to right. The solve stops when the largest relative residual of the local
problems in a sweep is at most `tol`; one further sweep without enrichment then
removes the directions added last.

The stopping test measures the residual within the current TT basis. The
enrichment brings the part of the residual outside that basis into the test, so
with `kickrank = 0`, or when `max_bond` limits the ranks, the test can be met
while `‖A x − b‖ / ‖b‖` is still above `tol`; `return_info = true` reports that
residual.

# Keyword arguments
- `tol=1e-6`: target relative residual of the local problems. It is the stopping
  threshold and the threshold of the rank truncation.
- `max_sweeps::Int=20`: largest number of sweeps.
- `max_bond=nothing`: largest bond dimension; `nothing` means no limit.
- `kickrank::Int=4`: rank of the residual approximation, which is the number of
  directions added to a bond per micro-step. `0` disables enrichment.
- `local_solver=:auto`, `local_threshold=nothing` (256 unknowns),
  `local_maxiter=200`: local solver settings; see [`ALS`](@ref). The iterative
  solver is GMRES for `linear_solve` and Lanczos for `eigen_solve`.
- `local_tol=nothing`: relative tolerance of the iterative local solver;
  `nothing` means `tol / 2`.
- `return_info::Bool=false` (`linear_solve` only): return
  `(x, (; residual, converged, sweeps))` instead of `x`, where `residual` is
  `‖A x − b‖ / ‖b‖`, `converged` tells whether the stopping test was met, and
  `sweeps` is the number of sweeps run.
- `verbosity::Int=1`: `0` silences the warning issued when `max_sweeps` is
  reached without meeting `tol`; `2` logs one line per sweep.
- `show_progress::Bool=true`: display a progress bar over the sweeps.

`eigen_solve` returns `(E, x, r_hist)`: the eigenvalue history (one entry per
micro-step), the eigenvector approximation, and the largest bond dimension
after each micro-step. Its residual is `‖A x − λ x‖` of the local problems for
unit-norm `x`, relative to the largest `‖A x‖` met during the solve.
"""
struct AMEn <: EigenSolverAlgorithm
    tol::Float64
    max_sweeps::Int
    max_bond::Union{Nothing, Int}
    kickrank::Int
    local_solver::Symbol
    local_threshold::Union{Nothing, Int}
    local_maxiter::Int
    local_tol::Union{Nothing, Float64}
    return_info::Bool
    verbosity::Int
    show_progress::Bool
end

function AMEn(;
        tol::Real = 1.0e-6,
        max_sweeps::Int = 20,
        max_bond::Union{Nothing, Int} = nothing,
        kickrank::Int = 4,
        local_solver::Symbol = :auto,
        local_threshold::Union{Nothing, Int} = nothing,
        local_maxiter::Int = 200,
        local_tol::Union{Nothing, Real} = nothing,
        return_info::Bool = false,
        verbosity::Int = 1,
        show_progress::Bool = true
    )
    tol ≥ 0 || throw(ArgumentError("`tol` must be ≥ 0; got $tol"))
    max_sweeps ≥ 1 || throw(ArgumentError("`max_sweeps` must be ≥ 1; got $max_sweeps"))
    kickrank ≥ 0 || throw(ArgumentError("`kickrank` must be ≥ 0; got $kickrank"))
    isnothing(max_bond) || max_bond ≥ 1 || throw(ArgumentError("`max_bond` must be ≥ 1; got $max_bond"))
    return AMEn(
        tol, max_sweeps, max_bond, kickrank, _check_local_solver(local_solver), local_threshold,
        local_maxiter, isnothing(local_tol) ? nothing : Float64(local_tol),
        return_info, verbosity, show_progress
    )
end

# Keyword arguments of `_amen_linsolve_impl` and `_amen_eigsolve_impl` for `alg`.
_amen_options(alg::AMEn) = (;
    alg.tol, alg.max_sweeps, max_bond = something(alg.max_bond, typemax(Int)), alg.kickrank,
    alg.local_solver, local_threshold = something(alg.local_threshold, 256), alg.local_maxiter,
    local_tol = something(alg.local_tol, alg.tol / 2), alg.verbosity, alg.show_progress,
)

"""
    DMRG(; kwargs...)

DMRG-style alternating scheme (White 1992; Oseledets & Dolgov 2012) that
optimizes `nsites` neighboring cores jointly at each micro-step. With
`nsites ≥ 2` the ranks adapt through truncated SVDs. One sweep moves left to
right and then right to left; a final micro-step on the first cores follows the
last sweep.

Pass to [`linear_solve`](@ref) or [`eigen_solve`](@ref). Both read the same
fields, except that `eigen_solve` rejects `return_info = true`.

# Keyword arguments
- `nsites::Int=2`: number of cores optimized per micro-step.
- `max_sweeps=1`: number of sweeps, or a vector with the sweeps of each rank
  stage; DMRG has no convergence test and runs all of them.
- `max_bond=nothing`: largest bond dimension, an integer or a per-stage vector;
  `nothing` means `isqrt(prod(dims))`.
- `trunc_tol=1e-12`: relative truncation tolerance of the SVDs, with the rule
  of [`tt_round!`](@ref) (used when `nsites ≥ 2`).
- `local_solver=:iterative`, `local_threshold=nothing` (256 unknowns),
  `local_maxiter=200`, `local_tol=1e-6`: local solver settings; see [`ALS`](@ref).
- `return_info::Bool=false`: for `linear_solve`, return `(x, (; residual))`,
  where `residual` is `‖A x − b‖ / ‖b‖`, instead of `x`.
- `verbosity::Int=1`: `2` logs one line per sweep, `3` also the rank and
  discarded weight at every core move.
- `show_progress::Bool=true`: display a progress bar over the sweeps.

`eigen_solve` returns `(E, x, r_hist)`: the eigenvalue history (one entry per
micro-step), the eigenvector approximation, and the largest bond dimension
after each micro-step.
"""
struct DMRG <: EigenSolverAlgorithm
    nsites::Int
    max_sweeps::Union{Int, Vector{Int}}
    max_bond::Union{Nothing, Int, Vector{Int}}
    trunc_tol::Float64
    local_solver::Symbol
    local_threshold::Union{Nothing, Int}
    local_maxiter::Int
    local_tol::Float64
    return_info::Bool
    verbosity::Int
    show_progress::Bool
end

function DMRG(;
        nsites::Int = 2,
        max_sweeps = 1,
        max_bond = nothing,
        trunc_tol::Real = 1.0e-12,
        local_solver::Symbol = :iterative,
        local_threshold = nothing,
        local_maxiter::Int = 200,
        local_tol::Real = 1.0e-6,
        return_info::Bool = false,
        verbosity::Int = 1,
        show_progress::Bool = true
    )
    return DMRG(
        nsites, _as_stage(max_sweeps, Int), isnothing(max_bond) ? nothing : _as_stage(max_bond, Int),
        trunc_tol, _check_local_solver(local_solver), local_threshold, local_maxiter,
        local_tol, return_info, verbosity, show_progress
    )
end

"""
    Krylov(; kwargs...)

Solve `A x = b` with a KrylovKit.jl Krylov method whose vectors are
`TTVector`s. Pass to [`linear_solve`](@ref); not supported by `eigen_solve`.

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
- `show_progress::Bool=true`: display a progress bar.
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
        show_progress::Bool = true
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
"Alias for [`AMEn`](@ref)."
const AMEnSolver = AMEn
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
function _check_linear_options(alg::ALS)
    _reject_unused(
        alg, "linear_solve", (:max_bond, :noise),
        "`max_sweeps`, `local_solver`, `local_threshold`, `local_maxiter`, `local_tol`, `return_info`, `verbosity`, and `show_progress`"
    )
    alg.max_sweeps isa Int ||
        throw(ArgumentError("ALS linear_solve runs a single stage; `max_sweeps` must be an integer, got $(alg.max_sweeps)"))
    return nothing
end
function _check_linear_options(alg::MALS)
    _reject_unused(
        alg, "linear_solve", (:local_solver, :local_threshold, :local_maxiter, :local_tol),
        "`max_sweeps`, `max_bond`, `trunc_tol`, `return_info`, `verbosity`, and `show_progress`"
    )
    alg.max_sweeps isa Int && !(alg.max_bond isa Vector) ||
        throw(ArgumentError("MALS linear_solve runs a single stage; `max_sweeps` and `max_bond` must be integers"))
    return nothing
end

# Seconds between redraws of the solver progress bars; a run shorter than this prints no bar.
const PROGRESS_DT = Ref(0.1)

_solver_progress(total::Integer, show_progress::Bool; desc::AbstractString = "") =
    Progress(max(total, 1); desc, enabled = show_progress, dt = PROGRESS_DT[])

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
        max_sweeps = alg.max_sweeps,
        local_solver = alg.local_solver,
        local_threshold = something(alg.local_threshold, typemax(Int)),
        local_maxiter = alg.local_maxiter,
        local_tol = alg.local_tol,
        return_info = alg.return_info,
        verbosity = alg.verbosity,
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
    return _mals_linsolve_impl(
        A, b, guess;
        max_sweeps = alg.max_sweeps,
        max_bond = something(alg.max_bond, round(Int, sqrt(prod(guess.ttv_dims)::Int))),
        trunc_tol = alg.trunc_tol,
        return_info = alg.return_info,
        verbosity = alg.verbosity,
        show_progress = alg.show_progress
    )
end

"""
    mals_linsolve(A, b, guess; kwargs...)

Compatibility wrapper for `linear_solve(A, b, guess, MALS(; kwargs...))`.
"""
function mals_linsolve(A, b, guess; kwargs...)
    return linear_solve(A, b, guess, MALS(; kwargs...))
end

linear_solve(A, b, guess, alg::AMEn) =
    _amen_linsolve_impl(A, b, guess; _amen_options(alg)..., return_info = alg.return_info)

# Keyword arguments of `_dmrg_linsolve_impl` and `_dmrg_eigsolve_impl` for `alg`.
function _dmrg_options(alg::DMRG, guess)
    st = _stages(; max_sweeps = alg.max_sweeps, max_bond = something(alg.max_bond, isqrt(prod(guess.ttv_dims)::Int)))
    return (;
        st..., nsites = alg.nsites, trunc_tol = alg.trunc_tol,
        local_solver = alg.local_solver, local_threshold = something(alg.local_threshold, 256),
        local_maxiter = alg.local_maxiter, local_tol = alg.local_tol,
        verbosity = alg.verbosity, show_progress = alg.show_progress,
    )
end

linear_solve(A, b, guess, alg::DMRG) =
    _dmrg_linsolve_impl(A, b, guess; _dmrg_options(alg, guess)..., return_info = alg.return_info)

"""
    dmrg_linsolve(A, b, guess; kwargs...)

Compatibility wrapper for `linear_solve(A, b, guess, DMRG(; kwargs...))`.
"""
function dmrg_linsolve(A, b, guess; kwargs...)
    return linear_solve(A, b, guess, DMRG(; kwargs...))
end

# Copy of `alg` for the linear solves inside a time stepper: the fields in
# `overrides` are replaced, and the copy returns a bare TTVector without a
# progress bar of its own.
function _stepper_algorithm(alg::LinearSolverAlgorithm; overrides...)
    fields = (f => getfield(alg, f) for f in fieldnames(typeof(alg)))
    return typeof(alg)(; fields..., overrides..., return_info = false, show_progress = false)
end

# A positive stepper `max_bond` also caps Krylov's operator applications;
# otherwise the algorithm object keeps its own rank cap.
_stepper_overrides(::Krylov, max_bond) = max_bond > 0 ? (; max_bond) : (;)
_stepper_overrides(::LinearSolverAlgorithm, max_bond) = (;)

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
        A::AbstractTTOperator, b::AbstractTTVector, guess::AbstractTTVector;
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
        show_progress::Bool = true,
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
        op = function (x::_RankBoundedTTVector)
            y = tt_compress!(A * x.tt, x.max_bond)
            return _RankBoundedTTVector(y, x.max_bond)
        end
        xb, info = linsolve(
            op,
            _RankBoundedTTVector(b, max_bond),
            _RankBoundedTTVector(guess, max_bond),
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
