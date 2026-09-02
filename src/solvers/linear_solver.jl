using KrylovKit
using ProgressMeter

struct _RankBoundedTTvector{V <: AbstractTTvector}
    tt::V
    max_bond::Int
end

abstract type LinearSolverAlgorithm end
abstract type EigenSolverAlgorithm <: LinearSolverAlgorithm end

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
        verbosity::Int = 0,
        show_progress::Bool = false
    )
    tol_value = isnothing(tol) ? nothing : Float64(tol)
    hermitian_value = isnothing(ishermitian) ? issymmetric : ishermitian
    return Krylov(max_bond, krylov_solver, krylovdim, maxiter, Float64(rtol), Float64(atol), tol_value, orth, issymmetric, hermitian_value, isposdef, verbosity, show_progress)
end

const TTLinearSolver = LinearSolverAlgorithm
const ALSSolver = ALS
const MALSSolver = MALS
const DMRGSolver = DMRG
const KrylovSolver = Krylov

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
        show_progress = alg.show_progress
    )
end

function linear_solve(A, b, guess, alg::ALS)
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
    sweep_schedule = isnothing(alg.sweep_schedule) ? [2] : alg.sweep_schedule
    rmax_schedule = isnothing(alg.rmax_schedule) ? [isqrt(prod(guess.ttv_dims)::Int)] : alg.rmax_schedule
    linsolv_tol = isnothing(alg.linsolv_tol) ? max(sqrt(alg.tol), 1.0e-8) : alg.linsolv_tol
    return _dmrg_linsolve_impl(
        A, b, guess;
        sweep_count = alg.sweep_count,
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

_stepper_linear_solve(A, b, guess, solver::Krylov; max_bond::Int, kwargs...) = linear_solve(A, b, guess, _stepper_algorithm(solver; max_bond = max_bond, kwargs...))
_stepper_linear_solve(A, b, guess, solver::LinearSolverAlgorithm; max_bond::Int, kwargs...) = linear_solve(A, b, guess, _stepper_algorithm(solver; kwargs...))

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
        verbosity::Int = 0,
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
        x, _ = linsolve(
            op,
            _RankBoundedTTvector(b, max_bond),
            _RankBoundedTTvector(guess, max_bond),
            alg;
            kwargs...
        )
        next!(progress)
        return tt_compress!(x.tt, max_bond)
    end
    x, _ = linsolve(x -> A * x, b, guess, alg; kwargs...)
    next!(progress)
    return x
end
