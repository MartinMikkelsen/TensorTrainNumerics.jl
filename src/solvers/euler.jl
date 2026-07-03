using TensorTrainNumerics
using ProgressMeter
using KrylovKit

# Rounding rank read by the VectorInterface extension's `add` ops during Krylov
# solves. 0 = no truncation, the default used everywhere else (Manopt, etc.).
const KRYLOV_ROUND_RANK = Ref{Int}(0)

"""
    TTLinearSolver

Abstract supertype for the linear-solver backends of the implicit time
steppers ([`implicit_euler_method`](@ref), [`crank_nicholson_method`](@ref)).
Pass an instance as `tt_solver` — e.g. `tt_solver = MALSSolver()`;
solver-specific options are forwarded through the stepper's trailing keyword
arguments. The string names `"als"`, `"mals"`, `"dmrg"`, `"krylov"` are still
accepted for backwards compatibility.
"""
abstract type TTLinearSolver end

"""Alternating linear scheme backend (fixed ranks), see [`als_linsolve`](@ref)."""
struct ALSSolver <: TTLinearSolver end

"""Modified ALS backend (rank-adaptive two-site sweeps), see [`mals_linsolve`](@ref)."""
struct MALSSolver <: TTLinearSolver end

"""DMRG backend (rank-adaptive), see [`dmrg_linsolve`](@ref)."""
struct DMRGSolver <: TTLinearSolver end

"""KrylovKit backend with rank rounding; respects the stepper's `max_bond`."""
struct KrylovSolver <: TTLinearSolver end

_tt_linsolver(s::TTLinearSolver) = s
function _tt_linsolver(name::AbstractString)
    name == "als" && return ALSSolver()
    name == "mals" && return MALSSolver()
    name == "dmrg" && return DMRGSolver()
    name == "krylov" && return KrylovSolver()
    throw(ArgumentError("Unknown TT solver: $name. Use \"als\", \"mals\", \"dmrg\", \"krylov\", or a TTLinearSolver instance."))
end

_tt_linsolve(::ALSSolver, M, rhs, guess; max_bond = 0, kwargs...) = als_linsolve(M, rhs, guess; kwargs...)
_tt_linsolve(::MALSSolver, M, rhs, guess; max_bond = 0, kwargs...) = mals_linsolve(M, rhs, guess; kwargs...)
_tt_linsolve(::DMRGSolver, M, rhs, guess; max_bond = 0, kwargs...) = dmrg_linsolve(M, rhs, guess; kwargs...)
_tt_linsolve(::KrylovSolver, M, rhs, guess; max_bond = 0, kwargs...) = krylov_linsolve(M, rhs, guess; max_bond = max_bond, kwargs...)

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
        kwargs...
    )
    # Keep the Krylov iterates from accumulating rank: rank(A*x) = rank(A)*rank(x),
    # and the VectorInterface ops otherwise only orthogonalize (no truncation), so
    # Krylov solves can blow up the bond dimension. We cap the matvec output here
    # and via KRYLOV_ROUND_RANK, which the extension's `add`/`add!` read, the
    # intermediate Krylov vectors.
    op = max_bond > 0 ? (x -> tt_compress!(A * x, max_bond)) : (x -> A * x)
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
    old = KRYLOV_ROUND_RANK[]
    KRYLOV_ROUND_RANK[] = max_bond
    try
        x, _ = linsolve(op, b, guess, alg; kwargs...)
        return x
    finally
        KRYLOV_ROUND_RANK[] = old
    end
end

"""
    euler_method(A, u₀, steps; normalize=true, return_error=false)

Explicit Euler time stepping `u ← u + h·A·u` in TT format.

With `return_error = true` also returns the relative defect of the last step,
`‖u_{n+1} − (I + hA)·u_n‖ / ‖u_{n+1}‖`, which measures the error introduced by
orthogonalization (and normalization when `normalize = true`) in that step.
"""
function euler_method(A::AbstractTToperator, u₀::AbstractTTvector, steps::Vector{Float64}; normalize::Bool = true, return_error::Bool = false)
    solution = (u₀)
    u_prev = (u₀)

    @showprogress for h in steps
        u_prev = solution
        update = A * solution
        solution = orthogonalize(solution + h * update)
        if normalize
            norm² = dot(solution, solution)
            solution = (1 / sqrt(norm²)) * solution
        end
    end

    if return_error
        isempty(steps) && return solution, 0.0
        h = steps[end]
        Iop = id_tto(eltype(A), A.N; n_dim = A.tto_dims[1])
        # Orthogonalize before taking the norm: the residual is a difference of
        # nearly equal TT vectors, and the plain dot-based norm has a ~√eps
        # cancellation floor on such inputs.
        residual = orthogonalize(solution - (Iop + h * A) * u_prev)
        rel_error = norm(residual) / max(norm(solution), eps())
        return solution, rel_error
    end

    return solution
end

"""
    implicit_euler_method(A, u₀, guess, steps; tt_solver=MALSSolver(), normalize=true, max_bond=0, return_error=false, kwargs...)

Implicit Euler time stepping: solve `(I − h·A)·u_{n+1} = u_n` at every step
with the TT linear solver selected by `tt_solver` (a [`TTLinearSolver`](@ref)
instance, or one of the strings `"als"`, `"mals"`, `"dmrg"`, `"krylov"`).
Remaining keyword arguments are forwarded to the solver.
"""
function implicit_euler_method(
        A::AbstractTToperator,
        u₀::AbstractTTvector,
        guess::AbstractTTvector,
        steps::Vector{Float64};
        normalize::Bool = true,
        return_error::Bool = false,
        tt_solver::Union{AbstractString, TTLinearSolver} = MALSSolver(),
        max_bond::Int = 0,
        kwargs...
    )
    solver = _tt_linsolver(tt_solver)
    solution = (u₀)
    u_prev = (u₀)
    I = id_tto(eltype(A), A.N)

    @showprogress for h in steps
        M = I - h * A

        next = _tt_linsolve(solver, M, solution, guess; max_bond = max_bond, kwargs...)::AbstractTTvector

        if normalize
            next = next / norm(next)
        end

        u_prev = solution
        solution = max_bond > 0 ? tt_compress!(next, max_bond) : orthogonalize(next)
        guess = solution
    end

    if return_error
        h = steps[end]
        M = I - h * A
        residual = M * solution - u_prev
        rel_error = norm(residual) / norm(solution)
        return solution, rel_error
    end

    return solution
end

"""
    crank_nicholson_method(A, u₀, guess, steps; tt_solver=MALSSolver(), normalize=true, max_bond=0, return_error=false, kwargs...)

Crank–Nicolson time stepping: solve `(I − h/2·A)·u_{n+1} = (I + h/2·A)·u_n` at
every step with the TT linear solver selected by `tt_solver` (a
[`TTLinearSolver`](@ref) instance, or one of the strings `"als"`, `"mals"`,
`"dmrg"`, `"krylov"`). Remaining keyword arguments are forwarded to the solver.
"""
function crank_nicholson_method(
        A::AbstractTToperator,
        u₀::AbstractTTvector,
        guess::AbstractTTvector,
        steps::Vector{Float64};
        normalize::Bool = true,
        return_error::Bool = false,
        tt_solver::Union{AbstractString, TTLinearSolver} = MALSSolver(),
        max_bond::Int = 0,
        kwargs...
    )
    solver = _tt_linsolver(tt_solver)
    solution = (u₀)
    u_prev = (u₀)
    I = id_tto(eltype(A), A.N)

    @showprogress for h in steps
        LHS = I - (h / 2) * A
        RHS = (I + (h / 2) * A) * solution

        next = _tt_linsolve(solver, LHS, RHS, guess; max_bond = max_bond, kwargs...)::AbstractTTvector

        if normalize
            next = next / norm(next)
        end

        u_prev = solution
        solution = max_bond > 0 ? tt_compress!(next, max_bond) : orthogonalize(next)
        guess = solution
    end

    if return_error
        h = steps[end]
        LHS = I - (h / 2) * A
        RHS = (I + (h / 2) * A) * u_prev
        residual = LHS * solution - RHS
        rel_error = norm(residual) / norm(solution)
        return solution, rel_error
    end

    return solution
end

"""
    rk4_method(A, u₀, steps, max_bond; normalize=true, return_error=false)

Classical fourth-order Runge–Kutta time stepping in TT format, compressing every
stage and the iterate to bond dimension `max_bond`.

With `return_error = true` also returns the relative defect of the last step,
`‖u_{n+1} − (u_n + Δu_n)‖ / ‖u_{n+1}‖`, which measures the error introduced by
rank truncation (and normalization when `normalize = true`) in that step.
"""
function rk4_method(
        A::AbstractTToperator, u₀::AbstractTTvector, steps::Vector{Float64}, max_bond::Int;
        normalize::Bool = true, return_error::Bool = false
    )
    u = u₀
    u_prev = u₀
    incr = u₀   # placeholder, overwritten on the first step
    @showprogress for h in steps
        k1 = A * u
        k2 = A * tt_compress!(u + (h / 2) * k1, max_bond)
        k3 = A * tt_compress!(u + (h / 2) * k2, max_bond)
        k4 = A * tt_compress!(u + h * k3, max_bond)
        incr = (h / 6) * tt_compress!(k1 + 2k2 + 2k3 + k4, max_bond)
        u_prev = u
        u_new = tt_compress!(u + incr, max_bond)
        if normalize
            u_new = (1 / sqrt(dot(u_new, u_new))) * u_new
        end
        u = u_new
    end
    if return_error
        isempty(steps) && return u, 0.0
        residual = orthogonalize(u - (u_prev + incr))
        rel_error = norm(residual) / max(norm(u), eps())
        return u, rel_error
    end
    return u
end
