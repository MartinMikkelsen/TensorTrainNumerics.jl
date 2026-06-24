using TensorTrainNumerics
using ProgressMeter
using KrylovKit

# Rounding rank read by the VectorInterface extension's `add` ops during Krylov
# solves. 0 = no truncation, the default used everywhere else (Manopt, etc.).
const KRYLOV_ROUND_RANK = Ref{Int}(0)

function _krylov_algorithm(krylov_solver::Symbol, max_bond::Int;
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
            verbosity = verbosity)
    elseif solver == :cg
        return KrylovKit.CG(; maxiter = krylovdim * maxiter, tol = tol, verbosity = verbosity)
    end
    throw(ArgumentError("Unknown Krylov solver: $krylov_solver. Use :auto, :bicgstab, :cg, or :gmres."))
end

function krylov_linsolve(A::AbstractTToperator, b::AbstractTTvector, guess::AbstractTTvector;
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
        kwargs...)
    # Keep the Krylov iterates from accumulating rank: rank(A*x) = rank(A)*rank(x),
    # and the VectorInterface ops otherwise only orthogonalize (no truncation), so
    # Krylov solves can blow up the bond dimension. We cap the matvec output here
    # and via KRYLOV_ROUND_RANK, which the extension's `add`/`add!` read, the
    # intermediate Krylov vectors.
    op = max_bond > 0 ? (x -> tt_compress!(A * x, max_bond)) : (x -> A * x)
    solver = krylov_solver == :auto && isposdef && (issymmetric || ishermitian) ? :cg : krylov_solver
    tol_value = isnothing(tol) ? max(atol, rtol * norm(b)) : tol
    alg = _krylov_algorithm(solver, max_bond;
        krylovdim = krylovdim,
        maxiter = maxiter,
        tol = tol_value,
        orth = orth,
        verbosity = verbosity)
    old = KRYLOV_ROUND_RANK[]
    KRYLOV_ROUND_RANK[] = max_bond
    try
        x, _ = linsolve(op, b, guess, alg; kwargs...)
        return x
    finally
        KRYLOV_ROUND_RANK[] = old
    end
end

function euler_method(A::AbstractTToperator, u₀::AbstractTTvector, steps::Vector{Float64}; normalize::Bool = true, return_error::Bool = false)
    solution = (u₀)
    I = id_tto(eltype(A), A.N)

    @showprogress for h in steps
        update = A * solution
        solution = orthogonalize(solution + h * update)
        if normalize
            norm² = dot(solution, solution)
            solution = (1 / sqrt(norm²)) * solution
        end
    end

    if return_error
        h = steps[end]
        residual = solution - (I + h * A) * solution
        rel_error = norm(residual) / norm(solution)
        return solution, rel_error
    end

    return solution
end

function implicit_euler_method(
        A::AbstractTToperator,
        u₀::AbstractTTvector,
        guess::AbstractTTvector,
        steps::Vector{Float64};
        normalize::Bool = true,
        return_error::Bool = false,
        tt_solver::String = "mals",
        max_bond::Int = 0,
        kwargs...
    )
    solution = (u₀)
    u_prev = (u₀)
    I = id_tto(eltype(A), A.N)

    @showprogress for h in steps
        M = I - h * A

        next = (tt_solver == "mals" ? mals_linsolve(M, solution, guess; kwargs...) :
            tt_solver == "als" ? als_linsolve(M, solution, guess; kwargs...) :
            tt_solver == "dmrg" ? dmrg_linsolve(M, solution, guess; kwargs...) :
            tt_solver == "krylov" ? krylov_linsolve(M, solution, guess; max_bond = max_bond, kwargs...) :
            error("Unknown TT solver: $tt_solver"))::AbstractTTvector

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

function crank_nicholson_method(
        A::AbstractTToperator,
        u₀::AbstractTTvector,
        guess::AbstractTTvector,
        steps::Vector{Float64};
        normalize::Bool = true,
        return_error::Bool = false,
        tt_solver::String = "mals",
        max_bond::Int = 0,
        kwargs...
    )
    solution = (u₀)
    u_prev = (u₀)
    I = id_tto(eltype(A), A.N)

    @showprogress for h in steps
        LHS = I - (h / 2) * A
        RHS = (I + (h / 2) * A) * solution

        next = (tt_solver == "mals" ? mals_linsolve(LHS, RHS, guess; kwargs...) :
            tt_solver == "als" ? als_linsolve(LHS, RHS, guess; kwargs...) :
            tt_solver == "dmrg" ? dmrg_linsolve(LHS, RHS, guess; kwargs...) :
            tt_solver == "krylov" ? krylov_linsolve(LHS, RHS, guess; max_bond = max_bond, kwargs...) :
            error("Unknown TT solver: $tt_solver"))::AbstractTTvector

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

function rk4_method(
        A::AbstractTToperator, u₀::AbstractTTvector, steps::Vector{Float64}, max_bond::Int;
        normalize::Bool = true, return_error::Bool = false
    )
    u = u₀
    @showprogress for h in steps
        k1 = A * u
        k2 = A * tt_compress!(u + (h / 2) * k1, max_bond)
        k3 = A * tt_compress!(u + (h / 2) * k2, max_bond)
        k4 = A * tt_compress!(u + h * k3, max_bond)
        incr = (h / 6) * tt_compress!(k1 + 2k2 + 2k3 + k4, max_bond)
        u_new = tt_compress!(u + incr, max_bond)
        if normalize
            u_new = (1 / sqrt(dot(u_new, u_new))) * u_new
        end
        u = u_new
    end
    if return_error
        h = steps[end]
        k1 = A * u
        k2 = A * tt_compress!(u + (h / 2) * k1, max_bond)
        k3 = A * tt_compress!(u + (h / 2) * k2, max_bond)
        k4 = A * tt_compress!(u + h * k3, max_bond)
        incr = (h / 6) * tt_compress!(k1 + 2k2 + 2k3 + k4, max_bond)
        residual = tt_compress!(u - (u - incr) - incr, max_bond)
        rel_error = norm(residual) / max(norm(u), eps())
        return u, rel_error
    end
    return u
end
