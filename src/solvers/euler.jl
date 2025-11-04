using TensorTrainNumerics
using ProgressMeter

function euler_method(A::TToperator, u₀::TTvector, steps::Vector{Float64}; normalize::Bool = true, return_error::Bool = false)
    solution = (u₀)
    I = id_tto(A.N)

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
        A::TToperator,
        u₀::TTvector,
        guess::TTvector,
        steps::Vector{Float64};
        normalize::Bool = true,
        return_error::Bool = false,
        tt_solver::String = "mals",
        kwargs...
    )
    solution = (u₀)
    u_prev = (u₀)
    I = id_tto(A.N)

    @showprogress for h in steps
        M = I - h * A

        next = tt_solver == "mals" ? mals_linsolve(M, solution, guess; kwargs...) :
            tt_solver == "als" ? als_linsolve(M, solution, guess; kwargs...) :
            tt_solver == "dmrg" ? dmrg_linsolve(M, solution, guess; kwargs...) :
            error("Unknown TT solver: $tt_solver")

        if normalize
            next = next / norm(next)
        end

        u_prev = solution
        solution = orthogonalize(next)
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
        A::TToperator,
        u₀::TTvector,
        guess::TTvector,
        steps::Vector{Float64};
        normalize::Bool = true,
        return_error::Bool = false,
        tt_solver::String = "mals",
        kwargs...
    )
    solution = (u₀)
    u_prev = (u₀)
    I = id_tto(A.N)

    @showprogress for h in steps
        LHS = I - (h / 2) * A
        RHS = (I + (h / 2) * A) * solution

        next = tt_solver == "mals" ? mals_linsolve(LHS, RHS, guess; kwargs...) :
            tt_solver == "als" ? als_linsolve(LHS, RHS, guess; kwargs...) :
            tt_solver == "dmrg" ? dmrg_linsolve(LHS, RHS, guess; kwargs...) :
            error("Unknown TT solver: $tt_solver")

        if normalize
            next = next / norm(next)
        end

        u_prev = solution
        solution = orthogonalize(next)
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
        A::TToperator, u₀::TTvector, steps::Vector{Float64}, max_bond::Int;
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
