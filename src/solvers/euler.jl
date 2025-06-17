using TensorTrainNumerics
using ProgressMeter

function euler_method(A::TToperator, u₀::TTvector, steps::Vector{Float64}; normalize::Bool=true, return_error::Bool=false)
    solution = copy(u₀)
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
        rel_error = sqrt(dot(residual, residual)) / sqrt(dot(solution, solution))
        return solution, rel_error
    end

    return solution
end

function implicit_euler_method(
    A::TToperator,
    u₀::TTvector,
    guess::TTvector,
    steps::Vector{Float64};
    normalize::Bool=true,
    return_error::Bool=false,
    tt_solver::String="mals"
)
    solution = copy(u₀)
    u_prev = copy(u₀)
    I = id_tto(A.N)

    @showprogress for h in steps
        M = I - h * A

        next = tt_solver == "mals" ? mals_linsolv(M, solution, guess) :
               tt_solver == "als"  ? als_linsolv(M, solution, guess) :
               tt_solver == "dmrg" ? dmrg_linsolve(M, solution, guess) :
               error("Unknown TT solver: $tt_solver")

        if normalize
            norm² = dot(next, next)
            next = (1 / sqrt(norm²)) * next
        end

        u_prev = solution
        solution = orthogonalize(next)
        guess = solution  
    end

    if return_error
        h = steps[end]
        M = I - h * A
        residual = M * solution - u_prev
        rel_error = sqrt(dot(residual, residual)) / sqrt(dot(solution, solution))
        return solution, rel_error
    end

    return solution
end
