using TensorTrainNumerics
using ProgressMeter
using CairoMakie

function tt_linsolve(solver::String, A, B, guess; kwargs...)
    if solver == "mals"
        return mals_linsolve(A, B, guess; kwargs...)
    elseif solver == "als"
        return als_linsolve(A, B, guess; kwargs...)
    elseif solver == "dmrg"
        return dmrg_linsolve(A, B, guess; kwargs...)
    else
        error("Unknown TT solver: $solver")
    end
end

function power_method(A::TToperator, guess::TTvector, RHS::Union{TToperator, Nothing}=nothing; repeats::Int=10, σ::Float64=0.999, tt_solver::String="mals", kwargs...)
    if RHS === nothing
        shift = A - σ * id_tto(A.N)
    else
        shift = A - σ * RHS
    end

    eigenvalues = Float64[]
    x = guess

    @showprogress for i in 1:repeats
        if RHS === nothing
            x = (tt_linsolve(tt_solver, shift, x, x))
            λ = dot(x, A * x)
        else
            x = (tt_linsolve(tt_solver, shift, x, RHS * x))
            λ = 1 / dot(x, RHS * x)
        end
        push!(eigenvalues, λ)
    end

    return eigenvalues, x
end
