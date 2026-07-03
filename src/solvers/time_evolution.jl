using ProgressMeter

"""
    euler_method(A, u₀, steps; normalize=true, return_error=false, show_progress=true)

Explicit Euler time stepping `u ← u + h·A·u` in TT format.

With `return_error = true` also returns the relative defect of the last step,
`‖u_{n+1} − (I + hA)·u_n‖ / ‖u_{n+1}‖`, which measures the error introduced by
orthogonalization (and normalization when `normalize = true`) in that step.
"""
function euler_method(
        A::AbstractTToperator, u₀::AbstractTTvector, steps::Vector{Float64};
        normalize::Bool = true, return_error::Bool = false,
        show_progress::Bool = true
    )
    solution = (u₀)
    u_prev = (u₀)
    progress = _solver_progress(length(steps), show_progress; desc = "Euler method")

    for h in steps
        u_prev = solution
        update = A * solution
        solution = orthogonalize(solution + h * update)
        if normalize
            norm² = dot(solution, solution)
            solution = (1 / sqrt(norm²)) * solution
        end
        next!(progress)
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
    implicit_euler_method(A, u₀, guess, steps; tt_solver=MALS(), normalize=true, max_bond=0, return_error=false, show_progress=true, kwargs...)

Implicit Euler time stepping: solve `(I − h·A)·u_{n+1} = u_n` at every step
with the TT linear solver selected by `tt_solver` (a [`LinearSolverAlgorithm`](@ref)
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
        tt_solver::Union{AbstractString, LinearSolverAlgorithm} = MALS(),
        max_bond::Int = 0,
        show_progress::Bool = true,
        kwargs...
    )
    solver = tt_solver isa AbstractString ? _linear_solver_algorithm(tt_solver) : tt_solver
    solution = (u₀)
    u_prev = (u₀)
    I = id_tto(eltype(A), A.N)
    progress = _solver_progress(length(steps), show_progress; desc = "Implicit Euler method")

    for h in steps
        M = I - h * A

        next = _stepper_linear_solve(M, solution, guess, solver; max_bond = max_bond, kwargs...)::AbstractTTvector

        if normalize
            next = next / norm(next)
        end

        u_prev = solution
        solution = max_bond > 0 ? tt_compress!(next, max_bond) : orthogonalize(next)
        guess = solution
        next!(progress)
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
    crank_nicholson_method(A, u₀, guess, steps; tt_solver=MALS(), normalize=true, max_bond=0, return_error=false, show_progress=true, kwargs...)

Crank–Nicolson time stepping: solve `(I − h/2·A)·u_{n+1} = (I + h/2·A)·u_n` at
every step with the TT linear solver selected by `tt_solver` (a
[`LinearSolverAlgorithm`](@ref) instance, or one of the strings `"als"`, `"mals"`,
`"dmrg"`, `"krylov"`). Remaining keyword arguments are forwarded to the solver.
"""
function crank_nicholson_method(
        A::AbstractTToperator,
        u₀::AbstractTTvector,
        guess::AbstractTTvector,
        steps::Vector{Float64};
        normalize::Bool = true,
        return_error::Bool = false,
        tt_solver::Union{AbstractString, LinearSolverAlgorithm} = MALS(),
        max_bond::Int = 0,
        show_progress::Bool = true,
        kwargs...
    )
    solver = tt_solver isa AbstractString ? _linear_solver_algorithm(tt_solver) : tt_solver
    solution = (u₀)
    u_prev = (u₀)
    I = id_tto(eltype(A), A.N)
    progress = _solver_progress(length(steps), show_progress; desc = "Crank-Nicholson method")

    for h in steps
        LHS = I - (h / 2) * A
        RHS = (I + (h / 2) * A) * solution

        next = _stepper_linear_solve(LHS, RHS, guess, solver; max_bond = max_bond, kwargs...)::AbstractTTvector

        if normalize
            next = next / norm(next)
        end

        u_prev = solution
        solution = max_bond > 0 ? tt_compress!(next, max_bond) : orthogonalize(next)
        guess = solution
        next!(progress)
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
    rk4_method(A, u₀, steps, max_bond; normalize=true, return_error=false, show_progress=true)

Classical fourth-order Runge–Kutta time stepping in TT format, compressing every
stage and the iterate to bond dimension `max_bond`.

With `return_error = true` also returns the relative defect of the last step,
`‖u_{n+1} − (u_n + Δu_n)‖ / ‖u_{n+1}‖`, which measures the error introduced by
rank truncation (and normalization when `normalize = true`) in that step.
"""
function rk4_method(
        A::AbstractTToperator, u₀::AbstractTTvector, steps::Vector{Float64}, max_bond::Int;
        normalize::Bool = true, return_error::Bool = false,
        show_progress::Bool = true
    )
    u = u₀
    u_prev = u₀
    incr = u₀   # placeholder, overwritten on the first step
    progress = _solver_progress(length(steps), show_progress; desc = "RK4 method")
    for h in steps
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
        next!(progress)
    end
    if return_error
        isempty(steps) && return u, 0.0
        residual = orthogonalize(u - (u_prev + incr))
        rel_error = norm(residual) / max(norm(u), eps())
        return u, rel_error
    end
    return u
end
