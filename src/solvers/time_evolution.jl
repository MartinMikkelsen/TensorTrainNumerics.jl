using ProgressMeter

"""
    euler_method(A, u₀, steps; normalize=false, return_info=false, show_progress=true)

Explicit Euler time stepping `u ← u + h·A·u` in TT format.

With `return_info = true` returns `(u, (; error))`, where `error` is the relative
defect of the last step, `‖u_{n+1} − (I + hA)·u_n‖ / ‖u_{n+1}‖`, which measures the error introduced by
orthogonalization (and normalization when `normalize = true`) in that step.
"""
function euler_method(
        A::AbstractTToperator, u₀::AbstractTTvector, steps::Vector{Float64};
        normalize::Bool = false, return_info::Bool = false,
        show_progress::Bool = true
    )
    solution = (u₀)
    u_prev = (u₀)
    progress = _solver_progress(length(steps), show_progress; desc = "Euler method")

    t = 0.0
    for (step, h) in enumerate(steps)
        u_prev = solution
        update = A * solution
        solution = orthogonalize(solution + h * update)
        if normalize
            norm² = dot(solution, solution)
            solution = (1 / sqrt(norm²)) * solution
        end
        t += h
        next!(progress; showvalues = [("step", "$step/$(length(steps))"), ("time", t), ("largest rank", maximum(solution.ttv_rks))])
    end

    if return_info
        isempty(steps) && return solution, (; error = 0.0)
        h = steps[end]
        Iop = _identity_like(A)
        # Orthogonalize before taking the norm: the residual is a difference of
        # nearly equal TT vectors, and the plain dot-based norm has a ~√eps
        # cancellation floor on such inputs.
        residual = orthogonalize(solution - (Iop + h * A) * u_prev)
        return solution, (; error = norm(residual) / max(norm(solution), eps()))
    end

    return solution
end

"""
    implicit_euler_method(A, u₀, guess, steps; alg=MALS(), normalize=false, max_bond=0, return_info=false, show_progress=true, kwargs...)

Implicit Euler time stepping: solve `(I − h·A)·u_{n+1} = u_n` at every step
with the TT linear solver `alg` (a [`LinearSolverAlgorithm`](@ref)).
Remaining keyword arguments replace fields of `alg` for these solves (for
example `max_sweeps = 2`). `max_bond > 0` compresses every step to that bond
dimension and, for [`Krylov`](@ref), also caps its operator applications. With
`return_info = true` returns `(u, (; error))`, where `error` is the relative
residual of the last step's linear system.
"""
function implicit_euler_method(
        A::AbstractTToperator,
        u₀::AbstractTTvector,
        guess::AbstractTTvector,
        steps::Vector{Float64};
        normalize::Bool = false,
        return_info::Bool = false,
        alg::LinearSolverAlgorithm = MALS(),
        max_bond::Int = 0,
        show_progress::Bool = true,
        kwargs...
    )
    step_alg = _stepper_algorithm(alg; _stepper_overrides(alg, max_bond)..., kwargs...)
    solution = (u₀)
    u_prev = (u₀)
    Id = _identity_like(A)
    progress = _solver_progress(length(steps), show_progress; desc = "Implicit Euler method")

    t = 0.0
    for (step, h) in enumerate(steps)
        M = Id - h * A

        next = linear_solve(M, solution, guess, step_alg)::AbstractTTvector

        if normalize
            next = next / norm(next)
        end

        u_prev = solution
        solution = max_bond > 0 ? tt_compress!(next, max_bond) : orthogonalize(next)
        guess = solution
        t += h
        next!(progress; showvalues = [("step", "$step/$(length(steps))"), ("time", t), ("largest rank", maximum(solution.ttv_rks))])
    end

    if return_info
        h = steps[end]
        M = Id - h * A
        residual = M * solution - u_prev
        return solution, (; error = norm(residual) / norm(solution))
    end

    return solution
end

"""
    crank_nicolson_method(A, u₀, guess, steps; alg=MALS(), normalize=false, max_bond=0, return_info=false, show_progress=true, kwargs...)

Crank–Nicolson time stepping: solve `(I − h/2·A)·u_{n+1} = (I + h/2·A)·u_n` at
every step with the TT linear solver `alg` (a [`LinearSolverAlgorithm`](@ref)).
Remaining keyword arguments replace fields of `alg` for these solves (for
example `max_sweeps = 2`). `max_bond > 0` compresses every step to that bond
dimension and, for [`Krylov`](@ref), also caps its operator applications. With
`return_info = true` returns `(u, (; error))`, where `error` is the relative
residual of the last step's linear system.
"""
function crank_nicolson_method(
        A::AbstractTToperator,
        u₀::AbstractTTvector,
        guess::AbstractTTvector,
        steps::Vector{Float64};
        normalize::Bool = false,
        return_info::Bool = false,
        alg::LinearSolverAlgorithm = MALS(),
        max_bond::Int = 0,
        show_progress::Bool = true,
        kwargs...
    )
    step_alg = _stepper_algorithm(alg; _stepper_overrides(alg, max_bond)..., kwargs...)
    solution = (u₀)
    u_prev = (u₀)
    Id = _identity_like(A)
    progress = _solver_progress(length(steps), show_progress; desc = "Crank-Nicolson method")

    t = 0.0
    for (step, h) in enumerate(steps)
        LHS = Id - (h / 2) * A
        RHS = (Id + (h / 2) * A) * solution

        next = linear_solve(LHS, RHS, guess, step_alg)::AbstractTTvector

        if normalize
            next = next / norm(next)
        end

        u_prev = solution
        solution = max_bond > 0 ? tt_compress!(next, max_bond) : orthogonalize(next)
        guess = solution
        t += h
        next!(progress; showvalues = [("step", "$step/$(length(steps))"), ("time", t), ("largest rank", maximum(solution.ttv_rks))])
    end

    if return_info
        h = steps[end]
        LHS = Id - (h / 2) * A
        RHS = (Id + (h / 2) * A) * u_prev
        residual = LHS * solution - RHS
        return solution, (; error = norm(residual) / norm(solution))
    end

    return solution
end

# Deprecated: renamed to `crank_nicolson_method`.
function crank_nicholson_method(args...; kwargs...)
    Base.depwarn("`crank_nicholson_method` is deprecated, use `crank_nicolson_method`.", :crank_nicholson_method)
    return crank_nicolson_method(args...; kwargs...)
end

"""
    rk4_method(A, u₀, steps; max_bond, normalize=false, return_info=false, show_progress=true)

Classical fourth-order Runge–Kutta time stepping in TT format, compressing every
stage and the iterate to bond dimension `max_bond`.

With `return_info = true` returns `(u, (; error))`, where `error` is the relative
defect of the last step, `‖u_{n+1} − (u_n + Δu_n)‖ / ‖u_{n+1}‖`, which measures the error introduced by
rank truncation (and normalization when `normalize = true`) in that step.
"""
function rk4_method(
        A::AbstractTToperator, u₀::AbstractTTvector, steps::Vector{Float64};
        max_bond::Int, normalize::Bool = false, return_info::Bool = false,
        show_progress::Bool = true
    )
    u = u₀
    u_prev = u₀
    incr = u₀   # placeholder, overwritten on the first step
    progress = _solver_progress(length(steps), show_progress; desc = "RK4 method")
    t = 0.0
    for (step, h) in enumerate(steps)
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
        t += h
        next!(progress; showvalues = [("step", "$step/$(length(steps))"), ("time", t), ("largest rank", maximum(u.ttv_rks))])
    end
    if return_info
        isempty(steps) && return u, (; error = 0.0)
        residual = orthogonalize(u - (u_prev + incr))
        return u, (; error = norm(residual) / max(norm(u), eps()))
    end
    return u
end
