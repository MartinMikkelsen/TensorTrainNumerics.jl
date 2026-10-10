using ProgressMeter

"""
    euler_method(A, u₀, steps; normalize=false, return_info=false, show_progress=true)

Explicit Euler time stepping `u ← u + h·A·u` in TT format.

With `return_info = true` returns `(u, (; error))`, where `error` is the relative
defect of the last step, `‖u_{n+1} − (I + hA)·u_n‖ / ‖u_{n+1}‖`, which measures the error introduced by
orthogonalization (and normalization when `normalize = true`) in that step.
"""
function euler_method(
        A::AbstractTTOperator, u₀::AbstractTTVector, steps::Vector{Float64};
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
        next!(progress; showvalues = [("step", "$step/$(length(steps))"), ("time", t), ("largest rank", maximum(solution.ranks))])
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
        A::AbstractTTOperator,
        u₀::AbstractTTVector,
        guess::AbstractTTVector,
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

        next = linear_solve(M, solution, guess, step_alg)::AbstractTTVector

        if normalize
            next = next / norm(next)
        end

        u_prev = solution
        solution = max_bond > 0 ? tt_compress!(next, max_bond) : orthogonalize(next)
        guess = solution
        t += h
        next!(progress; showvalues = [("step", "$step/$(length(steps))"), ("time", t), ("largest rank", maximum(solution.ranks))])
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
        A::AbstractTTOperator,
        u₀::AbstractTTVector,
        guess::AbstractTTVector,
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

        next = linear_solve(LHS, RHS, guess, step_alg)::AbstractTTVector

        if normalize
            next = next / norm(next)
        end

        u_prev = solution
        solution = max_bond > 0 ? tt_compress!(next, max_bond) : orthogonalize(next)
        guess = solution
        t += h
        next!(progress; showvalues = [("step", "$step/$(length(steps))"), ("time", t), ("largest rank", maximum(solution.ranks))])
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

"""
    rk4_method(A, u₀, steps; max_bond, normalize=false, return_info=false, show_progress=true)

Classical fourth-order Runge–Kutta time stepping in TT format, compressing every
stage and the iterate to bond dimension `max_bond`.

With `return_info = true` returns `(u, (; error))`, where `error` is the relative
defect of the last step, `‖u_{n+1} − (u_n + Δu_n)‖ / ‖u_{n+1}‖`, which measures the error introduced by
rank truncation (and normalization when `normalize = true`) in that step.
"""
function rk4_method(
        A::AbstractTTOperator, u₀::AbstractTTVector, steps::Vector{Float64};
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
        next!(progress; showvalues = [("step", "$step/$(length(steps))"), ("time", t), ("largest rank", maximum(u.ranks))])
    end
    if return_info
        isempty(steps) && return u, (; error = 0.0)
        residual = orthogonalize(u - (u_prev + incr))
        return u, (; error = norm(residual) / max(norm(u), eps()))
    end
    return u
end

"""
    TimeEvolutionAlgorithm

Supertype of algorithm objects accepted by [`time_evolve`](@ref):
[`Euler`](@ref), [`ImplicitEuler`](@ref), [`CrankNicolson`](@ref),
[`RK4`](@ref), and [`TDVP`](@ref).
"""
abstract type TimeEvolutionAlgorithm end

"""
    Euler(; normalize=false, return_info=false, show_progress=true)

Explicit Euler time stepping; see [`euler_method`](@ref) for the scheme and the
meaning of the keyword arguments.
"""
struct Euler <: TimeEvolutionAlgorithm
    normalize::Bool
    return_info::Bool
    show_progress::Bool
end

Euler(; normalize::Bool = false, return_info::Bool = false, show_progress::Bool = true) =
    Euler(normalize, return_info, show_progress)

"""
    ImplicitEuler(; linear_solver=MALS(), max_bond=0, normalize=false, return_info=false, show_progress=true)

Implicit Euler time stepping. Every step solves a TT linear system with
`linear_solver`, a [`LinearSolverAlgorithm`](@ref); see
[`implicit_euler_method`](@ref) for the scheme and the other keyword arguments.
"""
struct ImplicitEuler{S <: LinearSolverAlgorithm} <: TimeEvolutionAlgorithm
    linear_solver::S
    max_bond::Int
    normalize::Bool
    return_info::Bool
    show_progress::Bool
end

function ImplicitEuler(;
        linear_solver::LinearSolverAlgorithm = MALS(), max_bond::Int = 0,
        normalize::Bool = false, return_info::Bool = false, show_progress::Bool = true
    )
    return ImplicitEuler(linear_solver, max_bond, normalize, return_info, show_progress)
end

"""
    CrankNicolson(; linear_solver=MALS(), max_bond=0, normalize=false, return_info=false, show_progress=true)

Crank–Nicolson time stepping. Every step solves a TT linear system with
`linear_solver`, a [`LinearSolverAlgorithm`](@ref); see
[`crank_nicolson_method`](@ref) for the scheme and the other keyword arguments.
"""
struct CrankNicolson{S <: LinearSolverAlgorithm} <: TimeEvolutionAlgorithm
    linear_solver::S
    max_bond::Int
    normalize::Bool
    return_info::Bool
    show_progress::Bool
end

function CrankNicolson(;
        linear_solver::LinearSolverAlgorithm = MALS(), max_bond::Int = 0,
        normalize::Bool = false, return_info::Bool = false, show_progress::Bool = true
    )
    return CrankNicolson(linear_solver, max_bond, normalize, return_info, show_progress)
end

"""
    RK4(; max_bond, normalize=false, return_info=false, show_progress=true)

Classical fourth-order Runge–Kutta time stepping with every stage compressed to
bond dimension `max_bond`; see [`rk4_method`](@ref).
"""
struct RK4 <: TimeEvolutionAlgorithm
    max_bond::Int
    normalize::Bool
    return_info::Bool
    show_progress::Bool
end

RK4(; max_bond::Int, normalize::Bool = false, return_info::Bool = false, show_progress::Bool = true) =
    RK4(max_bond, normalize, return_info, show_progress)

"""
    TDVP(; nsites=1, kwargs...)

Time-dependent variational principle with one-site (`nsites = 1`, fixed ranks)
or two-site (`nsites = 2`, adaptive ranks) updates; see [`tdvp`](@ref) and
[`tdvp2`](@ref) for the schemes.

# Keyword arguments
- `nsites::Int=1`: `1` or `2`.
- `max_bond::Int=typemax(Int)`, `trunc_tol::Real=0.0`: rank truncation of the
  two-site updates. Setting either with `nsites = 1` throws an `ArgumentError`.
- `normalize`, `substeps`, `carry_env`, `imaginary_time`, `return_info`,
  `verbosity`, `show_progress`: as in [`tdvp`](@ref).
- Remaining keyword arguments are passed to `KrylovKit.exponentiate`.
"""
struct TDVP{K <: NamedTuple} <: TimeEvolutionAlgorithm
    nsites::Int
    max_bond::Int
    trunc_tol::Float64
    normalize::Bool
    substeps::Int
    carry_env::Bool
    imaginary_time::Bool
    return_info::Bool
    verbosity::Int
    show_progress::Bool
    exponentiate::K
end

function TDVP(;
        nsites::Int = 1, max_bond::Int = typemax(Int), trunc_tol::Real = 0.0,
        normalize::Bool = false, substeps::Int = 1, carry_env::Bool = true,
        imaginary_time::Bool = false, return_info::Bool = false,
        verbosity::Int = 1, show_progress::Bool = true, kwargs...
    )
    nsites in (1, 2) || throw(ArgumentError("`nsites` must be 1 or 2; got $nsites"))
    if nsites == 1 && (max_bond != typemax(Int) || trunc_tol != 0)
        throw(ArgumentError("`max_bond` and `trunc_tol` apply only to `nsites = 2`; one-site TDVP keeps the ranks fixed"))
    end
    _check_exponentiate_kwargs("TDVP", kwargs)
    return TDVP(
        nsites, max_bond, Float64(trunc_tol), normalize, substeps, carry_env,
        imaginary_time, return_info, verbosity, show_progress, NamedTuple(kwargs)
    )
end

"""
    time_evolve(A, u₀, steps, alg::TimeEvolutionAlgorithm) -> TTVector
    time_evolve(A, u₀, steps, alg; guess=u₀)

Evolve `u₀` under the generator `A` with the time stepper `alg`: an
[`Euler`](@ref), [`ImplicitEuler`](@ref), [`CrankNicolson`](@ref),
[`RK4`](@ref), or [`TDVP`](@ref) object.

`steps` holds the step sizes, not time points. [`Euler`](@ref),
[`ImplicitEuler`](@ref), [`CrankNicolson`](@ref), and [`RK4`](@ref) integrate
`du/dt = A u`; for [`TDVP`](@ref) the generator is `-iA` unless
`imaginary_time = true`.

For [`ImplicitEuler`](@ref) and [`CrankNicolson`](@ref), `guess` is the initial
guess of the first linear solve; later steps start from the previous solution.

If `alg` was built with `return_info = true`, the result is `(u, info)`.
"""
function time_evolve(A, u₀, steps, alg::Euler)
    (; normalize, return_info, show_progress) = alg
    return euler_method(A, u₀, collect(Float64, steps); normalize, return_info, show_progress)
end

function time_evolve(A, u₀, steps, alg::ImplicitEuler; guess = u₀)
    (; max_bond, normalize, return_info, show_progress) = alg
    return implicit_euler_method(
        A, u₀, guess, collect(Float64, steps);
        alg = alg.linear_solver, max_bond, normalize, return_info, show_progress
    )
end

function time_evolve(A, u₀, steps, alg::CrankNicolson; guess = u₀)
    (; max_bond, normalize, return_info, show_progress) = alg
    return crank_nicolson_method(
        A, u₀, guess, collect(Float64, steps);
        alg = alg.linear_solver, max_bond, normalize, return_info, show_progress
    )
end

function time_evolve(A, u₀, steps, alg::RK4)
    (; max_bond, normalize, return_info, show_progress) = alg
    return rk4_method(A, u₀, collect(Float64, steps); max_bond, normalize, return_info, show_progress)
end

function time_evolve(A, u₀, steps, alg::TDVP)
    (; normalize, substeps, carry_env, imaginary_time, return_info, verbosity, show_progress) = alg
    common = (; normalize, substeps, carry_env, imaginary_time, return_info, verbosity, show_progress)
    h = collect(Float64, steps)
    alg.nsites == 1 && return tdvp(A, u₀, h; common..., alg.exponentiate...)
    return tdvp2(A, u₀, h; common..., max_bond = alg.max_bond, trunc_tol = alg.trunc_tol, alg.exponentiate...)
end
