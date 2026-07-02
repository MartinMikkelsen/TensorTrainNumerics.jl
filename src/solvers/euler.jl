using TensorTrainNumerics
using ProgressMeter
using KrylovKit

# Rounding rank read by the VectorInterface extension's `add` ops during Krylov
# solves. 0 = no truncation, the default used everywhere else (Manopt, etc.).
const KRYLOV_ROUND_RANK = Ref{Int}(0)

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

        next = (
            tt_solver == "mals" ? mals_linsolve(M, solution, guess; kwargs...) :
                tt_solver == "als" ? als_linsolve(M, solution, guess; kwargs...) :
                tt_solver == "dmrg" ? dmrg_linsolve(M, solution, guess; kwargs...) :
                tt_solver == "krylov" ? krylov_linsolve(M, solution, guess; max_bond = max_bond, kwargs...) :
                error("Unknown TT solver: $tt_solver")
        )::AbstractTTvector

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

        next = (
            tt_solver == "mals" ? mals_linsolve(LHS, RHS, guess; kwargs...) :
                tt_solver == "als" ? als_linsolve(LHS, RHS, guess; kwargs...) :
                tt_solver == "dmrg" ? dmrg_linsolve(LHS, RHS, guess; kwargs...) :
                tt_solver == "krylov" ? krylov_linsolve(LHS, RHS, guess; max_bond = max_bond, kwargs...) :
                error("Unknown TT solver: $tt_solver")
        )::AbstractTTvector

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

function _global_cn_time_bits(steps::Vector{Float64})
    isempty(steps) && throw(ArgumentError("steps must be nonempty"))
    τ = steps[1]
    all(step -> step == τ, steps) || throw(ArgumentError("global_crank_nicholson_method requires constant time steps"))
    Nt = length(steps)
    Nt < 2 && throw(ArgumentError("global_crank_nicholson_method requires at least two time steps so the time index has at least one QTT bit"))
    ispow2(Nt) || throw(ArgumentError("global_crank_nicholson_method requires length(steps) to be a power of two"))
    return τ, round(Int, log2(Nt))
end

function _convert_ttv_eltype(::Type{T}, v::TTvector{S, N}) where {T <: Number, S <: Number, N}
    return TTvector{T, N}(
        v.N,
        [convert(Array{T, 3}, core) for core in v.ttv_vec],
        v.ttv_dims,
        copy(v.ttv_rks),
        copy(v.ttv_ot)
    )
end

function _space_time_guess(
        ::Type{T},
        guess::QTTvector,
        time_bits::Int,
        space_n_dims::Int,
        space_bits_per_dim::Int,
        space_ordering::Symbol
    ) where {T <: Number}
    @assert guess.n_dims == space_n_dims "guess n_dims must match u0"
    @assert guess.bits_per_dim == space_bits_per_dim "guess bits_per_dim must match u0"
    @assert guess.ordering == space_ordering "guess ordering must match u0"
    time_guess = ones_tt(T, ntuple(_ -> 2, time_bits))
    return time_guess ⊗ _convert_ttv_eltype(T, TTvector(guess))
end

function _space_time_guess(
        ::Type{T},
        guess::SpaceTimeQTTvector,
        time_bits::Int,
        space_n_dims::Int,
        space_bits_per_dim::Int,
        space_ordering::Symbol
    ) where {T <: Number}
    @assert guess.time_bits == time_bits "space-time guess time_bits must match length(steps)"
    @assert guess.space_n_dims == space_n_dims "space-time guess space_n_dims must match u0"
    @assert guess.space_bits_per_dim == space_bits_per_dim "space-time guess space_bits_per_dim must match u0"
    @assert guess.space_ordering == space_ordering "space-time guess ordering must match u0"
    return _convert_ttv_eltype(T, TTvector(guess))
end

function _global_tt_linsolve(
        A::AbstractTToperator,
        b::AbstractTTvector,
        guess::AbstractTTvector,
        tt_solver::String,
        max_bond::Int;
        kwargs...
    )
    return (
        tt_solver == "mals" ? mals_linsolve(A, b, guess; kwargs...) :
            tt_solver == "als" ? als_linsolve(A, b, guess; kwargs...) :
            tt_solver == "dmrg" ? dmrg_linsolve(A, b, guess; kwargs...) :
            tt_solver == "krylov" ? krylov_linsolve(A, b, guess; max_bond = max_bond, kwargs...) :
            error("Unknown TT solver: $tt_solver")
    )::AbstractTTvector
end

function global_crank_nicholson_method(
        A::QTToperator,
        u0::QTTvector,
        guess::Union{QTTvector, SpaceTimeQTTvector},
        steps::Vector{Float64};
        normalize::Bool = false,
        return_error::Bool = false,
        tt_solver::String = "mals",
        max_bond::Int = 0,
        kwargs...
    )
    normalize && throw(ArgumentError("global_crank_nicholson_method does not support per-step normalization"))
    check_compat(A, u0)

    τ, time_bits = _global_cn_time_bits(steps)
    T = promote_type(eltype(A), eltype(u0), typeof(τ))
    τT = convert(T, τ)

    A_tt = _convert_tto_eltype(T, TToperator(A))
    u0_tt = _convert_ttv_eltype(T, TTvector(u0))

    I_space = id_tto(T, A.N)
    L = I_space - (τT / 2) * A_tt
    R = I_space + (τT / 2) * A_tt

    I_time = qtt_time_identity(T, time_bits)
    S_time = qtt_lower_shift(T, time_bits)
    A_global = (I_time ⊗ L) - (S_time ⊗ R)

    first_rhs = R * u0_tt
    time_rhs = _convert_ttv_eltype(T, qtt_basis_vector(time_bits, 1))
    rhs = time_rhs ⊗ first_rhs

    guess_tt = _space_time_guess(
        T,
        guess,
        time_bits,
        u0.n_dims,
        u0.bits_per_dim,
        u0.ordering
    )

    solution_tt = _global_tt_linsolve(
        A_global,
        rhs,
        guess_tt,
        tt_solver,
        max_bond;
        kwargs...
    )
    if max_bond > 0
        solution_tt = tt_compress!(solution_tt, max_bond)
    else
        solution_tt = orthogonalize(solution_tt)
    end

    U = SpaceTimeQTTvector(
        solution_tt,
        time_bits,
        u0.n_dims,
        u0.bits_per_dim,
        u0.ordering
    )

    if return_error
        residual = norm(A_global * solution_tt - rhs) / max(norm(rhs), eps(real(T)))
        return U, residual
    end
    return U
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
