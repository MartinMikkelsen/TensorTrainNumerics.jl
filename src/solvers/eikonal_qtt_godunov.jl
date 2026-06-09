# TT-cross based variable-speed Eikonal components.
#
# These routines keep nonlinear Godunov pieces in QTT form by sampling them with
# TT-cross from pointwise TT evaluations.  The serial 2D layout is
#   first d bits  -> y index
#   second d bits -> x index
# matching the vectorization used by the dense validation path.

function eikonal_serial_bits_2d(i::Int, j::Int, d::Int)
    N = 2^d
    1 <= i <= N || throw(BoundsError("row index i must be in 1:$N"))
    1 <= j <= N || throw(BoundsError("column index j must be in 1:$N"))
    return vcat(_bits_for_qtt_index(i, d), _bits_for_qtt_index(j, d))
end

function _eikonal_bits_to_ij(bits::AbstractVector{<:Integer}, d::Int)
    length(bits) == 2d || throw(DimensionMismatch("expected $(2d) serial QTT bits"))
    i = tuple_to_index(Tuple(bits[1:d]))
    j = tuple_to_index(Tuple(bits[(d + 1):(2d)]))
    return i, j
end

function _eikonal_shifted_bits_2d(bits::AbstractVector{<:Integer}, d::Int, di::Int, dj::Int)
    i, j = _eikonal_bits_to_ij(bits, d)
    N = 2^d
    inew = i + di
    jnew = j + dj
    if !(1 <= inew <= N && 1 <= jnew <= N)
        return nothing
    end
    return eikonal_serial_bits_2d(inew, jnew, d)
end

function _eikonal_cross_domain_2d(d::Int)
    return [Float64[1.0, 2.0] for _ in 1:(2d)]
end

function _eikonal_bit_matrix_to_ints(coords::AbstractMatrix)
    return round.(Int, coords)
end

function eikonal_point_values_qtt(q::QTTvector, bit_indices::AbstractMatrix{<:Integer})
    q.ordering == :serial || throw(ArgumentError("point evaluation currently expects serial QTT ordering"))
    size(bit_indices, 2) == q.N || throw(DimensionMismatch("bit index matrix must have $(q.N) columns"))
    return real.(_evaluate_tt(q.ttv_vec, Matrix{Int}(bit_indices), q.N))
end

function _eikonal_point_value_qtt(q::QTTvector, bits::AbstractVector{<:Integer})
    return eikonal_point_values_qtt(q, reshape(collect(Int, bits), 1, :))[1]
end

function _eikonal_point_or_boundary(q::QTTvector, bits)
    bits === nothing && return 0.0
    return _eikonal_point_value_qtt(q, bits)
end

function eikonal_function_cross_qtt_2d(
        f::Function,
        d::Int;
        alg::CrossAlgorithm = DMRG(verbose = false, tol = 1.0e-10, rmax = 64, maxiter = 10, kickrank = nothing),
        ranks::Union{Int, Vector{Int}} = 2,
        val_size::Int = 1000
    )
    N = 2^d
    h = 1.0 / (N + 1)
    domain = _eikonal_cross_domain_2d(d)
    sampler = coords -> begin
        bits = _eikonal_bit_matrix_to_ints(coords)
        out = Vector{Float64}(undef, size(bits, 1))
        @inbounds for p in axes(bits, 1)
            i, j = _eikonal_bits_to_ij(view(bits, p, :), d)
            out[p] = f(j * h, i * h)
        end
        out
    end
    tt = tt_cross(sampler, domain, alg; ranks = ranks, val_size = val_size)
    return QTTvector(tt, 2, d, :serial)
end

function eikonal_speed_cross_qtt_2d(
        d::Int;
        speed,
        alg::CrossAlgorithm = DMRG(verbose = false, tol = 1.0e-10, rmax = 64, maxiter = 10, kickrank = nothing),
        ranks::Union{Int, Vector{Int}} = 2,
        val_size::Int = 1000
    )
    if speed isa QTTvector
        speed.n_dims == 2 || throw(ArgumentError("speed QTTvector must have n_dims = 2"))
        speed.bits_per_dim == d || throw(ArgumentError("speed QTTvector bits_per_dim must equal d"))
        speed.ordering == :serial || throw(ArgumentError("speed QTTvector must use :serial ordering"))
        return speed
    elseif speed isa TTvector
        if all(==(2), speed.ttv_dims) && speed.N == 2d
            return QTTvector(speed, 2, d, :serial)
        end
        return QTTvector(to_qtt(speed, [fill(2, d), fill(2, d)]), 2, d, :serial)
    elseif speed isa Function
        return eikonal_function_cross_qtt_2d(speed, d;
            alg = alg,
            ranks = ranks,
            val_size = val_size)
    else
        throw(ArgumentError("TT-cross Eikonal speed must be a function, TTvector, or QTTvector"))
    end
end

function eikonal_speed_interp_qtt_2d(
        d::Int;
        speed,
        degree::Int = 6,
        tolerance::Float64 = 1.0e-12,
        maxbonddim::Int = typemax(Int)
    )
    if speed isa QTTvector
        speed.n_dims == 2 || throw(ArgumentError("speed QTTvector must have n_dims = 2"))
        speed.bits_per_dim == d || throw(ArgumentError("speed QTTvector bits_per_dim must equal d"))
        speed.ordering == :serial || throw(ArgumentError("speed QTTvector must use :serial ordering"))
        return speed
    elseif speed isa TTvector
        if all(==(2), speed.ttv_dims) && speed.N == 2d
            return QTTvector(speed, 2, d, :serial)
        end
        return QTTvector(to_qtt(speed, [fill(2, d), fill(2, d)]), 2, d, :serial)
    elseif speed isa Function
        return eikonal_function_interp_qtt_2d(speed, d;
            degree = degree,
            tolerance = tolerance,
            maxbonddim = maxbonddim)
    else
        throw(ArgumentError("Interpolative Eikonal speed must be a function, TTvector, or QTTvector"))
    end
end

function _eikonal_1d_upwind_operators(d::Int; h::Real)
    I1 = id_tto(d)
    dminus = (1.0 / h) * ∇(d)
    dplus = (1.0 / h) * (I1 - shift(d))
    return (; minus = dminus, plus = dplus, identity = I1)
end

function eikonal_upwind_operators_qtt_2d(d::Int; h::Real)
    ops1 = _eikonal_1d_upwind_operators(d; h = h)
    Iy = ops1.identity
    Ix = ops1.identity
    return (
        x_minus = QTToperator(kron(Iy, ops1.minus), 2, d, :serial),
        x_plus = QTToperator(kron(Iy, ops1.plus), 2, d, :serial),
        y_minus = QTToperator(kron(ops1.minus, Ix), 2, d, :serial),
        y_plus = QTToperator(kron(ops1.plus, Ix), 2, d, :serial),
    )
end

function _eikonal_godunov_sample(u::QTTvector, speed::QTTvector, bits::AbstractVector{<:Integer}; h::Real)
    d = u.bits_per_dim
    ui = _eikonal_point_value_qtt(u, bits)
    left = _eikonal_point_or_boundary(u, _eikonal_shifted_bits_2d(bits, d, 0, -1))
    right = _eikonal_point_or_boundary(u, _eikonal_shifted_bits_2d(bits, d, 0, 1))
    down = _eikonal_point_or_boundary(u, _eikonal_shifted_bits_2d(bits, d, -1, 0))
    up = _eikonal_point_or_boundary(u, _eikonal_shifted_bits_2d(bits, d, 1, 0))

    dxm = (ui - left) / h
    dxp = (ui - right) / h
    dym = (ui - down) / h
    dyp = (ui - up) / h

    gx = max(dxm, dxp, 0.0)
    gy = max(dym, dyp, 0.0)
    s = _eikonal_point_value_qtt(speed, bits)
    return (; residual = gx^2 + gy^2 - s^2, dxm = dxm, dxp = dxp, dym = dym, dyp = dyp)
end

function _eikonal_cross_qtt_2d(
        f::Function,
        d::Int;
        alg::CrossAlgorithm,
        ranks::Union{Int, Vector{Int}},
        val_size::Int
    )
    domain = _eikonal_cross_domain_2d(d)
    sampler = coords -> begin
        bits = _eikonal_bit_matrix_to_ints(coords)
        out = Vector{Float64}(undef, size(bits, 1))
        @inbounds for p in axes(bits, 1)
            out[p] = f(view(bits, p, :))
        end
        out
    end
    tt = tt_cross(sampler, domain, alg; ranks = ranks, val_size = val_size)
    return QTTvector(tt, 2, d, :serial)
end

function eikonal_godunov_residual_cross_qtt_2d(
        u::QTTvector,
        speed::QTTvector;
        h::Real,
        alg::CrossAlgorithm = DMRG(verbose = false, tol = 1.0e-10, rmax = 64, maxiter = 10, kickrank = nothing),
        ranks::Union{Int, Vector{Int}} = 2,
        val_size::Int = 1000
    )
    check_compat(u, speed)
    return _eikonal_cross_qtt_2d(u.bits_per_dim;
        alg = alg,
        ranks = ranks,
        val_size = val_size) do bits
        _eikonal_godunov_sample(u, speed, bits; h = h).residual
    end
end

function eikonal_godunov_coefficients_cross_qtt_2d(
        u::QTTvector;
        h::Real,
        alg::CrossAlgorithm = DMRG(verbose = false, tol = 1.0e-10, rmax = 64, maxiter = 10, kickrank = nothing),
        ranks::Union{Int, Vector{Int}} = 2,
        val_size::Int = 1000
    )
    d = u.bits_per_dim
    coeff(which::Symbol) = _eikonal_cross_qtt_2d(d;
        alg = alg,
        ranks = ranks,
        val_size = val_size) do bits
        ui = _eikonal_point_value_qtt(u, bits)
        left = _eikonal_point_or_boundary(u, _eikonal_shifted_bits_2d(bits, d, 0, -1))
        right = _eikonal_point_or_boundary(u, _eikonal_shifted_bits_2d(bits, d, 0, 1))
        down = _eikonal_point_or_boundary(u, _eikonal_shifted_bits_2d(bits, d, -1, 0))
        up = _eikonal_point_or_boundary(u, _eikonal_shifted_bits_2d(bits, d, 1, 0))
        dxm = (ui - left) / h
        dxp = (ui - right) / h
        dym = (ui - down) / h
        dyp = (ui - up) / h
        if which == :x_minus
            return dxm >= max(dxp, 0.0) ? 2 * dxm : 0.0
        elseif which == :x_plus
            return dxp > max(dxm, 0.0) ? 2 * dxp : 0.0
        elseif which == :y_minus
            return dym >= max(dyp, 0.0) ? 2 * dym : 0.0
        else
            return dyp > max(dym, 0.0) ? 2 * dyp : 0.0
        end
    end
    return (
        x_minus = coeff(:x_minus),
        x_plus = coeff(:x_plus),
        y_minus = coeff(:y_minus),
        y_plus = coeff(:y_plus),
    )
end

function _qtt_diag_operator(coeff::QTTvector)
    return QTToperator(ttv_to_diag_tto(TTvector(coeff)), coeff.n_dims, coeff.bits_per_dim, coeff.ordering)
end

function _qtt_operator_product(A::QTToperator, B::QTToperator)
    check_compat(A, B)
    return QTToperator(TToperator(A) * TToperator(B), A.n_dims, A.bits_per_dim, A.ordering)
end

function eikonal_godunov_jacobian_cross_qtt_2d(
        u::QTTvector;
        h::Real,
        alg::CrossAlgorithm = DMRG(verbose = false, tol = 1.0e-10, rmax = 64, maxiter = 10, kickrank = nothing),
        ranks::Union{Int, Vector{Int}} = 2,
        val_size::Int = 1000
    )
    coeffs = eikonal_godunov_coefficients_cross_qtt_2d(u;
        h = h,
        alg = alg,
        ranks = ranks,
        val_size = val_size)
    ops = eikonal_upwind_operators_qtt_2d(u.bits_per_dim; h = h)
    J = _qtt_operator_product(_qtt_diag_operator(coeffs.x_minus), ops.x_minus)
    J = J + _qtt_operator_product(_qtt_diag_operator(coeffs.x_plus), ops.x_plus)
    J = J + _qtt_operator_product(_qtt_diag_operator(coeffs.y_minus), ops.y_minus)
    J = J + _qtt_operator_product(_qtt_diag_operator(coeffs.y_plus), ops.y_plus)
    return J
end

function _eikonal_tt_norm(q::QTTvector)
    return norm(TTvector(q))
end

function _eikonal_residual_norm_qtt(r::QTTvector, speed::QTTvector)
    speed_sq = hadamard(speed, speed)
    return _eikonal_tt_norm(r) / max(_eikonal_tt_norm(speed_sq), eps(Float64))
end

function _eikonal_backtracking_qtt(
        u::QTTvector,
        delta::QTTvector,
        speed::QTTvector,
        current_residual::Real;
        h::Real,
        max_bond::Int,
        alg::CrossAlgorithm,
        ranks::Union{Int, Vector{Int}},
        val_size::Int
    )
    best_u = u
    best_residual = Float64(current_residual)
    for α in (1.0, 0.5, 0.25, 0.125, 0.0625)
        candidate = tt_compress!(u + α * delta, max_bond)
        residual = eikonal_godunov_residual_cross_qtt_2d(candidate, speed;
            h = h,
            alg = alg,
            ranks = ranks,
            val_size = val_size)
        rel = _eikonal_residual_norm_qtt(residual, speed)
        if isfinite(rel) && rel < best_residual
            return candidate, residual, rel
        end
    end
    residual = eikonal_godunov_residual_cross_qtt_2d(best_u, speed;
        h = h,
        alg = alg,
        ranks = ranks,
        val_size = val_size)
    return best_u, residual, best_residual
end

function eikonal_godunov_mals_2d_qtt(
        d::Int;
        speed,
        residual_tol::Real = 1.0e-10,
        max_newton::Int = 20,
        max_bond::Int = 64,
        mals_tol::Real = 0.0,
        cross_alg::CrossAlgorithm = DMRG(verbose = false, tol = 1.0e-10, rmax = max_bond, maxiter = 10, kickrank = nothing),
        cross_ranks::Union{Int, Vector{Int}} = 2,
        cross_val_size::Int = 1000,
        continuation_steps::Int = 1,
        verbose::Bool = false
    )
    h = 1.0 / (2^d + 1)
    target_speed = eikonal_speed_cross_qtt_2d(d;
        speed = speed,
        alg = cross_alg,
        ranks = cross_ranks,
        val_size = cross_val_size)
    one_speed = eikonal_function_cross_qtt_2d((_, _) -> 1.0, d;
        alg = cross_alg,
        ranks = 1,
        val_size = min(cross_val_size, 128))
    u = eikonal_function_cross_qtt_2d((x, y) -> min(x, y, 1 - x, 1 - y), d;
        alg = cross_alg,
        ranks = cross_ranks,
        val_size = cross_val_size)
    u = tt_compress!(u, max_bond)

    residual_history = Float64[]
    rank_history = Int[]
    θ_values = continuation_steps <= 1 ? (1.0,) : range(0.0, 1.0; length = continuation_steps)

    for θ in θ_values
        current_speed = θ == 1.0 ? target_speed : tt_compress!((1 - θ) * one_speed + θ * target_speed, max_bond)
        for iter in 0:max_newton
            residual = eikonal_godunov_residual_cross_qtt_2d(u, current_speed;
                h = h,
                alg = cross_alg,
                ranks = cross_ranks,
                val_size = cross_val_size)
            rel = _eikonal_residual_norm_qtt(residual, current_speed)
            push!(residual_history, rel)
            push!(rank_history, maximum(u.ttv_rks))
            verbose && println("  θ = $(round(θ, sigdigits = 3))  Newton $iter  residual = $(round(rel, sigdigits = 4))  max_rank = $(maximum(u.ttv_rks))")
            rel < residual_tol && break

            J = eikonal_godunov_jacobian_cross_qtt_2d(u;
                h = h,
                alg = cross_alg,
                ranks = cross_ranks,
                val_size = cross_val_size)
            rhs = -1.0 * residual
            delta_tt = _nonsymmetric_mals_linsolve(TToperator(J), TTvector(rhs), TTvector(rhs);
                max_bond = max_bond,
                tol = Float64(mals_tol))
            delta = QTTvector(delta_tt, 2, d, :serial)
            u, _, rel_new = _eikonal_backtracking_qtt(u, delta, current_speed, rel;
                h = h,
                max_bond = max_bond,
                alg = cross_alg,
                ranks = cross_ranks,
                val_size = cross_val_size)
            rel_new >= rel && break
        end
    end

    info = (
        residual_history = residual_history,
        rank_history = rank_history,
        h = h,
        N = 2^d,
        speed = target_speed,
        newton_steps = length(residual_history),
        used_dense_jacobian = false,
        cross_algorithm = typeof(cross_alg),
    )
    return u, info
end

"""
    eikonal_godunov_mals_2d_qtt_interp(d; speed, interp_degree, ...)

Newton-MALS Godunov Eikonal solver that uses InterpolativeQTT (Lindsey's method)
to build the speed-field QTT and the constant-1 reference field.

For smooth speed functions this gives significantly lower bond dimension than
TT-cross sampling.  The Godunov residual and Jacobian coefficients (which are
non-smooth near the wavefront) are still sampled via TT-cross.

Requires the InterpolativeQTT extension to be loaded
(`using InterpolativeQTT` or `import InterpolativeQTT`).
"""
function eikonal_godunov_mals_2d_qtt_interp(
        d::Int;
        speed,
        interp_degree::Int = 6,
        interp_tol::Float64 = 1.0e-12,
        interp_maxbond::Int = typemax(Int),
        residual_tol::Real = 1.0e-10,
        max_newton::Int = 20,
        max_bond::Int = 64,
        mals_tol::Real = 0.0,
        cross_alg::CrossAlgorithm = DMRG(verbose = false, tol = 1.0e-10, rmax = max_bond, maxiter = 10, kickrank = nothing),
        cross_ranks::Union{Int, Vector{Int}} = 2,
        cross_val_size::Int = 1000,
        continuation_steps::Int = 1,
        verbose::Bool = false
    )
    h = 1.0 / (2^d + 1)

    target_speed = eikonal_speed_interp_qtt_2d(d;
        speed = speed,
        degree = interp_degree,
        tolerance = interp_tol,
        maxbonddim = interp_maxbond)

    one_speed = eikonal_function_interp_qtt_2d((_, _) -> 1.0, d;
        degree = 1,
        tolerance = interp_tol)

    u = eikonal_function_cross_qtt_2d((x, y) -> min(x, y, 1 - x, 1 - y), d;
        alg = cross_alg,
        ranks = cross_ranks,
        val_size = cross_val_size)
    u = tt_compress!(u, max_bond)

    residual_history = Float64[]
    rank_history = Int[]
    θ_values = continuation_steps <= 1 ? (1.0,) : range(0.0, 1.0; length = continuation_steps)

    for θ in θ_values
        current_speed = θ == 1.0 ? target_speed : tt_compress!((1 - θ) * one_speed + θ * target_speed, max_bond)
        for iter in 0:max_newton
            residual = eikonal_godunov_residual_cross_qtt_2d(u, current_speed;
                h = h,
                alg = cross_alg,
                ranks = cross_ranks,
                val_size = cross_val_size)
            rel = _eikonal_residual_norm_qtt(residual, current_speed)
            push!(residual_history, rel)
            push!(rank_history, maximum(u.ttv_rks))
            verbose && println("  θ = $(round(θ, sigdigits = 3))  Newton $iter  residual = $(round(rel, sigdigits = 4))  max_rank = $(maximum(u.ttv_rks))")
            rel < residual_tol && break

            J = eikonal_godunov_jacobian_cross_qtt_2d(u;
                h = h,
                alg = cross_alg,
                ranks = cross_ranks,
                val_size = cross_val_size)
            rhs = -1.0 * residual
            delta_tt = _nonsymmetric_mals_linsolve(TToperator(J), TTvector(rhs), TTvector(rhs);
                max_bond = max_bond,
                tol = Float64(mals_tol))
            delta = QTTvector(delta_tt, 2, d, :serial)
            u, _, rel_new = _eikonal_backtracking_qtt(u, delta, current_speed, rel;
                h = h,
                max_bond = max_bond,
                alg = cross_alg,
                ranks = cross_ranks,
                val_size = cross_val_size)
            rel_new >= rel && break
        end
    end

    info = (
        residual_history = residual_history,
        rank_history = rank_history,
        h = h,
        N = 2^d,
        speed = target_speed,
        newton_steps = length(residual_history),
        used_dense_jacobian = false,
        interp_degree = interp_degree,
    )
    return u, info
end
