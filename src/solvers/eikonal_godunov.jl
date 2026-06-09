# Variable-speed Eikonal solver components based on monotone Godunov upwinding.
#
# PDE: |grad u(x)| = speed(x), speed(x) > 0, with zero boundary values.

function godunov_residual_1d(u::AbstractVector, speed::AbstractVector; h::Real)
    N = length(u)
    @assert length(speed) == N
    T = promote_type(Float64, eltype(u), eltype(speed))
    r = zeros(T, N)
    hT = T(h)
    for i in 1:N
        uim = i == 1 ? zero(T) : T(u[i - 1])
        uip = i == N ? zero(T) : T(u[i + 1])
        ui = T(u[i])
        dm = (ui - uim) / hT
        dp = (uip - ui) / hT
        gx = max(dm, -dp, zero(T))
        r[i] = gx^2 - T(speed[i])^2
    end
    return r
end

function godunov_residual_2d(u::AbstractMatrix, speed::AbstractMatrix; h::Real)
    Ny, Nx = size(u)
    @assert size(speed) == size(u)
    T = promote_type(Float64, eltype(u), eltype(speed))
    r = zeros(T, Ny, Nx)
    hT = T(h)
    for j in 1:Nx
        for i in 1:Ny
            ui = T(u[i, j])
            uim = j == 1 ? zero(T) : T(u[i, j - 1])
            uip = j == Nx ? zero(T) : T(u[i, j + 1])
            ujm = i == 1 ? zero(T) : T(u[i - 1, j])
            ujp = i == Ny ? zero(T) : T(u[i + 1, j])
            dxm = (ui - uim) / hT
            dxp = (uip - ui) / hT
            dym = (ui - ujm) / hT
            dyp = (ujp - ui) / hT
            gx = max(dxm, -dxp, zero(T))
            gy = max(dym, -dyp, zero(T))
            r[i, j] = gx^2 + gy^2 - T(speed[i, j])^2
        end
    end
    return r
end

function _godunov_update(a::Real, b::Real, s::Real, h::Real)
    sh = s * h
    lo, hi = minmax(a, b)
    if hi - lo >= sh
        return lo + sh
    end
    disc = 2 * sh^2 - (hi - lo)^2
    return (lo + hi + sqrt(max(disc, 0.0))) / 2
end

function dense_fast_sweeping_eikonal_2d(
        speed::AbstractMatrix;
        h::Real,
        boundary::Symbol = :zero,
        max_sweeps::Int = 10_000,
        tol::Real = 1.0e-12
    )
    boundary == :zero || throw(ArgumentError("only boundary = :zero is supported"))
    Ny, Nx = size(speed)
    T = promote_type(Float64, eltype(speed))
    smax = maximum(T.(speed))
    u = fill(T((Ny + Nx + 2) * h * smax), Ny, Nx)
    hist = Float64[]
    row_orders = (1:Ny, Ny:-1:1)
    col_orders = (1:Nx, Nx:-1:1)

    for _ in 1:max_sweeps
        max_change = zero(T)
        for rows in row_orders
            for cols in col_orders
                for j in cols
                    for i in rows
                        left = j == 1 ? zero(T) : u[i, j - 1]
                        right = j == Nx ? zero(T) : u[i, j + 1]
                        down = i == 1 ? zero(T) : u[i - 1, j]
                        up = i == Ny ? zero(T) : u[i + 1, j]
                        candidate = T(_godunov_update(min(left, right), min(down, up), T(speed[i, j]), h))
                        old = u[i, j]
                        if candidate < old
                            u[i, j] = candidate
                            max_change = max(max_change, abs(old - candidate))
                        end
                    end
                end
            end
        end
        push!(hist, Float64(max_change))
        max_change < tol && break
    end
    return u, hist
end

function dense_fast_sweeping_eikonal_1d(
        speed::AbstractVector;
        h::Real,
        boundary::Symbol = :zero,
        tol::Real = 1.0e-12
    )
    boundary == :zero || throw(ArgumentError("only boundary = :zero is supported"))
    N = length(speed)
    T = promote_type(Float64, eltype(speed))
    hT = T(h)
    left = zeros(T, N)
    right = zeros(T, N)
    for i in 1:N
        left[i] = (i == 1 ? zero(T) : left[i - 1]) + hT * T(speed[i])
    end
    for i in N:-1:1
        right[i] = (i == N ? zero(T) : right[i + 1]) + hT * T(speed[i])
    end
    u = min.(left, right)
    residual = godunov_residual_1d(u, speed; h = h)
    return u, [Float64(maximum(abs.(residual)))]
end

function _vector_to_qtt(values::AbstractVector, d::Int; max_bond::Int = typemax(Int))
    length(values) == 2^d || throw(DimensionMismatch("expected vector of length 2^d"))
    tensor = zeros(eltype(values), ntuple(_ -> 2, d))
    for pos in 1:(2^d)
        tensor[CartesianIndex(Tuple(_bits_for_qtt_index(pos, d)))] = values[pos]
    end
    q = ttv_decomp(tensor)
    return max_bond == typemax(Int) ? q : tt_compress!(q, max_bond)
end

function _bits_for_qtt_index(pos::Int, d::Int)
    return reverse(digits(pos - 1, base = 2, pad = d)) .+ 1
end

function _matrix_to_qtto(matrix::AbstractMatrix, d::Int)
    size(matrix) == (2^d, 2^d) || throw(DimensionMismatch("expected a 2^d by 2^d matrix"))
    tensor = zeros(eltype(matrix), ntuple(_ -> 2, 2d))
    for col in 1:(2^d)
        col_bits = _bits_for_qtt_index(col, d)
        for row in 1:(2^d)
            row_bits = _bits_for_qtt_index(row, d)
            tensor[CartesianIndex(Tuple(vcat(row_bits, col_bits)))] = matrix[row, col]
        end
    end
    return tto_decomp(tensor)
end

function _grid_to_vector_2d(grid::AbstractMatrix)
    return vec(permutedims(grid))
end

function _vector_to_grid_2d(values::AbstractVector, d::Int)
    N = 2^d
    return permutedims(reshape(values, N, N))
end

function _grid_to_qtt_2d(grid::AbstractMatrix, d::Int; max_bond::Int = typemax(Int))
    return _vector_to_qtt(_grid_to_vector_2d(grid), 2d; max_bond = max_bond)
end

function _qtt_to_grid_2d(u::AbstractTTvector, d::Int)
    return _vector_to_grid_2d(real.(qtt_to_function(u)), d)
end

function godunov_residual_qtt_1d(
        u::TTvector,
        speed::TTvector;
        h::Real,
        max_bond::Int = 32
    )
    d = u.N
    uvals = real.(qtt_to_function(u))
    speed_vals = real.(qtt_to_function(speed))
    residual = godunov_residual_1d(uvals, speed_vals; h = h)
    return _vector_to_qtt(residual, d; max_bond = max_bond)
end

function _active_masks_1d_values(u::AbstractVector; h::Real)
    N = length(u)
    T = promote_type(Float64, eltype(u))
    hT = T(h)
    mminus = zeros(T, N)
    mplus = zeros(T, N)
    for i in 1:N
        uim = i == 1 ? zero(T) : T(u[i - 1])
        uip = i == N ? zero(T) : T(u[i + 1])
        ui = T(u[i])
        dm = (ui - uim) / hT
        neg_dp = (ui - uip) / hT
        if dm >= max(neg_dp, zero(T))
            mminus[i] = one(T)
        elseif neg_dp > max(dm, zero(T))
            mplus[i] = one(T)
        end
    end
    return (; minus = mminus, plus = mplus)
end

function godunov_active_masks_1d(
        u::TTvector;
        h::Real,
        max_bond::Int = 32
    )
    d = u.N
    masks = _active_masks_1d_values(real.(qtt_to_function(u)); h = h)
    return (
        minus = _vector_to_qtt(masks.minus, d; max_bond = max_bond),
        plus = _vector_to_qtt(masks.plus, d; max_bond = max_bond),
    )
end

function _godunov_jacobian_1d_matrix(u::AbstractVector; h::Real)
    N = length(u)
    T = promote_type(Float64, eltype(u))
    hT = T(h)
    masks = _active_masks_1d_values(u; h = h)
    J = zeros(T, N, N)
    for i in 1:N
        ui = T(u[i])
        if masks.minus[i] == one(T)
            uim = i == 1 ? zero(T) : T(u[i - 1])
            g = (ui - uim) / hT
            coeff = 2 * g / hT
            J[i, i] += coeff
            i > 1 && (J[i, i - 1] -= coeff)
        elseif masks.plus[i] == one(T)
            uip = i == N ? zero(T) : T(u[i + 1])
            g = (ui - uip) / hT
            coeff = 2 * g / hT
            J[i, i] += coeff
            i < N && (J[i, i + 1] -= coeff)
        end
    end
    return J
end

function godunov_jacobian_qtt_1d(
        u::TTvector;
        h::Real,
        max_bond::Int = 32
    )
    d = u.N
    J = _godunov_jacobian_1d_matrix(real.(qtt_to_function(u)); h = h)::Matrix{Float64}
    return _matrix_to_qtto(J, d)
end

function _speed_to_qtt_1d(speed, d::Int)
    h = 1.0 / (2^d + 1)
    if speed isa TTvector
        return speed
    elseif speed isa Function
        values = [speed(i * h) for i in 1:(2^d)]
        return _vector_to_qtt(values, d)
    else
        values = collect(speed)
        return _vector_to_qtt(values, d)
    end
end

function _speed_to_qtt_2d(speed, d::Int)
    if speed isa TTvector
        return speed
    end
    N = 2^d
    h = 1.0 / (N + 1)
    if speed isa Function
        grid = [speed(j * h, i * h) for i in 1:N, j in 1:N]
        return _grid_to_qtt_2d(grid, d)
    else
        return _grid_to_qtt_2d(speed, d)
    end
end

function _relative_godunov_residual_1d_values(uvals::AbstractVector, speed_vals::AbstractVector; h::Real)
    residual = godunov_residual_1d(uvals, speed_vals; h = h)
    denom = sqrt(sum(abs2, speed_vals .^ 2))
    return sqrt(sum(abs2, residual)) / max(denom, eps(Float64)), residual
end

function godunov_residual_qtt_2d(
        u::TTvector,
        speed::TTvector;
        h::Real,
        max_bond::Int = 48
    )
    d = div(u.N, 2)
    ugrid = _qtt_to_grid_2d(u, d)
    speed_grid = _qtt_to_grid_2d(speed, d)
    residual = godunov_residual_2d(ugrid, speed_grid; h = h)::Matrix{Float64}
    return _grid_to_qtt_2d(residual, d; max_bond = max_bond)
end

function _relative_godunov_residual_2d_values(ugrid::AbstractMatrix, speed_grid::AbstractMatrix; h::Real)
    residual = godunov_residual_2d(ugrid, speed_grid; h = h)
    denom = sqrt(sum(abs2, speed_grid .^ 2))
    return sqrt(sum(abs2, residual)) / max(denom, eps(Float64)), residual
end

function _initial_eikonal_1d(init::Symbol, speed_vals::AbstractVector, d::Int; h::Real, max_bond::Int)
    if init == :fast_sweeping
        uvals, _ = dense_fast_sweeping_eikonal_1d(speed_vals; h = h)
        return _vector_to_qtt(uvals, d; max_bond = max_bond)
    elseif init == :half_fast_sweeping
        uvals, _ = dense_fast_sweeping_eikonal_1d(speed_vals; h = h)
        return _vector_to_qtt(0.5 .* uvals, d; max_bond = max_bond)
    else
        throw(ArgumentError("unknown 1D Eikonal initializer $(repr(init))"))
    end
end

function _initial_eikonal_2d(init::Symbol, speed_grid::AbstractMatrix, d::Int; h::Real, max_bond::Int)
    if init == :fast_sweeping
        ugrid, _ = dense_fast_sweeping_eikonal_2d(speed_grid; h = h)
        return _grid_to_qtt_2d(ugrid, d; max_bond = max_bond)
    elseif init == :half_fast_sweeping
        ugrid, _ = dense_fast_sweeping_eikonal_2d(speed_grid; h = h)
        return _grid_to_qtt_2d(0.5 .* ugrid, d; max_bond = max_bond)
    else
        throw(ArgumentError("unknown 2D Eikonal initializer $(repr(init))"))
    end
end

function _godunov_jacobian_2d_matrix(u::AbstractMatrix; h::Real)
    Ny, Nx = size(u)
    T = promote_type(Float64, eltype(u))
    hT = T(h)
    idx(i, j) = j + (i - 1) * Nx
    J = zeros(T, Ny * Nx, Ny * Nx)
    for j in 1:Nx
        for i in 1:Ny
            row = idx(i, j)
            ui = T(u[i, j])

            left = j == 1 ? zero(T) : T(u[i, j - 1])
            right = j == Nx ? zero(T) : T(u[i, j + 1])
            dxm = (ui - left) / hT
            neg_dxp = (ui - right) / hT
            if dxm >= max(neg_dxp, zero(T))
                coeff = 2 * dxm / hT
                J[row, idx(i, j)] += coeff
                j > 1 && (J[row, idx(i, j - 1)] -= coeff)
            elseif neg_dxp > max(dxm, zero(T))
                coeff = 2 * neg_dxp / hT
                J[row, idx(i, j)] += coeff
                j < Nx && (J[row, idx(i, j + 1)] -= coeff)
            end

            down = i == 1 ? zero(T) : T(u[i - 1, j])
            up = i == Ny ? zero(T) : T(u[i + 1, j])
            dym = (ui - down) / hT
            neg_dyp = (ui - up) / hT
            if dym >= max(neg_dyp, zero(T))
                coeff = 2 * dym / hT
                J[row, idx(i, j)] += coeff
                i > 1 && (J[row, idx(i - 1, j)] -= coeff)
            elseif neg_dyp > max(dym, zero(T))
                coeff = 2 * neg_dyp / hT
                J[row, idx(i, j)] += coeff
                i < Ny && (J[row, idx(i + 1, j)] -= coeff)
            end
        end
    end
    return J
end

function godunov_jacobian_qtt_2d(
        u::TTvector;
        h::Real,
        max_bond::Int = 48
    )
    d = div(u.N, 2)
    J = _godunov_jacobian_2d_matrix(_qtt_to_grid_2d(u, d); h = h)::Matrix{Float64}
    return _matrix_to_qtto(J, 2d)
end

function _backtracking_update_1d(
        u::TTvector,
        delta::TTvector,
        speed_vals::AbstractVector;
        h::Real,
        current_residual::Real,
        max_bond::Int
    )
    best_u = u
    best_residual = Float64(current_residual)
    for α in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
        candidate = tt_compress!(u + α * delta, max_bond)
        candidate_vals = real.(qtt_to_function(candidate))
        candidate_residual, _ = _relative_godunov_residual_1d_values(candidate_vals, speed_vals; h = h)
        if isfinite(candidate_residual) && candidate_residual < best_residual
            return candidate, candidate_residual
        end
    end
    return best_u, best_residual
end

function _backtracking_update_2d(
        u::TTvector,
        delta::TTvector,
        speed_grid::AbstractMatrix,
        d::Int;
        h::Real,
        current_residual::Real,
        max_bond::Int
    )
    best_u = u
    best_residual = Float64(current_residual)
    for α in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
        candidate = tt_compress!(u + α * delta, max_bond)
        candidate_grid = _qtt_to_grid_2d(candidate, d)
        candidate_residual, _ = _relative_godunov_residual_2d_values(candidate_grid, speed_grid; h = h)
        if isfinite(candidate_residual) && candidate_residual < best_residual
            return candidate, candidate_residual
        end
    end
    return best_u, best_residual
end

function eikonal_godunov_mals_1d(
        d::Int;
        speed,
        boundary::Symbol = :zero,
        init::Symbol = :fast_sweeping,
        max_newton::Int = 30,
        residual_tol::Real = 1.0e-10,
        max_bond::Int = 64,
        mals_tol::Real = 0.0,
        damping::Symbol = :backtracking,
        verbose::Bool = false
    )
    boundary == :zero || throw(ArgumentError("only boundary = :zero is supported"))
    damping == :backtracking || throw(ArgumentError("only damping = :backtracking is supported"))
    h = 1.0 / (2^d + 1)
    speed_qtt = _speed_to_qtt_1d(speed, d)
    speed_vals = real.(qtt_to_function(speed_qtt))
    minimum(speed_vals) > 0 || throw(DomainError(minimum(speed_vals), "speed must be positive"))

    u = _initial_eikonal_1d(init, speed_vals, d; h = h, max_bond = max_bond)
    residual_history = Float64[]
    rank_history = Int[]

    for iter in 0:max_newton
        uvals = real.(qtt_to_function(u))
        rel_residual, residual_vals = _relative_godunov_residual_1d_values(uvals, speed_vals; h = h)
        push!(residual_history, rel_residual)
        push!(rank_history, maximum(u.ttv_rks))
        verbose && println("  Newton $iter  residual = $(round(rel_residual, sigdigits = 4))  max_rank = $(maximum(u.ttv_rks))")
        rel_residual < residual_tol && break

        J = godunov_jacobian_qtt_1d(u; h = h, max_bond = max_bond)
        rhs = _vector_to_qtt(-residual_vals, d; max_bond = max_bond)
        delta = _nonsymmetric_mals_linsolve(J, rhs, rhs;
            max_bond = max_bond,
            tol = Float64(mals_tol))
        u, _ = _backtracking_update_1d(u, delta, speed_vals;
            h = h,
            current_residual = rel_residual,
            max_bond = max_bond)
    end

    info = (
        residual_history = residual_history,
        rank_history = rank_history,
        h = h,
        speed = speed_qtt,
        newton_steps = max(length(residual_history) - 1, 0),
    )
    return u, info
end

function eikonal_godunov_mals_2d(
        d::Int;
        speed,
        boundary::Symbol = :zero,
        init::Symbol = :fast_sweeping,
        max_newton::Int = 30,
        residual_tol::Real = 1.0e-10,
        max_bond::Int = 64,
        mals_tol::Real = 0.0,
        damping::Symbol = :backtracking,
        verbose::Bool = false
    )
    boundary == :zero || throw(ArgumentError("only boundary = :zero is supported"))
    damping == :backtracking || throw(ArgumentError("only damping = :backtracking is supported"))
    h = 1.0 / (2^d + 1)
    speed_qtt = _speed_to_qtt_2d(speed, d)
    speed_grid = _qtt_to_grid_2d(speed_qtt, d)
    minimum(speed_grid) > 0 || throw(DomainError(minimum(speed_grid), "speed must be positive"))

    u = _initial_eikonal_2d(init, speed_grid, d; h = h, max_bond = max_bond)
    residual_history = Float64[]
    rank_history = Int[]

    for iter in 0:max_newton
        ugrid = _qtt_to_grid_2d(u, d)
        rel_residual, residual_grid = _relative_godunov_residual_2d_values(ugrid, speed_grid; h = h)
        push!(residual_history, rel_residual)
        push!(rank_history, maximum(u.ttv_rks))
        verbose && println("  Newton $iter  residual = $(round(rel_residual, sigdigits = 4))  max_rank = $(maximum(u.ttv_rks))")
        rel_residual < residual_tol && break

        J = godunov_jacobian_qtt_2d(u; h = h, max_bond = max_bond)
        rhs = _grid_to_qtt_2d(-residual_grid, d; max_bond = max_bond)
        delta = _nonsymmetric_mals_linsolve(J, rhs, rhs;
            max_bond = max_bond,
            tol = Float64(mals_tol))
        u, _ = _backtracking_update_2d(u, delta, speed_grid, d;
            h = h,
            current_residual = rel_residual,
            max_bond = max_bond)
    end

    info = (
        residual_history = residual_history,
        rank_history = rank_history,
        h = h,
        speed = speed_qtt,
        newton_steps = max(length(residual_history) - 1, 0),
    )
    return u, info
end
