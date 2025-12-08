using LinearAlgebra
using Random
using Maxvol

function _build_fiber_indices_left(lsets, rsets, j, Is, Rs, N)
    n_fibers = Rs[j] * Is[j] * Rs[j + 1]
    indices = Matrix{Int}(undef, n_fibers, N)
    idx = 1
    for r_right in 1:Rs[j + 1]
        for r_left in 1:Rs[j]
            for i in 1:Is[j]
                if j == 1
                    left_idx = Int[]
                else
                    left_idx = lsets[j][r_left, :]
                end
                if j == N
                    right_idx = Int[]
                else
                    right_idx = rsets[j][r_right, :]
                end
                indices[idx, :] = vcat(left_idx, [i], right_idx)
                idx += 1
            end
        end
    end
    return indices
end

function _build_fiber_indices_right(lsets, rsets, j, Is, Rs, N)
    n_fibers = Rs[j] * Is[j] * Rs[j + 1]
    indices = Matrix{Int}(undef, n_fibers, N)
    idx = 1
    for r_left in 1:Rs[j]
        for i in 1:Is[j]
            for r_right in 1:Rs[j + 1]
                if j == 1
                    left_idx = Int[]
                else
                    left_idx = lsets[j][r_left, :]
                end
                if j == N
                    right_idx = Int[]
                else
                    right_idx = rsets[j][r_right, :]
                end
                indices[idx, :] = vcat(left_idx, [i], right_idx)
                idx += 1
            end
        end
    end
    return indices
end

function _evaluate_on_domain(f, domain::Vector{<:AbstractVector}, indices::Matrix{Int})
    N = length(domain)
    n_points = size(indices, 1)
    T = eltype(domain[1])
    coords = Matrix{T}(undef, n_points, N)
    for p in 1:n_points
        for d in 1:N
            coords[p, d] = domain[d][indices[p, d]]
        end
    end
    return f(coords)
end

function _cap_ranks!(Rs, Is, rmax)
    N = length(Is)
    for n in 2:N
        Rs[n] = min(Rs[n - 1] * Is[n - 1], Rs[n], Is[n] * Rs[n + 1], rmax)
    end
    for n in (N - 1):-1:1
        Rs[n + 1] = min(Rs[n] * Is[n], Rs[n + 1], Is[n + 1] * Rs[n + 2], rmax)
    end
    return Rs
end

function _evaluate_tt(cores, indices, N)
    T = eltype(cores[1])
    n_points = size(indices, 1)
    result = zeros(T, n_points)
    for p in 1:n_points
        val = ones(T, 1)
        for d in 1:N
            idx = indices[p, d]
            core_slice = cores[d][idx, :, :]
            val = val' * core_slice
            val = vec(val)
        end
        result[p] = val[1]
    end
    return result
end

"""
    tt_cross(f::Function, domain::Vector{<:AbstractVector{T}}; kwargs...)

Performs TT-cross interpolation for constructing a tensor-train approximation of a multivariate function.

# Arguments
- `f::Function`: The multivariate function to approximate
- `domain::Vector{<:AbstractVector{T}}`: A vector of vectors defining the domain for each dimension

# Keyword Arguments
- `ranks_tt::Union{Int, Vector{Int}}`: Target TT-ranks (default: 2)
- `kickrank::Union{Nothing, Int}`: Rank increment for enrichment (default: 3)
- `rmax::Int`: Maximum allowed rank (default: 100)
- `eps::Float64`: Convergence tolerance (default: 1.0e-6)
- `max_iter::Int`: Maximum number of iterations (default: 25)

# References
- Ivan Oseledets, Eugene Tyrtyshnikov, "TT-cross approximation for multidimensional arrays"
"""
function tt_cross(
        f::Function,
        domain::Vector{<:AbstractVector{T}};
        ranks_tt::Union{Int, Vector{Int}} = 2,
        kickrank::Union{Nothing, Int} = 3,
        rmax::Int = 100,
        eps::Float64 = 1.0e-6,
        max_iter::Int = 25,
        val_size::Int = 1000,
        verbose::Bool = true
    ) where {T <: Number}

    N = length(domain)
    Is = [length(d) for d in domain]

    if isa(ranks_tt, Int)
        ranks_tt_vec = fill(ranks_tt, N - 1)
    else
        ranks_tt_vec = copy(ranks_tt)
    end
    Rs = vcat([1], ranks_tt_vec, [1])
    _cap_ranks!(Rs, Is, rmax)

    cores = Vector{Array{T, 3}}(undef, N)
    for n in 1:N
        cores[n] = randn(T, Is[n], Rs[n], Rs[n + 1])
    end

    lsets = Vector{Matrix{Int}}(undef, N)
    rsets = Vector{Matrix{Int}}(undef, N)
    lsets[1] = ones(Int, 1, 0)
    rsets[N] = ones(Int, 1, 0)

    max_R = maximum(Rs)
    randint = zeros(Int, max_R, N)
    for n in 1:N
        for r in 1:max_R
            randint[r, n] = rand(1:Is[n])
        end
    end

    for n in 1:(N - 1)
        rsets[n] = randint[1:Rs[n + 1], (n + 1):N]
    end
    for n in 2:N
        lsets[n] = zeros(Int, 0, 0)
    end

    Xs_val = Matrix{Int}(undef, val_size, N)
    for d in 1:N
        Xs_val[:, d] = rand(1:Is[d], val_size)
    end
    val_coords = Matrix{T}(undef, val_size, N)
    for p in 1:val_size
        for d in 1:N
            val_coords[p, d] = domain[d][Xs_val[p, d]]
        end
    end
    ys_val = f(val_coords)
    norm_ys_val = norm(ys_val)
    if norm_ys_val < eps
        norm_ys_val = one(T)
    end

    if verbose
        @info "Cross-interpolation over a $(N)D domain containing $(prod(Is)) grid points"
    end

    converged = false
    val_eps = Inf

    for iter in 1:max_iter
        if verbose
            @info "iterations: $(iter)"
        end

        left_locals = Vector{Vector{Int}}(undef, N - 1)

        for j in 1:(N - 1)
            indices = _build_fiber_indices_left(lsets, rsets, j, Is, Rs, N)
            V = _evaluate_on_domain(f, domain, indices)
            V = reshape(V, Rs[j] * Is[j], Rs[j + 1])

            Q, _ = qr(V)
            Q_mat = Matrix(Q)

            local_indices, _ = maxvol!(copy(Q_mat))
            left_locals[j] = local_indices

            V_new = Q_mat / Q_mat[local_indices, :]
            cores[j] = reshape(V_new, Is[j], Rs[j], Rs[j + 1])

            local_r = zeros(Int, length(local_indices))
            local_i = zeros(Int, length(local_indices))
            for (k, idx) in enumerate(local_indices)
                local_i[k] = ((idx - 1) % Is[j]) + 1
                local_r[k] = ((idx - 1) ÷ Is[j]) + 1
            end

            if j == 1
                lsets[j + 1] = reshape(local_i, length(local_i), 1)
            else
                new_lset = zeros(Int, length(local_indices), j)
                for k in 1:length(local_indices)
                    new_lset[k, 1:(j - 1)] = lsets[j][local_r[k], :]
                    new_lset[k, j] = local_i[k]
                end
                lsets[j + 1] = new_lset
            end

            Rs[j + 1] = length(local_indices)
        end

        for j in N:-1:2
            indices = _build_fiber_indices_right(lsets, rsets, j, Is, Rs, N)
            V = _evaluate_on_domain(f, domain, indices)
            V_mat = reshape(V, Is[j] * Rs[j + 1], Rs[j])

            Q, _ = qr(V_mat)
            Q_mat = Matrix(Q)

            local_indices, _ = maxvol!(copy(Q_mat))

            V_new = Q_mat / Q_mat[local_indices, :]
            core = zeros(T, Is[j], Rs[j], Rs[j + 1])
            for r in 1:Rs[j]
                for i in 1:Is[j]
                    for rp in 1:Rs[j + 1]
                        row_idx = (i - 1) * Rs[j + 1] + rp
                        core[i, r, rp] = V_new[row_idx, r]
                    end
                end
            end
            cores[j] = core

            local_i = zeros(Int, length(local_indices))
            local_r = zeros(Int, length(local_indices))
            for (k, idx) in enumerate(local_indices)
                local_r[k] = ((idx - 1) % Rs[j + 1]) + 1
                local_i[k] = ((idx - 1) ÷ Rs[j + 1]) + 1
            end

            if j == N
                rsets[j - 1] = reshape(local_i, length(local_i), 1)
            else
                new_rset = zeros(Int, length(local_indices), N - j + 1)
                for k in 1:length(local_indices)
                    new_rset[k, 1] = local_i[k]
                    new_rset[k, 2:end] = rsets[j][local_r[k], :]
                end
                rsets[j - 1] = new_rset
            end

            Rs[j] = length(local_indices)
        end

        indices_first = _build_fiber_indices_left(lsets, rsets, 1, Is, Rs, N)
        V_first = _evaluate_on_domain(f, domain, indices_first)
        cores[1] = reshape(V_first, Is[1], Rs[1], Rs[2])

        y_approx = _evaluate_tt(cores, Xs_val, N)
        val_eps = norm(ys_val - y_approx) / norm_ys_val

        if verbose
            @info "ε: $(val_eps), largest rank: $(maximum(Rs))"
        end

        if val_eps < eps
            converged = true
            if verbose
                @info "converged! ε < $(eps)"
            end
            break
        end

        if !converged && iter < max_iter && kickrank !== nothing
            newRs = copy(Rs)
            for n in 2:N
                newRs[n] = min(rmax, newRs[n] + kickrank)
            end
            _cap_ranks!(newRs, Is, rmax)

            for n in 1:(N - 1)
                if newRs[n + 1] > Rs[n + 1]
                    extra_rows = newRs[n + 1] - Rs[n + 1]
                    n_cols = N - n
                    extra = zeros(Int, extra_rows, n_cols)
                    for er in 1:extra_rows
                        for col in 1:n_cols
                            extra[er, col] = rand(1:Is[n + col])
                        end
                    end
                    rsets[n] = vcat(rsets[n], extra)
                end
            end
            Rs = newRs
        end
    end

    if !converged && verbose
        @warn "Warning: max_iter reached without convergence (eps = $(val_eps))"
    end

    dims = Tuple(Is)
    rks = copy(Rs)
    ot = zeros(Int, N)

    return TTvector{T, N}(N, cores, dims, rks, ot)
end

function tt_cross(
        f::Function,
        dims::NTuple{N, Int};
        kwargs...
    ) where {N}
    domain = [collect(1.0:Float64(d)) for d in dims]
    return tt_cross(f, domain; kwargs...)
end

function tt_cross(
        f::Function,
        dims::Vector{Int};
        kwargs...
    )
    domain = [collect(1.0:Float64(d)) for d in dims]
    return tt_cross(f, domain; kwargs...)
end
