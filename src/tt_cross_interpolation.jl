using LinearAlgebra
using Random
using Maxvol

abstract type CrossAlgorithm end
abstract type PivotAlgorithm end

const CROSS_MAXITER = Ref{Int}(50)
const CROSS_TOL = Ref{Float64}(1.0e-10)
const CROSS_RMAX = Ref{Int}(500)
const CROSS_KICKRANK = Ref{Int}(3)
const MAXVOL_TOL = Ref{Float64}(1.05)

struct MaxVolPivot{T <: Real} <: PivotAlgorithm
    tol::T
    maxiter::Int
end

function MaxVolPivot(; tol::Real = MAXVOL_TOL[], maxiter::Int = 100)
    return MaxVolPivot(tol, maxiter)
end

struct RandomPivot <: PivotAlgorithm
    nsamples::Int
    seed::Union{Nothing, Int}
end

function RandomPivot(; nsamples::Int = 1000, seed::Union{Nothing, Int} = nothing)
    return RandomPivot(nsamples, seed)
end

struct MaxVol{T <: Real, P <: PivotAlgorithm} <: CrossAlgorithm
    maxiter::Int
    tol::T
    rmax::Int
    kickrank::Union{Nothing, Int}
    verbose::Bool
    pivot::P
end

function MaxVol(;
        maxiter::Int = CROSS_MAXITER[],
        tol::Real = CROSS_TOL[],
        rmax::Int = CROSS_RMAX[],
        kickrank::Union{Nothing, Int} = CROSS_KICKRANK[],
        verbose::Bool = true,
        pivot::PivotAlgorithm = MaxVolPivot()
    )
    return MaxVol(maxiter, tol, rmax, kickrank, verbose, pivot)
end

struct Greedy{T <: Real, P <: PivotAlgorithm} <: CrossAlgorithm
    maxiter::Int
    tol::T
    rmax::Int
    verbose::Bool
    nsamples::Int
    pivot::P
end

function Greedy(;
        maxiter::Int = CROSS_MAXITER[],
        tol::Real = CROSS_TOL[],
        rmax::Int = CROSS_RMAX[],
        verbose::Bool = true,
        nsamples::Int = 1000,
        pivot::PivotAlgorithm = RandomPivot()
    )
    return Greedy(maxiter, tol, rmax, verbose, nsamples, pivot)
end

struct DMRG{T <: Real, P <: PivotAlgorithm} <: CrossAlgorithm
    maxiter::Int
    tol::T
    rmax::Int
    kickrank::Union{Nothing, Int}
    verbose::Bool
    pivot::P
    nsweeps_inner::Int
end

function DMRG(;
        maxiter::Int = CROSS_MAXITER[],
        tol::Real = CROSS_TOL[],
        rmax::Int = CROSS_RMAX[],
        kickrank::Union{Nothing, Int} = CROSS_KICKRANK[],
        verbose::Bool = true,
        pivot::PivotAlgorithm = MaxVolPivot(),
        nsweeps_inner::Int = 2
    )
    return DMRG(maxiter, tol, rmax, kickrank, verbose, pivot, nsweeps_inner)
end

function tt_cross(f::Function, domain, alg::MaxVol; kwargs...)
end

function tt_cross(f::Function, domain, alg::Greedy; kwargs...)
end

function tt_cross(f::Function, domain, alg::DMRG; kwargs...)
end

function tt_cross(f::Function, domain; alg::CrossAlgorithm = MaxVol(), kwargs...)
    return tt_cross(f, domain, alg; kwargs...)
end

function tt_cross(
        f::Function,
        domain::Vector{<:AbstractVector{T}},
        alg::MaxVol;
        ranks::Union{Int, Vector{Int}} = 2,
        val_size::Int = 1000
    ) where {T <: Number}

    N = length(domain)
    Is = [length(d) for d in domain]

    if isa(ranks, Int)
        Rs = vcat([1], fill(ranks, N - 1), [1])
    else
        Rs = vcat([1], ranks, [1])
    end
    _cap_ranks!(Rs, Is, alg.rmax)

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
    for n in 1:N, r in 1:max_R
        randint[r, n] = rand(1:Is[n])
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
    for p in 1:val_size, d in 1:N
        val_coords[p, d] = domain[d][Xs_val[p, d]]
    end
    ys_val = f(val_coords)
    norm_ys_val = norm(ys_val)
    if norm_ys_val < alg.tol
        norm_ys_val = one(T)
    end

    if alg.verbose
        @info "MaxVol cross-interpolation over $(N)D domain with $(prod(Is)) grid points"
    end

    converged = false
    val_eps = Inf

    for iter in 1:alg.maxiter
        for j in 1:(N - 1)
            indices = _build_fiber_indices_left(lsets, rsets, j, Is, Rs, N)
            V = _evaluate_on_domain(f, domain, indices)
            V = reshape(V, Rs[j] * Is[j], Rs[j + 1])

            Q, _ = qr(V)
            Q_mat = Matrix(Q)

            local_indices, _ = maxvol!(copy(Q_mat), alg.pivot.tol, alg.pivot.maxiter)

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

            local_indices, _ = maxvol!(copy(Q_mat), alg.pivot.tol, alg.pivot.maxiter)

            V_new = Q_mat / Q_mat[local_indices, :]
            core = zeros(T, Is[j], Rs[j], Rs[j + 1])
            for r in 1:Rs[j], i in 1:Is[j], rp in 1:Rs[j + 1]
                row_idx = (i - 1) * Rs[j + 1] + rp
                core[i, r, rp] = V_new[row_idx, r]
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

        if alg.verbose
            @info "Iteration $iter: ε = $(val_eps), max rank = $(maximum(Rs))"
        end

        if val_eps < alg.tol
            converged = true
            break
        end

        if !converged && iter < alg.maxiter && alg.kickrank !== nothing
            newRs = copy(Rs)
            for n in 2:N
                newRs[n] = min(alg.rmax, newRs[n] + alg.kickrank)
            end
            _cap_ranks!(newRs, Is, alg.rmax)

            for n in 1:(N - 1)
                if newRs[n + 1] > Rs[n + 1]
                    extra_rows = newRs[n + 1] - Rs[n + 1]
                    n_cols = N - n
                    extra = zeros(Int, extra_rows, n_cols)
                    for er in 1:extra_rows, col in 1:n_cols
                        extra[er, col] = rand(1:Is[n + col])
                    end
                    rsets[n] = vcat(rsets[n], extra)
                end
            end
            Rs = newRs
        end
    end

    if converged && alg.verbose
        @info "Converged: ε = $(val_eps) < $(alg.tol)"
    elseif !converged && alg.verbose
        @warn "Max iterations reached: ε = $(val_eps)"
    end

    return TTvector{T, N}(N, cores, Tuple(Is), copy(Rs), zeros(Int, N))
end

function tt_cross(f::Function, domain; alg::CrossAlgorithm = MaxVol(), kwargs...)
    return tt_cross(f, domain, alg; kwargs...)
end

function tt_cross(f::Function, dims::NTuple{N, Int}, alg::CrossAlgorithm; kwargs...) where {N}
    domain = [collect(1.0:Float64(d)) for d in dims]
    return tt_cross(f, domain, alg; kwargs...)
end

function tt_cross(f::Function, dims::Vector{Int}, alg::CrossAlgorithm; kwargs...)
    domain = [collect(1.0:Float64(d)) for d in dims]
    return tt_cross(f, domain, alg; kwargs...)
end

function _contract_with_weights(cores, weights)
    result = ones(eltype(cores[1]), 1)
    for k in eachindex(cores)
        contracted = sum(weights[k][i] .* cores[k][i, :, :] for i in axes(cores[k], 1))
        result = vec(result' * contracted)
    end
    return result[1]
end

function _gauss_legendre(n::Int, a::T, b::T) where {T}
    β = [k / sqrt(4k^2 - 1) for k in 1:(n - 1)]
    J = SymTridiagonal(zeros(n), β)
    λ, V = eigen(J)
    nodes = (b - a) / 2 .* λ .+ (a + b) / 2
    weights = (b - a) .* V[1, :] .^ 2
    return nodes, weights
end

"""
    tt_integrate(f::Function, bounds::Vector{Tuple{T, T}}; nquad::Int = 20, kwargs...) where {T <: AbstractFloat}

Compute the multidimensional integral of a function using Tensor Train cross-interpolation.

This function approximates the integral of a function `f` over a multidimensional rectangular domain
by first constructing a low-rank Tensor Train approximation using cross-interpolation, then integrating
the approximation using Gauss-Legendre quadrature.

# References

Vysotsky, Lev I., Alexander V. Smirnov, and Eugene E. Tyrtyshnikov. "Tensor-train numerical integration of multivariate functions with singularities." Lobachevskii Journal of Mathematics 42.7 (2021): 1608-1621.
"""
function tt_integrate(
        f::Function,
        bounds::Vector{Tuple{T, T}};
        nquad::Int = 20,
        kwargs...
    ) where {T <: AbstractFloat}

    d = length(bounds)
    nodes = [_gauss_legendre(nquad, bounds[k]...)[1] for k in 1:d]
    weights = [_gauss_legendre(nquad, bounds[k]...)[2] for k in 1:d]

    tt = tt_cross(f, nodes; kwargs...)
    cores = tt.ttv_vec

    result = ones(eltype(cores[1]), 1)
    for k in 1:d
        contracted = sum(weights[k][i] .* cores[k][i, :, :] for i in axes(cores[k], 1))
        result = vec(result' * contracted)
    end
    return result[1]
end

tt_integrate(f, d::Int, bound::Tuple{T, T} = (zero(T), one(T)); kw...) where {T <: AbstractFloat} = tt_integrate(f, fill(bound, d); kw...)
tt_integrate(f, d::Int; kw...) = tt_integrate(f, fill((0.0, 1.0), d); kw...)

function tt_integrate(f::Function, lower::Vector{T}, upper::Vector{T}; kwargs...) where {T <: AbstractFloat}
    bounds = collect(zip(lower, upper))
    return tt_integrate(f, bounds; kwargs...)
end

function tt_cross(
        f::Function,
        domain::Vector{<:AbstractVector{T}},
        alg::DMRG;
        ranks::Union{Int, Vector{Int}} = 2,
        val_size::Int = 1000
    ) where {T <: Number}

    N = length(domain)
    Is = [length(d) for d in domain]

    if isa(ranks, Int)
        Rs = vcat([1], fill(ranks, N - 1), [1])
    else
        Rs = vcat([1], ranks, [1])
    end
    _cap_ranks!(Rs, Is, alg.rmax)

    I_l = Vector{Matrix{Int}}(undef, N)
    I_g = Vector{Matrix{Int}}(undef, N)
    I_l[1] = ones(Int, 1, 0)
    I_g[N] = ones(Int, 1, 0)

    for k in 2:N
        I_l[k] = zeros(Int, Rs[k], k - 1)
        for r in 1:Rs[k], j in 1:(k - 1)
            I_l[k][r, j] = rand(1:Is[j])
        end
    end

    for k in 1:(N - 1)
        I_g[k] = zeros(Int, Rs[k + 1], N - k)
        for r in 1:Rs[k + 1], j in 1:(N - k)
            I_g[k][r, j] = rand(1:Is[k + j])
        end
    end

    cores = Vector{Array{T, 3}}(undef, N)
    for n in 1:N
        cores[n] = zeros(T, Is[n], Rs[n], Rs[n + 1])
    end

    Xs_val = Matrix{Int}(undef, val_size, N)
    for d in 1:N
        Xs_val[:, d] = rand(1:Is[d], val_size)
    end
    val_coords = Matrix{T}(undef, val_size, N)
    for p in 1:val_size, d in 1:N
        val_coords[p, d] = domain[d][Xs_val[p, d]]
    end
    ys_val = f(val_coords)
    norm_ys_val = norm(ys_val)
    if norm_ys_val < alg.tol
        norm_ys_val = one(T)
    end

    if alg.verbose
        @info "DMRG cross-interpolation over $(N)D domain with $(prod(Is)) grid points"
    end

    converged = false
    val_eps = Inf

    for iter in 1:alg.maxiter
        for k in 1:(N - 1)
            _dmrg_update!(cores, I_l, I_g, Rs, f, domain, Is, k, N, true, alg)
        end

        y_approx = _evaluate_tt(cores, Xs_val, N)
        val_eps = norm(ys_val - y_approx) / norm_ys_val

        if alg.verbose
            @info "Sweep $(2 * iter - 1) (L→R): ε = $(val_eps), max rank = $(maximum(Rs))"
        end

        if val_eps < alg.tol
            converged = true
            break
        end

        for k in (N - 1):-1:1
            _dmrg_update!(cores, I_l, I_g, Rs, f, domain, Is, k, N, false, alg)
        end

        y_approx = _evaluate_tt(cores, Xs_val, N)
        val_eps = norm(ys_val - y_approx) / norm_ys_val

        if alg.verbose
            @info "Sweep $(2 * iter) (R→L): ε = $(val_eps), max rank = $(maximum(Rs))"
        end

        if val_eps < alg.tol
            converged = true
            break
        end
    end

    if converged && alg.verbose
        @info "Converged: ε = $(val_eps) < $(alg.tol)"
    elseif !converged && alg.verbose
        @warn "Max iterations reached: ε = $(val_eps)"
    end

    return TTvector{T, N}(N, cores, Tuple(Is), copy(Rs), zeros(Int, N))
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

function _sample_superblock(f, domain, I_l, I_g, k, Is, N)
    r_l = size(I_l[k], 1)
    r_g = size(I_g[k + 1], 1)
    s1, s2 = Is[k], Is[k + 1]

    n_samples = r_l * s1 * s2 * r_g
    indices = Matrix{Int}(undef, n_samples, N)

    idx = 1
    for rg in 1:r_g, i2 in 1:s2, i1 in 1:s1, rl in 1:r_l
        if k > 1
            indices[idx, 1:(k - 1)] = I_l[k][rl, :]
        end
        indices[idx, k] = i1
        indices[idx, k + 1] = i2
        if k + 1 < N
            indices[idx, (k + 2):N] = I_g[k + 1][rg, :]
        end
        idx += 1
    end

    values = _evaluate_on_domain(f, domain, indices)
    return reshape(values, r_l, s1, s2, r_g)
end

function _combine_indices_left(I_l_k, s)
    r_l = size(I_l_k, 1)
    n_cols = size(I_l_k, 2)
    result = zeros(Int, r_l * s, n_cols + 1)
    idx = 1
    for i in 1:s, r in 1:r_l
        if n_cols > 0
            result[idx, 1:n_cols] = I_l_k[r, :]
        end
        result[idx, end] = i
        idx += 1
    end
    return result
end

function _combine_indices_right(s, I_g_k)
    r_g = size(I_g_k, 1)
    n_cols = size(I_g_k, 2)
    result = zeros(Int, s * r_g, n_cols + 1)
    idx = 1
    for r in 1:r_g, i in 1:s
        result[idx, 1] = i
        if n_cols > 0
            result[idx, 2:end] = I_g_k[r, :]
        end
        idx += 1
    end
    return result
end

function _dmrg_update!(cores, I_l, I_g, Rs, f, domain, Is, k, N, left_to_right, alg::DMRG)
    superblock = _sample_superblock(f, domain, I_l, I_g, k, Is, N)
    r_l, s1, s2, r_g = size(superblock)

    A = reshape(superblock, r_l * s1, s2 * r_g)
    U, S, Vt = _svdtrunc(A; max_bond = alg.rmax, truncerr = alg.tol)
    r = size(S, 1)

    return if left_to_right
        if k < N - 1
            Q, _ = qr(U)
            Q_mat = Matrix(Q)
            I_idx, _ = maxvol!(copy(Q_mat), alg.pivot.tol, alg.pivot.maxiter)
            G = Q_mat / Q_mat[I_idx, :]

            combined = _combine_indices_left(I_l[k], s1)
            I_l[k + 1] = combined[I_idx, :]
            Rs[k + 1] = length(I_idx)

            cores[k] = permutedims(reshape(G, r_l, s1, Rs[k + 1]), (2, 1, 3))
        else
            cores[k] = permutedims(reshape(U, r_l, s1, r), (2, 1, 3))
            cores[k + 1] = permutedims(reshape(S * Vt, r, s2, r_g), (2, 1, 3))
            Rs[k + 1] = r
        end
    else
        if k > 1
            Q, _ = qr(Vt')
            Q_mat = Matrix(Q)
            I_idx, _ = maxvol!(copy(Q_mat), alg.pivot.tol, alg.pivot.maxiter)
            G = Q_mat / Q_mat[I_idx, :]

            combined = _combine_indices_right(s2, I_g[k + 1])
            I_g[k] = combined[I_idx, :]
            Rs[k + 1] = length(I_idx)

            cores[k + 1] = permutedims(reshape(G', Rs[k + 1], s2, r_g), (2, 1, 3))
        else
            cores[k] = permutedims(reshape(U * S, r_l, s1, r), (2, 1, 3))
            cores[k + 1] = permutedims(reshape(Vt, r, s2, r_g), (2, 1, 3))
            Rs[k + 1] = r
        end
    end
end

function tt_cross(
        f::Function,
        domain::Vector{<:AbstractVector{T}},
        alg::Greedy;
        ranks::Union{Int, Vector{Int}} = 1,
        val_size::Int = 1000
    ) where {T <: Number}

    N = length(domain)
    Is = [length(d) for d in domain]

    I_l = Vector{Matrix{Int}}(undef, N)
    I_g = Vector{Matrix{Int}}(undef, N)
    J_l = Vector{Vector{Int}}(undef, N)
    J_g = Vector{Vector{Int}}(undef, N)

    I_l[1] = ones(Int, 1, 0)
    I_g[N] = ones(Int, 1, 0)
    J_l[1] = Int[]
    J_g[N] = Int[]

    init_point = [1 for _ in 1:N]
    for k in 2:N
        I_l[k] = reshape(init_point[1:k-1], 1, k-1)
        J_l[k] = [1]
    end
    for k in N-1:-1:1
        I_g[k] = reshape(init_point[k+1:N], 1, N-k)
        J_g[k] = [1]
    end

    Rs = ones(Int, N + 1)

    fibers = Vector{Array{T, 3}}(undef, N)
    for k in 1:N
        fibers[k] = _sample_fiber(f, domain, I_l, I_g, k, Is, N)
    end

    list_Q3 = Vector{Array{T, 3}}(undef, N - 1)
    list_R = Vector{Matrix{T}}(undef, N - 1)
    for k in 1:N-1
        list_Q3[k], list_R[k] = _fiber_to_Q3R(fibers[k])
    end

    cores = Vector{Array{T, 3}}(undef, N)
    for k in 1:N-1
        cores[k] = _Q3_to_core(list_Q3[k], J_l[k + 1])
    end
    cores[N] = fibers[N]

    Xs_val = Matrix{Int}(undef, val_size, N)
    for d in 1:N
        Xs_val[:, d] = rand(1:Is[d], val_size)
    end
    val_coords = Matrix{T}(undef, val_size, N)
    for p in 1:val_size, d in 1:N
        val_coords[p, d] = domain[d][Xs_val[p, d]]
    end
    ys_val = f(val_coords)
    norm_ys_val = norm(ys_val)
    if norm_ys_val < alg.tol
        norm_ys_val = one(T)
    end

    if alg.verbose
        @info "Greedy cross-interpolation over $(N)D domain with $(prod(Is)) grid points"
    end

    converged = false
    val_eps = Inf

    for iter in 1:alg.maxiter
        for k in 1:N-1
            _greedy_update!(cores, fibers, list_Q3, list_R, I_l, I_g, J_l, J_g, Rs, f, domain, Is, k, N, alg)
        end

        cores_eval = [permutedims(cores[k], (2, 1, 3)) for k in 1:N]
        y_approx = _evaluate_tt(cores_eval, Xs_val, N)
        val_eps = norm(ys_val - y_approx) / norm_ys_val

        if alg.verbose
            @info "Sweep $(2*iter - 1) (L→R): ε = $(val_eps), max rank = $(maximum(Rs))"
        end

        if val_eps < alg.tol
            converged = true
            break
        end

        for k in N-1:-1:1
            _greedy_update!(cores, fibers, list_Q3, list_R, I_l, I_g, J_l, J_g, Rs, f, domain, Is, k, N, alg)
        end

        cores_eval = [permutedims(cores[k], (2, 1, 3)) for k in 1:N]
        y_approx = _evaluate_tt(cores_eval, Xs_val, N)
        val_eps = norm(ys_val - y_approx) / norm_ys_val

        if alg.verbose
            @info "Sweep $(2*iter) (R→L): ε = $(val_eps), max rank = $(maximum(Rs))"
        end

        if val_eps < alg.tol
            converged = true
            break
        end
    end

    if converged && alg.verbose
        @info "Converged: ε = $(val_eps) < $(alg.tol)"
    elseif !converged && alg.verbose
        @warn "Max iterations reached: ε = $(val_eps)"
    end

    cores_out = [permutedims(cores[k], (2, 1, 3)) for k in 1:N]
    return TTvector{T, N}(N, cores_out, Tuple(Is), copy(Rs), zeros(Int, N))
end

function _sample_fiber(f, domain, I_l, I_g, k, Is, N)
    r_l = size(I_l[k], 1)
    r_g = size(I_g[k], 1)
    s = Is[k]

    n_samples = r_l * s * r_g
    indices = Matrix{Int}(undef, n_samples, N)

    idx = 1
    for rg in 1:r_g, i in 1:s, rl in 1:r_l
        if k > 1
            indices[idx, 1:(k - 1)] = I_l[k][rl, :]
        end
        indices[idx, k] = i
        if k < N
            indices[idx, (k + 1):N] = I_g[k][rg, :]
        end
        idx += 1
    end

    values = _evaluate_on_domain(f, domain, indices)
    return reshape(values, r_l, s, r_g)
end

function _fiber_to_Q3R(fiber::Array{T, 3}) where {T}
    r_l, s, r_g = size(fiber)
    F = qr(reshape(fiber, r_l * s, r_g))
    Q = Matrix(F.Q)
    R = F.R
    r = min(r_l * s, r_g)
    Q3 = reshape(Q[:, 1:r], r_l, s, r)
    return Q3, Matrix(R[1:r, :])
end

function _Q3_to_core(Q3::Array{T, 3}, row_indices::Vector{Int}) where {T}
    r_l, s, r_g = size(Q3)
    Q = reshape(Q3, r_l * s, r_g)
    if isempty(row_indices)
        return Q3
    end
    P = pinv(Q[row_indices, :])
    G = Q * P
    return reshape(G, r_l, s, length(row_indices))
end

function _combine_indices_left_fiber(I_l_k, s)
    r_l = size(I_l_k, 1)
    n_cols = size(I_l_k, 2)
    result = zeros(Int, r_l * s, n_cols + 1)
    idx = 1
    for i in 1:s, r in 1:r_l
        if n_cols > 0
            result[idx, 1:n_cols] = I_l_k[r, :]
        end
        result[idx, end] = i
        idx += 1
    end
    return result
end

function _combine_indices_right_fiber(s, I_g_k)
    r_g = size(I_g_k, 1)
    n_cols = size(I_g_k, 2)
    result = zeros(Int, s * r_g, n_cols + 1)
    idx = 1
    for r in 1:r_g, i in 1:s
        result[idx, 1] = i
        if n_cols > 0
            result[idx, 2:end] = I_g_k[r, :]
        end
        idx += 1
    end
    return result
end

function _get_row_indices(rows::Matrix{Int}, all_rows::Matrix{Int})
    if size(rows, 1) == 0 || size(all_rows, 1) == 0
        return Int[]
    end
    large_set = Dict{Vector{Int}, Int}()
    for idx in 1:size(all_rows, 1)
        large_set[all_rows[idx, :]] = idx
    end
    return [large_set[rows[r, :]] for r in 1:size(rows, 1)]
end

function _sample_superblock_greedy(f, domain, I_l, I_g, Is, k, N, j_l, j_g)
    i_ls = _combine_indices_left_fiber(I_l[k], Is[k])
    i_sg = _combine_indices_right_fiber(Is[k + 1], I_g[k + 1])

    i_ls_sel = i_ls[j_l, :]
    i_sg_sel = i_sg[j_g, :]

    n_l = length(j_l)
    n_g = length(j_g)
    indices = Matrix{Int}(undef, n_l * n_g, N)

    idx = 1
    for jg in 1:n_g, jl in 1:n_l
        if k > 1
            indices[idx, 1:(k - 1)] = i_ls_sel[jl, 1:(end - 1)]
        end
        indices[idx, k] = i_ls_sel[jl, end]
        indices[idx, k + 1] = i_sg_sel[jg, 1]
        if k + 1 < N
            indices[idx, (k + 2):N] = i_sg_sel[jg, 2:end]
        end
        idx += 1
    end

    values = _evaluate_on_domain(f, domain, indices)
    return reshape(values, n_l, n_g)
end

function _sample_skeleton_greedy(cores, fibers, k, j_l, j_g)
    core_k = cores[k]
    fiber_kp1 = fibers[k + 1]

    r_l, s1, chi = size(core_k)
    chi2, s2, r_g = size(fiber_kp1)

    if chi != chi2
        return zeros(eltype(core_k), length(j_l), length(j_g))
    end

    G = reshape(core_k, r_l * s1, chi)[j_l, :]
    R = reshape(fiber_kp1, chi2, s2 * r_g)[:, j_g]
    return G * R
end

function _greedy_update!(cores, fibers, list_Q3, list_R, I_l, I_g, J_l, J_g, Rs, f, domain, Is, k, N, alg::Greedy)
    n_ls = size(I_l[k], 1) * Is[k]
    n_sg = Is[k + 1] * size(I_g[k + 1], 1)

    if n_ls == 0 || n_sg == 0
        return
    end

    j_l_random = rand(1:n_ls, min(alg.nsamples, n_ls))
    j_g_random = rand(1:n_sg, min(alg.nsamples, n_sg))

    A_random = _sample_superblock_greedy(f, domain, I_l, I_g, Is, k, N, j_l_random, j_g_random)
    B_random = _sample_skeleton_greedy(cores, fibers, k, j_l_random, j_g_random)

    diff = abs.(A_random - B_random)
    if all(iszero, diff)
        return
    end

    max_idx = argmax(diff)
    i, j = Tuple(CartesianIndices(diff)[max_idx])
    j_l = j_l_random[i]
    j_g = j_g_random[j]

    c_A = zeros(eltype(A_random), 0)
    c_B = zeros(eltype(A_random), 0)
    r_A = zeros(eltype(A_random), 0)
    r_B = zeros(eltype(A_random), 0)

    for _ in 1:alg.pivot.nsamples
        c_A = vec(_sample_superblock_greedy(f, domain, I_l, I_g, Is, k, N, collect(1:n_ls), [j_g]))
        c_B = vec(_sample_skeleton_greedy(cores, fibers, k, collect(1:n_ls), [j_g]))
        new_j_l = argmax(abs.(c_A - c_B))
        if new_j_l == j_l
            break
        end
        j_l = new_j_l

        r_A = vec(_sample_superblock_greedy(f, domain, I_l, I_g, Is, k, N, [j_l], collect(1:n_sg)))
        r_B = vec(_sample_skeleton_greedy(cores, fibers, k, [j_l], collect(1:n_sg)))
        new_j_g = argmax(abs.(r_A - r_B))
        if new_j_g == j_g
            break
        end
        j_g = new_j_g
    end

    if isempty(c_A)
        c_A = vec(_sample_superblock_greedy(f, domain, I_l, I_g, Is, k, N, collect(1:n_ls), [j_g]))
        c_B = vec(_sample_skeleton_greedy(cores, fibers, k, collect(1:n_ls), [j_g]))
    end
    if isempty(r_A)
        r_A = vec(_sample_superblock_greedy(f, domain, I_l, I_g, Is, k, N, [j_l], collect(1:n_sg)))
        r_B = vec(_sample_skeleton_greedy(cores, fibers, k, [j_l], collect(1:n_sg)))
    end

    pivot_error = abs(c_A[j_l] - c_B[j_l])
    if pivot_error < alg.tol
        return
    end

    i_ls = _combine_indices_left_fiber(I_l[k], Is[k])
    i_sg = _combine_indices_right_fiber(Is[k + 1], I_g[k + 1])

    i_l_new = i_ls[j_l:j_l, :]
    i_g_new = i_sg[j_g:j_g, :]

    I_l[k + 1] = vcat(I_l[k + 1], i_l_new)
    J_l[k + 1] = vcat(J_l[k + 1], j_l)
    I_g[k] = vcat(I_g[k], i_g_new)
    J_g[k] = vcat(J_g[k], j_g)

    r_l, s1, chi = size(fibers[k])
    fiber_1_flat = reshape(fibers[k], r_l * s1, chi)
    fiber_1_new = hcat(fiber_1_flat, c_A)
    fibers[k] = reshape(fiber_1_new, r_l, s1, chi + 1)

    list_Q3[k], list_R[k] = _fiber_to_Q3R(fibers[k])
    cores[k] = _Q3_to_core(list_Q3[k], J_l[k + 1])

    chi2, s2, r_g = size(fibers[k + 1])
    fiber_2_flat = reshape(fibers[k + 1], chi2, s2 * r_g)
    fiber_2_new = vcat(fiber_2_flat, r_A')
    fibers[k + 1] = reshape(fiber_2_new, chi2 + 1, s2, r_g)

    if k < N - 1
        list_Q3[k + 1], list_R[k + 1] = _fiber_to_Q3R(fibers[k + 1])
        cores[k + 1] = _Q3_to_core(list_Q3[k + 1], J_l[k + 2])
    else
        cores[k + 1] = fibers[k + 1]
    end

    return Rs[k + 1] = size(I_l[k + 1], 1)
end
