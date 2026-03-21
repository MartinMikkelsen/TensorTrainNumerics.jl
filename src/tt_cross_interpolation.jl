using LinearAlgebra
using Random
using Maxvol

abstract type CrossAlgorithm end
abstract type PivotAlgorithm end

const CROSS_MAXITER = Ref{Int}(50)
const CROSS_TOL = Ref{Float64}(1.0e-10)
const CROSS_RMAX = Ref{Int}(500)
const CROSS_KICKRANK = Ref{Int}(5)
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
end

function DMRG(;
        maxiter::Int = CROSS_MAXITER[],
        tol::Real = CROSS_TOL[],
        rmax::Int = CROSS_RMAX[],
        kickrank::Union{Nothing, Int} = CROSS_KICKRANK[],
        verbose::Bool = true,
        pivot::PivotAlgorithm = MaxVolPivot()
    )
    return DMRG(maxiter, tol, rmax, kickrank, verbose, pivot)
end

function tt_cross(f::Function, domain; alg::CrossAlgorithm = MaxVol(), kwargs...)
    return tt_cross(f, domain, alg; kwargs...)
end

function tt_cross(f::Function, dims::NTuple{N, Int}; alg::CrossAlgorithm = MaxVol(), kwargs...) where {N}
    domain = [collect(1.0:Float64(d)) for d in dims]
    return tt_cross(f, domain, alg; kwargs...)
end

function tt_cross(f::Function, dims::Vector{Int}; alg::CrossAlgorithm = MaxVol(), kwargs...)
    domain = [collect(1.0:Float64(d)) for d in dims]
    return tt_cross(f, domain, alg; kwargs...)
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

function _evaluate_on_domain(f, domain::Vector{<:AbstractVector}, indices::Matrix{Int})
    N = length(domain)
    n_points = size(indices, 1)
    T = promote_type(map(eltype, domain)...)
    coords = Matrix{T}(undef, n_points, N)
    for p in 1:n_points, d in 1:N
        coords[p, d] = domain[d][indices[p, d]]
    end
    return vec(f(coords))
end

function _evaluate_tt(cores, indices, N)
    T = eltype(cores[1])
    n_points = size(indices, 1)
    state = ones(T, n_points, 1)
    for d in 1:N
        r_r = size(cores[d], 3)
        slices = cores[d][indices[:, d], :, :]
        state = reshape(sum(reshape(state, n_points, :, 1) .* slices, dims = 2), n_points, r_r)
    end
    return vec(state)
end

function _svdtrunc(A::AbstractMatrix{T}; max_bond::Int = typemax(Int), truncerr::Real = 0.0) where {T}
    F = svd(A)
    s = F.S
    r = length(s)
    if truncerr > 0
        nrm = norm(s)
        cum = zero(eltype(s))
        for i in r:-1:1
            cum += abs2(s[i])
            if sqrt(cum) > truncerr * nrm
                r = i
                break
            end
        end
    end
    r = min(r, max_bond)
    return F.U[:, 1:r], Diagonal(s[1:r]), F.Vt[1:r, :]
end

function _build_fiber_indices(lsets, rsets, j, Is, Rs, N)
    n_fibers = Rs[j] * Is[j] * Rs[j + 1]
    indices = Matrix{Int}(undef, n_fibers, N)
    n_left  = j - 1
    n_right = N - j
    idx = 1
    for r_right in 1:Rs[j + 1], r_left in 1:Rs[j], i in 1:Is[j]
        j > 1 && (indices[idx, 1:n_left]              = lsets[j][r_left, :])
        indices[idx, j]                               = i
        j < N && (indices[idx, (j + 1):(j + n_right)] = rsets[j][r_right, :])
        idx += 1
    end
    return indices
end

function _infer_value_type(f::Function, domain::Vector{<:AbstractVector})
    N = length(domain)
    probe_idx = ones(Int, 1, N)
    return eltype(_evaluate_on_domain(f, domain, probe_idx))
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
    Tv = _infer_value_type(f, domain)

    Rs = isa(ranks, Int) ? vcat([1], fill(ranks, N - 1), [1]) : vcat([1], ranks, [1])
    _cap_ranks!(Rs, Is, alg.rmax)

    cores = [randn(Tv, Is[n], Rs[n], Rs[n + 1]) for n in 1:N]

    lsets = Vector{Matrix{Int}}(undef, N)
    rsets = Vector{Matrix{Int}}(undef, N)
    lsets[1] = ones(Int, 1, 0)
    rsets[N] = ones(Int, 1, 0)

    max_R = maximum(Rs)
    randint = [rand(1:Is[n]) for _ in 1:max_R, n in 1:N]
    for n in 1:(N - 1)
        rsets[n] = randint[1:Rs[n + 1], (n + 1):N]
    end
    for n in 2:N
        lsets[n] = zeros(Int, 0, 0)
    end

    Xs_val = hcat([rand(1:Is[d], val_size) for d in 1:N]...)::Matrix{Int}
    ys_val = _evaluate_on_domain(f, domain, Xs_val)
    norm_ys_val = max(norm(ys_val), alg.tol)

    alg.verbose && @info "MaxVol cross-interpolation over $(N)D domain with $(prod(Is)) grid points"

    converged = false
    val_eps = Inf

    for iter in 1:alg.maxiter
        for j in 1:(N - 1)
            indices = _build_fiber_indices(lsets, rsets, j, Is, Rs, N)
            V = reshape(_evaluate_on_domain(f, domain, indices), Rs[j] * Is[j], Rs[j + 1])
            Q_mat = Matrix(first(qr(V)))
            local_indices, _ = maxvol!(copy(Q_mat), alg.pivot.tol, alg.pivot.maxiter)

            G = Q_mat / Q_mat[local_indices, :]
            cores[j] = reshape(G, Is[j], Rs[j], length(local_indices))

            local_i = [((idx - 1) % Is[j]) + 1 for idx in local_indices]
            local_r = [((idx - 1) ÷ Is[j]) + 1 for idx in local_indices]

            if j == 1
                lsets[j + 1] = reshape(local_i, :, 1)
            else
                new_lset = zeros(Int, length(local_indices), j)
                for k in eachindex(local_indices)
                    new_lset[k, 1:(j - 1)] = lsets[j][local_r[k], :]
                    new_lset[k, j] = local_i[k]
                end
                lsets[j + 1] = new_lset
            end
            Rs[j + 1] = length(local_indices)
        end

        for j in N:-1:2
            indices = _build_fiber_indices(lsets, rsets, j, Is, Rs, N)
            V = reshape(_evaluate_on_domain(f, domain, indices), Rs[j] * Is[j], Rs[j + 1])
            V_3d = reshape(V, Is[j], Rs[j], Rs[j + 1])
            V_right = reshape(permutedims(V_3d, (2, 1, 3)), Rs[j], Is[j] * Rs[j + 1])
            # Use a plain transpose here: TT cross uses bilinear products, so an
            # adjoint would introduce a spurious conjugation for complex targets.
            Q_mat = Matrix(first(qr(transpose(V_right))))
            local_indices, _ = maxvol!(copy(Q_mat), alg.pivot.tol, alg.pivot.maxiter)

            G = Q_mat / Q_mat[local_indices, :]
            G_3d = reshape(G, Is[j], Rs[j + 1], Rs[j])
            cores[j] = permutedims(G_3d, (1, 3, 2))

            local_i = [((idx - 1) % Is[j]) + 1 for idx in local_indices]
            local_r = [((idx - 1) ÷ Is[j]) + 1 for idx in local_indices]

            if j == N
                rsets[j - 1] = reshape(local_i, :, 1)
            else
                new_rset = zeros(Int, length(local_indices), N - j + 1)
                for k in eachindex(local_indices)
                    new_rset[k, 1] = local_i[k]
                    new_rset[k, 2:end] = rsets[j][local_r[k], :]
                end
                rsets[j - 1] = new_rset
            end
            Rs[j] = length(local_indices)
        end

        indices = _build_fiber_indices(lsets, rsets, 1, Is, Rs, N)
        V = _evaluate_on_domain(f, domain, indices)
        cores[1] = reshape(V, Is[1], Rs[1], Rs[2])

        y_approx = _evaluate_tt(cores, Xs_val, N)
        val_eps = norm(ys_val - y_approx) / norm_ys_val

        alg.verbose && @info "Iteration $iter: ε = $(val_eps), max rank = $(maximum(Rs))"

        if val_eps < alg.tol
            converged = true
            break
        end

        if alg.kickrank !== nothing
            newRs = copy(Rs)
            for n in 2:N
                newRs[n] = min(newRs[n] + alg.kickrank, alg.rmax)
            end
            _cap_ranks!(newRs, Is, alg.rmax)
            for n in 1:(N - 1)
                if newRs[n + 1] > Rs[n + 1]
                    extra = [rand(1:Is[n + col]) for _ in 1:(newRs[n + 1] - Rs[n + 1]), col in 1:(N - n)]
                    rsets[n] = vcat(rsets[n], extra)
                end
            end
            Rs = newRs
        end
    end

    converged && alg.verbose && @info "Converged: ε = $(val_eps) < $(alg.tol)"
    !converged && alg.verbose && @warn "Max iterations reached: ε = $(val_eps)"

    return TTvector{eltype(cores[1]), N}(N, cores, Tuple(Is), copy(Rs), zeros(Int, N))
end

function _indexmerge(J1::AbstractMatrix{Int}, J2::AbstractMatrix{Int})
    sz1, sz2 = max(size(J1, 1), 1), max(size(J2, 1), 1)
    return hcat(repeat(Matrix(J1), sz2, 1), repeat(Matrix(J2), inner = (sz1, 1)))
end

function _form_tensor(y, mid_inv_L, mid_inv_U, N, Rs, Is)
    cores = Vector{Array{eltype(y[1]), 3}}(undef, N)
    for i in 1:N
        yi = mid_inv_L[i] * reshape(y[i], Rs[i], Is[i] * Rs[i + 1])
        yi = reshape(yi, Rs[i] * Is[i], Rs[i + 1]) * mid_inv_U[i + 1]
        cores[i] = permutedims(reshape(yi, Rs[i], Is[i], Rs[i + 1]), (2, 1, 3))
    end
    return cores
end

function tt_cross(
        f::Function,
        domain::Vector{<:AbstractVector{T}},
        alg::Greedy;
        val_size::Int = 1000
    ) where {T <: Number}

    N = length(domain)
    Is = [length(d) for d in domain]
    Tv = _infer_value_type(f, domain)
    Rs = ones(Int, N + 1)
    rng = (alg.pivot isa RandomPivot && !isnothing(alg.pivot.seed)) ? MersenneTwister(alg.pivot.seed) : Random.default_rng()
    sample_budget = alg.pivot isa RandomPivot ? min(alg.nsamples, alg.pivot.nsamples) : alg.nsamples

    y = Vector{Array{Tv, 3}}(undef, N)
    mid_inv_U = Vector{Matrix{Tv}}(undef, N + 1)
    mid_inv_L = Vector{Matrix{Tv}}(undef, N + 1)

    for i in 1:(N + 1)
        mid_inv_U[i] = ones(Tv, 1, 1)
        mid_inv_L[i] = ones(Tv, 1, 1)
    end

    Jyl = Vector{Matrix{Int}}(undef, N + 1)
    Jyr = Vector{Matrix{Int}}(undef, N + 1)
    ilocl = Vector{Vector{Int}}(undef, N + 1)
    ilocr = Vector{Vector{Int}}(undef, N + 1)

    Jyl[1] = ones(Int, 1, 0)
    Jyr[N + 1] = ones(Int, 1, 0)
    ilocl[1] = Int[]
    ilocr[N + 1] = Int[]

    for i in 2:N
        ilocl[i] = [1]
        ilocr[i] = [1]
    end

    for i in 1:(N - 1)
        Jcand = _indexmerge(Jyl[i], reshape(collect(1:Is[i]), :, 1))
        row = argmax(abs.(domain[i]))
        Jyl[i + 1] = Jcand[row:row, :]
    end
    for i in N:-1:2
        Jcand = _indexmerge(reshape(collect(1:Is[i]), :, 1), Jyr[i + 1])
        row = argmax(abs.(domain[i]))
        Jyr[i] = Jcand[row:row, :]
    end

    for i in N:-1:1
        J = _indexmerge(_indexmerge(Jyl[i], reshape(collect(1:Is[i]), :, 1)), Jyr[i + 1])
        cry = _evaluate_on_domain(f, domain, J)
        if i > 1
            cry_mat = reshape(cry, Rs[i], Is[i] * Rs[i + 1])
            _, imax = findmax(abs.(cry_mat), dims = 2)
            ilocr[i] = [imax[1][2]]
            Jyr[i] = _indexmerge(reshape(collect(1:Is[i]), :, 1), Jyr[i + 1])[ilocr[i], :]
            piv = cry_mat[1, ilocr[i][1]]
            if abs(piv) > max(alg.tol, eps(real(float(abs(piv)))))
                mid_inv_L[i] = reshape([1 / piv], 1, 1)
            else
                mid_inv_L[i] = ones(Tv, 1, 1)
            end
        end
        y[i] = reshape(cry, Rs[i], Is[i], Rs[i + 1])
    end

    for i in 1:N
        J = _indexmerge(_indexmerge(Jyl[i], reshape(collect(1:Is[i]), :, 1)), Jyr[i + 1])
        cry = _evaluate_on_domain(f, domain, J)
        if i < N
            cry_mat = reshape(cry, Rs[i] * Is[i], Rs[i + 1])
            _, imax = findmax(abs.(cry_mat), dims = 1)
            ilocl[i + 1] = [imax[1][1]]
            Jyl[i + 1] = _indexmerge(Jyl[i], reshape(collect(1:Is[i]), :, 1))[ilocl[i + 1], :]
            piv = cry_mat[ilocl[i + 1][1], 1]
            if abs(piv) > max(alg.tol, eps(real(float(abs(piv)))))
                mid_inv_U[i + 1] = reshape([1 / piv], 1, 1)
            else
                mid_inv_U[i + 1] = ones(Tv, 1, 1)
            end
        end
        y[i] = reshape(cry, Rs[i], Is[i], Rs[i + 1])
    end

    Xs_val = hcat([rand(1:Is[d], val_size) for d in 1:N]...)
    ys_val = _evaluate_on_domain(f, domain, Xs_val)
    norm_ys_val = max(norm(ys_val), alg.tol)

    alg.verbose && @info "Greedy cross-interpolation over $(N)D domain with $(prod(Is)) grid points"

    converged = false
    val_eps = Inf

    for swp in 1:alg.maxiter
        max_dx = 0.0

        for i in 1:(N - 1)
            cind1 = setdiff(1:(Rs[i] * Is[i]), ilocl[i + 1])
            cind2 = setdiff(1:(Is[i + 1] * Rs[i + 2]), ilocr[i + 1])
            (isempty(cind1) || isempty(cind2)) && continue

            testsz = min(length(cind1), length(cind2), sample_budget)
            tind1 = cind1[rand(rng, 1:length(cind1), testsz)]
            tind2 = cind2[rand(rng, 1:length(cind2), testsz)]

            J1 = _indexmerge(Jyl[i], reshape(collect(1:Is[i]), :, 1))
            J2 = _indexmerge(reshape(collect(1:Is[i + 1]), :, 1), Jyr[i + 2])

            crt = _evaluate_on_domain(f, domain, hcat(J1[tind1, :], J2[tind2, :]))
            maxy = maximum(abs.(crt))

            cry1 = reshape(y[i], Rs[i] * Is[i], Rs[i + 1])
            cry2 = reshape(y[i + 1], Rs[i + 1], Is[i + 1] * Rs[i + 2])
            cre1, cre2 = cry1 * mid_inv_U[i + 1], mid_inv_L[i + 1] * cry2

            cry_approx = [sum(cre1[tind1[j], :] .* cre2[:, tind2[j]]) for j in 1:testsz]
            cre = crt - cry_approx

            _, imax_test = findmax(abs.(cre))
            j_g_best = tind2[imax_test]

            crt_col = _evaluate_on_domain(f, domain, hcat(J1[cind1, :], repeat(J2[j_g_best:j_g_best, :], length(cind1), 1)))
            cre_col = crt_col - cre1[cind1, :] * cre2[:, j_g_best]

            emax, imax1_local = findmax(abs.(cre_col))
            imax1 = cind1[imax1_local]
            dx = emax / max(maxy, eps(real(float(one(Tv)))))
            max_dx = max(max_dx, dx)

            if dx > alg.tol && Rs[i + 1] < alg.rmax
                J1m, J2m = J1[imax1:imax1, :], J2[j_g_best:j_g_best, :]
                cre1_new = reshape(_evaluate_on_domain(f, domain, hcat(J1, repeat(J2m, size(J1, 1), 1))), Rs[i] * Is[i], 1)
                cre2_new = reshape(_evaluate_on_domain(f, domain, hcat(repeat(J1m, size(J2, 1), 1), J2)), 1, Is[i + 1] * Rs[i + 2])

                uold, lold = mid_inv_U[i + 1], mid_inv_L[i + 1]
                erow = reshape(cry1[imax1, :], 1, Rs[i + 1])
                ecol = cre1_new[ilocl[i + 1], 1]
                alpha = cre1_new[imax1, 1] - sum(vec(erow * uold) .* vec(lold * ecol))
                (!isfinite(real(alpha)) || !isfinite(imag(alpha)) || abs(alpha) <= max(alg.tol, eps(real(float(abs(alpha)))))) && continue

                new_U = zeros(Tv, Rs[i + 1] + 1, Rs[i + 1] + 1)
                new_U[1:Rs[i + 1], 1:Rs[i + 1]] = uold
                new_U[1:Rs[i + 1], Rs[i + 1] + 1] = -(uold * (lold * ecol)) / alpha
                new_U[Rs[i + 1] + 1, Rs[i + 1] + 1] = 1 / alpha
                mid_inv_U[i + 1] = new_U

                new_L = zeros(Tv, Rs[i + 1] + 1, Rs[i + 1] + 1)
                new_L[1:Rs[i + 1], 1:Rs[i + 1]] = lold
                new_L[Rs[i + 1] + 1, 1:Rs[i + 1]] = -vec(erow * uold * lold)
                new_L[Rs[i + 1] + 1, Rs[i + 1] + 1] = 1
                mid_inv_L[i + 1] = new_L

                y[i] = reshape(hcat(cry1, cre1_new), Rs[i], Is[i], Rs[i + 1] + 1)
                y[i + 1] = reshape(vcat(cry2, cre2_new), Rs[i + 1] + 1, Is[i + 1], Rs[i + 2])
                Rs[i + 1] += 1

                Jyl[i + 1] = vcat(Jyl[i + 1], J1m)
                Jyr[i + 1] = vcat(Jyr[i + 1], J2m)
                push!(ilocl[i + 1], imax1)
                push!(ilocr[i + 1], j_g_best)
            end
        end

        cores_out = _form_tensor(y, mid_inv_L, mid_inv_U, N, Rs, Is)
        val_eps = norm(ys_val - _evaluate_tt(cores_out, Xs_val, N)) / norm_ys_val

        alg.verbose && @info "Sweep $swp: ε = $(val_eps), max_dx = $(max_dx), max rank = $(maximum(Rs))"

        if val_eps < alg.tol
            converged = true
            break
        end
    end

    converged && alg.verbose && @info "Converged: ε = $(val_eps) < $(alg.tol)"
    !converged && alg.verbose && @warn "Max iterations reached"

    fallback_tol = max(sqrt(alg.tol), 10 * alg.tol)
    if !converged && (!isfinite(val_eps) || val_eps > fallback_tol)
        alg.verbose && @warn "Greedy cross appears stalled/unstable (ε = $(val_eps)); retrying with DMRG cross"
        init_rank = min(maximum(Rs), alg.rmax)
        dmrg_alg = DMRG(maxiter = alg.maxiter, tol = alg.tol, rmax = alg.rmax, kickrank = nothing, verbose = alg.verbose)
        return tt_cross(f, domain, dmrg_alg; ranks = init_rank, val_size = val_size)
    end

    return TTvector{eltype(y[1]), N}(N, _form_tensor(y, mid_inv_L, mid_inv_U, N, Rs, Is), Tuple(Is), copy(Rs), zeros(Int, N))
end

function _sample_superblock(f, domain, I_l, I_g, k, Is, N)
    r_l, r_g = size(I_l[k], 1), size(I_g[k + 1], 1)
    s1, s2 = Is[k], Is[k + 1]
    indices = Matrix{Int}(undef, r_l * s1 * s2 * r_g, N)
    idx = 1
    for rg in 1:r_g, i2 in 1:s2, i1 in 1:s1, rl in 1:r_l
        k > 1 && (indices[idx, 1:(k - 1)] = I_l[k][rl, :])
        indices[idx, k] = i1
        indices[idx, k + 1] = i2
        k + 1 < N && (indices[idx, (k + 2):N] = I_g[k + 1][rg, :])
        idx += 1
    end
    return reshape(_evaluate_on_domain(f, domain, indices), r_l, s1, s2, r_g)
end

function _combine_indices_left(I_l_k, s)
    r_l, n_cols = size(I_l_k, 1), size(I_l_k, 2)
    result = zeros(Int, r_l * s, n_cols + 1)
    idx = 1
    for i in 1:s, r in 1:r_l
        n_cols > 0 && (result[idx, 1:n_cols] = I_l_k[r, :])
        result[idx, end] = i
        idx += 1
    end
    return result
end

function _combine_indices_right(s, I_g_k)
    r_g, n_cols = size(I_g_k, 1), size(I_g_k, 2)
    result = zeros(Int, s * r_g, n_cols + 1)
    idx = 1
    for r in 1:r_g, i in 1:s
        result[idx, 1] = i
        n_cols > 0 && (result[idx, 2:end] = I_g_k[r, :])
        idx += 1
    end
    return result
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
    Tv = _infer_value_type(f, domain)

    if N == 1
        coords = reshape(domain[1], :, 1)
        vals = vec(f(coords))
        return TTvector{eltype(vals), 1}(1, [reshape(vals, Is[1], 1, 1)], Tuple(Is), [1, 1], [0])
    end

    Rs = isa(ranks, Int) ? vcat([1], fill(ranks, N - 1), [1]) : vcat([1], ranks, [1])
    _cap_ranks!(Rs, Is, alg.rmax)

    I_l = Vector{Matrix{Int}}(undef, N)
    I_g = Vector{Matrix{Int}}(undef, N)
    I_l[1] = ones(Int, 1, 0)
    I_g[N] = ones(Int, 1, 0)

    for k in 2:N
        I_l[k] = [rand(1:Is[j]) for _ in 1:Rs[k], j in 1:(k - 1)]
    end
    for k in 1:(N - 1)
        I_g[k] = [rand(1:Is[k + j]) for _ in 1:Rs[k + 1], j in 1:(N - k)]
    end

    cores = [randn(Tv, Is[n], Rs[n], Rs[n + 1]) for n in 1:N]

    Xs_val = hcat([rand(1:Is[d], val_size) for d in 1:N]...)
    ys_val = _evaluate_on_domain(f, domain, Xs_val)
    norm_ys_val = max(norm(ys_val), alg.tol)

    alg.verbose && @info "DMRG cross-interpolation over $(N)D domain with $(prod(Is)) grid points"

    converged = false
    val_eps = Inf

    for iter in 1:alg.maxiter
        for k in 1:(N - 1)
            superblock = _sample_superblock(f, domain, I_l, I_g, k, Is, N)
            r_l, s1, s2, r_g = size(superblock)
            U, S, Vt = _svdtrunc(reshape(superblock, r_l * s1, s2 * r_g); max_bond = alg.rmax, truncerr = alg.tol)
            r = size(S, 1)

            if k < N - 1
                if alg.kickrank !== nothing
                    r_kick = min(r + alg.kickrank, alg.rmax, r_l * s1)
                    Q_mat = r_kick > r ?
                        Matrix(first(qr(hcat(U, randn(Tv, r_l * s1, r_kick - r)))))[:, 1:r_kick] :
                        Matrix(first(qr(U)))
                else
                    Q_mat = Matrix(first(qr(U)))
                end
                I_idx, _ = maxvol!(copy(Q_mat), alg.pivot.tol, alg.pivot.maxiter)
                I_l[k + 1] = _combine_indices_left(I_l[k], s1)[I_idx, :]
                Rs[k + 1] = length(I_idx)
                cores[k] = permutedims(reshape(Q_mat / Q_mat[I_idx, :], r_l, s1, Rs[k + 1]), (2, 1, 3))
            else
                cores[k] = permutedims(reshape(U, r_l, s1, r), (2, 1, 3))
                cores[k + 1] = permutedims(reshape(S * Vt, r, s2, r_g), (2, 1, 3))
                Rs[k + 1] = r
            end
        end

        val_eps = norm(ys_val - _evaluate_tt(cores, Xs_val, N)) / norm_ys_val
        alg.verbose && @info "Sweep $(2 * iter - 1) (L→R): ε = $(val_eps), max rank = $(maximum(Rs))"
        val_eps < alg.tol && (converged = true; break)

        for k in (N - 1):-1:1
            superblock = _sample_superblock(f, domain, I_l, I_g, k, Is, N)
            r_l, s1, s2, r_g = size(superblock)
            U, S, Vt = _svdtrunc(reshape(superblock, r_l * s1, s2 * r_g); max_bond = alg.rmax, truncerr = alg.tol)
            r = size(S, 1)

            if k > 1
                if alg.kickrank !== nothing
                    r_kick = min(r + alg.kickrank, alg.rmax, s2 * r_g)
                    Q_mat = r_kick > r ?
                        Matrix(first(qr(hcat(Vt', randn(Tv, s2 * r_g, r_kick - r)))))[:, 1:r_kick] :
                        Matrix(first(qr(Vt')))
                else
                    Q_mat = Matrix(first(qr(Vt')))
                end
                I_idx, _ = maxvol!(copy(Q_mat), alg.pivot.tol, alg.pivot.maxiter)
                I_g[k] = _combine_indices_right(s2, I_g[k + 1])[I_idx, :]
                Rs[k + 1] = length(I_idx)
                cores[k + 1] = permutedims(reshape((Q_mat / Q_mat[I_idx, :])', Rs[k + 1], s2, r_g), (2, 1, 3))
            else
                cores[k] = permutedims(reshape(U * S, r_l, s1, r), (2, 1, 3))
                cores[k + 1] = permutedims(reshape(Vt, r, s2, r_g), (2, 1, 3))
                Rs[k + 1] = r
            end
        end

        val_eps = norm(ys_val - _evaluate_tt(cores, Xs_val, N)) / norm_ys_val
        alg.verbose && @info "Sweep $(2 * iter) (R→L): ε = $(val_eps), max rank = $(maximum(Rs))"
        val_eps < alg.tol && (converged = true; break)
    end

    converged && alg.verbose && @info "Converged: ε = $(val_eps) < $(alg.tol)"
    !converged && alg.verbose && @warn "Max iterations reached: ε = $(val_eps)"

    return TTvector{eltype(cores[1]), N}(N, cores, Tuple(Is), copy(Rs), zeros(Int, N))
end

function tt_integrate(
        f::Function,
        lower::Vector{T},
        upper::Vector{T};
        alg::CrossAlgorithm = MaxVol(),
        nquad::Int = 20,
        kwargs...
    ) where {T <: Number}

    d = length(lower)
    @assert length(upper) == d "lower and upper bounds must have the same length"

    nodes = Vector{Vector{T}}(undef, d)
    weights = Vector{Vector{T}}(undef, d)
    for k in 1:d
        nodes[k], weights[k] = _gauss_legendre(nquad, lower[k], upper[k])
    end

    tt = tt_cross(f, nodes, alg; kwargs...)
    return _contract_with_weights(tt.ttv_vec, weights)
end

function tt_integrate(f::Function, d::Int; lower::T = 0.0, upper::T = 1.0, kwargs...) where {T <: Number}
    return tt_integrate(f, fill(lower, d), fill(upper, d); kwargs...)
end

function _contract_with_weights(cores::Vector{<:Array{T, 3}}, weights::Vector{<:Vector{T}}) where {T}
    result = ones(T, 1)
    for k in eachindex(cores)
        contracted = sum(weights[k][i] .* cores[k][i, :, :] for i in axes(cores[k], 1))
        result = vec(result' * contracted)
    end
    return result[1]
end

function _gauss_legendre(n::Int, a::T, b::T) where {T <: Number}
    β = T[k / sqrt(4k^2 - 1) for k in 1:(n - 1)]
    J = SymTridiagonal(zeros(T, n), β)
    λ, V = eigen(J)
    return (b - a) / 2 .* λ .+ (a + b) / 2, (b - a) .* V[1, :] .^ 2
end
