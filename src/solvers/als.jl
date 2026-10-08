using LinearMaps
using TensorOperations

"""
Implementation based on the presentation in 
Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider. "The alternating linear scheme for tensor optimization in the tensor train format." SIAM Journal on Scientific Computing 34.2 (2012): A683-A713.
"""

function init_H(x_tt::AbstractTTVector, A_tto::AbstractTTOperator)
    T = eltype(x_tt)
    d = nsites(x_tt)
    H = Array{Array{T}}(undef, d)
    H[d] = ones(T, 1, 1, 1)
    for i in d:-1:2
        H[i - 1] = zeros(T, A_tto.ranks[i], x_tt.ranks[i], x_tt.ranks[i])
        x_vec = x_tt.cores[i]
        A_vec = A_tto.cores[i]
        update_H!(x_vec, A_vec, H[i], H[i - 1])
    end
    return H
end

function update_H!(x_vec::Array{T, 3}, A_vec::Array{T, 4}, Hi::Array{T, 3}, Him::Array{T, 3}) where {T <: Number}
    @tensoropt((ϕ, χ), Him[a, α, β] = conj.(x_vec)[j, α, ϕ] * Hi[z, ϕ, χ] * x_vec[k, β, χ] * A_vec[j, k, a, z]) #size (rim, rim, rAim)
    return nothing
end

function init_Hb(x_tt::AbstractTTVector, b_tt::AbstractTTVector)
    T = eltype(x_tt)
    d = nsites(x_tt)
    H_b = Array{Array{T}}(undef, d)
    H_b[d] = ones(T, 1, 1)
    for i in d:-1:2
        H_b[i - 1] = zeros(T, x_tt.ranks[i], b_tt.ranks[i])
        b_vec = b_tt.cores[i]
        x_vec = x_tt.cores[i]
        update_Hb!(x_vec, b_vec, H_b[i], H_b[i - 1]) #size(rbim, rim)
    end
    return H_b
end

function update_Hb!(x_vec::Array{T, 3}, b_vec::Array{T, 3}, H_bi::Array{T, 2}, H_bim::Array{T, 2}) where {T <: Number}
    @tensoropt((ϕ, χ), H_bim[α, β] = H_bi[ϕ, χ] * b_vec[i, β, χ] * conj.(x_vec)[i, α, ϕ])
    return nothing
end

function update_G!(x_vec::Array{T, 3}, A_vec::Array{T, 4}, Gi::AbstractArray{<:Any, 5}, Gip::AbstractArray{<:Any, 5}) where {T <: Number}
    @tensor Gip[j, α, k, β, J] = (conj.(x_vec)[l, ϕ, α] * (Gi[l, ϕ, m, χ, L] * x_vec[m, χ, β])) * A_vec[j, k, L, J]
    return nothing
end

function update_Gb!(x_vec::Array{T, 3}, b_vec::Array{T, 3}, G_bi::AbstractArray{<:Any, 3}, G_bip::AbstractArray{<:Any, 3}) where {T <: Number}
    @tensoropt((ϕ, χ), G_bip[i, α, β] = b_vec[i, ϕ, β] * G_bi[j, χ, ϕ] * conj.(x_vec)[j, χ, α])
    return nothing
end

#full assemble of matrix K
function K_full(Gi::Array{T, 5}, Hi::Array{T, 3}, K_dims::NTuple{3, Int}) where {T <: Number}
    K = zeros(T, prod(K_dims), prod(K_dims))
    Krshp = reshape(K, (K_dims..., K_dims...))
    @tensor Krshp[a, b, c, d, e, f] = Gi[a, b, d, e, z] * Hi[z, c, f] #size (ni,rim,ri,ni,rim,ri)
    return K
end

function Ksolve(
        Gi::Array{T, 5}, G_bi::Array{T, 3}, Hi::Array{T, 3}, H_bi::Array{T, 2};
        local_solver::Symbol = :direct,
        local_threshold::Int = typemax(Int),
        local_maxiter::Int = 200,
        local_tol::Real = 1.0e-8
    ) where {T <: Number}
    K_dims = (size(Gi, 1), size(Gi, 2), size(Hi, 2))
    @tensor Pb[i, α1, α2] := G_bi[i, α1, β] * H_bi[α2, β] #size (ni,rim,ri)
    if _use_iterative(local_solver, prod(K_dims), local_threshold)
        function K_matfree!(y::AbstractVector{T}, x::AbstractVector{T})
            Yr = reshape(y, K_dims)
            Xr = reshape(x, K_dims)
            @tensor Yr[a, b, c] = Gi[a, b, d, e, z] * Xr[d, e, f] * Hi[z, c, f]
            return y
        end
        sol, _ = linsolve(
            LinearMap{T}(
                K_matfree!, prod(K_dims);
                issymmetric = true,
                ismutating = true
            ),
            Pb[:],
            zeros(T, prod(K_dims));
            issymmetric = true,
            isposdef = true,
            tol = local_tol,
            maxiter = local_maxiter,
        )
        return reshape(sol, K_dims)
    end
    K = K_full(Gi, Hi, K_dims)
    return reshape(K \ Pb[:], K_dims)
end

function K_eigmin(Gi::Array{T, 5}, Hi::Array{T, 3}, ttv_vec::Array{T, 3}; local_solver::Symbol = :direct, local_threshold::Int = typemax(Int), local_maxiter::Int = 200, local_tol::Real = 1.0e-6) where {T <: Number}
    K_dims = (size(Gi, 1), size(Gi, 2), size(Hi, 2))
    if _use_iterative(local_solver, prod(K_dims), local_threshold)
        H = zeros(T, prod(K_dims))
        function K_matfree(V::AbstractArray{T, 1}; Gi = Gi::Array{T, 5}, Hi = Hi::Array{T, 3}, K_dims = K_dims, H = H::AbstractArray{T, 1})
            Hrshp = reshape(H, K_dims)
            @tensoropt((b, c, e, f), Hrshp[a, b, c] = Gi[a, b, d, e, z] * reshape(V, K_dims)[d, e, f] * Hi[z, c, f])
            return H::AbstractArray{T, 1}
        end
        r = lobpcg(LinearMap(K_matfree, prod(K_dims); ishermitian = true), false, ttv_vec[:], 1; maxiter = local_maxiter, tol = local_tol)
        return r.λ[1]::Real, reshape(r.X[:, 1], K_dims)::Array{T, 3}
    else
        K = K_full(Gi, Hi, K_dims)
        F = eigen(Hermitian(K), 1:1)
        return real(F.values[1])::Real, reshape(F.vectors[:, 1], K_dims)::Array{T, 3}
    end
end

function K_eiggenmin(Gi, Hi, Ki, Li, ttv_vec; local_solver::Symbol = :auto, local_threshold::Int = 2500)
    @tensor begin
        K[a, b, c, d, e, f] := Gi[d, e, a, b, z] * Hi[z, f, c] #size (ni,rim,ri,ni,rim,ri)
        S[a, b, c, d, e, f] := Ki[d, e, a, b, z] * Li[z, f, c] #size (ni,rim,ri,ni,rim,ri)
    end
    if _use_iterative(local_solver, prod(size(K)[1:3]), local_threshold)
        r = lobpcg(reshape(K, prod(size(K)[1:3]), :), reshape(S, prod(size(S)[1:3]), :), false, ttv_vec[:], 1; maxiter = 500, tol = 1.0e-8)
        return r.λ[1], reshape(r.X[:, 1], size(K)[1:3])
    else
        F = eigen(reshape(K, prod(size(K)[1:3]), :), reshape(S, prod(size(K)[1:3]), :))
        return real(F.values[1]), reshape(F.vectors[:, 1], size(K)[1:3])
    end
end

function left_core_move(x_tt::AbstractTTVector, V::Array{T, 3}, i::Int, x_rks) where {T <: Number}
    rim, ri = x_rks[i], x_rks[i + 1]
    ni = x_tt.dims[i]

    # Prepare core movements
    QV, RV = qr(reshape(permutedims(V, [1 3 2]), ni * ri, :)) #QV: ni*ri x ni*ri; RV ni*ri x rim

    # Apply core movement 3.1
    x_tt.cores[i] = permutedims(reshape(Matrix(QV)[:, 1:rim], ni, ri, :), [1 3 2])

    # Apply core movement 3.2
    @tensoropt((b, c, z), Xim[a, b, c] := x_tt.cores[i - 1][a, b, z] * RV[1:rim, :][c, z]) #size (nim,rim2,rim_new)
    x_tt.cores[i - 1] = Xim
    _center_moved_left!(x_tt, i)
    return x_tt
end

function right_core_move(x_tt::AbstractTTVector, V::Array{T, 3}, i::Int, x_rks) where {T <: Number}
    rim, ri = x_rks[i], x_rks[i + 1]
    ni = x_tt.dims[i]
    QV, RV = qr(reshape(V, ni * rim, :)) #QV: ni*rim x ni*rim; RV ni*rim x ri

    # Apply core movement 3.1
    x_tt.cores[i] = reshape(Matrix(QV)[:, 1:ri], ni, rim, :)

    # Apply core movement 3.2
    @tensoropt((b, c, z), Xip[a, b, c] := RV[1:ri, :][b, z] * x_tt.cores[i + 1][a, z, c]) #size (nip,ri,rip)
    x_tt.cores[i + 1] = Xip
    _center_moved_right!(x_tt, i)
    return x_tt
end


# Implementation of `linear_solve(A, b, tt_start, ::ALS)`; see [`ALS`](@ref).
function _als_linsolve_impl(
        A::AbstractTTOperator, b::AbstractTTVector, tt_start::AbstractTTVector;
        max_sweeps::Int, local_solver::Symbol, local_threshold::Int,
        local_maxiter::Int, local_tol::Real,
        return_info::Bool, verbosity::Int, show_progress::Bool
    )
    T = eltype(tt_start)
    d = nsites(A)
    tt_opt = orthogonalize(tt_start)
    dims = tt_start.dims
    rks = copy(tt_start.ranks)

    G = Array{Array{T}}(undef, d)
    G_b = Array{Array{T}}(undef, d)
    for i in 1:d
        G[i] = zeros(T, dims[i], rks[i], dims[i], rks[i], A.ranks[i + 1])
        G_b[i] = zeros(dims[i], rks[i], b.ranks[i + 1])
    end
    G[1] = reshape(A.cores[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)
    G_b[1] = reshape(b.cores[1], dims[1], 1, :)
    H = init_H(tt_opt, A)
    H_b = init_Hb(tt_opt, b)

    local_opts = (; local_solver, local_threshold, local_maxiter, local_tol)
    progress = _solver_progress(max_sweeps, show_progress; desc = "ALS linear solve")
    for sweep in 1:max_sweeps
        for i in 1:(d - 1)
            V = Ksolve(G[i], G_b[i], H[i], H_b[i]; local_opts...)
            tt_opt = right_core_move(tt_opt, V, i, rks)
            update_G!(tt_opt.cores[i], A.cores[i + 1], G[i], G[i + 1])
            update_Gb!(tt_opt.cores[i], b.cores[i + 1], G_b[i], G_b[i + 1])
        end
        for i in d:(-1):2
            V = Ksolve(G[i], G_b[i], H[i], H_b[i]; local_opts...)
            tt_opt = left_core_move(tt_opt, V, i, rks)
            update_H!(tt_opt.cores[i], A.cores[i], H[i], H[i - 1])
            update_Hb!(tt_opt.cores[i], b.cores[i], H_b[i], H_b[i - 1])
        end
        max_rank = maximum(tt_opt.ranks)
        verbosity ≥ 2 && @info "ALS linear solve" sweep max_rank
        next!(progress; showvalues = [("sweep", "$sweep/$max_sweeps"), ("largest rank", max_rank)])
    end
    return return_info ? (tt_opt, (; residual = norm(A * tt_opt - b) / max(norm(b), eps(real(T))))) : tt_opt
end

# Implementation of `eigen_solve(A, tt_start, ::ALS)`; see [`ALS`](@ref).
function _als_eigsolve_impl(
        A::AbstractTTOperator, tt_start::AbstractTTVector;
        max_sweeps::Vector{Int}, max_bond::Vector{Int}, noise::Vector{Float64},
        local_solver::Symbol, local_threshold::Int, local_maxiter::Int, local_tol::Real,
        verbosity::Int, show_progress::Bool
    )
    T = eltype(tt_start)
    d = nsites(A)
    dims = tt_start.dims
    tt_opt = orthogonalize(tt_start)
    E = Float64[]
    local_opts = (; local_solver, local_threshold, local_maxiter, local_tol)
    progress = _solver_progress(sum(max_sweeps), show_progress; desc = "ALS eigen solve")
    sweep = 0
    for (stage, nsweeps) in enumerate(max_sweeps)
        r = maximum(tt_opt.ranks)
        max_bond[stage] < r && throw(
            ArgumentError(
                "ALS cannot lower ranks: stage $stage has max_bond = $(max_bond[stage]) but the current largest rank is $r"
            )
        )
        if max_bond[stage] > r
            tt_opt = orthogonalize(increase_ranks(tt_opt, max_bond[stage]; noise = noise[stage]))
        end
        # G[i] for i > 1 is overwritten during each left-to-right pass before use.
        G = Array{Array{T}}(undef, d)
        for i in 1:d
            G[i] = zeros(T, dims[i], tt_opt.ranks[i], dims[i], tt_opt.ranks[i], A.ranks[i + 1])
        end
        G[1] = reshape(A.cores[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)
        H = init_H(tt_opt, A)
        for _ in 1:nsweeps
            sweep += 1
            for i in 1:(d - 1)
                λ, V = K_eigmin(G[i], H[i], tt_opt.cores[i]; local_opts...)
                push!(E, λ)
                tt_opt = right_core_move(tt_opt, V, i, tt_opt.ranks)
                update_G!(tt_opt.cores[i], A.cores[i + 1], G[i], G[i + 1])
            end
            for i in d:(-1):2
                λ, V = K_eigmin(G[i], H[i], tt_opt.cores[i]; local_opts...)
                push!(E, λ)
                tt_opt = left_core_move(tt_opt, V, i, tt_opt.ranks)
                update_H!(tt_opt.cores[i], A.cores[i], H[i], H[i - 1])
            end
            eigenvalue = E[end]
            verbosity ≥ 2 && @info "ALS eigen solve" sweep max_rank = maximum(tt_opt.ranks) eigenvalue
            next!(progress; showvalues = [("sweep", "$sweep/$(sum(max_sweeps))"), ("eigenvalue", eigenvalue)])
        end
    end
    return E, tt_opt
end

function eigen_solve(A::AbstractTTOperator, guess::AbstractTTVector, alg::ALS)
    _reject_unused(alg, "eigen_solve", (:return_info,), "every option except `return_info`")
    st = _stages(;
        max_sweeps = alg.max_sweeps,
        max_bond = something(alg.max_bond, maximum(guess.ranks)),
        noise = alg.noise
    )
    return _als_eigsolve_impl(
        A, guess; st...,
        local_solver = alg.local_solver,
        local_threshold = something(alg.local_threshold, typemax(Int)),
        local_maxiter = alg.local_maxiter,
        local_tol = alg.local_tol,
        verbosity = alg.verbosity,
        show_progress = alg.show_progress
    )
end

"""
    als_gen_eigsolve(A, S, tt_start; sweep_schedule, max_bond, tol, local_solver, local_threshold)

Find the smallest generalized eigenpair `Ax = λ S x` using the ALS algorithm.

# Arguments
- `A::TTOperator{T}`: the operator on the left-hand side.
- `S::TTOperator{T}`: the positive-definite metric operator on the right-hand side.
- `tt_start::TTVector{T}`: initial guess for the eigenvector.

# Keyword arguments
- `sweep_schedule::Vector{Int}=[2]`: sweep count at which each rank stage ends.
- `max_bond::Vector{Int}`: maximum bond dimension at each stage.
- `tol::Float64=1e-10`: tolerance for the local generalized eigensolver.
- `local_solver::Symbol=:auto`: solver for the local eigenproblems: `:direct`,
  `:iterative`, or `:auto` (direct up to `local_threshold` unknowns).
- `local_threshold::Int=2500`: local problem size above which `:auto` solves iteratively.

# Returns
`(E, tt_opt)` where `E` is the eigenvalue history and `tt_opt` is the approximate
eigenvector, or `nothing` if the schedule is exhausted without a final return.
"""
function als_gen_eigsolve(
        A::AbstractTTOperator, S::AbstractTTOperator, tt_start::AbstractTTVector;
        sweep_schedule = [2], max_bond = [maximum(tt_start.ranks)],
        tol = 1.0e-10, local_solver::Symbol = :auto, local_threshold::Int = 2500,
        show_progress::Bool = false
    )
    T = eltype(tt_start)
    d = nsites(A)
    # Initialize the to be returned tensor in its tensor train format
    tt_opt = orthogonalize(tt_start)
    dims = tt_start.dims
    E = zeros(Float64, d * sweep_schedule[end]) #output eigenvalue
    # Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
    rks = tt_start.ranks

    # Initialize the arrays of G and K
    G = Array{Array{T}}(undef, d)
    K = Array{Array{T}}(undef, d)

    # Initialize G[1]
    for i in 1:d
        G[i] = zeros(dims[i], rks[i], dims[i], rks[i], A.ranks[i + 1])
        K[i] = zeros(dims[i], rks[i], dims[i], rks[i], S.ranks[i + 1])
    end
    G[1] = reshape(A.cores[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)
    K[1] = reshape(S.cores[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)

    #Initialize H and H_b
    H = init_H(tt_opt, A)
    L = init_H(tt_opt, S)

    nsweeps = 0 #sweeps counter
    i_schedule, i_μit = 1, 0
    progress = _solver_progress(max(sweep_schedule[end] - 1, 1), show_progress; desc = "ALS generalized eigen solve")
    while i_schedule <= length(sweep_schedule)
        nsweeps += 1
        if nsweeps == sweep_schedule[i_schedule]
            i_schedule += 1
            if i_schedule > length(sweep_schedule)
                return E[1:i_μit], tt_opt
            else
                tt_opt = increase_ranks(tt_opt, max_bond[i_schedule])
                tt_opt = orthogonalize(tt_opt)
                rks = copy(tt_opt.ranks)
                for i in 1:(d - 1)
                    Gtemp = zeros(T, dims[i + 1], rks[i + 1], dims[i + 1], rks[i + 1], A.ranks[i + 2])
                    Ktemp = zeros(T, dims[i + 1], rks[i + 1], dims[i + 1], rks[i + 1], S.ranks[i + 2])
                    Gtemp[1:size(G[i + 1], 1), 1:size(G[i + 1], 2), 1:size(G[i + 1], 3), 1:size(G[i + 1], 4), 1:size(G[i + 1], 5)] = G[i + 1]
                    Ktemp[1:size(K[i + 1], 1), 1:size(K[i + 1], 2), 1:size(K[i + 1], 3), 1:size(K[i + 1], 4), 1:size(K[i + 1], 5)] = K[i + 1]
                    G[i + 1] = Gtemp
                    K[i + 1] = Ktemp

                    Htemp = zeros(T, size(H[i], 1), rks[i + 1], rks[i + 1])
                    Ltemp = zeros(T, size(L[i], 1), rks[i + 1], rks[i + 1])
                    Htemp[1:size(H[i], 1), 1:size(H[i], 2), 1:size(H[i], 3)] = H[i]
                    Ltemp[1:size(L[i], 1), 1:size(L[i], 2), 1:size(L[i], 3)] = L[i]
                    H[i] = Htemp
                    L[i] = Ltemp
                end
            end
        end

        # First half sweep
        for i in 1:(d - 1)

            # Optimize core i only while it is the orthogonality center
            if _orthogonality_center(tt_opt) == i
                # Define V as solution of K*x=Pb in x
                i_μit += 1
                E[i_μit], V = K_eiggenmin(G[i], H[i], K[i], L[i], tt_opt.cores[i]; local_solver, local_threshold)
                tt_opt = right_core_move(tt_opt, V, i, tt_opt.ranks)
            end

            #update G and K
            update_G!(tt_opt.cores[i], A.cores[i + 1], G[i], G[i + 1])
            update_G!(tt_opt.cores[i], S.cores[i + 1], K[i], K[i + 1])
        end

        # Second half sweep
        for i in d:(-1):2
            # Define V as solution of K*x=Pb in x
            i_μit += 1
            E[i_μit], V = K_eiggenmin(G[i], H[i], K[i], L[i], tt_opt.cores[i]; local_solver, local_threshold)
            tt_opt = left_core_move(tt_opt, V, i, tt_opt.ranks)
            update_H!(tt_opt.cores[i], A.cores[i], H[i], H[i - 1])
            update_H!(tt_opt.cores[i], S.cores[i], L[i], L[i - 1])
        end
        next!(progress)
    end
    return E[1:i_μit], tt_opt
end
