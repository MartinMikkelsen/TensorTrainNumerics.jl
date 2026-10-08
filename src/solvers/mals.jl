using IterativeSolvers
using LinearMaps
using LinearAlgebra

"""
Implementation based on the presentation in 
Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider. "The alternating linear scheme for tensor optimization in the tensor train format." SIAM Journal on Scientific Computing 34.2 (2012): A683-A713.
"""

function updateH_mals!(x_vec::Array{T, 3}, A_vec::Array{T, 4}, Hi::AbstractArray{<:Any, 5}, Him::AbstractArray{<:Any, 5}) where {T <: Number}
    @tensor(Him[a, i, α, l, β] = conj.(x_vec)[j, α, x] * (Hi[z, j, x, k, y] * x_vec[k, β, y]) * A_vec[i, l, a, z])
    return nothing
end

function init_H_mals(x_tt::AbstractTTVector, A::AbstractTTOperator, rmax::Int)
    T = eltype(x_tt)
    d = nsites(x_tt)
    H = Array{Array{T, 5}}(undef, d - 1)
    # H[d-1] from the last operator core
    H[d - 1] = reshape(
        permutedims(A.tto_vec[d], [3, 1, 2, 4]),
        :, x_tt.ttv_dims[d], 1, x_tt.ttv_dims[d], 1
    )
    for i in (d - 1):-1:2
        rmax_i = min(rmax, prod(x_tt.ttv_dims[1:i]), prod(x_tt.ttv_dims[(i + 1):end]))
        H[i - 1] = zeros(
            T,
            A.tto_rks[i],
            x_tt.ttv_dims[i],
            rmax_i,
            x_tt.ttv_dims[i],
            rmax_i
        )
        # view the “active” slices of H[i] and H[i-1]
        Hi = @view(H[i][:, :, 1:x_tt.ttv_rks[i + 2], :, 1:x_tt.ttv_rks[i + 2]])
        Him = @view(H[i - 1][:, :, 1:x_tt.ttv_rks[i + 1], :, 1:x_tt.ttv_rks[i + 1]])
        updateH_mals!(x_tt.ttv_vec[i + 1], A.tto_vec[i], Hi, Him)
    end
    return H
end

# —— Corrected updateHb_mals! ——

function updateHb_mals!(
        xtt_vec::Array{T, 3}, btt_vec::Array{T, 3},
        Hbi::AbstractArray{<:Any, 3}, Hbim::AbstractArray{<:Any, 3}
    ) where {T <: Number}
    @tensor Hbim[β, i, χ] = conj.(xtt_vec)[j, χ, a] * Hbi[γ, j, a] * btt_vec[i, β, γ]
    return nothing
end

function init_Hb_mals(x_tt::AbstractTTVector, b::AbstractTTVector, rmax::Int)
    T = eltype(x_tt)
    d = nsites(x_tt)
    Hb = Array{Array{T, 3}}(undef, d - 1)
    # Base case: H_b[d-1] from the last vector core
    Hb[d - 1] = reshape(
        permutedims(b.ttv_vec[d], [2, 1, 3]),
        b.ttv_rks[d], b.ttv_dims[d], 1
    )
    for i in (d - 1):-1:2
        rmax_i = min(rmax, prod(x_tt.ttv_dims[1:i]), prod(x_tt.ttv_dims[(i + 1):end]))
        # Allocate H_b[i-1] as (b_rks[i], n_i, rmax_i)
        Hb[i - 1] = zeros(
            T,
            b.ttv_rks[i],
            b.ttv_dims[i],
            rmax_i
        )
        # view active slices of H_b[i] and H_b[i-1]
        Hbi = @view(Hb[i][:, :, 1:x_tt.ttv_rks[i + 2]])
        Hbim = @view(Hb[i - 1][:, :, 1:x_tt.ttv_rks[i + 1]])
        updateHb_mals!(x_tt.ttv_vec[i + 1], b.ttv_vec[i], Hbi, Hbim)
    end
    return Hb
end

function left_core_move_mals(
        xtt::AbstractTTVector, i::Integer, V::Array{T, 4},
        trunc_tol::Real, max_bond::Integer; trunc_err = nothing
    ) where {T <: Number}
    u_V, s_V, v_V = svd(reshape(V, prod(size(V)[1:2]), :))
    xtt.ttv_rks[i + 1] = _trunc_rank(s_V, trunc_tol, nsites(xtt), max_bond)
    isnothing(trunc_err) || (trunc_err[] = max(trunc_err[], _discarded_weight(s_V, xtt.ttv_rks[i + 1])))

    # Update the (i+1)-th core from truncated V-matrix
    xtt.ttv_vec[i + 1] = permutedims(
        reshape(
            v_V'[1:xtt.ttv_rks[i + 1], :],
            xtt.ttv_rks[i + 1], size(V, 3), size(V, 4)
        ), [2, 1, 3]
    )

    # Update the i-th core from truncated U * diag(s_trunc)
    xtt.ttv_vec[i] = reshape(
        u_V[:, 1:xtt.ttv_rks[i + 1]] * Diagonal(s_V[1:xtt.ttv_rks[i + 1]]),
        size(V, 1), size(V, 2), :
    )
    _center_moved_left!(xtt, i + 1)
    return xtt
end

function right_core_move_mals(
        xtt::AbstractTTVector, i::Integer, V::Array{T, 4},
        trunc_tol::Real, max_bond::Integer; trunc_err = nothing
    ) where {T <: Number}
    u_V, s_V, v_V = svd(reshape(V, prod(size(V)[1:2]), :))
    xtt.ttv_rks[i + 1] = _trunc_rank(s_V, trunc_tol, nsites(xtt), max_bond)
    isnothing(trunc_err) || (trunc_err[] = max(trunc_err[], _discarded_weight(s_V, xtt.ttv_rks[i + 1])))

    # Update the i-th core from truncated U
    xtt.ttv_vec[i] = reshape(
        u_V[:, 1:xtt.ttv_rks[i + 1]],
        size(V, 1), size(V, 2), xtt.ttv_rks[i + 1]
    )

    # Update the (i+1)-th core from diag(s_trunc) * V^T
    xtt.ttv_vec[i + 1] = permutedims(
        reshape(
            Diagonal(s_V[1:xtt.ttv_rks[i + 1]]) * v_V'[1:xtt.ttv_rks[i + 1], :],
            xtt.ttv_rks[i + 1], size(V, 3), size(V, 4)
        ), [2, 1, 3]
    )
    _center_moved_right!(xtt, i)
    return xtt
end

function K_full_mals(
        Gi::AbstractArray{<:Any, 5}, Hi::AbstractArray{<:Any, 5},
        K_dims::NTuple{4, Int}
    )
    T = promote_type(eltype(Gi), eltype(Hi))
    # Keep the matrix dimension explicit when the promoted element type is abstract.
    K = zeros(T, (prod(K_dims), prod(K_dims)))
    Krshp = reshape(K, (K_dims..., K_dims...))
    @tensor Krshp[a, b, c, d, e, f, g, h] = Gi[a, b, e, f, z] * Hi[z, c, d, g, h]
    return K
end

function Ksolve_mals(
        Gi::AbstractArray{T, 5}, Hi::AbstractArray{T, 5},
        G_bi::AbstractArray{T, 3}, H_bi::AbstractArray{T, 3}
    ) where {T <: Number}
    K_dims = (size(Gi, 1), size(Gi, 2), size(Hi, 2), size(Hi, 3))
    K = K_full_mals(Gi, Hi, K_dims)
    Pb = zeros(T, K_dims)
    @tensor Pb[a, b, c, d] = G_bi[a, b, z] * H_bi[z, c, d]
    V = reshape(K, prod(K_dims), :) \ Pb[:]
    return reshape(V, K_dims)
end

function K_eigmin_mals(
        Gi::Array{T, 5}, Hi::Array{T, 5},
        ttv_vec_i::Array{T, 3}, ttv_vec_ip::Array{T, 3};
        local_solver::Symbol = :auto, local_threshold::Int = 256,
        local_maxiter::Int = 200, local_tol::Real = 1.0e-6
    ) where {T <: Number}
    K_dims = (
        size(ttv_vec_i, 1), size(ttv_vec_i, 2),
        size(ttv_vec_ip, 1), size(ttv_vec_ip, 3),
    )
    Gtemp = @view(Gi[:, 1:K_dims[2], :, 1:K_dims[2], :])
    Htemp = @view(Hi[:, :, 1:K_dims[4], :, 1:K_dims[4]])
    if _use_iterative(local_solver, prod(K_dims), local_threshold)
        H = zeros(T, prod(K_dims))
        function K_matfree(
                V::AbstractArray{S, 1};
                K_dims::NTuple{4, Int} = K_dims,
                H::AbstractArray{S, 1} = H,
                Gtemp::AbstractArray{S, 5} = Gtemp,
                Htemp::AbstractArray{S, 5} = Htemp
            ) where {S <: Number}
            Hrshp = reshape(H, K_dims)
            @tensoropt(
                (f, h),
                Hrshp[a, b, c, d] = Gtemp[a, b, e, f, z] *
                    reshape(V, K_dims)[e, f, g, h] *
                    Htemp[z, c, d, g, h]
            )
            return H::AbstractArray{S, 1}
        end
        X0 = zeros(T, prod(K_dims))
        X0_temp = reshape(X0, K_dims)
        @tensor X0_temp[a, b, c, d] = ttv_vec_i[a, b, z] * ttv_vec_ip[c, z, d]
        r = lobpcg(
            LinearMap(
                K_matfree, prod(K_dims);
                ishermitian = true
            ),
            false, X0, 1; maxiter = local_maxiter, tol = local_tol
        )
        return r.λ[1]::Float64, reshape(r.X[:, 1], K_dims)::Array{T, 4}
    else
        K = K_full_mals(Gtemp, Htemp, K_dims)
        F = eigen(Hermitian(K), 1:1)
        return real(F.values[1])::Float64,
            reshape(F.vectors[:, 1], K_dims)::Array{T, 4}
    end
end

# Implementation of `linear_solve(A, b, tt_start, ::MALS)`; see [`MALS`](@ref).
function _mals_linsolve_impl(
        A::AbstractTTOperator, b::AbstractTTVector, tt_start::AbstractTTVector;
        max_sweeps::Int, max_bond::Int, trunc_tol::Real,
        return_info::Bool, verbosity::Int, show_progress::Bool
    )
    T = eltype(tt_start)
    d = nsites(b)

    tt_opt = orthogonalize(tt_start)
    dims = tt_start.ttv_dims
    A_rks = A.tto_rks
    b_rks = b.ttv_rks

    G = Array{Array{T, 5}}(undef, d)
    G_b = Array{Array{T, 3}}(undef, d)
    for i in 1:d
        rmax_i = min(max_bond, prod(dims[1:(i - 1)]), prod(dims[i:end]))
        G[i] = zeros(dims[i], rmax_i, dims[i], rmax_i, A_rks[i + 1])
        G_b[i] = zeros(dims[i], rmax_i, b_rks[i + 1])
    end
    G[1][:, 1:1, :, 1:1, :] = reshape(A.tto_vec[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)
    G_b[1] = reshape(b.ttv_vec[1], dims[1], 1, :)

    H = init_H_mals(tt_opt, A, max_bond)
    H_b = init_Hb_mals(tt_opt, b, max_bond)
    progress = _solver_progress(max_sweeps, show_progress; desc = "MALS linear solve")

    trunc_err = Ref(0.0)
    for sweep in 1:max_sweeps
        trunc_err[] = 0.0
        for i in 1:(d - 1)
            Gi = @view(G[i][:, 1:tt_opt.ttv_rks[i], :, 1:tt_opt.ttv_rks[i], :])
            Hi = @view(H[i][:, :, 1:tt_opt.ttv_rks[i + 2], :, 1:tt_opt.ttv_rks[i + 2]])
            G_bi = @view(G_b[i][:, 1:tt_opt.ttv_rks[i], :])
            H_bi = @view(H_b[i][:, :, 1:tt_opt.ttv_rks[i + 2]])

            V = Ksolve_mals(Gi, Hi, G_bi, H_bi)
            tt_opt = right_core_move_mals(tt_opt, i, V, trunc_tol, max_bond; trunc_err)

            Gip = @view(G[i + 1][:, 1:tt_opt.ttv_rks[i + 1], :, 1:tt_opt.ttv_rks[i + 1], :])
            G_bip = @view(G_b[i + 1][:, 1:tt_opt.ttv_rks[i + 1], :])
            update_G!(tt_opt.ttv_vec[i], A.tto_vec[i + 1], Gi, Gip)
            update_Gb!(tt_opt.ttv_vec[i], b.ttv_vec[i + 1], G_bi, G_bip)
        end

        for i in (d - 1):-1:1
            Gi = @view(G[i][:, 1:tt_opt.ttv_rks[i], :, 1:tt_opt.ttv_rks[i], :])
            Hi = @view(H[i][:, :, 1:tt_opt.ttv_rks[i + 2], :, 1:tt_opt.ttv_rks[i + 2]])
            G_bi = @view(G_b[i][:, 1:tt_opt.ttv_rks[i], :])
            H_bi = @view(H_b[i][:, :, 1:tt_opt.ttv_rks[i + 2]])

            V = Ksolve_mals(Gi, Hi, G_bi, H_bi)
            tt_opt = left_core_move_mals(tt_opt, i, V, trunc_tol, max_bond; trunc_err)

            if i > 1
                Him = @view(H[i - 1][:, :, 1:tt_opt.ttv_rks[i + 1], :, 1:tt_opt.ttv_rks[i + 1]])
                updateH_mals!(tt_opt.ttv_vec[i + 1], A.tto_vec[i], Hi, Him)

                H_bim = @view(H_b[i - 1][:, :, 1:tt_opt.ttv_rks[i + 1]])
                updateHb_mals!(tt_opt.ttv_vec[i + 1], b.ttv_vec[i], H_bi, H_bim)
            end
        end
        max_rank = maximum(tt_opt.ttv_rks)
        verbosity ≥ 2 && @info "MALS linear solve" sweep max_rank truncation_error = trunc_err[]
        next!(progress; showvalues = [("sweep", "$sweep/$max_sweeps"), ("largest rank", max_rank), ("truncation error", trunc_err[])])
    end
    return return_info ? (tt_opt, (; residual = norm(A * tt_opt - b) / max(norm(b), eps(real(T))))) : tt_opt
end

# Implementation of `eigen_solve(A, tt_start, ::MALS)`; see [`MALS`](@ref).
function _mals_eigsolve_impl(
        A::AbstractTTOperator, tt_start::AbstractTTVector;
        max_sweeps::Vector{Int}, max_bond::Vector{Int}, trunc_tol::Real,
        local_solver::Symbol, local_threshold::Int, local_maxiter::Int, local_tol::Real,
        verbosity::Int, show_progress::Bool
    )
    T = eltype(tt_start)
    d = nsites(A)
    tt_opt = orthogonalize(tt_start)
    dims = tt_start.ttv_dims
    E = Float64[]
    r_hist = Int[]

    G = Array{Array{T, 5}}(undef, d)
    rmax = maximum(max_bond)
    for i in 1:d
        rmax_i = min(rmax, prod(dims[1:(i - 1)]), prod(dims[i:end]))
        G[i] = zeros(dims[i], rmax_i, dims[i], rmax_i, A.tto_rks[i + 1])
    end
    G[1][:, 1:1, :, 1:1, :] = reshape(A.tto_vec[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)
    H = init_H_mals(tt_opt, A, rmax)

    local_opts = (; local_solver, local_threshold, local_maxiter, local_tol)
    progress = _solver_progress(sum(max_sweeps), show_progress; desc = "MALS eigen solve")
    sweep = 0
    trunc_err = Ref(0.0)
    for (stage, nsweeps) in enumerate(max_sweeps), _ in 1:nsweeps
        sweep += 1
        trunc_err[] = 0.0
        for i in 1:(d - 1)
            λ, V = K_eigmin_mals(G[i], H[i], tt_opt.ttv_vec[i], tt_opt.ttv_vec[i + 1]; local_opts...)
            push!(E, λ)
            tt_opt = right_core_move_mals(tt_opt, i, V, trunc_tol, max_bond[stage]; trunc_err)
            push!(r_hist, maximum(tt_opt.ttv_rks))

            Gi = @view(G[i][:, 1:tt_opt.ttv_rks[i], :, 1:tt_opt.ttv_rks[i], :])
            Gip = @view(G[i + 1][:, 1:tt_opt.ttv_rks[i + 1], :, 1:tt_opt.ttv_rks[i + 1], :])
            update_G!(tt_opt.ttv_vec[i], A.tto_vec[i + 1], Gi, Gip)
        end

        for i in (d - 1):-1:1
            λ, V = K_eigmin_mals(G[i], H[i], tt_opt.ttv_vec[i], tt_opt.ttv_vec[i + 1]; local_opts...)
            push!(E, λ)
            tt_opt = left_core_move_mals(tt_opt, i, V, trunc_tol, max_bond[stage]; trunc_err)
            push!(r_hist, maximum(tt_opt.ttv_rks))

            if i > 1
                Hi = @view(H[i][:, :, 1:tt_opt.ttv_rks[i + 2], :, 1:tt_opt.ttv_rks[i + 2]])
                Him = @view(H[i - 1][:, :, 1:tt_opt.ttv_rks[i + 1], :, 1:tt_opt.ttv_rks[i + 1]])
                updateH_mals!(tt_opt.ttv_vec[i + 1], A.tto_vec[i], Hi, Him)
            end
        end
        eigenvalue = E[end]
        verbosity ≥ 2 && @info "MALS eigen solve" sweep max_rank = maximum(tt_opt.ttv_rks) eigenvalue truncation_error = trunc_err[]
        next!(progress; showvalues = [("sweep", "$sweep/$(sum(max_sweeps))"), ("eigenvalue", eigenvalue), ("truncation error", trunc_err[])])
    end
    return E, tt_opt, r_hist
end

function eigen_solve(A::AbstractTTOperator, guess::AbstractTTVector, alg::MALS)
    _reject_unused(alg, "eigen_solve", (:return_info,), "every option except `return_info`")
    st = _stages(;
        max_sweeps = alg.max_sweeps,
        max_bond = something(alg.max_bond, round(Int, sqrt(prod(guess.ttv_dims)::Int)))
    )
    return _mals_eigsolve_impl(
        A, guess; st...,
        trunc_tol = alg.trunc_tol,
        local_solver = alg.local_solver,
        local_threshold = something(alg.local_threshold, 256),
        local_maxiter = alg.local_maxiter,
        local_tol = alg.local_tol,
        verbosity = alg.verbosity,
        show_progress = alg.show_progress
    )
end
