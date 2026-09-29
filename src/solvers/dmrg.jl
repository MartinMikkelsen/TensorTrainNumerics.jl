using LinearMaps
using TensorOperations
using KrylovKit

"""
Implementation based on the presentation in 
Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider. "The alternating linear scheme for tensor optimization in the tensor train format." SIAM Journal on Scientific Computing 34.2 (2012): A683-A713.
"""

function init_H(x_tt::AbstractTTvector, A_tto::AbstractTToperator, N::Int, rmax)
    T = eltype(x_tt)
    d = x_tt.N
    H = Array{Array{T, 3}, 1}(undef, d + 1 - N)
    H[d + 1 - N] = ones(T, 1, 1, 1)
    rks = r_and_d_to_rks(vcat(1, rmax * ones(Int, d - 1), 1), x_tt.ttv_dims; rmax = rmax)
    for i in (d + 1 - N):-1:2
        H[i - 1] = zeros(T, A_tto.tto_rks[i + N - 1], rks[i + N - 1], rks[i + N - 1])
        Hi_view = @view(H[i][:, 1:x_tt.ttv_rks[i + N], 1:x_tt.ttv_rks[i + N]])
        Him = @view(H[i - 1][:, 1:x_tt.ttv_rks[i + N - 1], 1:x_tt.ttv_rks[i + N - 1]])
        x_vec = x_tt.ttv_vec[i + N - 1]
        A_vec = A_tto.tto_vec[i + N - 1]
        update_H!(x_vec, A_vec, Hi_view, Him)
    end
    return H
end

function update_H!(x_vec::Array{T, 3}, A_vec::Array{T, 4}, Hi::AbstractArray{<:Any, 3}, Him::AbstractArray{<:Any, 3}) where {T <: Number}
    @tensoropt((ϕ, χ), Him[a, α, β] = conj.(x_vec)[j, α, ϕ] * Hi[z, ϕ, χ] * x_vec[k, β, χ] * A_vec[j, k, a, z]) #size (rAim, rim, rim)
    return nothing
end

function update_G!(x_vec::Array{T, 3}, A_vec::Array{T, 4}, Gi::AbstractArray{<:Any, 3}, Gip::AbstractArray{<:Any, 3}) where {T <: Number}
    @tensoropt((ϕ, χ), Gip[a, α, β] = conj.(x_vec)[j, ϕ, α] * Gi[z, ϕ, χ] * x_vec[k, χ, β] * A_vec[j, k, z, a]) #size (rAi, ri, ri)
    return nothing
end

#returns the contracted tensor A_i[\\mu_i] ⋯ A_j[\\mu_j] ∈ R^{R^A_{i-1} × n_i × n_i × ⋯ × n_j × n_j ×  R^A_j}
function Amid(A_tto::AbstractTToperator, i::Int, j::Int)
    A = permutedims(A_tto.tto_vec[i], (3, 1, 2, 4))
    for k in (i + 1):j
        C = reshape(A, A_tto.tto_rks[i], prod(A_tto.tto_dims[i:(k - 1)]), :, A_tto.tto_rks[k])
        @tensor Atemp[αk, Ik, ik, Jk, jk, βk] := A_tto.tto_vec[k][ik, jk, ξk, βk] * C[αk, Ik, Jk, ξk]
        A = reshape(Atemp, A_tto.tto_rks[i], prod(A_tto.tto_dims[i:k]), :, A_tto.tto_rks[k + 1])
    end
    return A #size R^A_{i-1} × (n_i⋯n_j) × (n_i⋯n_j) × R^A_j
end

#full assemble of matrix K
function K_full(Gi::AbstractArray{T, 3}, Hi::AbstractArray{T, 3}, Amid_tensor::AbstractArray{T, 4}) where {T <: Number}
    K_dims = (size(Gi, 2), size(Amid_tensor, 2), size(Hi, 2))
    K = zeros(T, K_dims..., K_dims...)
    @tensoropt((a, c, d, f), K[a, b, c, d, e, f] = Gi[y, a, d] * Hi[z, c, f] * Amid_tensor[y, b, e, z]) #size (r^X_{i-1},n_i⋯n_j,r^X_j)
    return reshape(K, prod(K_dims), prod(K_dims))
end

function init_Hb(x_tt::AbstractTTvector, b_tt::AbstractTTvector, N::Integer, rmax)
    T = eltype(x_tt)
    d = x_tt.N
    H_b = Array{Array{T, 2}, 1}(undef, d + 1 - N)
    H_b[d + 1 - N] = ones(T, 1, 1)
    rks = r_and_d_to_rks(vcat(1, rmax * ones(Int, d - 1), 1), x_tt.ttv_dims; rmax = rmax)
    for i in (d + 1 - N):-1:2
        H_b[i - 1] = zeros(T, rks[i + N - 1], b_tt.ttv_rks[i + N - 1])
        b_vec = b_tt.ttv_vec[i + N - 1]
        x_vec = x_tt.ttv_vec[i + N - 1]
        Hbi = @view(H_b[i][1:x_tt.ttv_rks[i + N], :])
        Hbim = @view(H_b[i - 1][1:x_tt.ttv_rks[i + N - 1], :])
        update_Hb!(x_vec, b_vec, Hbi, Hbim) #size(r^X_{i-1},r^b_{i-1})
    end
    return H_b
end

function update_Hb!(x_vec::Array{T, 3}, b_vec::Array{T, 3}, H_bi::AbstractArray{T, 2}, H_bim::AbstractArray{T, 2}) where {T <: Number}
    @tensoropt((ϕ, χ), H_bim[α, β] = H_bi[ϕ, χ] * b_vec[i, β, χ] * conj.(x_vec)[i, α, ϕ])
    return nothing
end

function update_Gb!(x_vec::Array{T, 3}, b_vec::Array{T, 3}, G_bi::AbstractArray{T, 2}, G_bip::AbstractArray{T, 2}) where {T <: Number}
    @tensoropt((ϕ, χ), G_bip[α, β] = G_bi[ϕ, χ] * b_vec[i, χ, β] * conj.(x_vec)[i, ϕ, α])
    return nothing
end

function b_mid(b_tt::AbstractTTvector, i::Integer, j::Integer)
    b_out = permutedims(b_tt.ttv_vec[i], (2, 1, 3))
    for k in (i + 1):j
        @tensor btemp[αk, ik, jk, βk] := b_out[αk, ik, ξk] * b_tt.ttv_vec[k][jk, ξk, βk]
        b_out = reshape(btemp, b_tt.ttv_rks[i], :, b_tt.ttv_rks[k + 1]) #size r^b_{i-1} × (n_i⋯n_k) × r^b_k
    end
    return b_out
end

function Ksolve!(Gi_view::AbstractArray{T, 3}, G_bi::AbstractArray{T, 2}, Hi_view::AbstractArray{T, 3}, H_bi::AbstractArray{T, 2}, Amid_tensor::AbstractArray{T, 4}, Bmid::AbstractArray{T, 3}, Pb, V0::AbstractArray{T, 3}, Vapp::AbstractArray{T, 3}; local_solver::Symbol = :iterative, local_threshold::Int = 256, local_maxiter::Int = 200, local_tol::Real = 1.0e-6) where {T <: Number}
    K_dims = (size(Gi_view, 2), size(Amid_tensor, 2), size(Hi_view, 2))
    @tensoropt Pb[α1, i, α2] = G_bi[α1, β1] * Bmid[β1, i, β2] * H_bi[α2, β2] #size (r^X_{i-1},n_i⋯n_j,r^X_j)

    if _use_iterative(local_solver, prod(K_dims), local_threshold)
        function K_matfree(Vout, V)
            Hrshp = reshape(Vout, K_dims)
            @tensoropt((a, c, d, f), Hrshp[a, b, c] = Gi_view[y, a, d] * Amid_tensor[y, b, e, z] * reshape(V, K_dims)[d, e, f] * Hi_view[z, c, f])
            return nothing
        end
        Vapp[:], _ = linsolve(
            LinearMap{T}(K_matfree, prod(K_dims); ismutating = true),
            Pb[:], V0[:], KrylovKit.GMRES(; tol = local_tol, maxiter = local_maxiter)
        )
        return nothing
    else
        K = K_full(Gi_view, Hi_view, Amid_tensor)
        Vapp[:] = K \ Pb[:]
        return nothing
    end
end

function right_core_move!(x_tt::AbstractTTvector, V, V_move, i::Int, trunc_tol::Real, r_max::Integer; verbose::Bool = false, trunc_err = nothing)
    u_V, s_V, v_V = svd(reshape(V, x_tt.ttv_rks[i] * x_tt.ttv_dims[i], :))
    x_tt.ttv_rks[i + 1] = _trunc_rank(s_V, trunc_tol, x_tt.N, r_max)
    δ = _discarded_weight(s_V, x_tt.ttv_rks[i + 1])
    isnothing(trunc_err) || (trunc_err[] = max(trunc_err[], δ))
    verbose && @info "DMRG core move" bond = i + 1 rank = x_tt.ttv_rks[i + 1] max_rank = r_max truncation_error = δ

    x_tt.ttv_vec[i] = permutedims(reshape(u_V[:, 1:x_tt.ttv_rks[i + 1]], x_tt.ttv_rks[i], x_tt.ttv_dims[i], :), (2, 1, 3))
    x_tt.ttv_ot[i] = 1
    x_tt.ttv_ot[i + 1] = 0
    mid_size = div(size(v_V, 1), size(V, 3))  # = dim[i+1] for N≥2, = 1 for N=1
    V_moveview = @view(V_move[1:x_tt.ttv_rks[i + 1], 1:mid_size, 1:size(V, 3)])
    @tensor V_moveview[αk, ik, βk] = reshape(v_V'[1:x_tt.ttv_rks[i + 1], :], x_tt.ttv_rks[i + 1], :, size(V, 3))[αk, ik, βk]
    for ak in axes(V_moveview, 1)
        V_moveview[ak, :, :] = V_moveview[ak, :, :] * (s_V[ak])
    end
    #	return x_tt, reshape(Diagonal(s_V[1:x_tt.ttv_rks[i+1]])*v_V'[1:x_tt.ttv_rks[i+1],:],x_tt.ttv_rks[i+1],:,size(V,3))
    return nothing
end

function left_core_move!(x_tt::AbstractTTvector, V, V_move, j::Int, trunc_tol::Real, r_max::Integer; verbose::Bool = false, trunc_err = nothing)
    u_V, s_V, v_V = svd(reshape(V, :, x_tt.ttv_dims[j] * x_tt.ttv_rks[j + 1]))
    x_tt.ttv_rks[j] = _trunc_rank(s_V, trunc_tol, x_tt.N, r_max)
    δ = _discarded_weight(s_V, x_tt.ttv_rks[j])
    isnothing(trunc_err) || (trunc_err[] = max(trunc_err[], δ))
    verbose && @info "DMRG core move" bond = j rank = x_tt.ttv_rks[j] max_rank = r_max truncation_error = δ

    x_tt.ttv_vec[j] = permutedims(reshape(v_V'[1:x_tt.ttv_rks[j], :], x_tt.ttv_rks[j], :, x_tt.ttv_rks[j + 1]), (2, 1, 3))
    x_tt.ttv_ot[j] = -1
    x_tt.ttv_ot[j - 1] = 0
    mid_size = div(size(u_V, 1), size(V, 1))  # = dim[j-1] for N≥2, = 1 for N=1
    V_moveview = @view(V_move[1:size(V, 1), 1:mid_size, 1:x_tt.ttv_rks[j]])
    @tensor V_moveview[αk, ik, βk] = reshape(u_V[:, 1:x_tt.ttv_rks[j]], size(V, 1), :, x_tt.ttv_rks[j])[αk, ik, βk]
    for bk in axes(V_moveview, 3)
        V_moveview[:, :, bk] = V_moveview[:, :, bk] * s_V[bk]
    end
    return nothing
end


function K_eigmin(Gi_view::AbstractArray{<:Any, 3}, Hi_view::AbstractArray{<:Any, 3}, V0::AbstractArray{<:Any, 3}, Amid_tensor::AbstractArray{<:Any, 4}, V; local_solver::Symbol = :iterative, local_threshold::Int = 256, local_maxiter::Int = 200, local_tol::Real = 1.0e-6)
    T = promote_type(eltype(Gi_view), eltype(Hi_view), eltype(V0), eltype(Amid_tensor))
    K_dims = size(V0)
    λ = zero(T)
    if _use_iterative(local_solver, prod(K_dims), local_threshold)
        function K_matfree(Vout, V::AbstractArray{S, 1}; Gi = Gi_view::AbstractArray{S, 3}, Hi = Hi_view::AbstractArray{S, 3}, K_dims = K_dims::NTuple{3, Int}, Amid_tensor = Amid_tensor::AbstractArray{S, 4}) where {S <: Number}
            Hrshp = reshape(Vout, K_dims)
            @tensoropt((a, c, d, f), Hrshp[a, b, c] = Gi[y, a, d] * Amid_tensor[y, b, e, z] * reshape(V, K_dims)[d, e, f] * Hi[z, c, f])
            return nothing
        end
        r = eigsolve(LinearMap{T}(K_matfree, prod(K_dims); ishermitian = true, ismutating = true), copy(V0[:]), 1, :SR; ishermitian = true, tol = local_tol, maxiter = local_maxiter)
        for i in eachindex(V)
            V[i] = reshape(r[2][1], K_dims)[i]
        end
        λ = real(r[1][1])
    else
        K = K_full(Gi_view, Hi_view, Amid_tensor)
        F = eigen(Hermitian(K), 1:1)
        for i in eachindex(V)
            V[i] = reshape(F.vectors[:, 1], K_dims)[i]
        end
        λ = F.values[1]
    end
    return λ
end

function init_dmrg(A::AbstractTToperator, tt_opt::AbstractTTvector, rks, N::Integer)
    T = eltype(tt_opt)
    d = tt_opt.N
    rmax = maximum(rks)
    G = Array{Array{T, 3}, 1}(undef, d + 1 - N)
    Amid_list = Array{Array{T, 4}, 1}(undef, d + 1 - N)
    for i in 1:(d + 1 - N)
        G[i] = zeros(T, A.tto_rks[i], rks[i], rks[i])
        Amid_list[i] = Amid(A, i, i + N - 1)
    end
    G[1] = ones(T, size(G[1]))
    H = init_H(tt_opt, A, N, rmax)

    V0 = zeros(T, rmax, maximum(tt_opt.ttv_dims)^N, rmax)
    V0_view = @view(V0[1:tt_opt.ttv_rks[1], 1:prod(tt_opt.ttv_dims[1:N]), 1:tt_opt.ttv_rks[1 + N]])
    V0_view = b_mid(tt_opt, 1, N)
    V = zeros(T, rmax, maximum(tt_opt.ttv_dims)^N, rmax)
    V_move = zeros(T, rmax, maximum(tt_opt.ttv_dims), rmax)
    V_temp = zeros(T, rmax, maximum(tt_opt.ttv_dims), maximum(tt_opt.ttv_dims), rmax)
    return G, Amid_list, H, V0, V, V_move, V_temp, V0_view
end

function init_dmrg_b(b::AbstractTTvector, tt_opt::AbstractTTvector, rks, N)
    T = eltype(b)
    d = b.N
    rmax = maximum(rks)
    G_b = zeros.(T, rks[1:(d + 1 - N)], b.ttv_rks[1:(d + 1 - N)]) #Array{Array{T,2},1}(undef, d+1-N)
    bmid_list = Array{Array{T, 3}, 1}(undef, d + 1 - N)
    for i in 1:(d + 1 - N)
        bmid_list[i] = b_mid(b, i, i + N - 1)
    end
    G_b[1] = ones(T, size(G_b[1]))
    Pb_temp = zeros(T, rmax, maximum(tt_opt.ttv_dims)^N, rmax)
    H_b = init_Hb(tt_opt, b, N, rmax)
    return G_b, bmid_list, H_b, Pb_temp
end

function update_G_H_V(Gi, Hi, V, tt_dims, tt_rks, i, N)
    Gi_view = @view(Gi[:, 1:tt_rks[i], 1:tt_rks[i]])
    Hi_view = @view(Hi[:, 1:tt_rks[i + N], 1:tt_rks[i + N]])
    V_view = @view(V[1:tt_rks[i], 1:prod(tt_dims[i:(i + N - 1)]), 1:tt_rks[i + N]])
    return Gi_view, Hi_view, V_view
end

function update_G_H_V_b(Gbi, Hbi, Pb_temp, tt_dims, tt_rks, i, N)
    G_bi_view = @view(Gbi[1:tt_rks[i], :])
    H_bi_view = @view(Hbi[1:tt_rks[i + N], :])
    Pb_view = @view(Pb_temp[1:tt_rks[i], 1:prod(tt_dims[i:(i + N - 1)]), 1:tt_rks[i + N]])
    return G_bi_view, H_bi_view, Pb_view
end

function update_right(tt_opt, V0, V_view, V_move, V_temp, i, N, trunc_tol, rmax, Ai, Gi_view, Gip; verbose::Bool = false, trunc_err = nothing)
    right_core_move!(tt_opt, V_view, V_move, i, trunc_tol, rmax; verbose, trunc_err)

    V_moveview = @view(V_move[1:tt_opt.ttv_rks[i + 1], 1:prod(tt_opt.ttv_dims[(i + 1):(i + N - 1)]), 1:tt_opt.ttv_rks[i + N]])
    V_tempview = @view(V_temp[1:size(V_moveview, 1), 1:size(V_moveview, 2), 1:tt_opt.ttv_dims[i + N], 1:tt_opt.ttv_rks[i + 1 + N]])
    @tensor V_tempview[αk, J, ik, γk] = V_moveview[αk, J, βk] * tt_opt.ttv_vec[i + N][ik, βk, γk]
    V0_view = @view(V0[1:tt_opt.ttv_rks[i + 1], 1:prod(tt_opt.ttv_dims[(i + 1):(i + N - 1)]), 1:tt_opt.ttv_rks[i + N]])
    V0_view = reshape(V_tempview, size(V_tempview, 1), :, size(V_tempview, 4))

    #update G[i+1]
    Gip_view = @view(Gip[:, 1:tt_opt.ttv_rks[i + 1], 1:tt_opt.ttv_rks[i + 1]])
    update_G!(tt_opt.ttv_vec[i], Ai, Gi_view, Gip_view)
    return V0_view
end

function update_left(tt_opt, V0, V_view, V_move, V_temp, i, N, trunc_tol, rmax, Aip, Hi_view, Him; verbose::Bool = false, trunc_err = nothing)
    left_core_move!(tt_opt, V_view, V_move, i + N - 1, trunc_tol, rmax; verbose, trunc_err)

    #update the initialization
    V_moveview = @view(V_move[1:tt_opt.ttv_rks[i], 1:prod(tt_opt.ttv_dims[i:(i + N - 2)]), 1:tt_opt.ttv_rks[i + N - 1]])
    V_tempview = @view(V_temp[1:tt_opt.ttv_rks[i - 1], 1:size(V_moveview, 2), 1:tt_opt.ttv_dims[i - 1], 1:size(V_moveview, 3)])
    @tensor V_tempview[αk, J, ik, γk] = V_moveview[βk, J, γk] * tt_opt.ttv_vec[i - 1][ik, αk, βk]
    V0_view = @view(V0[1:tt_opt.ttv_rks[i], 1:prod(tt_opt.ttv_dims[i:(i + N - 2)]), 1:tt_opt.ttv_rks[i + N - 1]])
    V0_view = reshape(V_tempview, size(V_tempview, 1), :, size(V_tempview, 4))

    #update H[i-1]
    Him_view = @view(Him[:, 1:tt_opt.ttv_rks[i + N - 1], 1:tt_opt.ttv_rks[i + N - 1]])
    update_H!(tt_opt.ttv_vec[i + N - 1], Aip, Hi_view, Him_view)
    return V0_view
end

#function K_eiggenmin(Gi,Hi,Ki,Li,ttv_vec;it_solver=false,itslv_thresh=2500)
#	@tensor begin
#		K[a,b,c,d,e,f] := Gi[d,e,a,b,z]*Hi[z,f,c] #size (ni,rim,ri,ni,rim,ri)
#		S[a,b,c,d,e,f] := Ki[d,e,a,b,z]*Li[z,f,c] #size (ni,rim,ri,ni,rim,ri)
#	end
#	if it_solver || prod(size(K)[1:3]) > itslv_thresh
#		r = lobpcg(reshape(K,prod(size(K)[1:3]),:),reshape(S,prod(size(S)[1:3]),:),false,ttv_vec[:],1;maxiter=500,tol=1e-8)
#		return r.λ[1], reshape(r.X[:,1],size(K)[1:3])
#	else
#		F = eigen(reshape(K,prod(size(K)[1:3]),:),reshape(S,prod(size(K)[1:3]),:),)
#		return real(F.values[1]),reshape(F.vectors[:,1],size(K)[1:3])
#	end
#end

# Lines shown under the DMRG progress bar; one-site DMRG does not truncate.
function _dmrg_progress_values(sweep, total, quantity::Pair, nsites, trunc_err)
    values = Any[("sweep", "$sweep/$total"), (first(quantity), last(quantity))]
    nsites ≥ 2 && push!(values, ("truncation error", trunc_err))
    return values
end

# Micro-step on the first `nsites` cores after the last sweep; it leaves the
# orthogonality center on core 1.
function _dmrg_final_core!(tt_opt, V, V_view, V_move, nsites, trunc_tol, max_bond; verbose)
    if nsites == 1
        tt_opt.ttv_vec[1] = permutedims(copy(V_view), (2, 1, 3))
    else
        for i in nsites:-1:2
            V_view = @view(V[1:tt_opt.ttv_rks[i - nsites + 1], 1:prod(tt_opt.ttv_dims[(i - nsites + 1):i]), 1:tt_opt.ttv_rks[i + 1]])
            left_core_move!(tt_opt, V_view, V_move, i, trunc_tol, max_bond; verbose)
        end
        V_moveview = @view(V_move[1:tt_opt.ttv_rks[1], 1:prod(tt_opt.ttv_dims[1:(nsites - 1)]), 1:tt_opt.ttv_rks[nsites]])
        tt_opt.ttv_vec[1] = permutedims(reshape(V_moveview, 1, tt_opt.ttv_dims[1], :), (2, 1, 3))
    end
    tt_opt.ttv_ot[1] = 0
    return tt_opt
end

# Implementation of `linear_solve(A, b, tt_start, ::DMRG)`; see [`DMRG`](@ref).
function _dmrg_linsolve_impl(
        A::AbstractTToperator, b::AbstractTTvector, tt_start::AbstractTTvector;
        nsites::Int, max_sweeps::Vector{Int}, max_bond::Vector{Int}, trunc_tol::Real,
        local_solver::Symbol, local_threshold::Int, local_maxiter::Int, local_tol::Real,
        return_info::Bool, verbosity::Int, show_progress::Bool
    )
    T = eltype(tt_start)
    d = b.N
    rmax = maximum(max_bond)
    if nsites == 1
        tt_start = increase_ranks(tt_start, rmax)
    end
    tt_opt = orthogonalize(tt_start)
    dims = tt_start.ttv_dims
    rks = r_and_d_to_rks(vcat(1, rmax * ones(Int, d - 1), 1), dims; rmax = rmax)

    G, Amid_list, H, V0, V, V_move, V_temp, V0_view = init_dmrg(A, tt_opt, rks, nsites)
    G_b, bmid_list, H_b, Pb_temp = init_dmrg_b(b, tt_opt, rks, nsites)

    local_opts = (; local_solver, local_threshold, local_maxiter, local_tol)
    verbose = verbosity ≥ 3
    progress = _solver_progress(sum(max_sweeps), show_progress; desc = "DMRG linear solve")
    sweep = 0
    trunc_err = Ref(0.0)
    for (stage, nsweeps) in enumerate(max_sweeps), _ in 1:nsweeps
        sweep += 1
        trunc_err[] = 0.0
        for i in 1:(d - nsites)
            Gi_view, Hi_view, V_view = update_G_H_V(G[i], H[i], V, tt_opt.ttv_dims, tt_opt.ttv_rks, i, nsites)
            G_bi_view, H_bi_view, Pb_view = update_G_H_V_b(G_b[i], H_b[i], Pb_temp, tt_opt.ttv_dims, tt_opt.ttv_rks, i, nsites)
            Ksolve!(Gi_view, G_bi_view, Hi_view, H_bi_view, Amid_list[i], bmid_list[i], Pb_view, V0_view, V_view; local_opts...)
            V0_view = update_right(tt_opt, V0, V_view, V_move, V_temp, i, nsites, trunc_tol, max_bond[stage], A.tto_vec[i], Gi_view, G[i + 1]; verbose, trunc_err)
            G_bip = @view(G_b[i + 1][1:tt_opt.ttv_rks[i + 1], :])
            update_Gb!(tt_opt.ttv_vec[i], b.ttv_vec[i], G_bi_view, G_bip)
        end
        for i in (d + 1 - nsites):(-1):2
            Gi_view, Hi_view, V_view = update_G_H_V(G[i], H[i], V, tt_opt.ttv_dims, tt_opt.ttv_rks, i, nsites)
            G_bi_view, H_bi_view, Pb_view = update_G_H_V_b(G_b[i], H_b[i], Pb_temp, tt_opt.ttv_dims, tt_opt.ttv_rks, i, nsites)
            Ksolve!(Gi_view, G_bi_view, Hi_view, H_bi_view, Amid_list[i], bmid_list[i], Pb_view, V0_view, V_view; local_opts...)
            V0_view = update_left(tt_opt, V0, V_view, V_move, V_temp, i, nsites, trunc_tol, max_bond[stage], A.tto_vec[i + nsites - 1], Hi_view, H[i - 1]; verbose, trunc_err)
            H_bim = @view(H_b[i - 1][1:tt_opt.ttv_rks[i + nsites - 1], :])
            update_Hb!(tt_opt.ttv_vec[i + nsites - 1], b.ttv_vec[i + nsites - 1], H_bi_view, H_bim)
        end
        max_rank = maximum(tt_opt.ttv_rks)
        verbosity ≥ 2 && @info "DMRG linear solve" sweep max_rank truncation_error = trunc_err[]
        next!(progress; showvalues = _dmrg_progress_values(sweep, sum(max_sweeps), "largest rank" => max_rank, nsites, trunc_err[]))
    end
    Gi_view, Hi_view, V_view = update_G_H_V(G[1], H[1], V, tt_opt.ttv_dims, tt_opt.ttv_rks, 1, nsites)
    G_bi_view, H_bi_view, Pb_view = update_G_H_V_b(G_b[1], H_b[1], Pb_temp, tt_opt.ttv_dims, tt_opt.ttv_rks, 1, nsites)
    Ksolve!(Gi_view, G_bi_view, Hi_view, H_bi_view, Amid_list[1], bmid_list[1], Pb_view, V0_view, V_view; local_opts...)
    _dmrg_final_core!(tt_opt, V, V_view, V_move, nsites, trunc_tol, max_bond[end]; verbose)
    finish!(progress)
    return return_info ? (tt_opt, (; residual = norm(A * tt_opt - b) / max(norm(b), eps(real(T))))) : tt_opt
end

# Implementation of `eigen_solve(A, tt_start, ::DMRG)`; see [`DMRG`](@ref).
function _dmrg_eigsolve_impl(
        A::AbstractTToperator, tt_start::AbstractTTvector;
        nsites::Int, max_sweeps::Vector{Int}, max_bond::Vector{Int}, trunc_tol::Real,
        local_solver::Symbol, local_threshold::Int, local_maxiter::Int, local_tol::Real,
        verbosity::Int, show_progress::Bool
    )
    d = tt_start.N
    tt_opt = orthogonalize(tt_start)
    dims = tt_start.ttv_dims
    rmax = maximum(max_bond)
    rks = r_and_d_to_rks(vcat(1, rmax * ones(Int, d - 1), 1), dims; rmax = rmax)
    E = Float64[]
    r_hist = Int64[]
    G, Amid_list, H, V0, V, V_move, V_temp, V0_view = init_dmrg(A, tt_opt, rks, nsites)

    local_opts = (; local_solver, local_threshold, local_maxiter, local_tol)
    verbose = verbosity ≥ 3
    progress = _solver_progress(sum(max_sweeps), show_progress; desc = "DMRG eigen solve")
    sweep = 0
    trunc_err = Ref(0.0)
    for (stage, nsweeps) in enumerate(max_sweeps), _ in 1:nsweeps
        sweep += 1
        trunc_err[] = 0.0
        for i in 1:(d - nsites)
            Gi_view, Hi_view, V_view = update_G_H_V(G[i], H[i], V, tt_opt.ttv_dims, tt_opt.ttv_rks, i, nsites)
            λ = K_eigmin(Gi_view, Hi_view, V0_view, Amid_list[i], V_view; local_opts...)
            push!(E, λ)
            V0_view = update_right(tt_opt, V0, V_view, V_move, V_temp, i, nsites, trunc_tol, max_bond[stage], A.tto_vec[i], Gi_view, G[i + 1]; verbose, trunc_err)
            push!(r_hist, maximum(tt_opt.ttv_rks))
        end
        for i in (d - nsites + 1):(-1):2
            Gi_view, Hi_view, V_view = update_G_H_V(G[i], H[i], V, tt_opt.ttv_dims, tt_opt.ttv_rks, i, nsites)
            λ = K_eigmin(Gi_view, Hi_view, V0_view, Amid_list[i], V_view; local_opts...)
            push!(E, λ)
            V0_view = update_left(tt_opt, V0, V_view, V_move, V_temp, i, nsites, trunc_tol, max_bond[stage], A.tto_vec[i + nsites - 1], Hi_view, H[i - 1]; verbose, trunc_err)
            push!(r_hist, maximum(tt_opt.ttv_rks))
        end
        # With nsites = d a sweep has no micro-steps; the final micro-step computes E.
        eigenvalue = isempty(E) ? NaN : E[end]
        verbosity ≥ 2 && @info "DMRG eigen solve" sweep max_rank = maximum(tt_opt.ttv_rks) eigenvalue truncation_error = trunc_err[]
        next!(progress; showvalues = _dmrg_progress_values(sweep, sum(max_sweeps), "eigenvalue" => eigenvalue, nsites, trunc_err[]))
    end
    Gi_view, Hi_view, V_view = update_G_H_V(G[1], H[1], V, tt_opt.ttv_dims, tt_opt.ttv_rks, 1, nsites)
    λ = K_eigmin(Gi_view, Hi_view, V0_view, Amid_list[1], V_view; local_opts...)
    push!(E, λ)
    push!(r_hist, maximum(tt_opt.ttv_rks))
    _dmrg_final_core!(tt_opt, V, V_view, V_move, nsites, trunc_tol, max_bond[end]; verbose)
    finish!(progress)
    return E, tt_opt, r_hist
end

function eigen_solve(A::AbstractTToperator, guess::AbstractTTvector, alg::DMRG)
    _reject_unused(alg, "eigen_solve", (:return_info,), "every option except `return_info`")
    return _dmrg_eigsolve_impl(A, guess; _dmrg_options(alg, guess)...)
end
