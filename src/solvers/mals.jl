using IterativeSolvers
using LinearMaps
using LinearAlgebra

"""
Implementation based on the presentation in 
Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider. "The alternating linear scheme for tensor optimization in the tensor train format." SIAM Journal on Scientific Computing 34.2 (2012): A683-A713.
"""

function updateH_mals!(x_vec::Array{T,3}, A_vec::Array{T,4}, Hi::AbstractArray{T,5}, Him::AbstractArray{T,5}) where T<:Number
	@tensor(Him[a,i,α,l,β] = conj.(x_vec)[j,α,x] * (Hi[z,j,x,k,y] * x_vec[k,β,y]) * A_vec[i,l,a,z])
	nothing
end

function init_H_mals(x_tt::TTvector{T}, A::TToperator{T}, rmax::Int) where {T<:Number}
	d = x_tt.N
	H = Array{Array{T,5}}(undef, d-1)
	# H[d-1] from the last operator core
	H[d-1] = reshape(permutedims(A.tto_vec[d], [3,1,2,4]),
	                 :, x_tt.ttv_dims[d], 1, x_tt.ttv_dims[d], 1)
	for i = (d-1):-1:2
		rmax_i = min(rmax, prod(x_tt.ttv_dims[1:i]), prod(x_tt.ttv_dims[i+1:end]))
		H[i-1] = zeros(T,
		              A.tto_rks[i],
		              x_tt.ttv_dims[i],
		              rmax_i,
		              x_tt.ttv_dims[i],
		              rmax_i)
		# view the “active” slices of H[i] and H[i-1]
		Hi  = @view(H[i][:, :, 1:x_tt.ttv_rks[i+2], :, 1:x_tt.ttv_rks[i+2]])
		Him = @view(H[i-1][:, :, 1:x_tt.ttv_rks[i+1], :, 1:x_tt.ttv_rks[i+1]])
		updateH_mals!(x_tt.ttv_vec[i+1], A.tto_vec[i], Hi, Him)
	end
	return H
end

function sv_trunc(s::Array{Float64}, tol::Float64)
	if tol == 0.0
		return s
	else
		d = length(s)
		i = 0
		weight = 0.0
		norm2 = sum(abs2, s)
		while (i < d) && (weight < tol * norm2)
			weight += s[d-i]^2
			i += 1
		end
		return s[1:(d - i + 1)]
	end
end

# —— Corrected updateHb_mals! ——

function updateHb_mals!(xtt_vec::Array{T,3}, btt_vec::Array{T,3},
                       Hbi::AbstractArray{T,3}, Hbim::AbstractArray{T,3}) where T<:Number
	@tensor Hbim[β, i, χ] = conj.(xtt_vec)[j, χ, a] * Hbi[γ, j, a] * btt_vec[i, β, γ]
	nothing
end

function init_Hb_mals(x_tt::TTvector{T}, b::TTvector{T}, rmax::Int) where {T<:Number}
	d = x_tt.N
	Hb = Array{Array{T,3}}(undef, d-1)
	# Base case: H_b[d-1] from the last vector core
	Hb[d-1] = reshape(permutedims(b.ttv_vec[d], [2,1,3]),
	                  b.ttv_rks[d], b.ttv_dims[d], 1)
	for i = (d-1):-1:2
		rmax_i = min(rmax, prod(x_tt.ttv_dims[1:i]), prod(x_tt.ttv_dims[i+1:end]))
		# Allocate H_b[i-1] as (b_rks[i], n_i, rmax_i)
		Hb[i-1] = zeros(T,
		               b.ttv_rks[i],
		               b.ttv_dims[i],
		               rmax_i)
		# view active slices of H_b[i] and H_b[i-1]
		Hbi  = @view(Hb[i][:, :, 1:x_tt.ttv_rks[i+2]])
		Hbim = @view(Hb[i-1][:, :, 1:x_tt.ttv_rks[i+1]])
		updateHb_mals!(x_tt.ttv_vec[i+1], b.ttv_vec[i], Hbi, Hbim)
	end
	return Hb
end

function left_core_move_mals(xtt::TTvector{T}, i::Integer, V::Array{T,4},
                             tol::Float64, rmax::Integer) where {T<:Number}
	# Perform the truncated SVD on the 2-core block V
	u_V, s_V, v_V = svd(reshape(V, prod(size(V)[1:2]), :))
	s_trunc = sv_trunc(s_V, tol)
	xtt.ttv_rks[i+1] = min(length(s_trunc), rmax)

	# Update the (i+1)-th core from truncated V-matrix
	xtt.ttv_vec[i+1] = permutedims(reshape(
		v_V'[1:xtt.ttv_rks[i+1], :],
		xtt.ttv_rks[i+1], size(V,3), size(V,4)
	), [2,1,3])

	# Update the i-th core from truncated U * diag(s_trunc)
	xtt.ttv_vec[i] = reshape(
		u_V[:, 1:xtt.ttv_rks[i+1]] * Diagonal(s_trunc[1:xtt.ttv_rks[i+1]]),
		size(V,1), size(V,2), :)
	xtt.ttv_ot[i+1] = 1
	xtt.ttv_ot[i] = 0
	return xtt
end

function right_core_move_mals(xtt::TTvector{T}, i::Integer, V::Array{T,4},
                              tol::Float64, rmax::Integer) where {T<:Number}
	# Perform the truncated SVD on the 2-core block V
	u_V, s_V, v_V = svd(reshape(V, prod(size(V)[1:2]), :))
	s_trunc = sv_trunc(s_V, tol)
	xtt.ttv_rks[i+1] = min(length(s_trunc), rmax)

	# Update the i-th core from truncated U
	xtt.ttv_vec[i] = reshape(
		u_V[:, 1:xtt.ttv_rks[i+1]],
		size(V,1), size(V,2), xtt.ttv_rks[i+1]
	)
	xtt.ttv_ot[i] = -1

	# Update the (i+1)-th core from diag(s_trunc) * V^T
	xtt.ttv_vec[i+1] = permutedims(reshape(
		Diagonal(s_trunc[1:xtt.ttv_rks[i+1]]) * v_V'[1:xtt.ttv_rks[i+1], :],
		xtt.ttv_rks[i+1], size(V,3), size(V,4)
	), [2,1,3])
	xtt.ttv_ot[i+1] = 0
	return xtt
end

function K_full_mals(Gi::AbstractArray{T,5}, Hi::AbstractArray{T,5},
                     K_dims::NTuple{4,Int}) where T<:Number
	K = zeros(T, prod(K_dims), prod(K_dims))
	Krshp = reshape(K, (K_dims..., K_dims...))
	@tensor Krshp[a,b,c,d,e,f,g,h] = Gi[a,b,e,f,z] * Hi[z,c,d,g,h]
	return Hermitian(K)
end

function Ksolve_mals(Gi::AbstractArray{T,5}, Hi::AbstractArray{T,5},
                     G_bi::AbstractArray{T,3}, H_bi::AbstractArray{T,3}) where T<:Number
	K_dims = (size(Gi,1), size(Gi,2), size(Hi,2), size(Hi,3))
	K = K_full_mals(Gi, Hi, K_dims)
	Pb = zeros(T, K_dims)
	@tensor Pb[a,b,c,d] = G_bi[a,b,z] * H_bi[z,c,d]
	V = reshape(K, prod(K_dims), :) \ Pb[:]
	return reshape(V, K_dims)
end

function K_eigmin_mals(Gi::Array{T,5}, Hi::Array{T,5},
                       ttv_vec_i::Array{T,3}, ttv_vec_ip::Array{T,3};
                       it_solver::Bool=false, itslv_thresh::Int=256,
                       maxiter::Int=200, tol::Float64=1e-6) where T<:Number
	K_dims = (size(ttv_vec_i,1), size(ttv_vec_i,2),
	          size(ttv_vec_ip,1), size(ttv_vec_ip,3))
	Gtemp = @view(Gi[:, 1:K_dims[2], :, 1:K_dims[2], :])
	Htemp = @view(Hi[:, :, 1:K_dims[4], :, 1:K_dims[4]])
	if it_solver || prod(K_dims) > itslv_thresh
		H = zeros(T, prod(K_dims))
		function K_matfree(V::AbstractArray{S,1};
		                   K_dims::NTuple{4,Int}=K_dims,
		                   H::AbstractArray{S,1}=H,
		                   Gtemp::AbstractArray{S,5}=Gtemp,
		                   Htemp::AbstractArray{S,5}=Htemp) where S<:Number
			Hrshp = reshape(H, K_dims)
			@tensoropt((f,h),
				Hrshp[a,b,c,d] = Gtemp[a,b,e,f,z] *
				               reshape(V, K_dims)[e,f,g,h] *
				               Htemp[z,c,d,g,h]
			)
			return H::AbstractArray{S,1}
		end
		X0 = zeros(T, prod(K_dims))
		X0_temp = reshape(X0, K_dims)
		@tensor X0_temp[a,b,c,d] = ttv_vec_i[a,b,z] * ttv_vec_ip[c,z,d]
		r = lobpcg(LinearMap(K_matfree, prod(K_dims);
		                     ishermitian = true),
		           false, X0, 1; maxiter = maxiter, tol = tol)
		return r.λ[1]::Float64, reshape(r.X[:,1], K_dims)::Array{T,4}
	else
		K = K_full_mals(Gtemp, Htemp, K_dims)
		F = eigen(K, 1:1)
		return real(F.values[1])::Float64,
		       reshape(F.vectors[:,1], K_dims)::Array{T,4}
	end
end

"""
Returns the solution `tt_opt :: TTvector` of Ax=b using the MALS algorithm where A is given as `TToperator` and `b`, `tt_start` are `TTvector`.
The ranks are adapted at each microstep by keeping the singular values larger than `tol`.
"""
function mals_linsolve(A :: TToperator{T}, b :: TTvector{T},
                      tt_start :: TTvector{T};
                      tol::Float64 = 1e-12,
                      rmax::Int = round(Int, sqrt(prod(tt_start.ttv_dims)))) where {T<:Number}

	d = b.N

	# Initialize the TT iterate
	tt_opt = orthogonalize(tt_start)
	dims = tt_start.ttv_dims
	A_rks = A.tto_rks
	b_rks = b.ttv_rks

	# Allocate G, G_b
	G   = Array{Array{T,5}}(undef, d)
	G_b = Array{Array{T,3}}(undef, d)
	for i in 1:d
		rmax_i = min(rmax, prod(dims[1:i-1]), prod(dims[i:end]))
		G[i]   = zeros(dims[i], rmax_i, dims[i], rmax_i, A_rks[i+1])
		G_b[i] = zeros(dims[i], rmax_i, b_rks[i+1])
	end
	G[1][:,1:1,:,1:1,:] = reshape(A.tto_vec[1][:,:,1,:], dims[1], 1, dims[1], 1, :)
	G_b[1] = reshape(b.ttv_vec[1], dims[1], 1, :)

	H   = init_H_mals(tt_opt, A, rmax)
	H_b = init_Hb_mals(tt_opt, b, rmax)

	# One sweep: can be wrapped in a loop if repeats > 1
	# First half sweep
	for i in 1:(d-1)
		Gi   = @view(G[i][:, 1:tt_opt.ttv_rks[i], :, 1:tt_opt.ttv_rks[i], :])
		Hi   = @view(H[i][:, :, 1:tt_opt.ttv_rks[i+2], :, 1:tt_opt.ttv_rks[i+2]])
		G_bi = @view(G_b[i][:, 1:tt_opt.ttv_rks[i], :])
		H_bi = @view(H_b[i][:, :, 1:tt_opt.ttv_rks[i+2]])

		# Solve local 2-core system
		V = Ksolve_mals(Gi, Hi, G_bi, H_bi)
		tt_opt = right_core_move_mals(tt_opt, i, V, tol, rmax)

		# Update G[i+1], G_b[i+1]
		Gip   = @view(G[i+1][:, 1:tt_opt.ttv_rks[i+1], :, 1:tt_opt.ttv_rks[i+1], :])
		G_bip = @view(G_b[i+1][:, 1:tt_opt.ttv_rks[i+1], :])
		update_G!(tt_opt.ttv_vec[i], A.tto_vec[i+1], Gi,   Gip)
		update_Gb!(tt_opt.ttv_vec[i], b.ttv_vec[i+1], G_bi, G_bip)
	end

	# Second half sweep
	for i in (d-1):-1:1
		Gi   = @view(G[i][:, 1:tt_opt.ttv_rks[i], :, 1:tt_opt.ttv_rks[i], :])
		Hi   = @view(H[i][:, :, 1:tt_opt.ttv_rks[i+2], :, 1:tt_opt.ttv_rks[i+2]])
		G_bi = @view(G_b[i][:, 1:tt_opt.ttv_rks[i], :])
		H_bi = @view(H_b[i][:, :, 1:tt_opt.ttv_rks[i+2]])

		V = Ksolve_mals(Gi, Hi, G_bi, H_bi)
		tt_opt = left_core_move_mals(tt_opt, i, V, tol, rmax)

		if i > 1
			Him  = @view(H[i-1][:, :, 1:tt_opt.ttv_rks[i+1], :, 1:tt_opt.ttv_rks[i+1]])
			updateH_mals!(tt_opt.ttv_vec[i+1], A.tto_vec[i], Hi, Him)

			H_bim = @view(H_b[i-1][:, :, 1:tt_opt.ttv_rks[i+1]])
			updateHb_mals!(tt_opt.ttv_vec[i+1], b.ttv_vec[i], H_bi, H_bim)
		end
	end

	return tt_opt
end

"""
Returns the list of the approximate smallest eigenvalue at each microstep, the corresponding eigenvector as a `TTvector` and the list of the maximum rank at each microstep.

`A` is given as `TToperator` and `tt_start` is a `TTvector`.
The ranks are adapted at each microstep by keeping the singular values larger than `tol`.
The number of total sweeps is given by `sweep_schedule[end]`. The maximum rank is prescribed at each sweep `sweep_schedule[k] ≤ i < sweep_schedule[k+1]` by `rmax_schedule[k]`.
"""
function mals_eigsolve(A :: TToperator{T}, tt_start :: TTvector{T};
                      tol::Float64 = 1e-12,
                      sweep_schedule::Vector{Int} = [2],
                      rmax_schedule::Vector{Int} = [round(Int, sqrt(prod(tt_start.ttv_dims)))],
                      it_solver::Bool = false,
                      linsolv_maxiter::Int = 200,
                      linsolv_tol::Float64 = max(sqrt(tol), 1e-8),
                      itslv_thresh::Int = 256) where {T<:Number}

	d = A.N
	@assert(length(rmax_schedule) == length(sweep_schedule),
	        "Sweep schedule error")

	tt_opt = orthogonalize(tt_start)
	dims = tt_start.ttv_dims

	E = Float64[]
	r_hist = Int[]

	# Allocate G
	G = Array{Array{T,5}}(undef, d)
	rmax = maximum(rmax_schedule)
	for i in 1:d
		rmax_i = min(rmax, prod(dims[1:i-1]), prod(dims[i:end]))
		G[i] = zeros(dims[i], rmax_i, dims[i], rmax_i, A.tto_rks[i+1])
	end
	G[1][:,1:1,:,1:1,:] = reshape(A.tto_vec[1][:,:,1,:], dims[1], 1, dims[1], 1, :)

	H = init_H_mals(tt_opt, A, rmax)

	nsweeps = 0
	i_schedule = 1
	while i_schedule <= length(sweep_schedule)
		nsweeps += 1

		if nsweeps == sweep_schedule[i_schedule]
			i_schedule += 1
			if i_schedule > length(sweep_schedule)
				return E, tt_opt, r_hist
			end
		end

		# First half sweep
		for i in 1:(d-1)
			λ, V = K_eigmin_mals(G[i], H[i],
			                    tt_opt.ttv_vec[i],
			                    tt_opt.ttv_vec[i+1];
			                    it_solver=it_solver,
			                    maxiter=linsolv_maxiter,
			                    tol=linsolv_tol)
			push!(E, λ)

			tt_opt = right_core_move_mals(tt_opt, i, V, tol, rmax_schedule[i_schedule])
			push!(r_hist, maximum(tt_opt.ttv_rks))

			Gi  = @view(G[i][:, 1:tt_opt.ttv_rks[i], :, 1:tt_opt.ttv_rks[i], :])
			Gip = @view(G[i+1][:, 1:tt_opt.ttv_rks[i+1], :, 1:tt_opt.ttv_rks[i+1], :])
			update_G!(tt_opt.ttv_vec[i], A.tto_vec[i+1], Gi, Gip)
		end

		# Second half sweep
		for i in (d-1):-1:1
			λ, V = K_eigmin_mals(G[i], H[i],
			                    tt_opt.ttv_vec[i],
			                    tt_opt.ttv_vec[i+1];
			                    it_solver=it_solver,
			                    maxiter=linsolv_maxiter,
			                    tol=linsolv_tol)
			push!(E, λ)

			tt_opt = left_core_move_mals(tt_opt, i, V, tol, rmax_schedule[i_schedule])
			push!(r_hist, maximum(tt_opt.ttv_rks))

			if i > 1
				Hi  = @view(H[i][:, :, 1:tt_opt.ttv_rks[i+2], :, 1:tt_opt.ttv_rks[i+2]])
				Him = @view(H[i-1][:, :, 1:tt_opt.ttv_rks[i+1], :, 1:tt_opt.ttv_rks[i+1]])
				updateH_mals!(tt_opt.ttv_vec[i+1], A.tto_vec[i], Hi, Him)
			end
		end
	end

	return E, tt_opt, r_hist
end
