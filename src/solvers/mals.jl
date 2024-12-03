using IterativeSolvers
using LinearMaps

"""
Implementation based on the presentation in 
Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider. "The alternating linear scheme for tensor optimization in the tensor train format." SIAM Journal on Scientific Computing 34.2 (2012): A683-A713.
"""

function updateH_mals!(x_vec::Array{T,3}, A_vec::Array{T,4}, Hi::AbstractArray{T,5}, Him::AbstractArray{T,5}) where T<:Number
	@tensor(Him[a,i,α,l,β] = conj.(x_vec)[j,α,x]*(Hi[z,j,x,k,y]*x_vec[k,β,y])*A_vec[i,l,a,z]) #size(rAim,ri,ni,ri,ni)
	nothing
end

function init_H_mals(x_tt::TTvector{T},A::TToperator{T},rmax::Int) where {T<:Number}
	d = x_tt.N
	H = Array{Array{T,5}}(undef, d-1)
	H[d-1] = reshape(permutedims(A.tto_vec[d],[3,1,2,4]), :, x_tt.ttv_dims[d], 1, x_tt.ttv_dims[d], 1) #size(R^A_d, n_d, r_{d+1}, n_d, r_{d+1})
	for i = (d-1) : -1 : 2
		# Update H[i-1]
		rmax_i = min(rmax,prod(x_tt.ttv_dims[1:i]),prod(x_tt.ttv_dims[i+1:end]))
		H[i-1] = zeros(T,A.tto_rks[i],x_tt.ttv_dims[i],rmax_i,x_tt.ttv_dims[i],rmax_i)
		Hi = @view(H[i][:,:,1:x_tt.ttv_rks[i+2],:,1:x_tt.ttv_rks[i+2]])
		Him = @view(H[i-1][:,:,1:x_tt.ttv_rks[i+1],:,1:x_tt.ttv_rks[i+1]])
		updateH_mals!(x_tt.ttv_vec[i+1], A.tto_vec[i], Hi, Him)
	end
	return H
end

function updateHb_mals!(xtt_vec::Array{T,3}, btt_vec::Array{T,3}, Hbi::AbstractArray{T,3}, Hbim::AbstractArray{T,3}) where T<:Number
	@tensor Hbim[β,i,χ] = conj.(xtt_vec)[j,χ,a]*Hbi[γ,j,a]*btt_vec[i,β,γ]
	nothing
end

function init_Hb_mals(x_tt::TTvector{T},b::TTvector{T},rmax::Int) where {T<:Number}
	d = x_tt.N
	Hb = Array{Array{T,3}}(undef, d-1)
	Hb[d-1] = reshape(permutedims(b.ttv_vec[d],[2,1,3]),b.ttv_rks[d],b.ttv_dims[d],1)
	for i in d-1:-1:2
		rmax_i = min(rmax,prod(x_tt.ttv_dims[1:i]),prod(x_tt.ttv_dims[i+1:end]))
		Hb[i-1] = zeros(T,b.ttv_rks[i+1],b.ttv_dims[i+1],rmax_i)
		Hbi = @view(Hb[i][:,:,1:x_tt.ttv_rks[i+2]])
		Hbim = @view(Hb[i-1][:,:,1:x_tt.ttv_rks[i+1]])
		updateHb_mals!(x_tt.ttv_vec[i+1],b.ttv_vec[i],Hbi,Hbim)
	end
	return Hb
end

function left_core_move_mals(xtt::TTvector{T},i::Integer,V::Array{T,4},tol::Float64,rmax::Integer) where {T<:Number}
	# Perform the truncated svd
	u_V, s_V, v_V = svd(reshape(V, prod(size(V)[1:2]), :))
	# Determine the truncated rank
	s_trunc = sv_trunc(s_V,tol)
	# Update the ranks to the truncated one
	xtt.ttv_rks[i+1] = min(length(s_trunc),rmax)
	println("Rank: $(xtt.ttv_rks[i+1]),	Max rank=$rmax")
	println("Discarded weight: $((norm(s_V)-norm(s_V[1:xtt.ttv_rks[i+1]]))/norm(s_V))")

	# xtt.ttv_vec[i+1] = truncated Transpose(v_V)
	xtt.ttv_vec[i+1] = permutedims(reshape(v_V'[1:xtt.ttv_rks[i+1],:],xtt.ttv_rks[i+1],size(V,3),size(V,4)),[2 1 3])
	# xtt.ttv_vec[i] = truncated u_V * Diagonal(s_V)
	xtt.ttv_vec[i] = reshape(u_V[:, 1:xtt.ttv_rks[i+1]] * Diagonal(s_trunc[1:xtt.ttv_rks[i+1]]),size(V,1),size(V,2),:)
	xtt.ttv_ot[i+1] = 1
	xtt.ttv_ot[i] = 0
	return xtt
end

function right_core_move_mals(xtt::TTvector{T},i::Integer,V::Array{T,4},tol::Float64,rmax::Integer) where {T<:Number}
	# Perform the truncated svd
	u_V, s_V, v_V = svd(reshape(V, prod(size(V)[1:2]), :))
	# Determine the truncated rank
	s_trunc = sv_trunc(s_V,tol)
	# Update the ranks to the truncated one
	xtt.ttv_rks[i+1] = min(length(s_trunc),rmax)
	println("Rank: $(xtt.ttv_rks[i+1]),	Max rank=$rmax")
	println("Discarded weight: $((norm(s_V)-norm(s_V[1:xtt.ttv_rks[i+1]]))/norm(s_V))")

	# xtt.ttv_vec[i] = truncated u_V
	xtt.ttv_vec[i] = reshape(u_V[:, 1:xtt.ttv_rks[i+1]], size(V,1), size(V,2), xtt.ttv_rks[i+1])
	xtt.ttv_ot[i] = -1

	# xtt.ttv_vec[i+1] = truncated Diagonal(s_V) * Transpose(v_V)
	xtt.ttv_vec[i+1] = permutedims(reshape(Diagonal(s_trunc[1:xtt.ttv_rks[i+1]]) * v_V'[1:xtt.ttv_rks[i+1],:],xtt.ttv_rks[i+1],size(V,3),size(V,4)), [2 1 3])
	xtt.ttv_ot[i+1] = 0
	return xtt
end

function K_full_mals(Gi::AbstractArray{T,5},Hi::AbstractArray{T,5},K_dims::NTuple{4,Int}) where T<:Number
	K = zeros(T,prod(K_dims),prod(K_dims))
	Krshp = reshape(K,(K_dims...,K_dims...))
	@tensor Krshp[a,b,c,d,e,f,g,h] = Gi[a,b,e,f,z]*Hi[z,c,d,g,h] #size(K)=(ni,rim,nip,rip)
	return Hermitian(K)
end

function Ksolve_mals(Gi::AbstractArray{T,5}, Hi::AbstractArray{T,5}, G_bi::AbstractArray{T,3}, H_bi::AbstractArray{T,3}) where T<:Number
	K_dims = (size(Gi,1),size(Gi,2),size(Hi,2),size(Hi,3))
	K = K_full_mals(Gi,Hi,K_dims)
	Pb = zeros(T,K_dims)
	@tensor Pb[a,b,c,d] = G_bi[a,b,z]*H_bi[z,c,d] #size(ni,rim,nip,rip)
	V = reshape(K,prod(K_dims),:) \ Pb[:]
	return reshape(V,K_dims)
end

function K_eigmin_mals(Gi::Array{T,5},Hi::Array{T,5},ttv_vec_i::Array{T,3},ttv_vec_ip::Array{T,3};it_solver=false,itslv_thresh=256::Int64,maxiter=200::Int64,tol=1e-6::Float64) where T<:Number
	K_dims = (size(ttv_vec_i,1),size(ttv_vec_i,2),size(ttv_vec_ip,1),size(ttv_vec_ip,3))
	Gtemp = @view(Gi[:,1:K_dims[2],:,1:K_dims[2],:])
	Htemp = @view(Hi[:,:,1:K_dims[4],:,1:K_dims[4]])
	if it_solver || prod(K_dims) > itslv_thresh
		H = zeros(T,prod(K_dims))
		function K_matfree(V::AbstractArray{S,1};K_dims=K_dims::NTuple{4,Int},H=H::AbstractArray{S,1},Gtemp=Gtemp::AbstractArray{S,5},Htemp=Htemp::AbstractArray{S,5}) where S<:Number
			Hrshp = reshape(H,K_dims)
			@tensoropt((f,h), Hrshp[a,b,c,d] = Gtemp[a,b,e,f,z]*reshape(V,K_dims)[e,f,g,h]*Htemp[z,c,d,g,h])
			return H::AbstractArray{S,1}
		end
		X0 = zeros(T,prod(K_dims))
		X0_temp = reshape(X0,K_dims)
		@tensor X0_temp[a,b,c,d] = ttv_vec_i[a,b,z]*ttv_vec_ip[c,z,d]
		r = lobpcg(LinearMap(K_matfree,prod(K_dims);ishermitian = true),false,X0,1;maxiter=maxiter,tol=tol)
		return r.λ[1]::Float64, reshape(r.X[:,1],K_dims)::Array{T,4}
	else
		K = K_full_mals(Gtemp,Htemp,K_dims)
		F = eigen(K,1:1)
		return real(F.values[1])::Float64,reshape(F.vectors[:,1],K_dims)::Array{T,4}
	end	
end

"""
Returns the solution `tt_opt :: TTvector` of Ax=b using the MALS algorithm where A is given as `TToperator` and `b`, `tt_start` are `TTvector`.
The ranks are adapted at each microstep by keeping the singular values larger than `tol`.
"""
function mals_linsolv(A :: TToperator{T}, b :: TTvector{T}, tt_start :: TTvector{T}; tol=1e-12::Float64,rmax=round(Int,sqrt(prod(tt_start.ttv_dims)))) where {T<:Number}
	# mals finds the minimum of the operator J(x)=1/2*<Ax,x> - <x,b>
	# input:
	# 	A: the tensor operator in its tensor train format
	#   b: the tensor in its tensor train format
	#	tt_start: start value in its tensor train format
	#	opt_rks: rank vector considered to be optimal enough
	#	tol: tolerated inaccuracy
	# output:
	#	tt_opt: stationary point of J up to tolerated inaccuracy
	# 			in its tensor train format
	d = b.N

	# Initialize the to be returned tensor in its tensor train format
	tt_opt = orthogonalize(tt_start)
	dims = tt_start.ttv_dims
	# Define the array of ranks of A [R_0=1,R_1,...,R_d]
	A_rks = A.tto_rks
	# Define the array of ranks of b [R^b_0=1,R^b_1,...,R^b_d]
	b_rks = b.ttv_rks

	# Initialize the arrays of G and H
	G = Array{Array{T,5}}(undef, d)
	# Initialize the arrays of G_b and H_b
	G_b = Array{Array{T,3}}(undef, d)
	# Initialize G[1], G_b[1], H[d] and H_b[d]	
	for i in 1:d
		rmax_i = min(rmax,prod(dims[1:i-1]),prod(dims[i:end]))
		G[i] = zeros(dims[i],rmax_i,dims[i],rmax_i,A_rks[i+1])
		G_b[i] = zeros(dims[i],rmax_i,b_rks[i+1])
	end
	G[1][:,1:1,:,1:1,:] = reshape(A.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)
	G_b[1] = reshape(b.ttv_vec[1], dims[1], 1, :)

	H = init_H_mals(tt_opt,A,rmax)
	H_b = init_Hb_mals(tt_opt,b,rmax)

	#while 1==1 #TODO make it work for real
		# First half sweep
		for i = 1:(d-1)
			Gi = @view(G[i][:,1:tt_opt.ttv_rks[i],:,1:tt_opt.ttv_rks[i],:])
			Hi = @view(H[i][:,:,1:tt_opt.ttv_rks[i+2],:,1:tt_opt.ttv_rks[i+2]])
			G_bi = @view(G_b[i][:,1:tt_opt.ttv_rks[i],:])
			H_bi = @view(H_b[i][:,:,1:tt_opt.ttv_rks[i+2]])
			# Define V as solution of K*x=P2b in x
			V = Ksolve_mals(Gi,Hi,G_bi,H_bi)
			tt_opt = right_core_move_mals(tt_opt,i,V,tol,rmax)
			# Update G[i+1],G_b[i+1]
			Gip = @view(G[i+1][:,1:tt_opt.ttv_rks[i+1],:,1:tt_opt.ttv_rks[i+1],:])
			G_bip = @view(G_b[i+1][:,1:tt_opt.ttv_rks[i+1],:])
			update_G!(tt_opt.ttv_vec[i],A.tto_vec[i+1],Gi,Gip)
			update_Gb!(tt_opt.ttv_vec[i],b.ttv_vec[i+1],G_bi,G_bip)
		end

		# Second half sweep
		for i = d-1:(-1):1
			Gi = @view(G[i][:,1:tt_opt.ttv_rks[i],:,1:tt_opt.ttv_rks[i],:])
			Hi = @view(H[i][:,:,1:tt_opt.ttv_rks[i+2],:,1:tt_opt.ttv_rks[i+2]])
			G_bi = @view(G_b[i][:,1:tt_opt.ttv_rks[i],:])
			H_bi = @view(H_b[i][:,:,1:tt_opt.ttv_rks[i+2]])
			# Define V as solution of K*x=P2b in x
			V = Ksolve_mals(Gi,Hi,G_bi,H_bi)
			tt_opt = left_core_move_mals(tt_opt,i,V,tol,rmax)
			# Update H[i-1], H_b[i-1]
			if i > 1
				Him = @view(H[i-1][:,:,1:tt_opt.ttv_rks[i+1],:,1:tt_opt.ttv_rks[i+1]])
				updateH_mals!(tt_opt.ttv_vec[i+1], A.tto_vec[i], Hi, Him)
				H_bim = @view(H_b[i-1][:,:,1:tt_opt.ttv_rks[i+1]])
				updateHb_mals!(tt_opt.ttv_vec[i+1], b.ttv_vec[i], H_bi,H_bim)
			end
		end
	#end
	return tt_opt
end

"""
Returns the list of the approximate smallest eigenvalue at each microstep, the corresponding eigenvector as a `TTvector` and the list of the maximum rank at each microstep.

`A` is given as `TToperator` and `tt_start` is a `TTvector`.
The ranks are adapted at each microstep by keeping the singular values larger than `tol`.
The number of total sweeps is given by `sweep_schedule[end]`. The maximum rank is prescribed at each sweep `sweep_schedule[k]≤ i <sweep_schedule[k+1]` by `rmax_schedule[k]`.
"""
function mals_eigsolv(A :: TToperator{T}, tt_start :: TTvector{T}; tol=1e-12::Float64,sweep_schedule=[2]::Array{Int64,1},rmax_schedule=[round(Int,sqrt(prod(tt_start.ttv_dims)))]::Array{Int64,1},it_solver=false::Bool,linsolv_maxiter=200::Int64,linsolv_tol=max(sqrt(tol),1e-8)::Float64,itslv_thresh=256::Int) where {T<:Number}

	d = A.N
	@assert(length(rmax_schedule)==length(sweep_schedule),"Sweep schedule error")	
	# Initialize the to be returned tensor in its tensor train format
	tt_opt = orthogonalize(tt_start)
	dims = tt_start.ttv_dims
	# Initialize the output objects
	E = Float64[]
	r_hist = Int64[]
	# Initialize the arrays of G
	G = Array{Array{T}}(undef, d)
	rmax = maximum(rmax_schedule)
	# Initialize G[i]
	for i in 1:d
		rmax_i = min(rmax,prod(dims[1:i-1]),prod(dims[i:end]))
		G[i] = zeros(dims[i],rmax_i,dims[i],rmax_i,A.tto_rks[i+1])
	end
	G[1][:,1:1,:,1:1,:] = reshape(A.tto_vec[1][:,:,1,:], dims[1],1,dims[1], 1, :)

	H = init_H_mals(tt_opt,A,rmax)

	nsweeps = 0 #sweeps counter
	i_schedule = 1
	while i_schedule <= length(sweep_schedule) 
		nsweeps+=1
		println("Macro-iteration $nsweeps; bond dimension $(rmax_schedule[i_schedule])")

		if nsweeps == sweep_schedule[i_schedule]
			i_schedule+=1
			if i_schedule > length(sweep_schedule)
				return E::Array{Float64,1}, tt_opt::TTvector{T}, r_hist::Array{Int,1}
			end
		end

		# First half sweep
		for i = 1:(d-1)
			println("Forward sweep: core optimization $i out of $(d-1)")
			# Define V as solution of K V= λ V for smallest λ
			λ,V = K_eigmin_mals(G[i],H[i],tt_opt.ttv_vec[i],tt_opt.ttv_vec[i+1];it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol)
			println("Eigenvalue: $λ")
			E = vcat(E,λ)
			tt_opt = right_core_move_mals(tt_opt,i,V,tol,rmax_schedule[i_schedule])
			r_hist = vcat(r_hist,maximum(tt_opt.ttv_rks))
			# Update G[i+1],G_b[i+1]
			Gi = @view(G[i][:,1:tt_opt.ttv_rks[i],:,1:tt_opt.ttv_rks[i],:])
			Gip = @view(G[i+1][:,1:tt_opt.ttv_rks[i+1],:,1:tt_opt.ttv_rks[i+1],:])
			update_G!(tt_opt.ttv_vec[i],A.tto_vec[i+1],Gi,Gip)
		end

		# Second half sweep
		for i = d-1:(-1):1
			println("Backward sweep: core optimization $(d-i) out of $(d-1)")
			# Define V as solution of K*x=P2b in x
			λ,V = K_eigmin_mals(G[i],H[i],tt_opt.ttv_vec[i],tt_opt.ttv_vec[i+1];it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol)
			println("Eigenvalue: $λ")
			E = vcat(E,λ)
			tt_opt = left_core_move_mals(tt_opt,i,V,tol,rmax_schedule[i_schedule])
			r_hist = vcat(r_hist,maximum(tt_opt.ttv_rks))
			# Update H[i-1]
			if i > 1
				Hi = @view(H[i][:,:,1:tt_opt.ttv_rks[i+2],:,1:tt_opt.ttv_rks[i+2]])
				Him = @view(H[i-1][:,:,1:tt_opt.ttv_rks[i+1],:,1:tt_opt.ttv_rks[i+1]])
				updateH_mals!(tt_opt.ttv_vec[i+1], A.tto_vec[i], Hi, Him)
			end
		end
	end
	return E::Array{Float64,1}, tt_opt::TTvector{T}, r_hist::Array{Int,1}
end