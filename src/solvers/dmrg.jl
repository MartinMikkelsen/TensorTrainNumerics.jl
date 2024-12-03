using LinearMaps
using TensorOperations
using KrylovKit

"""
Implementation based on the presentation in 
Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider. "The alternating linear scheme for tensor optimization in the tensor train format." SIAM Journal on Scientific Computing 34.2 (2012): A683-A713.
"""

function init_H(x_tt::TTvector{T},A_tto::TToperator{T},N::Int,rmax) where {T<:Number}
	d = x_tt.N
	H = Array{Array{T,3},1}(undef, d+1-N)
	H[d+1-N] = ones(T,1,1,1)
	rks = r_and_d_to_rks(vcat(1,rmax*ones(Int,d-1),1),x_tt.ttv_dims;rmax=rmax)
	for i = d+1-N:-1:2
		H[i-1] = zeros(T,A_tto.tto_rks[i+N-1],rks[i+N-1],rks[i+N-1])
		Hi = @view(H[i][:,1:x_tt.ttv_rks[i+N],1:x_tt.ttv_rks[i+N]])
		Him = @view(H[i-1][:,1:x_tt.ttv_rks[i+N-1],1:x_tt.ttv_rks[i+N-1]])
		x_vec = x_tt.ttv_vec[i+N-1]
		A_vec = A_tto.tto_vec[i+N-1]
		update_H!(x_vec,A_vec,Hi,Him)
	end
	return H
end

function update_H!(x_vec::Array{T,3},A_vec::Array{T,4},Hi::AbstractArray{T,3},Him::AbstractArray{T,3}) where T<:Number
	@tensoropt((ϕ,χ), Him[a,α,β] = conj.(x_vec)[j,α,ϕ]*Hi[z,ϕ,χ]*x_vec[k,β,χ]*A_vec[j,k,a,z]) #size (rAim, rim, rim)
	nothing
end

function update_G!(x_vec::Array{T,3},A_vec::Array{T,4},Gi::AbstractArray{T,3},Gip::AbstractArray{T,3}) where T<:Number
	@tensoropt((ϕ,χ), Gip[a,α,β] = conj.(x_vec)[j,ϕ,α]*Gi[z,ϕ,χ]*x_vec[k,χ,β]*A_vec[j,k,z,a]) #size (rAi, ri, ri)
	nothing
end

#returns the contracted tensor A_i[\\mu_i] ⋯ A_j[\\mu_j] ∈ R^{R^A_{i-1} × n_i × n_i × ⋯ × n_j × n_j ×  R^A_j}
function Amid(A_tto::TToperator{T},i::Int,j::Int) where {T<:Number}
	A = permutedims(A_tto.tto_vec[i],(3,1,2,4))
	for k in i+1:j
		C = reshape(A,A_tto.tto_rks[i],prod(A_tto.tto_dims[i:k-1]),:,A_tto.tto_rks[k])
		@tensor Atemp[αk,Ik,ik,Jk,jk,βk] := A_tto.tto_vec[k][ik,jk,ξk,βk]*C[αk,Ik,Jk,ξk]
		A = reshape(Atemp,A_tto.tto_rks[i],prod(A_tto.tto_dims[i:k]),:,A_tto.tto_rks[k+1])
	end
	return A #size R^A_{i-1} × (n_i⋯n_j) × (n_i⋯n_j) × R^A_j
end

#full assemble of matrix K
function K_full(Gi::AbstractArray{T,3},Hi::AbstractArray{T,3},Amid_tensor::AbstractArray{T,4}) where {T<:Number,d}
	K_dims = (size(Gi,2),size(Amid_tensor,2),size(Hi,2))
	K = zeros(T,K_dims...,K_dims...)
	@tensoropt((a,c,d,f), K[a,b,c,d,e,f] = Gi[y,a,d]*Hi[z,c,f]*Amid_tensor[y,b,e,z]) #size (r^X_{i-1},n_i⋯n_j,r^X_j)
	return Hermitian(reshape(K,prod(K_dims),prod(K_dims)))
end

function init_Hb(x_tt::TTvector{T},b_tt::TTvector{T},N::Integer,rmax) where {T<:Number}
	d = x_tt.N
	H_b = Array{Array{T,2},1}(undef, d+1-N) 
	H_b[d+1-N] = ones(T,1,1)
	rks = r_and_d_to_rks(vcat(1,rmax*ones(Int,d-1),1),x_tt.ttv_dims;rmax=rmax)
	for i = d+1-N:-1:2
		H_b[i-1] = zeros(T,rks[i],b_tt.ttv_rks[i+N-1])
		b_vec = b_tt.ttv_vec[i+N-1]
		x_vec = x_tt.ttv_vec[i+N-1]
		Hbi = @view(H_b[i][1:x_tt.ttv_rks[i+N],:])
		Hbim = @view(H_b[i-1][1:x_tt.ttv_rks[i+N-1],:])
		update_Hb!(x_vec,b_vec,Hbi,Hbim) #size(r^X_{i-1},r^b_{i-1}) 
	end
	return H_b
end

function update_Hb!(x_vec::Array{T,3},b_vec::Array{T,3},H_bi::AbstractArray{T,2},H_bim::AbstractArray{T,2}) where T<:Number
	@tensoropt((ϕ,χ), H_bim[α,β] = H_bi[ϕ,χ]*b_vec[i,β,χ]*conj.(x_vec)[i,α,ϕ])
	nothing
end

function update_Gb!(x_vec::Array{T,3},b_vec::Array{T,3},G_bi::AbstractArray{T,2},G_bip::AbstractArray{T,2}) where T<:Number
	@tensoropt((ϕ,χ), G_bip[α,β] = G_bi[ϕ,χ]*b_vec[i,χ,β]*conj.(x_vec)[i,ϕ,α])
	nothing
end

function b_mid(b_tt::TTvector{T},i::Integer,j::Integer) where {T<:Number}
	b_out = permutedims(b_tt.ttv_vec[i],(2,1,3))
	for k in i+1:j
		@tensor btemp[αk,ik,jk,βk] := b_out[αk,ik,ξk]*b_tt.ttv_vec[k][jk,ξk,βk]
		b_out = reshape(btemp,b_tt.ttv_rks[i],:,b_tt.ttv_rks[k+1]) #size r^b_{i-1} × (n_i⋯n_k) × r^b_k
	end
	return b_out
end

function Ksolve(Gi::AbstractArray{T,3},G_bi::AbstractArray{T,2},Hi::AbstractArray{T,3},H_bi::AbstractArray{T,2},Amid_tensor::AbstractArray{T,4},Bmid::AbstractArray{T,3},V0::AbstractArray{T,3};it_solver=false,maxiter=200,tol=1e-6,itslv_thresh=256) where T<:Number
	K_dims = (size(Gi,2),size(Amid_tensor,2),size(Hi,2))
	@tensor Pb[α1,i,α2] := G_bi[α1,β1]*Bmid[β1,i,β2]*H_bi[α2,β2] #size (r^X_{i-1},n_i⋯n_j,r^X_j)
	if it_solver || prod(K_dims) > itslv_thresh
		Gtemp = @view(Gi[:,1:K_dims[1],1:K_dims[1]])
		Htemp = @view(Hi[:,1:K_dims[3],1:K_dims[3]])
		Vout = zeros(T,prod(K_dims))

		function K_matfree(V::AbstractArray{S,1};Gi=Gtemp::AbstractArray{S,3},Hi=Htemp::AbstractArray{S,3},K_dims=K_dims::NTuple{3,Int},Amid_tensor=Amid_tensor::AbstractArray{S,4},Vout=Vout::AbstractArray{S,1}) where S<:Number
			Hrshp = reshape(Vout,K_dims)
			@tensoropt((a,c,d,f), Hrshp[a,b,c] = Gi[y,a,d]*Hi[z,c,f]*Amid_tensor[y,b,e,z]*reshape(V,K_dims)[d,e,f] + Gi[y,d,a]*Hi[z,f,c]*Amid_tensor[y,e,b,z]*reshape(V,K_dims)[d,e,f])
			return 0.5*Vout
		end

		Vout,info = linsolve(LinearMap{T}(K_matfree,prod(K_dims);issymmetric = true),Pb[:];issymmetric=true,tol=tol,maxiter=maxiter)
		return reshape(Vout,K_dims)
	else
		K = K_full(Gi,Hi,Amid_tensor)
		return reshape(K\Pb[:],K_dims)
	end
end

function right_core_move(x_tt::TTvector{T},V::Array{T,3},i::Int,tol::Float64,rmax::Integer) where {T<:Number}
	# Perform the truncated svd
	u_V, s_V, v_V = svd(reshape(V,x_tt.ttv_rks[i]*x_tt.ttv_dims[i],:))
	# Update the ranks to the truncated one
	x_tt.ttv_rks[i+1] = min(cut_off_index(s_V,tol),rmax)
	println("Rank: $(x_tt.ttv_rks[i+1]),	Max rank=$rmax")
	println("Discarded weight: $((norm(s_V)-norm(s_V[1:x_tt.ttv_rks[i+1]]))/norm(s_V))")

	x_tt.ttv_vec[i] = permutedims(reshape(u_V[:,1:x_tt.ttv_rks[i+1]],x_tt.ttv_rks[i],x_tt.ttv_dims[i],:),(2,1,3))
	x_tt.ttv_ot[i] = 1
	x_tt.ttv_ot[i+1]=0
	return x_tt, reshape(Diagonal(s_V[1:x_tt.ttv_rks[i+1]])*v_V'[1:x_tt.ttv_rks[i+1],:],x_tt.ttv_rks[i+1],:,size(V,3))
end

function left_core_move(x_tt::TTvector{T},V::Array{T,3},j::Int,tol::Float64,rmax::Integer) where {T<:Number}
	# Perform the truncated svd
	W = reshape(V, :, x_tt.ttv_dims[j], x_tt.ttv_rks[j+1])
	u_V, s_V, v_V = svd(reshape(W,:, x_tt.ttv_dims[j]*x_tt.ttv_rks[j+1]))
	# Update the ranks to the truncated one
	x_tt.ttv_rks[j] = min(cut_off_index(s_V,tol),rmax)
	println("Rank: $(x_tt.ttv_rks[j]),	Max rank=$rmax")
	println("Discarded weight: $((norm(s_V)-norm(s_V[1:x_tt.ttv_rks[j]]))/norm(s_V))")

	x_tt.ttv_vec[j] = permutedims(reshape(v_V'[1:x_tt.ttv_rks[j],:],x_tt.ttv_rks[j],:,x_tt.ttv_rks[j+1]),(2,1,3))
	x_tt.ttv_ot[j] = -1
	x_tt.ttv_ot[j-1]=0
	return x_tt, reshape(u_V[:,1:x_tt.ttv_rks[j]]*Diagonal(s_V[1:x_tt.ttv_rks[j]]),size(V,1),:,x_tt.ttv_rks[j])
end


function K_eigmin(Gi::AbstractArray{T,3},Hi::AbstractArray{T,3},V0::AbstractArray{T,3},Amid_tensor::AbstractArray{T,4};it_solver=false::Bool,itslv_thresh=256::Int64,maxiter=200::Int64,tol=1e-6::Float64) where T<:Number
	K_dims = size(V0)
	Gtemp = @view(Gi[:,1:K_dims[1],1:K_dims[1]])
	Htemp = @view(Hi[:,1:K_dims[3],1:K_dims[3]])
	if it_solver || prod(K_dims) > itslv_thresh
		Vout = zeros(T,prod(K_dims))
		function K_matfree(V::AbstractArray{S,1};Gi=Gtemp::AbstractArray{S,3},Hi=Htemp::AbstractArray{S,3},K_dims=K_dims::NTuple{3,Int},Amid_tensor=Amid_tensor::AbstractArray{S,4},Vout=Vout::AbstractArray{S,1}) where S<:Number
			Hrshp = reshape(Vout,K_dims)
			@tensoropt((a,c,d,f), Hrshp[a,b,c] = Gi[y,a,d]*Hi[z,c,f]*Amid_tensor[y,b,e,z]*reshape(V,K_dims)[d,e,f] + Gi[y,d,a]*Hi[z,f,c]*Amid_tensor[y,e,b,z]*reshape(V,K_dims)[d,e,f])
			return 0.5*Vout
		end
		r = eigsolve(LinearMap{T}(K_matfree,prod(K_dims);issymmetric = true),copy(V0[:]),1,:SR,issymmetric=true,tol=tol,maxiter=maxiter)
#		r = lobpcg(LinearMap(K_matfree,prod(K_dims);ishermitian = true),false,copy(V0[:]),3;maxiter=maxiter,tol=tol)
#		return r.λ[1]::Float64, reshape(r.X[:,1],K_dims)::Array{T,3}
#		@assert abs(imag(r[1][1])) <1e-12
#		@assert norm(imag.(r[2])) <1e-12
		return real(r[1][1])::Float64, reshape(real.(r[2][1]),K_dims)::Array{T,3}
	else
		K = K_full(Gtemp,Htemp,Amid_tensor)
		F = eigen(K,1:1)
		return F.values[1]::Float64,reshape(F.vectors[:,1],K_dims)::Array{T,3}
	end	
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

"""
Solve Ax=b using the ALS algorithm where A is given as `TToperator` and `b`, `tt_start` are `TTvector`.
The ranks of the solution is the same as `tt_start`.
`sweep_count` is the number of total sweeps in the ALS.
"""
function dmrg_linsolv(A :: TToperator{T}, b :: TTvector{T}, tt_start :: TTvector{T};sweep_count=2,N=2,tol=1e-12::Float64,rmax=isqrt(prod(tt_start.ttv_dims)),it_solver=false,
	linsolv_maxiter=200::Int64, #maximum of iterations for the iterative solver
	linsolv_tol=max(sqrt(tol),1e-8)::Float64, #tolerance of the iterative linear solver
	itslv_thresh=256::Int #switch from full to iterative
	) where {T<:Number}
	# als finds the minimum of the operator J:1/2*<Ax,Ax> - <x,b>
	# input:
	# 	A: the tensor operator in its tensor train format
	#   b: the tensor in its tensor train format
	#	tt_start: start value in its tensor train format
	#	tt_opt: stationary point of J up to tolerated rank opt_rks
	# 			in its tensor train format

	# Initialize the to be returned tensor in its tensor train format
	d = b.N
	if N==1
		tt_start = tt_up_rks(tt_start,rmax)
	end
	tt_opt = orthogonalize(tt_start)
	dims = tt_start.ttv_dims

	# Initialize the arrays of G and G_b
	G = Array{Array{T,3},1}(undef, d+1-N)
	G_b = Array{Array{T,2},1}(undef, d+1-N)
	Amid_list = Array{Array{T,4},1}(undef, d+1-N)
	bmid_list = Array{Array{T,3},1}(undef, d+1-N)

	rks = r_and_d_to_rks(vcat(1,rmax*ones(Int,d-1),1),dims;rmax=rmax)
	# Initialize G[1], G_b[1], H[d] and H_b[d]
	for i in 1:d+1-N
		G[i] = zeros(T,A.tto_rks[i],rks[i],rks[i])
		G_b[i] = zeros(T,rks[i],b.ttv_rks[i])
		Amid_list[i] = Amid(A,i,i+N-1)
		bmid_list[i] = b_mid(b,i,i+N-1)
	end
	G[1] = ones(T,size(G[1]))
	G_b[1] = ones(T,size(G_b[1]))

	#Initialize H and H_b
	H = init_H(tt_opt,A,N,rmax)
	H_b = init_Hb(tt_opt,b,N,rmax)

	#Initialize V0
	V0 = b_mid(tt_opt,1,N)
	V = zeros(T,tt_opt.ttv_rks[1],prod(tt_opt.ttv_dims[1:N]),tt_opt.ttv_rks[N+1])
	nsweeps = 0 #sweeps counter
	while nsweeps < sweep_count
		nsweeps+=1
		# First half sweep
		for i = 1:d-N
			println("Forward sweep: core optimization $i out of $(d+1-N)")
			Gi = @view(G[i][:,1:tt_opt.ttv_rks[i],1:tt_opt.ttv_rks[i]])
			G_bi = @view(G_b[i][1:tt_opt.ttv_rks[i],:])
			Hi = @view(H[i][:,1:tt_opt.ttv_rks[i+N],1:tt_opt.ttv_rks[i+N]])
			H_bi = @view(H_b[i][1:tt_opt.ttv_rks[i+N],:])
			# Define V as solution of K*x=Pb in x
			V = Ksolve(Gi,G_bi,Hi,H_bi,Amid_list[i],bmid_list[i],V0;it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh=itslv_thresh)
			println("solved")
			tt_opt, V = right_core_move(tt_opt,V,i,tol,rmax)

			#Update the next initialization
			@tensor W[αk,J,ik,γk] := V[αk,J,βk]*tt_opt.ttv_vec[i+N][ik,βk,γk]
			V0 = copy(reshape(W,size(W,1),:,size(W,4)))
			println("V0")

			#update G,G_b
			Gip = @view(G[i+1][:,1:tt_opt.ttv_rks[i+1],1:tt_opt.ttv_rks[i+1]])
			G_bip = @view(G_b[i+1][1:tt_opt.ttv_rks[i+1],:])
			update_G!(tt_opt.ttv_vec[i],A.tto_vec[i],Gi,Gip)
			update_Gb!(tt_opt.ttv_vec[i],b.ttv_vec[i],G_bi,G_bip)
		end

		if nsweeps == sweep_count
			return tt_opt
		else
			nsweeps+=1
			# Second half sweep
			for i = (d+1-N):(-1):2
				println("Backward sweep: core optimization $i out of $(d+1-N)")
				# Define V as solution of K*x=Pb in x
				Gi = @view(G[i][:,1:tt_opt.ttv_rks[i],1:tt_opt.ttv_rks[i]])
				G_bi = @view(G_b[i][1:tt_opt.ttv_rks[i],:])
				Hi = @view(H[i][:,1:tt_opt.ttv_rks[i+N],1:tt_opt.ttv_rks[i+N]])
				H_bi = @view(H_b[i][1:tt_opt.ttv_rks[i+N],:])
				# Define V as solution of K*x=Pb in x
				V = Ksolve(Gi,G_bi,Hi,H_bi,Amid_list[i],bmid_list[i],V0;it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh=itslv_thresh)
				tt_opt,V = left_core_move(tt_opt,V,i+N-1,tol,rmax)
				#update the initialization
				@tensor W[αk,ik,J,γk] := V[βk,J,γk]*tt_opt.ttv_vec[i-1][ik,αk,βk]
				V0 = copy(reshape(W,size(W,1),:,size(W,4)))

				#update H[i-1]
				Him = @view(H[i-1][:,1:tt_opt.ttv_rks[i+N-1],1:tt_opt.ttv_rks[i+N-1]])
				H_bim = @view(H_b[i-1][1:tt_opt.ttv_rks[i+N-1],:])
				update_H!(tt_opt.ttv_vec[i+N-1],A.tto_vec[i+N-1],Hi,Him)
				update_Hb!(tt_opt.ttv_vec[i+N-1],b.ttv_vec[i+N-1],H_bi,H_bim)
			end
		end
	end
	#last step to complete the sweep
	Gi = @view(G[1][:,1:tt_opt.ttv_rks[1],1:tt_opt.ttv_rks[1]])
	G_bi = @view(G_b[1][1:tt_opt.ttv_rks[1],:])
	Hi = @view(H[1][:,1:tt_opt.ttv_rks[1+N],1:tt_opt.ttv_rks[1+N]])
	H_bi = @view(H_b[1][1:tt_opt.ttv_rks[1+N],:])
	# Define V as solution of K*x=Pb in x
	V = Ksolve(Gi,G_bi,Hi,H_bi,Amid_list[1],bmid_list[1],V0;it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh=itslv_thresh)
	for i in N:-1:2
		tt_opt, V = left_core_move(tt_opt,V,i,tol,rmax)
	end
	tt_opt.ttv_vec[1] = permutedims(reshape(V,1,tt_opt.ttv_dims[1],:),(2,1,3))
	tt_opt.ttv_ot[1] = 0
	return tt_opt
end

"""
Returns the lowest eigenvalue of A by minimizing the Rayleigh quotient in the ALS algorithm.

The ranks can be increased in the course of the ALS: if `sweep_schedule[k] ≤ i <sweep_schedule[k+1]` is the current number of sweeps then the ranks is given by `rmax_schedule[k]`.
"""
function dmrg_eigsolv(A :: TToperator{T},
	tt_start :: TTvector{T} ; #TT initial guess
	N=2::Integer, #Number of open sites, N=1 is one-site DMRG, N=2 is two-site DMRG...
	tol=1e-12::Float64, #truncation in left or right core move (doesn't matter for N=1)
	sweep_schedule=[2]::Array{Int64,1}, #Number of sweeps for each bond dimension in rmax_schedule
	rmax_schedule=[isqrt(prod(tt_start.ttv_dims))]::Array{Int64,1}, #maximum rank in sweep_schedule
	it_solver=false::Bool, #linear solver for the microstep
	linsolv_maxiter=200::Int64, #maximum of iterations for the iterative solver
	linsolv_tol=max(sqrt(tol),1e-8)::Float64, #tolerance of the iterative linear solver
	itslv_thresh=256::Int #switch from full to iterative
	)  where {T<:Number} 
	@assert(length(rmax_schedule)==length(sweep_schedule),"Sweep schedule error")	

	d = tt_start.N
	# Initialize the to be returned tensor in its tensor train format
	tt_opt = orthogonalize(tt_start)
	dims = tt_start.ttv_dims
	rmax = maximum(rmax_schedule)
	rks = r_and_d_to_rks(vcat(1,rmax*ones(Int,d-1),1),dims;rmax=rmax)
	# Initialize the output objects
	E = Float64[]
	r_hist = Int64[]

	# Initialize the arrays of G
	G = Array{Array{T,3},1}(undef, d+1-N)
	Amid_list = Array{Array{T,4},1}(undef, d+1-N)

	# Initialize G
	for i in 1:d+1-N
		G[i] = zeros(T,A.tto_rks[i],rks[i],rks[i])
		Amid_list[i] = Amid(A,i,i+N-1)
	end
	G[1] = ones(T,size(G[1]))
	#Initialize H
	H = init_H(tt_opt,A,N,rmax)
	#Initialize V0
	V0 = b_mid(tt_opt,1,N)
	V = zeros(T,size(V0))

	nsweeps = 0 #sweeps counter
	i_schedule = 1
	while i_schedule <= length(sweep_schedule) 
		nsweeps+=1
		println("Macro-iteration $nsweeps; bond dimension $(rmax_schedule[i_schedule])")

		if nsweeps == sweep_schedule[i_schedule]
			i_schedule+=1
			if i_schedule > length(sweep_schedule)
				#last step to complete the sweep
				λ,V = K_eigmin(G[1],H[1],V0,Amid_list[1] ;it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh=itslv_thresh)
				println("Eigenvalue: $λ")
				E = vcat(E,λ)
				r_hist = vcat(r_hist,maximum(tt_opt.ttv_rks))
				for i in N:-1:2
					tt_opt, V = left_core_move(tt_opt,V,i,tol,rmax_schedule[end])
				end
				tt_opt.ttv_vec[1] = permutedims(reshape(V,1,tt_opt.ttv_dims[1],:),(2,1,3))
				tt_opt.ttv_ot[1] = 0
				return E::Array{Float64,1}, tt_opt::TTvector{T}, r_hist::Array{Int,1}
			end
		end
		# First half sweep
		for i = 1:(d-N)
			println("Forward sweep: core optimization $i out of $(d-N)")
			# Define V as solution of K V= λ V for smallest λ
			λ,V = K_eigmin(G[i],H[i],V0, Amid_list[i]; it_solver=it_solver, maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh=itslv_thresh)
			println("Eigenvalue: $λ")
			E = vcat(E,λ)
			tt_opt, V = right_core_move(tt_opt,V,i,tol,rmax_schedule[i_schedule])
			r_hist = vcat(r_hist,maximum(tt_opt.ttv_rks))
			#Update the next initialization
			@tensor W[αk,J,ik,γk] := V[αk,J,βk]*tt_opt.ttv_vec[i+N][ik,βk,γk]
			V0 = copy(reshape(W,size(W,1),:,size(W,4)))
			# Update G[i+1],G_b[i+1]
			Gi = @view(G[i][:,1:tt_opt.ttv_rks[i],1:tt_opt.ttv_rks[i]])
			Gip = @view(G[i+1][:,1:tt_opt.ttv_rks[i+1],1:tt_opt.ttv_rks[i+1]])
			update_G!(tt_opt.ttv_vec[i],A.tto_vec[i],Gi,Gip)
		end

		# Second half sweep
		for i = d-N+1:(-1):2
			println("Backward sweep: core optimization $(i-1) out of $(d-N)")
			# Define V as solution of K*x=P2b in x
			λ,V = K_eigmin(G[i],H[i],V0,Amid_list[i] ;it_solver=it_solver,maxiter=linsolv_maxiter,tol=linsolv_tol,itslv_thresh=itslv_thresh)
			println("Eigenvalue: $λ")
			E = vcat(E,λ)
			tt_opt,V = left_core_move(tt_opt,V,i+N-1,tol,rmax_schedule[i_schedule])
			r_hist = vcat(r_hist,maximum(tt_opt.ttv_rks))
			#update the initialization
			@tensor W[αk,ik,J,γk] := V[βk,J,γk]*tt_opt.ttv_vec[i-1][ik,αk,βk]
			V0 = copy(reshape(W,size(W,1),:,size(W,4)))
			# Update H[i-1]
			Hi = @view(H[i][:,1:tt_opt.ttv_rks[i+N],1:tt_opt.ttv_rks[i+N]])
			Him = @view(H[i-1][:,1:tt_opt.ttv_rks[i+N-1],1:tt_opt.ttv_rks[i+N-1]])
			update_H!(tt_opt.ttv_vec[i+N-1],A.tto_vec[i+N-1],Hi,Him)
		end
	end
	return E::Array{Float64,1}, tt_opt::TTvector{T}, r_hist::Array{Int,1}
end

