using Random
using LinearAlgebra
using Base.Threads
using TensorOperations
import Base.eltype
import Base.copy

"""
Tensor rings (TR), aka periodic MPS. TR are stored in 'vec' as order three tensors of size A_k ∈ R^{n_k × r_{k-1} × r_k} with r_0 = r_d.
"""
abstract type AbstractTRvector end

struct TRvector{T<:Number,M} <:AbstractTRvector
	N :: Int
	ttv_vec :: Array{Array{T,3},1}
	ttv_dims :: NTuple{M,Int64}
	ttv_rks :: Array{Int64,1}
	ttv_ot :: Array{Int64,1}
	function TRvector{T,M}(N,vec,dims,rks,ot) where {T,M}
		@assert M isa Int64
		new{T,M}(N,vec,dims,rks,ot)
	end
end

Base.eltype(::TRvector{T,N}) where {T<:Number,N} = T 

#returns the tensor represented by 'x_tr'
function tr_to_tensor(x_tr::TRvector{T,N}) where {T<:Number,N}
	d = x_tr.N
	# Initialize the to be returned tensor
	tensor = zeros(T, x_tr.ttv_dims...)
	rmax = maximum(x_tr.ttv_rks)
	# Fill in the tensor for every t=(x_1,...,x_d)
	for t in CartesianIndices(tensor)
		a = collect(Tuple(t))
		curr = zeros(T,rmax,rmax)
		curr[1:x_tr.ttv_rks[d],1:x_tr.ttv_rks[d+1]] = copy(x_tr.ttv_vec[d][a[d],:,:])
		for i = d-1:-1:1
			curr[1:x_tr.ttv_rks[i],1:x_tr.ttv_rks[i+1]] = x_tr.ttv_vec[i][a[i],:,:]*curr[1:x_tr.ttv_rks[i+1],1:x_tr.ttv_rks[i+2]]
		end
		tensor[t] = tr(curr)
	end
	return tensor
end

#returns a zero TRvector
function zeros_tr(::Type{T},dims::NTuple{N,Int64},rks) where {T,N}
	@assert length(rks)==length(dims)+1
	vec = [zeros(T,dims[i],rks[i],rks[i+1]) for i in eachindex(dims)]	
	return TRvector{T,N}(N,vec,dims,deepcopy(rks),zeros(Int,N))
end

function Base.copy(x_tr::TRvector{T,N}) where {T<:Number,N}
	return TRvector{T,N}(x_tr.N,copy(x_tr.ttv_vec),x_tr.ttv_dims,copy(x_tr.ttv_rks),copy(x_tr.ttv_ot))
end

"""
Returns a random 'TRvector' of dimensions 'dims' and ranks 'rks'
"""
function rand_tr(dims,rks;normalise=true)
	return rand_tr(Float64,dims,rks;normalise=normalise)
end

function rand_tr(::Type{T},dims,rks;normalise=true) where T
	@assert length(dims)+1==length(rks) "Incompatible dimensions and ranks"
	@assert rks[1] == rks[end] "Incompatible first and last ranks"
	d = length(dims)
	if normalise
		tt_vec = [randn(T,dims[i],rks[i],rks[i+1])/sqrt(dims[i]*rks[i]*rks[i+1]) for i in eachindex(dims)]
	else
		tt_vec = [randn(T,dims[i],rks[i],rks[i+1]) for i in eachindex(dims)]
	end
	return TRvector{T,d}(d,tt_vec,dims,copy(rks),zeros(Int64,d))
end

"""
Returns a random 'TRvector' of dimensions 'dims' and maximal rank 'rmax'
"""
function rand_tr(dims,rmax::Int;normalise=true)
	return rand_tr(dims,rmax*ones(Int,length(dims)+1);normalise=normalise)
end