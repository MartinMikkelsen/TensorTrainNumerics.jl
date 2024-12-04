using Random
using LinearAlgebra
using Base.Threads
using IterativeSolvers
using TensorOperations
import Base.isempty
import Base.eltype
import Base.copy
import Base.complex

"""
TT constructor of a tensor
``C[\\mu_1,…,\\mu_d] = A_1[1][\\mu_1]*⋯*A_d[d][\\mu_d] \\in \\mathbb{K}^{n_1 \\times ... \\times n_d}``
The following properties are stored
	* ttv_vec: the TT cores A_k as a list of 3-order tensors ``(A_1,...,A_d)`` where ``A_k = A_k[\\mu_k,\\alpha_{k-1},\\alpha_k]``, ``1 \\leq \\alpha_{k-1} \\leq r_{k-1}``, ``1 \\leq \\alpha_{k} \\leq r_{k}``, ``1 \\leq \\mu_k \\leq n_k``
	* ttv_dims: the dimension of the tensor along each mode
	* ttv_rks: the TT ranks ``(r_0,...,r_d)`` where ``r_0=r_d=1``
	* ttv_ot: the orthogonality of the TT where 
		* ttv_ot[i] = 1 iff ``A_i`` is left-orthogonal *i.e.* ``\\sum_{\\mu_i} A_i[\\mu_i]^T A_i[\\mu_i] = I_{r_i}``
		* ttv_ot[i] = -1 iff ``A_i`` is right-orthogonal *i.e.* ``\\sum_{\\mu_i} A_i[\\mu_i] A_i[\\mu_i]^T = I_{r_{i-1}}``
		* ttv_ot[i] = 0 if nothing is known
"""
abstract type AbstractTTvector end
struct TTvector{T<:Number,M} <: AbstractTTvector
	N :: Int64
	ttv_vec :: Vector{Array{T,3}}
	ttv_dims :: NTuple{M,Int64}
	ttv_rks :: Vector{Int64}
	ttv_ot :: Vector{Int64}
	function TTvector{T,M}(N,vec,dims,rks,ot) where {T,M}
		@assert M isa Int64
		new{T,M}(N,vec,dims,rks,ot)
	end
end

Base.eltype(::TTvector{T,N}) where {T<:Number,N} = T 


"""	
TT constructor of a matrix ``M[i_1,..,i_d;j_1,..,j_d] \\in \\mathbb{K}^{n_1 \\cdots n_d \\times n_1 \\cdots n_d}`` where 
``M[i_1,..,i_d;j_1,..,j_d] = A_1[i_1,j_1] ... A_L[i_d,j_d]``
The following properties are stored
	* tto_vec: the TT cores A_k as a list of 4-order tensors ``(A_1,...,A_d)`` of dimensions A_k \\in \\mathbb{K}^{n_k × n_k × r_{k-1} × r_k}
	* ttv_dims: the dimension of the tensor along each mode
	* ttv_rks: the TT ranks ``(r_0,...,r_d)`` where ``r_0=r_d=1``
"""
abstract type AbstractTToperator end
struct TToperator{T<:Number,M} <: AbstractTToperator
	N :: Int64
	tto_vec :: Array{Array{T,4},1}
	tto_dims :: NTuple{M,Int64}
	tto_rks :: Array{Int64,1}
	tto_ot :: Array{Int64,1}
	function TToperator{T,M}(N,vec,dims,rks,ot) where {T,M}
		@assert M isa Int64
		new{T,M}(N,vec,dims,rks,ot)
	end
end

Base.eltype(::TToperator{T,M}) where {T,M} = T 
function Base.complex(A::TToperator{T,M}) where {T,M}
	return TToperator{Complex{T},M}(A.N,complex(A.tto_vec),A.tto_dims,A.tto_rks,A.tto_ot)
end

function Base.complex(v::TTvector{T,N}) where {T,N}
	return TTvector{Complex{T},N}(v.N,complex(v.ttv_vec),v.ttv_dims,v.ttv_rks,v.ttv_ot)
end

function QTTvector(vec::Vector{<:Array{<:Number, 3}}, rks::Vector{Int64}, ot::Vector{Int64})
    T = eltype(eltype(vec))
    N = length(vec)
    dims = ntuple(_ -> 2, N)  
    for core in vec
        @assert size(core, 1) == 2 "Each core must have physical dimension 2."
    end
    return TTvector{T, N}(N, vec, dims, rks, ot)
end

function is_qtt(tt::TTvector)
    all(dim == 2 for dim in tt.ttv_dims)
end

function QTToperator(vec::Vector{Array{T,4}}, rks::Vector{Int64}, ot::Vector{Int64}) where {T}
    N = length(vec)
    dims = ntuple(_ -> 2, N)
    for core in vec
        @assert size(core, 1) == 2 && size(core, 2) == 2 "Each core must have physical dimension 2."
    end
    return TToperator{T,N}(N, vec, dims, rks, ot)
end

function is_qtt_operator(op::TToperator)
    all(dim == 2 for dim in op.tto_dims)
end

"""
returns a zero TTvector with dimensions `dims` and ranks `rks`
"""
function zeros_tt(dims,rks;ot=zeros(Int64,length(dims)))
	return zeros_tt(Float64,dims,rks;ot=ot)
end

function zeros_tt(::Type{T},dims::NTuple{N,Int64},rks;ot=zeros(Int64,length(dims))) where {T,N}
	#@assert length(dims)+1==length(rks) "Dimensions and ranks are not compatible"
	tt_vec = [zeros(T,dims[i],rks[i],rks[i+1]) for i in eachindex(dims)]
	return TTvector{T,N}(N,tt_vec,dims,deepcopy(rks),deepcopy(ot))
end

"""
returns the ones tensor in TT format
"""
function ones_tt(dims)
	return ones_tt(Float64,dims)
end

function ones_tt(::Type{T},dims) where T
	N = length(dims)
	vec = [ones(T,n,1,1) for n in dims]
	rks = ones(Int64,N+1)
	ot = zeros(Int64,N)
	return TTvector{T,N}(N,vec,dims,rks,ot)
end

function ones_tt(n::Integer,d::Integer)
	dims = n*ones(Int64,d)
	return ones_tt(dims)
end

"""
returns a zero TToperator with dimensions `dims` and ranks `rks`
"""
function zeros_tto(dims,rks)
	return zeros_tto(Float64,dims,rks)
end

function zeros_tto(::Type{T},dims::NTuple{N,Int64},rks)  where {T,N}
	@assert length(dims)+1==length(rks) "Dimensions and ranks are not compatible"
	vec = [zeros(T,dims[i],dims[i],rks[i],rks[i+1]) for i in eachindex(dims)]
	return TToperator{T,N}(N,vec,dims,rks,zeros(Int64,N))
end

#returns partial isometry Q ∈ R^{n x m}
function rand_orthogonal(n,m;T=Float64)
    N = max(n,m)
    q,r = qr(rand(T,N,N))
    return Matrix(q)[1:n,1:m]
end

"""
Returns a random TTvector with dimensions `dims` and ranks `rks`
"""
function rand_tt(dims,rks;normalise=false,orthogonal=false)
	return rand_tt(Float64,dims,rks;normalise=normalise,orthogonal=orthogonal)
end

function rand_tt(::Type{T},dims,rks;normalise=false,orthogonal=false) where T
	y = zeros_tt(T,dims,rks)
	@simd for i in eachindex(y.ttv_vec)
		y.ttv_vec[i] = randn(T,dims[i],rks[i],rks[i+1])
		if normalise
			y.ttv_vec[i] *= 1/sqrt(dims[i]*rks[i+1])
		if orthogonal
			q,_ = qr(reshape(permutedims(y.ttv_vec[i],(1,3,2)),dims[i]*rks[i+1],rks[i]))
			y.ttv_vec[i] = permutedims(reshape(Matrix(q),dims[i],rks[i+1],rks[i]),(1,3,2))
		end
		end
	end
	return y
end

"""
Returns a random TTvector with dimensions `dims` and maximal rank `rmax`
"""
function rand_tt(dims,rmax::Int;T=Float64,normalise=false,orthogonal=false)
	d = length(dims)
	tt_vec = Vector{Array{T,3}}(undef,d)
	rks = rmax*ones(Int,d+1)
	rks = r_and_d_to_rks(rks,dims;rmax=rmax)
	for i in eachindex(tt_vec) 
		tt_vec[i] = randn(T,dims[i],rks[i],rks[i+1])
		if normalise
			tt_vec[i] *= 1/sqrt(dims[i]*rks[i+1])
		end
		if orthogonal
			q,_ = qr(reshape(permutedims(tt_vec[i],(1,3,2)),dims[i]*rks[i+1],rks[i]))
			tt_vec[i] = reshape(permutedims(Matrix(q),(1,2,3)),dims[i],rks[i],rks[i+1])
		end
	end
	return TTvector{T,d}(d,tt_vec,dims,rks,zeros(Int,d))
end

function rand_tt(x_tt::TTvector{T,N};ε=convert(T,1e-3)) where {T,N}
	tt_vec = copy(x_tt.ttv_vec)
	for i in eachindex(x_tt.ttv_vec)
		tt_vec[i] += ε*randn(x_tt.ttv_dims[i],x_tt.ttv_rks[i],x_tt.ttv_rks[i+1])
	end
	return TTvector{T,N}(N,tt_vec,x_tt.ttv_dims,x_tt.ttv_rks,zeros(Int,N))
end

function Base.copy(x_tt::TTvector{T,N}) where {T<:Number,N}
	y_tt = zeros_tt(T,x_tt.ttv_dims,x_tt.ttv_rks;ot=x_tt.ttv_ot)
	@threads for i in eachindex(x_tt.ttv_dims)
		y_tt.ttv_vec[i] = copy(x_tt.ttv_vec[i])
	end
	return y_tt
end

"""
TT decomposition by the Hierarchical SVD algorithm 
	* Oseledets, I. V. (2011). Tensor-train decomposition. *SIAM Journal on Scientific Computing*, 33(5), 2295-2317.
	* Schollwöck, U. (2011). The density-matrix renormalization group in the age of matrix product states. *Annals of physics*, 326(1), 96-192.
The *root* of the TT decomposition is at index *i.e.* ``A_i`` for ``i < index`` are left-orthogonal and ``A_i`` for ``i > index`` are right-orthogonal. Singular values lower than tol are discarded.
"""
function ttv_decomp(tensor::Array{T,d};index=1,tol=1e-12) where {T<:Number,d}
	# Decomposes a tensor into its tensor train with core matrices at i=index
	dims = size(tensor) #dims = [n_1,...,n_d]
	ttv_vec = Array{Array{T,3}}(undef,d)
	# ttv_ot[i]= -1 if i < index
	# ttv_ot[i] = 0 if i = index
	# ttv_ot[i] = 1 if i > index
	ttv_ot = -ones(Int64,d)
	ttv_ot[index] = 0
	if index < d
		ttv_ot[index+1:d] = ones(d-index)
	end
	rks = ones(Int64, d+1) 
	tensor_curr = tensor
	# Calculate ttv_vec[i] for i < index
	for i = 1 : (index - 1)
		# Reshape the currently left tensor
		tensor_curr = reshape(tensor_curr, Int(rks[i] * dims[i]), :)
		# Perform the singular value decomposition
		u, s, v = svd(tensor_curr)
		# Define the i-th rank
		rks[i+1] = length(s[s .>= tol])
		# Initialize ttv_vec[i]
		ttv_vec[i] = zeros(T,dims[i],rks[i],rks[i+1])
		# Fill in the ttv_vec[i]
		for x = 1 : dims[i]
			ttv_vec[i][x, :, :] = u[(rks[i]*(x-1) + 1):(rks[i]*x), :]
		end
		# Update the currently left tensor
		tensor_curr = Diagonal(s[1:rks[i+1]])*v'[1:rks[i+1],:]
	end

	# Calculate ttv_vec[i] for i > index
	if index < d
		for i = d : (-1) : (index + 1)
			# Reshape the currently left tensor
			tensor_curr = reshape(tensor_curr, :, dims[i] * rks[i+1])
			# Perform the singular value decomposition
			u, s, v = svd(tensor_curr)
			# Define the (i-1)-th rank
			rks[i]=length(s[s .>= tol])
			# Initialize ttv_vec[i]
			ttv_vec[i] = zeros(T, dims[i], rks[i], rks[i+1])
			# Fill in the ttv_vec[i]
			i_vec = zeros(Int,rks[i+1])
			for x = 1 : dims[i]
				i_vec = dims[i]*((1:rks[i+1])-ones(Int, rks[i+1])) + x*ones(Int,rks[i+1])
				ttv_vec[i][x, :, :] = v'[1:rks[i],i_vec] #(rks[i+1]*(x-1)+1):(rks[i+1]*x)
			end
			# Update the current left tensor
			tensor_curr = u[:,1:rks[i]]*Diagonal(s[1:rks[i]])
		end
	end
	# Calculate ttv_vec[i] for i = index
	# Reshape the current left tensor
	tensor_curr = reshape(tensor_curr, Int(dims[index]*rks[index]),:)
	# Initialize ttv_vec[i]
	ttv_vec[index] = zeros(T, dims[index], rks[index], rks[index+1])
	# Fill in the ttv_vec[i]
	for x = 1 : dims[index]
		ttv_vec[index][x, :, :] =
		tensor_curr[Int(rks[index]*(x-1) + 1):Int(rks[index]*x), 1:rks[index+1]]
	end

	# Define the return value as a TTvector
	return TTvector{T,d}(d,ttv_vec, dims, rks, ttv_ot)
end

"""
Returns the tensor corresponding to x_tt
"""
function ttv_to_tensor(x_tt :: TTvector{T,N}) where {T<:Number,N}
	d = length(x_tt.ttv_dims)
	r_max = maximum(x_tt.ttv_rks)
	# Initialize the to be returned tensor
	tensor = zeros(T, x_tt.ttv_dims)
	# Fill in the tensor for every t=(x_1,...,x_d)
	@simd for t in CartesianIndices(tensor)
		curr = ones(T,r_max)
		a = collect(Tuple(t))
		for i = d:-1:1
			curr[1:x_tt.ttv_rks[i]] = x_tt.ttv_vec[i][a[i],:,:]*curr[1:x_tt.ttv_rks[i+1]]
		end
		tensor[t] = curr[1]
	end
	return tensor
end

"""
Transforms a TToperator into a TTvector
"""
function tto_to_ttv(A::TToperator{T,N}) where {T<:Number,N}
	d = A.N
	xtt_vec = Array{Array{T,3},1}(undef,d)
	A_rks = A.tto_rks
	for i in eachindex(xtt_vec)
		xtt_vec[i] = reshape(A.tto_vec[i],A.tto_dims[i]^2,A_rks[i],A_rks[i+1])
	end
	return TTvector{T,N}(d,xtt_vec,A.tto_dims.^2,A.tto_rks,A.tto_ot)
end

"""
Transforms a TTvector (coming from a TToperator) into a TToperator
"""
function ttv_to_tto(x::TTvector{T,N}) where {T<:Number,N}
	@assert(isqrt.(x.ttv_dims).^2 == x.ttv_dims, DimensionMismatch)
	d = x.N
	Att_vec = Array{Array{T,4},1}(undef,d)
	x_rks = x.ttv_rks
	A_dims = isqrt.(x.ttv_dims)
	for i in eachindex(A_dims)
		Att_vec[i] = reshape(x.ttv_vec[i],A_dims[i],A_dims[i],x_rks[i],x_rks[i+1])
	end
	return TToperator{T,N}(d,Att_vec,A_dims,x.ttv_rks,x.ttv_ot)
end

"""
Returns the TT decomposition of a matrix using the HSVD algorithm
"""
function tto_decomp(tensor::Array{T,N}; index=1) where {T<:Number,N}
	# Decomposes a tensor operator into its tensor train
	# with core matrices at i=index
	# The tensor is given as tensor[x_1,...,x_d,y_1,...,y_d]
	d = Int(ndims(tensor)/2)
	tto_dims = size(tensor)[1:d]
	dims_sq = tto_dims.^2
	# The tensor is reorder  into tensor[x_1,y_1,...,x_d,y_d],
	# reshaped into tensor[(x_1,y_1),...,(x_d,y_d)]
	# and decomposed into its tensor train with core matrices at i= index
	index_sorted = reshape(Transpose(reshape(1:(2*d),:,2)),1,:)
	ttv = ttv_decomp(reshape(permutedims(tensor,index_sorted),(dims_sq[1:(end-1)]...), :); index=index)
	# Define the array of ranks [r_0=1,r_1,...,r_d]
	rks = ttv.ttv_rks
	# Initialize tto_vec
	tto_vec = Array{Array{T}}(undef,d)
	# Fill in tto_vec
	for i = 1:d
		# Initialize tto_vec[i]
		tto_vec[i] = zeros(T, tto_dims[i], tto_dims[i], rks[i], rks[i+1])
		# Fill in tto_vec[i]
		tto_vec[i] = reshape(ttv.ttv_vec[i], tto_dims[i], tto_dims[i], :, rks[i+1])
	end
	return TToperator{T,d}(d,tto_vec, tto_dims, rks, ttv.ttv_ot)
end

function tto_to_tensor(tto :: TToperator{T,N}) where {T<:Number,N}
	d = tto.N
	# Define the array of ranks [r_0=1,r_1,...,r_d]
	rks = tto.tto_rks
	r_max = maximum(rks)
	# The tensor has dimensions [n_1,...,n_d,n_1,...,n_d]
	tensor = zeros(T,(tto.tto_dims...,tto.tto_dims...))
	# Fill in the tensor for every t=(x_1,...,x_d,y_1,...,y_d)
	curr = ones(T,r_max)
	@simd for t in CartesianIndices(tensor)
		curr[1] = one(T)
		for i = d:-1:1
			curr[1:rks[i]] = tto.tto_vec[i][t[i], t[d + i], :, :]*curr[1:rks[i+1]]
		end
		tensor[t] = curr[1]
	end
	return tensor
end

#TTO representation of the identity matrix
function id_tto(d;n_dim=2)
	return id_tto(Float64,d;n_dim=n_dim)
end

function id_tto(::Type{T},d;n_dim=2) where {T}
	dims = Tuple(n_dim*ones(Int64,d))
	A = Array{Array{T,4},1}(undef,d)
	for j in 1:d
		A[j] = zeros(T,2,2,1,1)
		A[j][:,:,1,1] = Matrix{T}(I,2,2)
	end
	return TToperator{T,d}(d,A,dims,ones(Int64,d+1),zeros(d))
end

function rand_tto(dims,rmax::Int;T=Float64)
	d = length(dims)
	tt_vec = Vector{Array{T,4}}(undef,d)
	rks = ones(Int,d+1)
	for i in eachindex(tt_vec) 
		ri = min(prod(dims[1:i-1]),prod(dims[i:d]),rmax)
		rip = min(prod(dims[1:i]),prod(dims[i+1:d]),rmax)
		rks[i+1] = rip
		tt_vec[i] = randn(T,dims[i],dims[i],ri,rip)
	end
	return TToperator{T,d}(d,tt_vec,dims,rks,zeros(Int,d))
end

function visualize(tt::TTvector)
    N = tt.N
    dims = collect(tt.ttv_dims)
    ranks = tt.ttv_rks

    max_rank_len = maximum(length.(string.(ranks)))
    max_dim_len = maximum(length.(string.(dims)))

    rwidth = max(max_rank_len, 2)  # Minimum width of 2 for readability
    dwidth = max(max_dim_len, 1)

    seg_length = rwidth + 6 + rwidth  # '-- C --' is 6 characters

    line1 = lpad(string(ranks[1]), rwidth)
    line2 = " " ^ length(line1)
    line3 = " " ^ length(line1)

    for i in 1:N
        segment = "-- • --"
        rank_right = lpad(string(ranks[i+1]), rwidth)
        line1 *= segment * rank_right

        position_C = length(line1) - rwidth - 3  # 'C' is 3 characters before the right rank

        line2_len = length(line2)
        spaces_needed = position_C - line2_len
        line2 *= repeat(" ", spaces_needed-1) * "|"

        dim_str = string(dims[i])
        dim_len = length(dim_str)
        line3_len = length(line3)
        spaces_needed = position_C - line3_len - div(dim_len, 2)
        line3 *= repeat(" ", spaces_needed-1) * dim_str
    end

    diagram = line1 * "\n" * line2 * "\n" * line3

    println(diagram)
end

function visualize(tt::TToperator)
    N = tt.N
    dims = collect(tt.tto_dims)
    ranks = tt.tto_rks

    # Initialize top dimensions line (line0)
    total_length = 0
    line0_chars = []
    positions_C = []

    # Build the first line (ranks and nodes)
    line1 = ""
    for i in 1:N
        # Create rank-node-rank segment
        if i == 1
            segment = " $(ranks[i])-- • --$(ranks[i+1])"
        else
            segment = "-- • --$(ranks[i+1])"
        end
        line1 *= segment

        # Find the position of '•' in the segment
        pos_C = total_length + findfirst(isequal('•'), segment)
        push!(positions_C, pos_C)
        total_length += length(segment)

    end

    # Top Dimensions (line0)
    line0_chars = fill(' ', total_length)
    for i in 1:N
        pos_C = positions_C[i]
        dim_str = "$(dims[i])"
        dim_len = length(dim_str)

        # Center dimension above the vertical line
        start_pos = pos_C - div(dim_len, 2)
        for j in 1:dim_len
            idx = start_pos + j - 1
            if idx >= 1 && idx <= total_length
                line0_chars[idx] = dim_str[j]
            end
        end
    end
    line0 = String(line0_chars)

    # Build the second line (vertical connections to top dimensions)
    line2_chars = fill(' ', total_length)
    for pos_C in positions_C
        line2_chars[pos_C] = '|'
    end
    line2 = String(line2_chars)

    # Reuse the vertical line for line3
    line3 = line2

    # Bottom Dimensions (line4)
    line4_chars = fill(' ', total_length)
    for i in 1:N
        pos_C = positions_C[i]
        dim_str = "$(dims[i])"
        dim_len = length(dim_str)

        # Center dimension under the vertical line
        start_pos = pos_C - div(dim_len, 2)
        for j in 1:dim_len
            idx = start_pos + j - 1
            if idx >= 1 && idx <= total_length
                line4_chars[idx] = dim_str[j]
            end
        end
    end
    line4 = String(line4_chars)

    # Combine all the lines
    diagram = line0 * "\n" * line2 * "\n" * line1 * "\n" * line3 * "\n" * line4

    # Display the diagram
    println(diagram)
end

"""
returns the singular values of the reshaped tensor x[μ_1⋯μ_k;μ_{k+1}⋯μ_d] for all 1≤ k ≤ d
"""
function tt_svdvals(x_tt::TTvector{T,N};tol=1e-14) where {T<:Number,N}
	d = x_tt.N
	Σ = Array{Array{Float64,1},1}(undef,d-1)
	y_tt = orthogonalize(x_tt)
	y_rks = y_tt.ttv_rks
	for j in 1:d-1
		A = zeros(y_tt.ttv_dims[j],y_rks[j],y_tt.ttv_dims[j+1],y_rks[j+2])
		@tensor A[a,b,c,d] = y_tt.ttv_vec[j][a,b,z]*y_tt.ttv_vec[j+1][c,z,d]
		u,s,v = svd(reshape(A,size(A,1)*size(A,2),:),alg=LinearAlgebra.QRIteration())
		Σ[j],y_rks[j+1] = floor(s,tol)
		y_tt.ttv_vec[j+1] = permutedims(reshape(Diagonal(Σ[j])*v'[s.>tol,:],:,y_tt.ttv_dims[j+1],y_rks[j+2]),[2 1 3])
	end
	return Σ
end

function tt2qtt(tt_tensor::TToperator{T,N}, row_dims::Vector{Vector{Int}}, col_dims::Vector{Vector{Int}}, threshold::Float64=0.0) where {T<:Number,N}
    
    qtt_cores = Array{Array{T,4}}(undef, 0)
    tto_rks = [1]
    tto_dims = Int64[]

    # For each core in tt_tensor
    for i in 1:tt_tensor.N

        # Get core, rank_prev, rank_next, row_dim, col_dim
        core = permutedims(tt_tensor.tto_vec[i], (3,1,2,4))  # Now core is (r_{k-1}, n_k_row, n_k_col, r_k)
        rank_prev = tto_rks[end]
        rank_next = tt_tensor.tto_rks[i+1]
        row_dim = tt_tensor.tto_dims[i]
        col_dim = tt_tensor.tto_dims[i]  # Assuming square dimensions

        # Begin splitting
        for j in 1:(length(row_dims[i]) - 1)

            # Update row_dim and col_dim
            row_dim = div(row_dim, row_dims[i][j])
            col_dim = div(col_dim, col_dims[i][j])

            # Reshape and permute core
            core = reshape(core, (rank_prev, row_dims[i][j], row_dim, col_dims[i][j], col_dim, rank_next))
            core = permutedims(core, (1,2,4,3,5,6))  # Now core is (r_{k-1}, row_dims[i][j], col_dims[i][j], row_dim, col_dim, r_k)

            # Reshape core into 2D matrix for SVD
            core_reshaped = reshape(core, (rank_prev * row_dims[i][j] * col_dims[i][j], row_dim * col_dim * rank_next))

            # Compute SVD
            F = svd(core_reshaped; full=false)
            U = F.U
            S = F.S
            Vt = F.Vt

            # Rank reduction
            if threshold != 0.0
                indices = findall(S ./ S[1] .> threshold)
                U = U[:, indices]
                S = S[indices]
                Vt = Vt[indices, :]
            end

            # Update rank
            new_rank = length(S)

            # Reshape U into core and append to qtt_cores
            U_reshaped = reshape(U, (rank_prev, row_dims[i][j], col_dims[i][j], new_rank))
            # Permute back to (n_k_row, n_k_col, r_{k-1}, r_k)
            core_to_append = permutedims(U_reshaped, (2,3,1,4))
            push!(qtt_cores, core_to_append)

            # Update tto_rks
            push!(tto_rks, new_rank)

            # Update tto_dims
            push!(tto_dims, row_dims[i][j])
            push!(tto_dims, col_dims[i][j])

            # Update core for next iteration
            core = Diagonal(S) * Vt
            rank_prev = new_rank
        end

        # For the last QTT core
        core = reshape(core, (rank_prev, row_dim, col_dim, rank_next))
        core_to_append = permutedims(core, (2,3,1,4))
        push!(qtt_cores, core_to_append)

        # Update tto_dims
        push!(tto_dims, row_dim)
        push!(tto_dims, col_dim)

        # Update tto_rks
        push!(tto_rks, rank_next)
    end

    N_qtt = length(qtt_cores)
    M = length(tto_dims)
    tto_ot = zeros(Int64, N_qtt)

    qtt_tensor = TToperator{T,M}(N_qtt, qtt_cores, Tuple(tto_dims), tto_rks, tto_ot)

    return qtt_tensor
end

function tt2qtt(tt_tensor::TTvector{T,N}, dims::Vector{Vector{Int}}, threshold::Float64=0.0) where {T<:Number,N}
    
    qtt_cores = Array{Array{T,3}}(undef, 0)
    ttv_rks = [1]
    ttv_dims = Int64[]

    # For each core in tt_tensor
    for i in 1:tt_tensor.N

        # Get core, rank_prev, rank_next, dim
        core = permutedims(tt_tensor.ttv_vec[i], (2,1,3))  # Now core is (r_{k-1}, n_k, r_k)
        rank_prev = ttv_rks[end]
        rank_next = tt_tensor.ttv_rks[i+1]
        dim = tt_tensor.ttv_dims[i]

        # Begin splitting
        for j in 1:(length(dims[i]) - 1)

            # Update dim
            dim = div(dim, dims[i][j])

            # Reshape and permute core
            core = reshape(core, (rank_prev, dims[i][j], dim, rank_next))
            core = permutedims(core, (1,2,3,4))  # Now core is (r_{k-1}, dims[i][j], dim, r_k)

            # Reshape core into 2D matrix for SVD
            core_reshaped = reshape(core, (rank_prev * dims[i][j], dim * rank_next))

            # Compute SVD
            F = svd(core_reshaped; full=false)
            U = F.U
            S = F.S
            Vt = F.Vt

            # Rank reduction
            if threshold != 0.0
                indices = findall(S ./ S[1] .> threshold)
                U = U[:, indices]
                S = S[indices]
                Vt = Vt[indices, :]
            end

            # Update rank
            new_rank = length(S)

            # Reshape U into core and append to qtt_cores
            U_reshaped = reshape(U, (rank_prev, dims[i][j], new_rank))
            # Permute back to (n_k, r_{k-1}, r_k)
            core_to_append = permutedims(U_reshaped, (2,1,3))
            push!(qtt_cores, core_to_append)

            # Update ttv_rks
            push!(ttv_rks, new_rank)

            # Update ttv_dims
            push!(ttv_dims, dims[i][j])

            # Update core for next iteration
            core = Diagonal(S) * Vt
            rank_prev = new_rank
        end

        # For the last QTT core
        core = reshape(core, (rank_prev, dim, rank_next))
        core_to_append = permutedims(core, (2,1,3))
        push!(qtt_cores, core_to_append)

        # Update ttv_dims
        push!(ttv_dims, dim)

        # Update ttv_rks
        push!(ttv_rks, rank_next)
    end

    N_qtt = length(qtt_cores)
    M = length(ttv_dims)
    ttv_ot = zeros(Int64, N_qtt)

    qtt_tensor = TTvector{T,M}(N_qtt, qtt_cores, Tuple(ttv_dims), ttv_rks, ttv_ot)

    return qtt_tensor
end

function matricize(tt::TToperator{T, M}) where {T, M}
    first_core = tt.tto_vec[1]
    if prod(size(first_core)) != tt.tto_dims[1] * tt.tto_dims[1] * tt.tto_rks[2]
        error("First core size mismatch: expected $(tt.tto_dims[1] * tt.tto_dims[1] * tt.tto_rks[2]), got $(prod(size(first_core)))")
    end
    tt_mat = reshape(first_core, (tt.tto_dims[1], tt.tto_dims[1], tt.tto_rks[2]))

    for i in 2:tt.N
        next_core = tt.tto_vec[i]
        @tensor temp[a, b, c, d, r2] := tt_mat[a, b, r1] * next_core[c, d, r1, r2]
        tt_mat = permutedims(temp, (1, 3, 2, 4, 5))
        tt_mat = reshape(tt_mat, prod(tt.tto_dims[1:i]), prod(tt.tto_dims[1:i]), tt.tto_rks[i+1])
    end

    m = prod(tt.tto_dims)
    n = prod(tt.tto_dims)
    return reshape(tt_mat, m, n)
end

function matricize(tt::TTvector{T, M}) where {T, M}
    first_core = tt.ttv_vec[1]
    if prod(size(first_core)) != tt.ttv_dims[1] * tt.ttv_rks[2]
        error("First core size mismatch: expected $(tt.ttv_dims[1] * tt.ttv_rks[2]), got $(prod(size(first_core)))")
    end
    tt_mat = reshape(first_core, (tt.ttv_dims[1], tt.ttv_rks[2]))

    for i in 2:tt.N
        next_core = tt.ttv_vec[i]
        @tensor temp[a, b, r2] := tt_mat[a, r1] * next_core[b, r1, r2]
        tt_mat = permutedims(temp, (1, 3, 2))
        tt_mat = reshape(tt_mat, prod(tt.ttv_dims[1:i]), tt.ttv_rks[i+1])
    end

    m = prod(tt.ttv_dims)
    return reshape(tt_mat, m * tt.ttv_rks[end])
end

"""
	concatenate(tt1::TTvector, tt2::TTvector) -> TTvector

Concatenates two TTvector objects `tt1` and `tt2` into a single TTvector. 

# Arguments
- `tt1::TTvector`: The first TTvector to concatenate.
- `tt2::TTvector`: The second TTvector to concatenate.

# Returns
- `TTvector`: A new TTvector that is the result of concatenating `tt1` and `tt2`.

# Throws
- `ArgumentError`: If the final rank of `tt1` does not equal the initial rank of `tt2`.

"""
function concatenate(tt1::TTvector, tt2::TTvector)
    if tt1.ttv_rks[end] != tt2.ttv_rks[1]
        throw(ArgumentError("The final rank of the first TTvector must equal the initial rank of the second TTvector."))
    end

    N = tt1.N + tt2.N  
    ttv_vec = vcat(tt1.ttv_vec, tt2.ttv_vec)  
    ttv_dims = (tt1.ttv_dims..., tt2.ttv_dims...)  
    ttv_rks = vcat(tt1.ttv_rks[1:end-1], tt2.ttv_rks)  
    ttv_ot = vcat(tt1.ttv_ot, tt2.ttv_ot) 

    return TTvector{eltype(tt1), length(ttv_dims)}(N, ttv_vec, ttv_dims, ttv_rks, ttv_ot)
end

"""
	concatenate(tt1::TToperator, tt2::TToperator) -> TToperator

Concatenates two TToperators `tt1` and `tt2` into a single TToperator.

# Arguments
- `tt1::TToperator`: The first TToperator.
- `tt2::TToperator`: The second TToperator.

# Returns
- `TToperator`: A new TToperator that is the concatenation of `tt1` and `tt2`.

# Throws
- `ArgumentError`: If the final rank of `tt1` does not equal the initial rank of `tt2`.

# Description
This function concatenates two TToperators by combining their tensor train vectors, dimensions, ranks, and operator types. The resulting TToperator has the combined properties of the input TToperators.

"""
	function concatenate(tt1::TToperator, tt2::TToperator)
    if tt1.tto_rks[end] != tt2.tto_rks[1]
        throw(ArgumentError("The final rank of the first TToperator must equal the initial rank of the second TToperator."))
    end

    N = tt1.N + tt2.N  
    tto_vec = vcat(tt1.tto_vec, tt2.tto_vec)  
    tto_dims = (tt1.tto_dims..., tt2.tto_dims...)  
    tto_rks = vcat(tt1.tto_rks[1:end-1], tt2.tto_rks)  
    tto_ot = vcat(tt1.tto_ot, tt2.tto_ot) 

    return TToperator{eltype(tt1), length(tto_dims)}(N, tto_vec, tto_dims, tto_rks, tto_ot)
end
