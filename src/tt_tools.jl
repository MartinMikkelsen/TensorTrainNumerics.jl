using Random
using LinearAlgebra
using Base.Threads
using IterativeSolvers
using TensorOperations
import Base.isempty
import Base.eltype
import Base.copy
import Base.complex
import KrylovKit: orthogonalize

abstract type AbstractTTvector end
"""
A structure representing a Tensor Train (TT) vector.

# Fields
- `N::Int64`: The number of elements in the TT vector.
- `ttv_vec::Vector{Array{T,3}}`: A vector of 3-dimensional arrays representing the TT d.
- `ttv_dims::NTuple{M,Int64}`: A tuple containing the dimensions of the TT vector.
- `ttv_rks::Vector{Int64}`: A vector containing the TT ranks.
- `ttv_ot::Vector{Int64}`: A vector containing the orthogonalization information.

# Type Parameters
- `T<:Number`: The type of the elements in the TT vector.
"""
mutable struct TTvector{T <: Number, M} <: AbstractTTvector
    N::Int64
    ttv_vec::Vector{Array{T, 3}}
    ttv_dims::NTuple{M, Int64}
    ttv_rks::Vector{Int64}
    ttv_ot::Vector{Int64}
end

Base.eltype(::TTvector{T, N}) where {T <: Number, N} = T


abstract type AbstractTToperator end
"""
A structure representing a Tensor Train (TT) operator.

# Fields
- `N::Int64`: The number of dimensions of the TT operator.
- `tto_vec::Array{Array{T,4},1}`: A vector of 4-dimensional arrays representing the TT d.
- `tto_dims::NTuple{M,Int64}`: A tuple containing the dimensions of the TT operator.
- `tto_rks::Array{Int64,1}`: An array containing the TT ranks.
- `tto_ot::Array{Int64,1}`: An array containing the output dimensions of the TT operator.

# Type Parameters
- `T<:Number`: The type of the elements in the TT vector.
"""
struct TToperator{T <: Number, M} <: AbstractTToperator
    N::Int64
    tto_vec::Array{Array{T, 4}, 1}
    tto_dims::NTuple{M, Int64}
    tto_rks::Array{Int64, 1}
    tto_ot::Array{Int64, 1}
end

Base.eltype(::TToperator{T, M}) where {T, M} = T
Base.eltype(::TTvector{T, N}) where {T, N} = T

function Base.complex(A::TToperator{T, M}) where {T, M}
    return TToperator{Complex{T}, M}(A.N, complex.(A.tto_vec), A.tto_dims, A.tto_rks, A.tto_ot)
end

function Base.complex(v::TTvector{T, M}) where {T, M}
    return TTvector{Complex{T}, M}(v.N, complex.(v.ttv_vec), v.ttv_dims, v.ttv_rks, v.ttv_ot)
end


"""
    QTTvector(vec::Vector{<:Array{<:Number, 3}}, rks::Vector{Int64}, ot::Vector{Int64})

Constructs a Quantized Tensor Train (QTT) vector from a given vector of 3-dimensional arrays (d).

# Arguments
- `vec::Vector{<:Array{<:Number, 3}}`: A vector containing the d of the QTT vector. Each core must be a 3-dimensional array with the first dimension equal to 2.
- `rks::Vector{Int64}`: A vector of integer ranks for the QTT vector.
- `ot::Vector{Int64}`: A vector of integer orthogonalization types for the QTT vector.

# Returns
- `TTvector{T, N}`: A tensor train vector of type `T` and length `N`.

# Throws
- `AssertionError`: If any core in `vec` does not have the first dimension equal to 2.

"""
function QTTvector(vec::Vector{<:Array{<:Number, 3}}, rks::Vector{Int64}, ot::Vector{Int64})
    T = eltype(eltype(vec))
    N = length(vec)
    dims = ntuple(_ -> 2, N)
    for core in vec
        @assert size(core, 1) == 2 "Each core must have physical dimension 2."
    end
    return TTvector{T, N}(N, vec, dims, rks, ot)
end

"""
    is_qtt(tt::TTvector) -> Bool

Check if a given TTvector is a QTT (Quantized Tensor Train) vector.

# Arguments
- `tt::TTvector`: The tensor train vector to be checked.

# Returns
- `Bool`: Returns `true` if all dimensions of the tensor train vector are equal to 2, indicating it is a QTT vector, otherwise returns `false`.

# Example
"""
function is_qtt(tt::TTvector)
    return all(dim == 2 for dim in tt.ttv_dims)
end
"""
    QTToperator(vec::Vector{Array{T,4}}, rks::Vector{Int64}, ot::Vector{Int64}) where {T}

Constructs a Quantum Tensor Train (QTT) operator from a vector of 4-dimensional arrays (d).

# Arguments
- `vec::Vector{Array{T,4}}`: A vector containing the d of the QTT operator. Each core must be a 4-dimensional array with the first two dimensions equal to 2.
- `rks::Vector{Int64}`: A vector containing the rank sizes of the QTT operator.
- `ot::Vector{Int64}`: A vector containing the operator types.

# Returns
- `TToperator{T,N}`: A QTT operator constructed from the provided d, rank sizes, and operator types.

# Throws
- `AssertionError`: If any core in `vec` does not have the first two dimensions equal to 2.

"""
function QTToperator(vec::Vector{Array{T, 4}}, rks::Vector{Int64}, ot::Vector{Int64}) where {T}
    N = length(vec)
    dims = ntuple(_ -> 2, N)
    for core in vec
        @assert size(core, 1) == 2 && size(core, 2) == 2 "Each core must have physical dimension 2."
    end
    return TToperator{T, N}(N, vec, dims, rks, ot)
end

"""
    is_qtt_operator(op::TToperator) -> Bool

Check if a given `TToperator` is a Quantum Tensor Train (QTT) operator.

# Arguments
- `op::TToperator`: The tensor train operator to be checked.

# Returns
- `Bool`: Returns `true` if all dimensions of the tensor train operator are equal to 2, indicating it is a QTT operator. Otherwise, returns `false`.
"""
function is_qtt_operator(op::TToperator)
    return all(dim == 2 for dim in op.tto_dims)
end


"""
    rand_orthogonal(n, m; T=Float64)

Generate a random orthogonal matrix of size `n` by `m`.

# Arguments
- `n::Int`: Number of rows of the resulting matrix.
- `m::Int`: Number of columns of the resulting matrix.
- `T::Type{<:AbstractFloat}`: (Optional) The element type of the matrix. Defaults to `Float64`.

# Returns
- `Matrix{T}`: A random orthogonal matrix of size `n` by `m`.
"""
function rand_orthogonal(n, m; T = Float64)
    N = max(n, m)
    q, r = qr(rand(T, N, N))
    return Matrix(q)[1:n, 1:m]
end

"""
    rand_tt(dims, rks; normalise=false, orthogonal=false)

Generate a random Tensor Train (TT) format tensor with specified dimensions and ranks.

# Arguments
- `dims::Vector{Int}`: A vector specifying the dimensions of the tensor.
- `rks::Vector{Int}`: A vector specifying the TT-ranks.
- `normalise::Bool`: A keyword argument to indicate whether the tensor should be normalised. Default is `false`.
- `orthogonal::Bool`: A keyword argument to indicate whether the tensor should be orthogonal. Default is `false`.

# Returns
- A random tensor in TT format with the specified properties.
"""
function rand_tt(dims, rks; normalise = false, orthogonal = false)
    return rand_tt(Float64, dims, rks; normalise = normalise, orthogonal = orthogonal)
end

"""
    rand_tt(::Type{T}, dims, rks; normalise=false, orthogonal=false) where T

Generate a random Tensor Train (TT) tensor with specified dimensions and ranks.

# Arguments
- `::Type{T}`: The data type of the tensor elements.
- `dims`: A vector specifying the dimensions of the tensor.
- `rks`: A vector specifying the TT-ranks.
- `normalise`: A boolean flag indicating whether to normalize the TT-d. Default is `false`.
- `orthogonal`: A boolean flag indicating whether to orthogonalize the TT-d. Default is `false`.

# Returns
- A TT tensor with random elements of type `T`.
"""
function rand_tt(::Type{T}, dims, rks; normalise = false, orthogonal = false) where {T}
    y = zeros_tt(T, dims, rks)
    @simd for i in eachindex(y.ttv_vec)
        y.ttv_vec[i] = randn(T, dims[i], rks[i], rks[i + 1])
        if normalise
            y.ttv_vec[i] *= 1 / sqrt(dims[i] * rks[i + 1])
            if orthogonal
                q, _ = qr(reshape(permutedims(y.ttv_vec[i], (1, 3, 2)), dims[i] * rks[i + 1], rks[i]))
                y.ttv_vec[i] = permutedims(reshape(Matrix(q), dims[i], rks[i + 1], rks[i]), (1, 3, 2))
            end
        end
    end
    return y
end

"""
    rand_tt(dims, rmax::Int; T=Float64, normalise=false, orthogonal=false)

Generate a random Tensor Train (TT) vector with specified dimensions and rank.

# Arguments
- `dims::Vector{Int}`: A vector specifying the dimensions of each mode of the tensor.
- `rmax::Int`: The maximum TT-rank.
- `T::Type` (optional): The element type of the tensor (default is `Float64`).
- `normalise::Bool` (optional): If `true`, normalizes each core tensor (default is `false`).
- `orthogonal::Bool` (optional): If `true`, orthogonalizes each core tensor (default is `false`).

# Returns
- `TTvector{T,d}`: A TTvector object containing the generated TT d, dimensions, ranks, and a zero vector for the TT ranks.
"""
function rand_tt(dims, rmax::Int; T = Float64, normalise = false, orthogonal = false)
    d = length(dims)
    tt_vec = Vector{Array{T, 3}}(undef, d)
    rks = rmax * ones(Int, d + 1)
    rks = r_and_d_to_rks(rks, dims; rmax = rmax)
    for i in eachindex(tt_vec)
        tt_vec[i] = randn(T, dims[i], rks[i], rks[i + 1])
        if normalise
            tt_vec[i] *= 1 / sqrt(dims[i] * rks[i + 1])
        end
        if orthogonal
            q, _ = qr(reshape(permutedims(tt_vec[i], (1, 3, 2)), dims[i] * rks[i + 1], rks[i]))
            tt_vec[i] = reshape(permutedims(Matrix(q), (1, 2, 3)), dims[i], rks[i], rks[i + 1])
        end
    end
    return TTvector{T, d}(d, tt_vec, dims, rks, zeros(Int, d))
end
"""
    rand_tt(x_tt::TTvector{T,N}; ε=convert(T,1e-3)) -> TTvector{T,N}

Generate a random tensor train (TT) vector by adding Gaussian noise to the input TT vector `x_tt`.

# Arguments
- `x_tt::TTvector{T,N}`: The input TT vector to which noise will be added.
- `ε`: The standard deviation of the Gaussian noise to be added. Default is `1e-3` converted to type `T`.

# Returns
- `TTvector{T,N}`: A new TT vector with added Gaussian noise.
"""
function rand_tt(x_tt::TTvector{T, N}; ε = convert(T, 1.0e-3)) where {T, N}
    tt_vec = copy(x_tt.ttv_vec)
    for i in eachindex(x_tt.ttv_vec)
        tt_vec[i] += ε * randn(x_tt.ttv_dims[i], x_tt.ttv_rks[i], x_tt.ttv_rks[i + 1])
    end
    return TTvector{T, N}(N, tt_vec, x_tt.ttv_dims, x_tt.ttv_rks, zeros(Int, N))
end

"""
    Base.copy(x_tt::TTvector{T,N}) where {T<:Number,N}

Create a deep copy of a `TTvector` object.

# Arguments
- `x_tt::TTvector{T,N}`: The `TTvector` object to be copied, where `T` is a subtype of `Number` and `N` is the dimensionality.

# Returns
- A new `TTvector` object that is a deep copy of `x_tt`.
"""
function Base.copy(x_tt::TTvector{T, N}) where {T <: Number, N}
    y_tt = zeros_tt(T, x_tt.ttv_dims, x_tt.ttv_rks; ot = x_tt.ttv_ot)
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
function ttv_decomp(tensor::Array{T, d}; index = 1, tol = 1.0e-12) where {T <: Number, d}
    # Decomposes a tensor into its tensor train with core matrices at i=index
    dims = size(tensor) #dims = [n_1,...,n_d]
    ttv_vec = Array{Array{T, 3}}(undef, d)
    # ttv_ot[i]= -1 if i < index
    # ttv_ot[i] = 0 if i = index
    # ttv_ot[i] = 1 if i > index
    ttv_ot = -ones(Int64, d)
    ttv_ot[index] = 0
    if index < d
        ttv_ot[(index + 1):d] = ones(d - index)
    end
    rks = ones(Int64, d + 1)
    tensor_curr = tensor
    # Calculate ttv_vec[i] for i < index
    for i in 1:(index - 1)
        # Reshape the currently left tensor
        tensor_curr = reshape(tensor_curr, Int(rks[i] * dims[i]), :)
        # Perform the singular value decomposition
        u, s, v = svd(tensor_curr)
        # Define the i-th rank
        rks[i + 1] = length(s[s .>= tol])
        # Initialize ttv_vec[i]
        ttv_vec[i] = zeros(T, dims[i], rks[i], rks[i + 1])
        # Fill in the ttv_vec[i]
        for x in 1:dims[i]
            ttv_vec[i][x, :, :] = u[(rks[i] * (x - 1) + 1):(rks[i] * x), :]
        end
        # Update the currently left tensor
        tensor_curr = Diagonal(s[1:rks[i + 1]]) * v'[1:rks[i + 1], :]
    end

    # Calculate ttv_vec[i] for i > index
    if index < d
        for i in d:(-1):(index + 1)
            # Reshape the currently left tensor
            tensor_curr = reshape(tensor_curr, :, dims[i] * rks[i + 1])
            # Perform the singular value decomposition
            u, s, v = svd(tensor_curr)
            # Define the (i-1)-th rank
            rks[i] = length(s[s .>= tol])
            # Initialize ttv_vec[i]
            ttv_vec[i] = zeros(T, dims[i], rks[i], rks[i + 1])
            # Fill in the ttv_vec[i]
            i_vec = zeros(Int, rks[i + 1])
            for x in 1:dims[i]
                i_vec = dims[i] * ((1:rks[i + 1]) - ones(Int, rks[i + 1])) + x * ones(Int, rks[i + 1])
                ttv_vec[i][x, :, :] = v'[1:rks[i], i_vec] #(rks[i+1]*(x-1)+1):(rks[i+1]*x)
            end
            # Update the current left tensor
            tensor_curr = u[:, 1:rks[i]] * Diagonal(s[1:rks[i]])
        end
    end
    # Calculate ttv_vec[i] for i = index
    # Reshape the current left tensor
    tensor_curr = reshape(tensor_curr, Int(dims[index] * rks[index]), :)
    # Initialize ttv_vec[i]
    ttv_vec[index] = zeros(T, dims[index], rks[index], rks[index + 1])
    # Fill in the ttv_vec[i]
    for x in 1:dims[index]
        ttv_vec[index][x, :, :] =
            tensor_curr[Int(rks[index] * (x - 1) + 1):Int(rks[index] * x), 1:rks[index + 1]]
    end

    # Define the return value as a TTvector
    return TTvector{T, d}(d, ttv_vec, dims, rks, ttv_ot)
end

"""
    ttv_to_tensor(x_tt::TTvector{T,N}) where {T<:Number, N}

Convert a TTvector (Tensor Train vector) to a full tensor.

# Arguments
- `x_tt::TTvector{T,N}`: The input TTvector to be converted. `T` is the element type, and `N` is the number of dimensions.

# Returns
- A tensor of type `Array{T,N}` with the same dimensions as specified in `x_tt.ttv_dims`.
"""
function ttv_to_tensor(x_tt::TTvector{T, N}) where {T <: Number, N}
    d = x_tt.N
    tensor = zeros(T, x_tt.ttv_dims)
    @simd for t in CartesianIndices(tensor)
        # Start with the last core
        i = d
        curr = view(x_tt.ttv_vec[i], t[i], :, :)
        # Contract backwards through the TT d
        for j in (d - 1):-1:1
            curr = view(x_tt.ttv_vec[j], t[j], :, :) * curr
        end
        tensor[t] = curr[1, 1]
    end
    return tensor
end

"""
    tto_to_ttv(A::TToperator{T,N}) where {T<:Number,N}

Convert a `TToperator` to a `TTvector`.

# Arguments
- `A::TToperator{T,N}`: The TToperator to be converted. `T` is the element type, and `N` is the number of dimensions.

# Returns
- `TTvector{T,N}`: The resulting TTvector.

# Details
This function takes a `TToperator` and converts it into a `TTvector`. It reshapes the internal tensor d of the `TToperator` and constructs a `TTvector` with the appropriate dimensions and ranks.

"""
function tto_to_ttv(A::TToperator{T, N}) where {T <: Number, N}
    d = A.N
    xtt_vec = Array{Array{T, 3}, 1}(undef, d)
    A_rks = A.tto_rks
    for i in eachindex(xtt_vec)
        xtt_vec[i] = reshape(A.tto_vec[i], A.tto_dims[i]^2, A_rks[i], A_rks[i + 1])
    end
    return TTvector{T, N}(d, xtt_vec, A.tto_dims .^ 2, A.tto_rks, A.tto_ot)
end

"""
    ttv_to_tto(x::TTvector{T,N}) where {T<:Number,N}

Convert a `TTvector` to a `TToperator`.

# Arguments
- `x::TTvector{T,N}`: The input `TTvector` object to be converted. `T` is the element type, and `N` is the number of dimensions.

# Returns
- `TToperator{T,N}`: The resulting `TToperator` object.

# Throws
- `DimensionMismatch`: If the dimensions of the input `TTvector` are not perfect squares.

# Description
This function converts a `TTvector` to a `TToperator` by reshaping the core tensors of the `TTvector` into 4-dimensional arrays. The reshaping is done such that the first two dimensions of each core tensor are the square roots of the original dimensions, and the last two dimensions are the ranks of the `TTvector`.
"""
function ttv_to_tto(x::TTvector{T, N}) where {T <: Number, N}
    @assert(isqrt.(x.ttv_dims) .^ 2 == x.ttv_dims, DimensionMismatch)
    d = x.N
    Att_vec = Array{Array{T, 4}, 1}(undef, d)
    x_rks = x.ttv_rks
    A_dims = isqrt.(x.ttv_dims)
    for i in eachindex(A_dims)
        Att_vec[i] = reshape(x.ttv_vec[i], A_dims[i], A_dims[i], x_rks[i], x_rks[i + 1])
    end
    return TToperator{T, N}(d, Att_vec, A_dims, x.ttv_rks, x.ttv_ot)
end

"""
Returns the TT decomposition of a matrix using the HSVD algorithm
"""
function tto_decomp(tensor::Array{T, N}; index = 1) where {T <: Number, N}
    # Decomposes a tensor operator into its tensor train
    # with core matrices at i=index
    # The tensor is given as tensor[x_1,...,x_d,y_1,...,y_d]
    d = Int(ndims(tensor) / 2)
    tto_dims = size(tensor)[1:d]
    dims_sq = tto_dims .^ 2
    # The tensor is reorder  into tensor[x_1,y_1,...,x_d,y_d],
    # reshaped into tensor[(x_1,y_1),...,(x_d,y_d)]
    # and decomposed into its tensor train with core matrices at i= index
    index_sorted = reshape(Transpose(reshape(1:(2 * d), :, 2)), 1, :)
    ttv = ttv_decomp(reshape(permutedims(tensor, index_sorted), (dims_sq[1:(end - 1)]...), :); index = index)
    # Define the array of ranks [r_0=1,r_1,...,r_d]
    rks = ttv.ttv_rks
    # Initialize tto_vec
    tto_vec = Array{Array{T}}(undef, d)
    # Fill in tto_vec
    for i in 1:d
        # Initialize tto_vec[i]
        tto_vec[i] = zeros(T, tto_dims[i], tto_dims[i], rks[i], rks[i + 1])
        # Fill in tto_vec[i]
        tto_vec[i] = reshape(ttv.ttv_vec[i], tto_dims[i], tto_dims[i], :, rks[i + 1])
    end
    return TToperator{T, d}(d, tto_vec, tto_dims, rks, ttv.ttv_ot)
end
"""
    tto_to_tensor(tto::TToperator{T,N}) where {T<:Number, N}

Convert a TToperator to a full tensor.

# Arguments
- `tto::TToperator{T,N}`: The TToperator to be converted, where `T` is a subtype of `Number` and `N` is the order of the tensor.

# Returns
- A tensor of type `Array{T, 2N}` with dimensions `[n_1, ..., n_d, n_1, ..., n_d]`, where `n_i` are the dimensions of the TToperator.
"""
function tto_to_tensor(tto::TToperator{T, N}) where {T <: Number, N}
    d = tto.N
    # Define the array of ranks [r_0=1,r_1,...,r_d]
    rks = tto.tto_rks
    r_max = maximum(rks)
    # The tensor has dimensions [n_1,...,n_d,n_1,...,n_d]
    tensor = zeros(T, (tto.tto_dims..., tto.tto_dims...))
    # Fill in the tensor for every t=(x_1,...,x_d,y_1,...,y_d)
    curr = ones(T, r_max)
    @simd for t in CartesianIndices(tensor)
        curr[1] = one(T)
        for i in d:-1:1
            curr[1:rks[i]] = tto.tto_vec[i][t[i], t[d + i], :, :] * curr[1:rks[i + 1]]
        end
        tensor[t] = curr[1]
    end
    return tensor
end

"""
	r_and_d_to_rks(rks, dims; rmax=1024)

Adjusts the ranks `rks` based on the dimensions `dims` and an optional maximum rank `rmax`.

# Arguments
- `rks::AbstractVector`: A vector of ranks.
- `dims::AbstractVector`: A vector of dimensions.
- `rmax::Int`: An optional maximum rank (default is 1024).

# Returns
- `new_rks::Vector`: A vector of adjusted ranks.
"""
function r_and_d_to_rks(rks, dims; rmax = 1024)
    new_rks = ones(eltype(rks), length(rks))
    @simd for i in eachindex(dims)
        if prod(dims[i:end]) > 0
            if prod(dims[1:(i - 1)]) > 0
                new_rks[i] = min(rks[i], prod(dims[1:(i - 1)]), prod(dims[i:end]), rmax)
            else
                new_rks[i] = min(rks[i], prod(dims[i:end]), rmax)
            end
        else
            if prod(dims[1:(i - 1)]) > 0
                new_rks[i] = min(rks[i], prod(dims[1:(i - 1)]), rmax)
            else
                new_rks[i] = min(rks[i], rmax)
            end
        end
    end
    return new_rks
end

"""
    tt_up_rks_noise(tt_vec, tt_ot_i, rkm, rk, ϵ_wn)

Update a tensor train (TT) vector with random noise.

# Arguments
- `tt_vec::Array`: The input TT vector.
- `tt_ot_i::Int`: An integer parameter that is modified within the function.
- `rkm::Int`: The new rank for the second dimension.
- `rk::Int`: The new rank for the third dimension.
- `ϵ_wn::Float64`: The noise level.

# Returns
- `vec_out::Array`: The updated TT vector with added noise.
"""
function tt_up_rks_noise(tt_vec, tt_ot_i, rkm, rk, ϵ_wn)
    vec_out = zeros(eltype(tt_vec), size(tt_vec, 1), rkm, rk)
    vec_out[:, 1:size(tt_vec, 2), 1:size(tt_vec, 3)] = tt_vec
    if !iszero(ϵ_wn)
        if rkm == size(tt_vec, 2) && rk > size(tt_vec, 3)
            Q = rand_orthogonal(size(tt_vec, 1) * rkm, rk - size(tt_vec, 3))
            vec_out[:, :, (size(tt_vec, 3) + 1):rk] = ϵ_wn * reshape(Q, size(tt_vec, 1), rkm, rk - size(tt_vec, 3))
            tt_ot_i = 0
        elseif rk == size(tt_vec, 3) && rkm > size(tt_vec, 2)
            Q = rand_orthogonal(rkm - size(tt_vec, 2), size(tt_vec, 1) * rk)
            vec_out[:, (size(tt_vec, 2) + 1):rkm, :] = ϵ_wn * reshape(Q, size(tt_vec, 1), rkm - size(tt_vec, 2), rk)
            tt_ot_i = 0
        elseif rk > size(tt_vec, 3) && rkm > size(tt_vec, 2)
            Q = rand_orthogonal((rkm - size(tt_vec, 2)) * size(tt_vec, 1), (rk - size(tt_vec, 3)))
            vec_out[:, (size(tt_vec, 2) + 1):rkm, (size(tt_vec, 3) + 1):rk] = ϵ_wn * reshape(Q, size(tt_vec, 1), rkm - size(tt_vec, 2), rk - size(tt_vec, 3))
        end
    end
    return vec_out
end

"""
    tt_up_rks(x_tt::TTvector{T,N}, rk_max::Int; rks=vcat(1, rk_max*ones(Int, length(x_tt.ttv_dims)-1), 1), ϵ_wn=0.0) where {T<:Number, N}

Increase the rank of a Tensor Train (TT) vector `x_tt` to a specified maximum rank `rk_max`.

# Arguments
- `x_tt::TTvector{T,N}`: The input TT vector.
- `rk_max::Int`: The maximum rank to which the TT vector should be increased.
- `rks`: Optional. A vector specifying the ranks at each step. Defaults to a vector with `1` at the boundaries and `rk_max` in between.
- `ϵ_wn::Float64`: Optional. The noise level to be added during the rank increase. Defaults to `0.0`.

# Returns
- `TTvector{T,N}`: A new TT vector with increased ranks.
"""
function tt_up_rks(x_tt::TTvector{T, N}, rk_max::Int; rks = vcat(1, rk_max * ones(Int, length(x_tt.ttv_dims) - 1), 1), ϵ_wn = 0.0) where {T <: Number, N}
    d = x_tt.N
    vec_out = Array{Array{T}}(undef, d)
    out_ot = zeros(Int64, d)
    @assert(rk_max > maximum(x_tt.ttv_rks), "New bond dimension too low")
    rks = r_and_d_to_rks(rks, x_tt.ttv_dims; rmax = rk_max)
    for i in 1:d
        vec_out[i] = tt_up_rks_noise(x_tt.ttv_vec[i], x_tt.ttv_ot[i], rks[i], rks[i + 1], ϵ_wn)
    end
    return TTvector{T, N}(d, vec_out, x_tt.ttv_dims, rks, out_ot)
end

"""
    orthogonalize(x_tt::TTvector{T,N}; i=1::Int) where {T<:Number, N}

Orthogonalizes the given Tensor Train (TT) vector `x_tt` with respect to the `i`-th core. The orthogonalization process involves QR and LQ decompositions to ensure that the TT d are orthogonal.

# Arguments
- `x_tt::TTvector{T,N}`: The input TT vector to be orthogonalized.
- `i::Int=1`: The core index with respect to which the orthogonalization is performed. Defaults to 1.

# Returns
- `y_tt`: The orthogonalized TT vector.

"""
function orthogonalize(x_tt::TTvector{T, N}; i = 1::Int) where {T <: Number, N}
    d = x_tt.N
    @assert(1 ≤ i ≤ d, DimensionMismatch("Impossible orthogonalization"))
    y_rks = r_and_d_to_rks(x_tt.ttv_rks, x_tt.ttv_dims)
    y_tt = zeros_tt(T, x_tt.ttv_dims, y_rks)
    FR = ones(T, 1, 1)
    yleft_temp = zeros(T, maximum(x_tt.ttv_rks), maximum(x_tt.ttv_dims), maximum(x_tt.ttv_rks))
    for j in 1:(i - 1)
        y_tt.ttv_ot[j] = 1
        @tensoropt((βⱼ₋₁, αⱼ), yleft_temp[1:y_tt.ttv_rks[j], 1:x_tt.ttv_dims[j], 1:x_tt.ttv_rks[j + 1]][αⱼ₋₁, iⱼ, αⱼ] = FR[αⱼ₋₁, βⱼ₋₁] * x_tt.ttv_vec[j][iⱼ, βⱼ₋₁, αⱼ])
        F = qr(reshape(yleft_temp[1:y_tt.ttv_rks[j], 1:x_tt.ttv_dims[j], 1:x_tt.ttv_rks[j + 1]], x_tt.ttv_dims[j] * y_tt.ttv_rks[j], :))
        y_tt.ttv_rks[j + 1] = size(Matrix(F.Q), 2)
        y_tt.ttv_vec[j] = permutedims(reshape(Matrix(F.Q), y_tt.ttv_rks[j], x_tt.ttv_dims[j], y_tt.ttv_rks[j + 1]), [2 1 3])
        FR = F.R[1:y_tt.ttv_rks[j + 1], :]
    end
    FL = ones(T, 1, 1)
    (i < x_tt.N) && (yright_temp = zeros(T, maximum(x_tt.ttv_rks), maximum(y_tt.ttv_rks), maximum(x_tt.ttv_dims)))
    for j in d:-1:(i + 1)
        y_tt.ttv_ot[j] = -1
        yright_temp = zeros(T, x_tt.ttv_rks[j], y_tt.ttv_rks[j + 1], x_tt.ttv_dims[j])
        @tensoropt((αⱼ₋₁, αⱼ), yright_temp[1:x_tt.ttv_rks[j], 1:y_tt.ttv_rks[j + 1], 1:x_tt.ttv_dims[j]][αⱼ₋₁, βⱼ, iⱼ] = x_tt.ttv_vec[j][iⱼ, αⱼ₋₁, αⱼ] * FL[αⱼ, βⱼ])
        F = lq(reshape(yright_temp[1:x_tt.ttv_rks[j], 1:y_tt.ttv_rks[j + 1], 1:x_tt.ttv_dims[j]], x_tt.ttv_rks[j], :))
        y_tt.ttv_rks[j] = size(Matrix(F.Q), 1)
        y_tt.ttv_vec[j] = permutedims(reshape(Matrix(F.Q), y_tt.ttv_rks[j], y_tt.ttv_rks[j + 1], x_tt.ttv_dims[j]), [3 1 2])
        FL = F.L[:, 1:y_tt.ttv_rks[j]]
    end
    y_tt.ttv_ot[i] = 0
    y_tt.ttv_vec[i] = zeros(T, y_tt.ttv_dims[i], y_tt.ttv_rks[i], y_tt.ttv_rks[i + 1])
    @simd for k in 1:x_tt.ttv_dims[i]
        y_tt.ttv_vec[i][k, :, :] = FR * x_tt.ttv_vec[i][k, :, :] * FL
    end
    return y_tt
end


"""
    visualize(tt::TTvector)

Visualizes a Tensor Train (TT) vector by creating a textual representation of its dimensions and ranks.

# Arguments
- `tt::TTvector`: A Tensor Train vector object.
"""
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
    line2 = " "^length(line1)
    line3 = " "^length(line1)

    for i in 1:N
        segment = "-- • --"
        rank_right = lpad(string(ranks[i + 1]), rwidth)
        line1 *= segment * rank_right

        position_C = length(line1) - rwidth - 3  # 'C' is 3 characters before the right rank

        line2_len = length(line2)
        spaces_needed = position_C - line2_len
        line2 *= repeat(" ", spaces_needed - 1) * "|"

        dim_str = string(dims[i])
        dim_len = length(dim_str)
        line3_len = length(line3)
        spaces_needed = position_C - line3_len - div(dim_len, 2)
        line3 *= repeat(" ", spaces_needed - 1) * dim_str
    end

    diagram = line1 * "\n" * line2 * "\n" * line3

    return println(diagram)
end
"""
    visualize(tt::TToperator)

Visualizes a Tensor Train (TT) operator by creating a textual representation of its dimensions and ranks.

# Arguments
- `tt::TToperator`: A Tensor Train operator object.
"""
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
            segment = " $(ranks[i])-- • --$(ranks[i + 1])"
        else
            segment = "-- • --$(ranks[i + 1])"
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
    return println(diagram)
end

"""
    tt2qtt(tt_tensor::TToperator{T,N}, row_dims::Vector{Vector{Int}}, col_dims::Vector{Vector{Int}}, threshold::Float64=0.0) where {T<:Number,N}

Convert a TT (Tensor Train) operator to a QTT (Quantized Tensor Train) operator.

# Arguments
- `tt_tensor::TToperator{T,N}`: The input TT operator.
- `row_dims::Vector{Vector{Int}}`: A vector of vectors specifying the row dimensions for each core.
- `col_dims::Vector{Vector{Int}}`: A vector of vectors specifying the column dimensions for each core.
- `threshold::Float64=0.0`: A threshold for rank reduction during SVD. Default is 0.0, meaning no rank reduction.

# Returns
- `qtt_tensor::TToperator{T,M}`: The resulting QTT operator.
# Details
This function converts a given TT operator into a QTT operator by splitting each core of the TT operator according to the specified row and column dimensions. It performs SVD on reshaped d and applies rank reduction based on the given threshold. The resulting QTT d are then used to construct the QTT operator.
"""
function tt2qtt(tt_tensor::TToperator{T, N}, row_dims::Vector{Vector{Int}}, col_dims::Vector{Vector{Int}}, threshold::Float64 = 0.0) where {T <: Number, N}

    qtt_cores = Array{Array{T, 4}}(undef, 0)
    tto_rks = [1]
    tto_dims = Int64[]

    # For each core in tt_tensor
    for i in 1:tt_tensor.N

        # Get core, rank_prev, rank_next, row_dim, col_dim
        core = permutedims(tt_tensor.tto_vec[i], (3, 1, 2, 4))  # Now core is (r_{k-1}, n_k_row, n_k_col, r_k)
        rank_prev = tto_rks[end]
        rank_next = tt_tensor.tto_rks[i + 1]
        row_dim = tt_tensor.tto_dims[i]
        col_dim = tt_tensor.tto_dims[i]  # Assuming square dimensions

        # Begin splitting
        for j in 1:(length(row_dims[i]) - 1)

            # Update row_dim and col_dim
            row_dim = div(row_dim, row_dims[i][j])
            col_dim = div(col_dim, col_dims[i][j])

            # Reshape and permute core
            core = reshape(core, (rank_prev, row_dims[i][j], row_dim, col_dims[i][j], col_dim, rank_next))
            core = permutedims(core, (1, 2, 4, 3, 5, 6))  # Now core is (r_{k-1}, row_dims[i][j], col_dims[i][j], row_dim, col_dim, r_k)

            # Reshape core into 2D matrix for SVD
            core_reshaped = reshape(core, (rank_prev * row_dims[i][j] * col_dims[i][j], row_dim * col_dim * rank_next))

            # Compute SVD
            F = svd(core_reshaped; full = false)
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
            core_to_append = permutedims(U_reshaped, (2, 3, 1, 4))
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
        core_to_append = permutedims(core, (2, 3, 1, 4))
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

    qtt_tensor = TToperator{T, M}(N_qtt, qtt_cores, Tuple(tto_dims), tto_rks, tto_ot)

    return qtt_tensor
end
"""
    tt2qtt(tt_tensor::TTvector{T,N}, dims::Vector{Vector{Int}}, threshold::Float64=0.0) where {T<:Number,N}

Convert a Tensor Train (TT) tensor to a Quantized Tensor Train (QTT) tensor.

# Arguments
- `tt_tensor::TTvector{T,N}`: The input TT tensor to be converted.
- `dims::Vector{Vector{Int}}`: A vector of vectors specifying the dimensions for each core in the QTT tensor.
- `threshold::Float64=0.0`: A threshold for rank reduction during the SVD step. Default is 0.0, meaning no rank reduction.

# Returns
- `qtt_tensor::TTvector{T,M}`: The resulting QTT tensor.

# Description
This function converts a given TT tensor into a QTT tensor by splitting each core of the TT tensor according to the specified dimensions. It performs Singular Value Decomposition (SVD) on reshaped d and applies rank reduction based on the given threshold. The resulting QTT d are then assembled into a new QTT tensor.
"""
function tt2qtt(tt_tensor::TTvector{T, N}, dims::Vector{Vector{Int}}, threshold::Float64 = 0.0) where {T <: Number, N}

    qtt_cores = Array{Array{T, 3}}(undef, 0)
    ttv_rks = [1]
    ttv_dims = Int64[]

    # For each core in tt_tensor
    for i in 1:tt_tensor.N

        # Get core, rank_prev, rank_next, dim
        core = permutedims(tt_tensor.ttv_vec[i], (2, 1, 3))  # Now core is (r_{k-1}, n_k, r_k)
        rank_prev = ttv_rks[end]
        rank_next = tt_tensor.ttv_rks[i + 1]
        dim = tt_tensor.ttv_dims[i]

        # Begin splitting
        for j in 1:(length(dims[i]) - 1)

            # Update dim
            dim = div(dim, dims[i][j])

            # Reshape and permute core
            core = reshape(core, (rank_prev, dims[i][j], dim, rank_next))
            core = permutedims(core, (1, 2, 3, 4))  # Now core is (r_{k-1}, dims[i][j], dim, r_k)

            # Reshape core into 2D matrix for SVD
            core_reshaped = reshape(core, (rank_prev * dims[i][j], dim * rank_next))

            # Compute SVD
            F = svd(core_reshaped; full = false)
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
            core_to_append = permutedims(U_reshaped, (2, 1, 3))
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
        core_to_append = permutedims(core, (2, 1, 3))
        push!(qtt_cores, core_to_append)

        # Update ttv_dims
        push!(ttv_dims, dim)

        # Update ttv_rks
        push!(ttv_rks, rank_next)
    end

    N_qtt = length(qtt_cores)
    M = length(ttv_dims)
    ttv_ot = zeros(Int64, N_qtt)

    qtt_tensor = TTvector{T, M}(N_qtt, qtt_cores, Tuple(ttv_dims), ttv_rks, ttv_ot)

    return qtt_tensor
end

"""
    matricize(qtt::TTvector{Float64}, core::Int)::Vector{Float64}

Convert a TTvector to a vector of Float64 values by extracting a specific core.

# Arguments
- `qtt::TTvector{Float64}`: The TTvector to be converted.
- `core::Int`: The core index to be used for the conversion.

# Returns
- `Vector{Float64}`: A vector of Float64 values representing the specified core of the TTvector.

# Description
This function converts a given TTvector into a vector of Float64 values by extracting the specified core. It first converts the TTvector to a full tensor using `ttv_to_tensor`, then calculates the dyadic points and binary indices to extract the values from the tensor.
"""
function matricize(qtt::TTvector{T}, core::Int)::Vector{T} where {T <: Number}
    full_tensor = ttv_to_tensor(qtt)
    n = 2^core
    values = zeros(T, n)

    for i in 1:n
        x_le_p = sum(((i >> (k - 1)) & 1) / 2^k for k in 1:core)  # Calculate the dyadic point
        index_bits = bitstring(i - 1)[(end - core + 1):end]  # Binary representation
        indices = [parse(Int, bit) + 1 for bit in index_bits]  # Indices for CartesianIndex
        values[i] = full_tensor[CartesianIndex(indices...)]
    end
    return values
end


function concatenate(tt1::TTvector, tt2::TTvector)
    if tt1.ttv_rks[end] != tt2.ttv_rks[1]
        throw(ArgumentError("The final rank of the first TTvector must equal the initial rank of the second TTvector."))
    end

    N = tt1.N + tt2.N
    ttv_vec = vcat(tt1.ttv_vec, tt2.ttv_vec)
    ttv_dims = (tt1.ttv_dims..., tt2.ttv_dims...)
    ttv_rks = vcat(tt1.ttv_rks[1:(end - 1)], tt2.ttv_rks)
    ttv_ot = vcat(tt1.ttv_ot, tt2.ttv_ot)

    return TTvector{eltype(tt1), length(ttv_dims)}(N, ttv_vec, ttv_dims, ttv_rks, ttv_ot)
end


function concatenate(tt1::TToperator, tt2::TToperator)
    if tt1.tto_rks[end] != tt2.tto_rks[1]
        throw(ArgumentError("The final rank of the first TToperator must equal the initial rank of the second TToperator."))
    end

    N = tt1.N + tt2.N
    tto_vec = vcat(tt1.tto_vec, tt2.tto_vec)
    tto_dims = (tt1.tto_dims..., tt2.tto_dims...)
    tto_rks = vcat(tt1.tto_rks[1:(end - 1)], tt2.tto_rks)
    tto_ot = vcat(tt1.tto_ot, tt2.tto_ot)

    return TToperator{eltype(tt1), length(tto_dims)}(N, tto_vec, tto_dims, tto_rks, tto_ot)
end
