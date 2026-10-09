using Random
using LinearAlgebra
using IterativeSolvers
using TensorOperations
import Base: isempty, eltype, copy, complex
import KrylovKit: orthogonalize

"""
    AbstractTTVector{T, M}

Supertype of tensor-train vectors with element type `T` and `M` sites: [`TTVector`](@ref) and the QTT wrapper
[`QTTVector`](@ref).
"""
abstract type AbstractTTVector{T <: Number, M} end
"""
    TTVector{T<:Number, M}

A tensor in tensor-train (TT) format, also called a matrix product state:

    x[i₁, …, i_M] = G₁[i₁, :, :] * G₂[i₂, :, :] * ⋯ * G_M[i_M, :, :]

where core `Gₖ` has size `(nₖ, rₖ₋₁, rₖ)` and `r₀ = r_M = 1`.

# Fields
- `cores::Vector{Array{T,3}}`: the cores, each of size `(dims[k], ranks[k], ranks[k+1])`.
- `dims::NTuple{M,Int64}`: physical dimensions `(n₁, …, n_M)`.
- `ranks::Vector{Int64}`: TT ranks `(r₀, r₁, …, r_M)`, of length `M + 1`.
- `orthogonality::Vector{Int64}`: the interval `[left, right]`, with
  `1 ≤ left ≤ right ≤ M`. The cores `k < left` are left-orthogonal and the cores
  `k > right` are right-orthogonal; nothing is recorded about the cores inside
  the interval. `[c, c]` is an orthogonality center at core `c`, and `[1, M]`
  records nothing.

The fields cannot be reassigned, but in-place operations such as
[`tt_round!`](@ref) and [`add!`](@ref) replace the contents of `cores`,
`ranks`, and `orthogonality`; any other object built on the same vectors (such
as a [`QTTVector`](@ref) wrapping this `TTVector`) sees the change.

The number of cores (sites) is the type parameter `M`, returned by `nsites(x)`.

# Constructor

    TTVector{T, M}(cores, dims, ranks; orthogonality = (1, M))

A `Vector{Int64}` passed as `orthogonality` is stored without copying. The
constructor checks the orthogonality interval but not that the cores, `dims`,
and `ranks` are consistent with each other or that the cores have the recorded
orthogonality. Use constructors such as [`tt_decomp`](@ref), [`rand_tt`](@ref),
or [`zeros_tt`](@ref) to build valid instances.
"""
struct TTVector{T <: Number, M} <: AbstractTTVector{T, M}
    cores::Vector{Array{T, 3}}
    dims::NTuple{M, Int64}
    ranks::Vector{Int64}
    orthogonality::Vector{Int64}
    function TTVector{T, M}(cores, dims, ranks; orthogonality = (1, M)) where {T <: Number, M}
        return new{T, M}(cores, dims, ranks, _orthogonality_storage(orthogonality, M))
    end
end

TTVector(cores::Vector{Array{T, 3}}, dims::NTuple{M, Int64}, ranks; kwargs...) where {T <: Number, M} =
    TTVector{T, M}(cores, dims, ranks; kwargs...)

Base.eltype(::AbstractTTVector{T}) where {T} = T


"""
    AbstractTTOperator{T, M}

Supertype of tensor-train operators with element type `T` and `M` sites: [`TTOperator`](@ref) and the QTT wrapper
[`QTTOperator`](@ref).
"""
abstract type AbstractTTOperator{T <: Number, M} end
"""
    TTOperator{T<:Number, M}

A linear operator in tensor-train format, also called a matrix product operator:

    A[(i₁, …, i_M), (j₁, …, j_M)] = A₁[i₁, j₁, :, :] * ⋯ * A_M[i_M, j_M, :, :]

where core `Aₖ` has size `(mₖ, nₖ, rₖ₋₁, rₖ)` (output index, input index, left
rank, right rank) and `r₀ = r_M = 1`.

# Fields
- `cores::Vector{Array{T,4}}`: the cores, each of size
  `(row_dims[k], col_dims[k], ranks[k], ranks[k+1])`.
- `row_dims::NTuple{M,Int64}`: output (row) dimensions `(m₁, …, m_M)`.
- `col_dims::NTuple{M,Int64}`: input (column) dimensions `(n₁, …, n_M)`.
- `ranks::Vector{Int64}`: TT ranks `(r₀, r₁, …, r_M)`, of length `M + 1`.
- `orthogonality::Vector{Int64}`: the orthogonality interval `[left, right]`,
  with the same meaning as for [`TTVector`](@ref).

The number of cores (sites) is the type parameter `M`, returned by `nsites(A)`.

# Constructor

    TTOperator{T, M}(cores, row_dims, col_dims, ranks; orthogonality = (1, M))
    TTOperator{T, M}(cores, dims, ranks; orthogonality = (1, M))

The second form builds a square operator with `row_dims == col_dims == dims`.
A `Vector{Int64}` passed as `orthogonality` is stored without copying. The
constructor checks the orthogonality interval but not that the other fields are
consistent with each other.
"""
struct TTOperator{T <: Number, M} <: AbstractTTOperator{T, M}
    cores::Vector{Array{T, 4}}
    row_dims::NTuple{M, Int64}
    col_dims::NTuple{M, Int64}
    ranks::Vector{Int64}
    orthogonality::Vector{Int64}
    function TTOperator{T, M}(cores, row_dims, col_dims, ranks; orthogonality = (1, M)) where {T <: Number, M}
        return new{T, M}(cores, row_dims, col_dims, ranks, _orthogonality_storage(orthogonality, M))
    end
end

TTOperator{T, M}(cores, dims, ranks; kwargs...) where {T <: Number, M} =
    TTOperator{T, M}(cores, dims, dims, ranks; kwargs...)
TTOperator(cores::Vector{Array{T, 4}}, row_dims::NTuple{M, Int64}, col_dims::NTuple{M, Int64}, ranks; kwargs...) where {T <: Number, M} =
    TTOperator{T, M}(cores, row_dims, col_dims, ranks; kwargs...)
TTOperator(cores::Vector{Array{T, 4}}, dims::NTuple{M, Int64}, ranks; kwargs...) where {T <: Number, M} =
    TTOperator{T, M}(cores, dims, dims, ranks; kwargs...)

Base.eltype(::AbstractTTOperator{T}) where {T} = T

# Give the plain tensor train `x`, computed from `template` (or from `a` and
# `b`), the wrapper type and metadata of its inputs. Plain inputs leave `x`
# unchanged, and so does a mix of plain and wrapped inputs.
_rewrap(template, x) = x
_rewrap(a, b, x) = x

# Check that two wrapped tensor trains describe the same grid; plain ones always do.
_check_qtt(a, b) = nothing

"""
    nsites(x::Union{AbstractTTVector, AbstractTTOperator}) -> Int

Number of cores (sites) of the tensor train `x`.
"""
function nsites end

nsites(x::AbstractTTVector) = length(x.dims)::Int
nsites(A::AbstractTTOperator) = length(A.row_dims)::Int

# Physical dimensions of a square operator.
function _square_dims(A::AbstractTTOperator)
    A.row_dims == A.col_dims ||
        throw(DimensionMismatch("expected a square TTOperator; got row dimensions $(A.row_dims) and column dimensions $(A.col_dims)"))
    return A.row_dims
end

# Storage for the orthogonality interval of an `M`-site tensor train. A
# `Vector{Int64}` is stored as is, so that wrappers can share it.
function _orthogonality_storage(o::Vector{Int64}, M)
    length(o) == 2 || throw(ArgumentError("orthogonality must be an interval (left, right); got $o"))
    _check_orthogonality(o[1], o[2], M)
    return o
end
_orthogonality_storage(o, M) = _orthogonality_storage(collect(Int64, o), M)

function _check_orthogonality(left, right, M)
    1 ≤ left ≤ right ≤ M ||
        throw(ArgumentError("orthogonality must be an interval (left, right) with 1 ≤ left ≤ right ≤ $M; got ($left, $right)"))
    return nothing
end

# Orthogonality interval `(left, right)` of `x`: the cores `k < left` are
# left-orthogonal and the cores `k > right` are right-orthogonal. Nothing is
# recorded about the cores inside the interval.
function _orthogonality(x::Union{AbstractTTVector, AbstractTTOperator})
    o = x.orthogonality
    return o[1], o[2]
end

# Overwrite the entries of the interval; objects sharing it with `x` see the update.
function _set_orthogonality!(x::Union{AbstractTTVector, AbstractTTOperator}, left::Integer, right::Integer)
    o = x.orthogonality
    _check_orthogonality(left, right, nsites(x))
    o[1] = left
    o[2] = right
    return x
end

_forget_orthogonality!(x) = _set_orthogonality!(x, 1, nsites(x))

# The orthogonality center of `x`, or `nothing` if the interval spans several cores.
function _orthogonality_center(x)
    left, right = _orthogonality(x)
    return left == right ? left : nothing
end

# Record that core `i` has been replaced by an arbitrary core.
function _core_replaced!(x, i::Integer)
    left, right = _orthogonality(x)
    return _set_orthogonality!(x, min(left, i), max(right, i))
end

# Record that core `i` has been made left-orthogonal and core `i + 1` replaced.
function _center_moved_right!(x, i::Integer)
    left, right = _orthogonality(x)
    return _set_orthogonality!(x, left ≥ i ? i + 1 : left, max(right, i + 1))
end

# Record that core `i` has been made right-orthogonal and core `i - 1` replaced.
function _center_moved_left!(x, i::Integer)
    left, right = _orthogonality(x)
    return _set_orthogonality!(x, min(left, i - 1), right ≤ i ? i - 1 : right)
end

# Interval of the train formed by the cores of `a` followed by the cores of `b`:
# the left-orthogonal cores of `a` and the right-orthogonal cores of `b` keep
# their orthogonality.
function _joined_orthogonality(a, b)
    return _orthogonality(a)[1], nsites(a) + _orthogonality(b)[2]
end

function Base.complex(A::AbstractTTOperator{T, M}) where {T, M}
    Ac = TTOperator{Complex{real(T)}, M}(
        complex.(A.cores), A.row_dims, A.col_dims, copy(A.ranks); orthogonality = copy(A.orthogonality)
    )
    return _rewrap(A, Ac)
end

function Base.complex(v::AbstractTTVector{T, M}) where {T, M}
    vc = TTVector{Complex{real(T)}, M}(
        complex.(v.cores), v.dims, copy(v.ranks); orthogonality = copy(v.orthogonality)
    )
    return _rewrap(v, vc)
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
    return Matrix(qr(rand(T, N, N)).Q)[1:n, 1:m]
end

"""
    rand_tt([T=Float64,] dims, ranks; normalize=false, orthogonal=false)
    rand_tt(dims, max_bond::Int; normalize=false, orthogonal=false)

Generate a random [`TTVector`](@ref) with element type `T`, physical dimensions
`dims`, and TT ranks `ranks` (a vector of length `length(dims) + 1` with
`ranks[1] == ranks[end] == 1`). Core entries are drawn from `randn`.

With an integer `max_bond`, every interior rank is set to `max_bond`, reduced where the
dimensions force a smaller rank (see [`admissible_ranks`](@ref)).

# Keyword arguments
- `normalize::Bool=false`: scale core `k` by `1/√(dims[k]·ranks[k+1])`.
- `orthogonal::Bool=false`: when `normalize` is also `true`, replace every core by
  a right-orthogonal core from a QR factorization. Has no effect otherwise.
"""
function rand_tt(dims, ranks; kwargs...)
    return rand_tt(Float64, dims, ranks; kwargs...)
end

function rand_tt(::Type{T}, dims, ranks; normalize = false, orthogonal = false) where {T}
    y = zeros_tt(T, dims, ranks)
    @inbounds for i in eachindex(y.cores)
        y.cores[i] = randn(T, dims[i], ranks[i], ranks[i + 1])
        if normalize
            y.cores[i] *= 1 / sqrt(dims[i] * ranks[i + 1])
            if orthogonal
                q, _ = qr(reshape(permutedims(y.cores[i], (1, 3, 2)), dims[i] * ranks[i + 1], ranks[i]))
                y.cores[i] = permutedims(reshape(Matrix(q), dims[i], ranks[i + 1], ranks[i]), (1, 3, 2))
            end
        end
    end
    return y
end

function rand_tt(dims, max_bond::Int; kwargs...)
    d = length(dims)
    ranks = max_bond * ones(Int, d + 1)
    ranks = admissible_ranks(ranks, dims; max_bond = max_bond)
    return rand_tt(dims, ranks; kwargs...)
end

"""
    rand_tt(x_tt::TTVector{T,N}; ε=convert(T,1e-5)) -> TTVector{T,N}

Generate a random tensor train (TT) vector by adding Gaussian noise to the input TT vector `x_tt`.

# Arguments
- `x_tt::TTVector{T,N}`: The input TT vector to which noise will be added.
- `ε`: The standard deviation of the Gaussian noise to be added. Default is `1e-5` converted to type `T`.

# Returns
- `TTVector{T,N}`: A new TT vector with added Gaussian noise and independent mutable storage.
"""
function rand_tt(x_tt::AbstractTTVector{T, N}; ε = convert(T, 1.0e-5)) where {T, N}
    tt_vec = copy(x_tt.cores)
    for i in eachindex(x_tt.cores)
        tt_vec[i] += ε * randn(x_tt.dims[i], x_tt.ranks[i], x_tt.ranks[i + 1])
    end
    return _rewrap(x_tt, TTVector{T, N}(tt_vec, x_tt.dims, copy(x_tt.ranks)))
end

"""
    Base.copy(x_tt::TTVector{T,N}) where {T<:Number,N}

Create a deep copy of a `TTVector` object.

# Arguments
- `x_tt::TTVector{T,N}`: The `TTVector` object to be copied, where `T` is a subtype of `Number` and `N` is the dimensionality.

# Returns
- A new `TTVector` object that is a deep copy of `x_tt`.
"""
Base.copy(x_tt::AbstractTTVector{T, N}) where {T <: Number, N} =
    _rewrap(x_tt, TTVector{T, N}(copy.(x_tt.cores), x_tt.dims, copy(x_tt.ranks); orthogonality = copy(x_tt.orthogonality)))

Base.copy(A::AbstractTTOperator{T, N}) where {T <: Number, N} =
    _rewrap(A, TTOperator{T, N}(copy.(A.cores), A.row_dims, A.col_dims, copy(A.ranks); orthogonality = copy(A.orthogonality)))

"""
    tt_decomp(tensor::Array; index=1, tol=1e-12) -> TTVector

Decompose a dense `tensor` into a [`TTVector`](@ref) with the TT-SVD
(hierarchical SVD) algorithm of Oseledets (2011); see also Schollwöck (2011).

The cores `k < index` are left-orthogonal, the cores `k > index` are
right-orthogonal, and core `index` carries the norm. At every SVD, singular
values smaller than `tol` are discarded; `tol` is an absolute threshold, not
relative to the norm of `tensor`.

* Oseledets, I. V. (2011). Tensor-train decomposition. *SIAM Journal on Scientific Computing*, 33(5), 2295-2317.
* Schollwöck, U. (2011). The density-matrix renormalization group in the age of matrix product states. *Annals of Physics*, 326(1), 96-192.
"""
function tt_decomp(tensor::Array{T, d}; index = 1, tol = 1.0e-12) where {T <: Number, d}
    # Decomposes a tensor into its tensor train with core matrices at i=index
    dims = size(tensor) #dims = [n_1,...,n_d]
    ttv_vec = Array{Array{T, 3}}(undef, d)
    rks = ones(Int64, d + 1)
    tensor_curr = tensor
    # Calculate ttv_vec[i] for i < index
    @inbounds for i in 1:(index - 1)
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

    # Define the return value as a TTVector
    return TTVector{T, d}(ttv_vec, dims, rks; orthogonality = (index, index))
end

"""
    tt_to_tensor(x_tt::TTVector{T,N}) where {T<:Number, N}

Convert a TTVector (Tensor Train vector) to a full tensor.

# Arguments
- `x_tt::TTVector{T,N}`: The input TTVector to be converted. `T` is the element type, and `N` is the number of dimensions.

# Returns
- A tensor of type `Array{T,N}` with the same dimensions as specified in `x_tt.dims`.
"""
function tt_to_tensor(x_tt::AbstractTTVector{T, N}) where {T <: Number, N}
    d = nsites(x_tt)
    # Progressive contraction: P holds the partial contraction of cores 1:k as a
    # (prod(dims[1:k]), r_k) matrix with the first physical index fastest, so the
    # final reshape matches Julia's column-major tensor layout. O(d·n·r²·prod)
    # instead of one O(d·r²) chain per entry.
    P = x_tt.cores[1][:, 1, :]
    for k in 2:d
        G = x_tt.cores[k]
        nk = size(G, 1)
        rk = size(G, 3)
        Pnew = Array{T, 3}(undef, size(P, 1), nk, rk)
        for s in 1:nk
            Pnew[:, s, :] = P * G[s, :, :]
        end
        P = reshape(Pnew, size(Pnew, 1) * nk, rk)
    end
    return reshape(P[:, 1], x_tt.dims)
end

"""
    tto_to_tt(A::TTOperator{T,N}) where {T<:Number,N}

Convert a `TTOperator` to a `TTVector`.

# Arguments
- `A::TTOperator{T,N}`: The TTOperator to be converted. `T` is the element type, and `N` is the number of dimensions.

# Returns
- `TTVector{T,N}`: The resulting TTVector.

# Details
This function takes a `TTOperator` and converts it into a `TTVector`. It reshapes the cores of the `TTOperator` and constructs a `TTVector` with the appropriate dimensions and ranks.
The result owns its cores, ranks, and orthogonality interval; mutating it does not change `A`.

"""
function tto_to_tt(A::AbstractTTOperator{T, N}) where {T <: Number, N}
    d = nsites(A)
    xtt_vec = Array{Array{T, 3}, 1}(undef, d)
    A_rks = A.ranks
    for i in eachindex(xtt_vec)
        xtt_vec[i] = reshape(copy(A.cores[i]), A.row_dims[i] * A.col_dims[i], A_rks[i], A_rks[i + 1])
    end
    return TTVector{T, N}(xtt_vec, A.row_dims .* A.col_dims, copy(A.ranks); orthogonality = copy(A.orthogonality))
end

"""
    tt_to_tto(x::TTVector{T,N}) where {T<:Number,N}

Convert a `TTVector` to a `TTOperator`.

# Arguments
- `x::TTVector{T,N}`: The input `TTVector` object to be converted. `T` is the element type, and `N` is the number of dimensions.

# Returns
- `TTOperator{T,N}`: The resulting `TTOperator` object.

# Throws
- `DimensionMismatch`: If the dimensions of the input `TTVector` are not perfect squares.

# Description
This function converts a `TTVector` to a `TTOperator` by reshaping the core tensors of the `TTVector` into 4-dimensional arrays. The reshaping is done such that the first two dimensions of each core tensor are the square roots of the original dimensions, and the last two dimensions are the ranks of the `TTVector`.
The result owns its cores, ranks, and orthogonality interval; mutating it does not change `x`.
"""
function tt_to_tto(x::AbstractTTVector{T, N}) where {T <: Number, N}
    @assert(isqrt.(x.dims) .^ 2 == x.dims, DimensionMismatch)
    d = nsites(x)
    Att_vec = Array{Array{T, 4}, 1}(undef, d)
    x_rks = x.ranks
    A_dims = isqrt.(x.dims)
    for i in eachindex(A_dims)
        Att_vec[i] = reshape(copy(x.cores[i]), A_dims[i], A_dims[i], x_rks[i], x_rks[i + 1])
    end
    return TTOperator{T, N}(Att_vec, A_dims, copy(x.ranks); orthogonality = copy(x.orthogonality))
end

"""
    tto_decomp(tensor::Array; index=1) -> TTOperator

Decompose a dense operator into a [`TTOperator`](@ref) with the TT-SVD algorithm.
`tensor` has `2d` indices ordered `[i₁, …, i_d, j₁, …, j_d]` (output indices
first, then input indices). The index pairs are interleaved to
`[(i₁, j₁), …, (i_d, j_d)]` and decomposed with [`tt_decomp`](@ref) using its
default tolerance.
"""
function tto_decomp(tensor::Array{T, N}; index = 1) where {T <: Number, N}
    d = Int(ndims(tensor) / 2)
    row_dims = size(tensor)[1:d]
    col_dims = size(tensor)[(d + 1):(2 * d)]
    fused_dims = row_dims .* col_dims
    index_sorted = vec(Transpose(reshape(1:(2 * d), :, 2)))
    ttv = tt_decomp(reshape(permutedims(tensor, index_sorted), fused_dims); index = index)
    rks = ttv.ranks
    cores = [reshape(ttv.cores[i], row_dims[i], col_dims[i], rks[i], rks[i + 1]) for i in 1:d]
    return TTOperator{T, d}(cores, row_dims, col_dims, rks; orthogonality = ttv.orthogonality)
end

"""
    tto_to_tensor(tto::TTOperator{T,N}) where {T<:Number, N}

Convert a TTOperator to a full tensor.

# Arguments
- `tto::TTOperator{T,N}`: The TTOperator to be converted, where `T` is a subtype of `Number` and `N` is the order of the tensor.

# Returns
- A tensor of type `Array{T, 2N}` with dimensions `[m_1, ..., m_d, n_1, ..., n_d]`, where `m_i` are the row dimensions and `n_i` the column dimensions of the TTOperator.
"""
function tto_to_tensor(tto::AbstractTTOperator{T, N}) where {T <: Number, N}
    d = nsites(tto)
    # Fuse each core's (i, j) pair, contract progressively as a TT vector, then
    # split the fused axes back and sort them into [i_1,…,i_d, j_1,…,j_d].
    fused = tt_to_tensor(tto_to_tt(tto))
    pairs = ntuple(i -> isodd(i) ? tto.row_dims[cld(i, 2)] : tto.col_dims[cld(i, 2)], 2 * d)
    split = reshape(fused, pairs)
    perm = (ntuple(k -> 2k - 1, d)..., ntuple(k -> 2k, d)...)
    return permutedims(split, perm)
end

"""
    admissible_ranks(ranks, dims; max_bond=1024)

Return a copy of `ranks` in which every rank is reduced to the largest value a
tensor train with physical dimensions `dims` can have at that bond, and to at
most `max_bond`.

`ranks` has length `length(dims) + 1`. Rank `i` is capped by the product of the
dimensions to its left and by the product of the dimensions to its right; the
boundary ranks are 1.
"""
function admissible_ranks(ranks, dims; max_bond = 1024)
    new_rks = ones(eltype(ranks), length(ranks))
    @simd for i in eachindex(dims)
        if prod(dims[i:end]) > 0
            if prod(dims[1:(i - 1)]) > 0
                new_rks[i] = min(ranks[i], prod(dims[1:(i - 1)]), prod(dims[i:end]), max_bond)
            else
                new_rks[i] = min(ranks[i], prod(dims[i:end]), max_bond)
            end
        else
            if prod(dims[1:(i - 1)]) > 0
                new_rks[i] = min(ranks[i], prod(dims[1:(i - 1)]), max_bond)
            else
                new_rks[i] = min(ranks[i], max_bond)
            end
        end
    end
    return new_rks
end

"""
    increase_ranks_noise(tt_vec, rkm, rk, noise)

Pad a single TT core to larger bond dimensions `(rkm, rk)`, filling the new
entries with `noise`-scaled random orthogonal values (private helper).

# Arguments
- `tt_vec::Array`: The input TT core.
- `rkm::Int`: The new rank for the second dimension.
- `rk::Int`: The new rank for the third dimension.
- `noise::Float64`: The noise level.

# Returns
- `vec_out::Array`: The padded TT core.
"""
function increase_ranks_noise(tt_vec, rkm, rk, noise)
    vec_out = zeros(eltype(tt_vec), size(tt_vec, 1), rkm, rk)
    vec_out[:, 1:size(tt_vec, 2), 1:size(tt_vec, 3)] = tt_vec
    if !iszero(noise)
        if rkm == size(tt_vec, 2) && rk > size(tt_vec, 3)
            Q = rand_orthogonal(size(tt_vec, 1) * rkm, rk - size(tt_vec, 3))
            vec_out[:, :, (size(tt_vec, 3) + 1):rk] = noise * reshape(Q, size(tt_vec, 1), rkm, rk - size(tt_vec, 3))
        elseif rk == size(tt_vec, 3) && rkm > size(tt_vec, 2)
            Q = rand_orthogonal(rkm - size(tt_vec, 2), size(tt_vec, 1) * rk)
            vec_out[:, (size(tt_vec, 2) + 1):rkm, :] = noise * reshape(Q, size(tt_vec, 1), rkm - size(tt_vec, 2), rk)
        elseif rk > size(tt_vec, 3) && rkm > size(tt_vec, 2)
            Q = rand_orthogonal((rkm - size(tt_vec, 2)) * size(tt_vec, 1), (rk - size(tt_vec, 3)))
            vec_out[:, (size(tt_vec, 2) + 1):rkm, (size(tt_vec, 3) + 1):rk] = noise * reshape(Q, size(tt_vec, 1), rkm - size(tt_vec, 2), rk - size(tt_vec, 3))
        end
    end
    return vec_out
end

"""
    increase_ranks(x_tt::TTVector{T,N}, max_bond::Int; ranks=vcat(1, max_bond*ones(Int, length(x_tt.dims)-1), 1), noise=0.0) where {T<:Number, N}

Increase the bond ranks of a Tensor Train (TT) vector `x_tt` up to `max_bond`,
padding the new rank dimensions with `noise`-scaled random orthogonal values.
With `noise=0` this is exact (zero-padding); with `noise>0` it enriches the TT
so a fixed-rank solver (e.g. ALS) has room to develop higher-rank structure.

# Arguments
- `x_tt::TTVector{T,N}`: The input TT vector.
- `max_bond::Int`: The maximum bond dimension to increase the ranks to.
- `ranks`: Optional. A vector specifying the target ranks. Defaults to `1` at the boundaries and `max_bond` in between.
- `noise::Float64`: Optional. The noise level added to the new rank dimensions. Defaults to `0.0` (exact zero-padding).

# Returns
- `TTVector{T,N}`: A new TT vector with increased ranks.
"""
function increase_ranks(x_tt::AbstractTTVector{T, N}, max_bond::Int; ranks = vcat(1, max_bond * ones(Int, length(x_tt.dims) - 1), 1), noise = 0.0) where {T <: Number, N}
    d = nsites(x_tt)
    vec_out = Array{Array{T}}(undef, d)
    @assert(max_bond > maximum(x_tt.ranks), "New bond dimension too low")
    ranks = admissible_ranks(ranks, x_tt.dims; max_bond = max_bond)
    for i in 1:d
        vec_out[i] = increase_ranks_noise(x_tt.cores[i], ranks[i], ranks[i + 1], noise)
    end
    return _rewrap(x_tt, TTVector{T, N}(vec_out, x_tt.dims, ranks))
end

"""
    orthogonalize(x_tt::TTVector{T,N}; i=1::Int) where {T<:Number, N}

Orthogonalizes the given Tensor Train (TT) vector `x_tt` with respect to the `i`-th core. The orthogonalization process involves QR and LQ decompositions so that the cores left of `i` are left-orthogonal and the cores right of `i` are right-orthogonal.

# Arguments
- `x_tt::TTVector{T,N}`: The input TT vector to be orthogonalized.
- `i::Int=1`: The core index with respect to which the orthogonalization is performed. Defaults to 1.

# Returns
- `y_tt`: The orthogonalized TT vector.

"""
function orthogonalize(x_tt::AbstractTTVector{T, N}; i = 1::Int) where {T <: Number, N}
    d = nsites(x_tt)
    @assert(1 ≤ i ≤ d, DimensionMismatch("Impossible orthogonalization"))
    y_rks = admissible_ranks(x_tt.ranks, x_tt.dims)
    y_tt = zeros_tt(T, x_tt.dims, y_rks)
    FR = ones(T, 1, 1)
    yleft_temp = zeros(T, maximum(x_tt.ranks), maximum(x_tt.dims), maximum(x_tt.ranks))
    for j in 1:(i - 1)
        @tensoropt((βⱼ₋₁, αⱼ), yleft_temp[1:y_tt.ranks[j], 1:x_tt.dims[j], 1:x_tt.ranks[j + 1]][αⱼ₋₁, iⱼ, αⱼ] = FR[αⱼ₋₁, βⱼ₋₁] * x_tt.cores[j][iⱼ, βⱼ₋₁, αⱼ])
        F = qr(reshape(yleft_temp[1:y_tt.ranks[j], 1:x_tt.dims[j], 1:x_tt.ranks[j + 1]], x_tt.dims[j] * y_tt.ranks[j], :))
        y_tt.ranks[j + 1] = size(Matrix(F.Q), 2)
        y_tt.cores[j] = permutedims(reshape(Matrix(F.Q), y_tt.ranks[j], x_tt.dims[j], y_tt.ranks[j + 1]), [2 1 3])
        FR = F.R[1:y_tt.ranks[j + 1], :]
    end
    FL = ones(T, 1, 1)
    (i < nsites(x_tt)) && (yright_temp = zeros(T, maximum(x_tt.ranks), maximum(y_tt.ranks), maximum(x_tt.dims)))
    for j in d:-1:(i + 1)
        yright_temp = zeros(T, x_tt.ranks[j], y_tt.ranks[j + 1], x_tt.dims[j])
        @tensoropt((αⱼ₋₁, αⱼ), yright_temp[1:x_tt.ranks[j], 1:y_tt.ranks[j + 1], 1:x_tt.dims[j]][αⱼ₋₁, βⱼ, iⱼ] = x_tt.cores[j][iⱼ, αⱼ₋₁, αⱼ] * FL[αⱼ, βⱼ])
        F = lq(reshape(yright_temp[1:x_tt.ranks[j], 1:y_tt.ranks[j + 1], 1:x_tt.dims[j]], x_tt.ranks[j], :))
        y_tt.ranks[j] = size(Matrix(F.Q), 1)
        y_tt.cores[j] = permutedims(reshape(Matrix(F.Q), y_tt.ranks[j], y_tt.ranks[j + 1], x_tt.dims[j]), [3 1 2])
        FL = F.L[:, 1:y_tt.ranks[j]]
    end
    _set_orthogonality!(y_tt, i, i)
    y_tt.cores[i] = zeros(T, y_tt.dims[i], y_tt.ranks[i], y_tt.ranks[i + 1])
    @simd for k in 1:x_tt.dims[i]
        y_tt.cores[i][k, :, :] = FR * x_tt.cores[i][k, :, :] * FL
    end
    return _rewrap(x_tt, y_tt)
end

"""
    entanglement_entropy(ψ::TTVector; base=exp(1.0))

Compute the von Neumann entanglement entropy across every bond of an MPS.

The returned vector has length `nsites(ψ) - 1`; entry `k` is the entropy of the
bipartition `1:k | k+1:N`. The input state is not mutated. Use `base = 2`
to return entropy in bits.
"""
function entanglement_entropy(ψ::AbstractTTVector; base::Real = exp(1.0))
    @assert base > 0 && base != 1 "base must be positive and not equal to 1"

    N = nsites(ψ)
    entropy = zeros(Float64, max(N - 1, 0))
    N <= 1 && return entropy
    logscale = log(base)

    canonical = orthogonalize(ψ; i = 1)
    cores = [permutedims(copy(core), (2, 1, 3)) for core in canonical.cores]

    for k in 1:(N - 1)
        A = cores[k]
        r_left, n, r_right = size(A)
        F = svd(reshape(A, r_left * n, r_right))

        probabilities = abs2.(F.S)
        total = sum(probabilities)
        if total > 0
            probabilities ./= total
            entropy[k] = -sum(p -> p > 0 ? p * log(p) : 0.0, probabilities) / logscale
        end

        if k < N - 1
            transfer = Diagonal(F.S) * F.Vt
            B = cores[k + 1]
            cores[k + 1] = reshape(
                transfer * reshape(B, size(B, 1), :),
                length(F.S), size(B, 2), size(B, 3)
            )
        end
    end
    return entropy
end

_dims_description(A::AbstractTTOperator) =
    A.row_dims == A.col_dims ? string(A.row_dims) : "$(A.row_dims) × $(A.col_dims)"

function _orthogonality_description(x)
    left, right = _orthogonality(x)
    left == right && return "center @ site $left"
    (left, right) == (1, nsites(x)) && return "none"
    return "center within sites $left:$right"
end

function Base.show(io::IO, tt::TTVector{T, N}) where {T <: Number, N}
    return print(io, "MPS{$T}($(nsites(tt)) sites)")
end

function Base.show(io::IO, tto::TTOperator{T, N}) where {T <: Number, N}
    return print(io, "MPO{$T}($(nsites(tto)) sites)")
end

function Base.show(io::IO, ::MIME"text/plain", tt::TTVector{T, N}) where {T <: Number, N}
    println(io, "MPS{$T} with $(nsites(tt)) sites")
    println(io, "  Physical dims : $(tt.dims)")
    println(io, "  Bond dims     : $(tt.ranks)")
    return print(io, "  Orthogonality : $(_orthogonality_description(tt))")
end

function Base.show(io::IO, ::MIME"text/plain", tto::TTOperator{T, N}) where {T <: Number, N}
    println(io, "MPO{$T} with $(nsites(tto)) sites")
    println(io, "  Physical dims : $(_dims_description(tto))")
    println(io, "  Bond dims     : $(tto.ranks)")
    return print(io, "  Orthogonality : $(_orthogonality_description(tto))")
end

"""
    visualize(tt)

Print an ASCII bond diagram of the TT structure to stdout.
"""
function visualize(tt::AbstractTTVector)
    N = nsites(tt)
    dims = collect(Int, tt.dims)::Vector{Int}
    ranks = tt.ranks::Vector{Int}
    rwidth = max(maximum(length.(string.(ranks))), 2)
    line1 = lpad(string(ranks[1]), rwidth)
    line2 = " "^length(line1)
    line3 = " "^length(line1)
    for i in 1:N
        rank_right = lpad(string(ranks[i + 1]), rwidth)
        line1 *= "-- • --" * rank_right
        position_C = length(line1) - rwidth - 3
        line2 *= repeat(" ", position_C - length(line2) - 1) * "|"
        dim_str = string(dims[i])
        line3 *= repeat(" ", position_C - length(line3) - div(length(dim_str), 2) - 1) * dim_str
    end
    println(line1); println(line2)
    return println(line3)
end

function visualize(tt::AbstractTTOperator)
    N = nsites(tt)
    ranks = tt.ranks::Vector{Int}
    total_length = 0
    positions_C = Int[]
    line1 = ""
    for i in 1:N
        seg = i == 1 ? " $(ranks[i])-- • --$(ranks[i + 1])" : "-- • --$(ranks[i + 1])"
        line1 *= seg
        push!(positions_C, total_length + findfirst(isequal('•'), seg))
        total_length += length(seg)
    end
    function dim_line(dims)
        buf = fill(' ', total_length)
        for i in 1:N
            s = string(dims[i]); start = positions_C[i] - div(length(s), 2)
            for (j, c) in enumerate(s)
                idx = start + j - 1
                1 ≤ idx ≤ total_length && (buf[idx] = c)
            end
        end
        return String(buf)
    end
    vline = String([p in positions_C ? '|' : ' ' for p in 1:total_length])
    println(dim_line(tt.row_dims)); println(vline); println(line1); println(vline)
    return println(dim_line(tt.col_dims))
end

"""
    matricize(qtt::TTVector, core::Int) -> Vector

Evaluate a binary (QTT) tensor train on the coarse grid spanned by its first
`core` bits, returning `2^core` values in big-endian order (bit 1 is the most
significant, matching `tuple_to_index`). The remaining sites are fixed at
physical index 1 (bit value 0). For `core == nsites(qtt)` this is the full grid
vector, identical to [`qtt_to_vector`](@ref).

The contraction is progressive — O(d·r²·2^core) — and never materializes the
full tensor.
"""
function matricize(qtt::AbstractTTVector{T}, core::Int)::Vector{T} where {T <: Number}
    d = nsites(qtt)
    @assert 1 ≤ core ≤ d "core must be in 1:$(d)"
    @assert all(==(2), qtt.dims) "matricize expects binary (QTT) physical dimensions"

    # Contract the trailing cores at physical index 1 into a boundary vector.
    v = ones(T, 1)
    for k in d:-1:(core + 1)
        v = qtt.cores[k][1, :, :] * v
    end

    # Progressive contraction over the first `core` cores, appending each bit
    # as the least-significant index (big-endian, as in `qtt_to_vector`).
    P = qtt.cores[1][:, 1, :]
    for k in 2:core
        G = qtt.cores[k]
        n_prev = size(P, 1)
        Pn = similar(P, 2 * n_prev, size(G, 3))
        @views begin
            Pn[1:2:end, :] .= P * G[1, :, :]
            Pn[2:2:end, :] .= P * G[2, :, :]
        end
        P = Pn
    end
    return P * v
end

"""
    concatenate(tt1::TTVector, tt2::TTVector) -> TTVector
    concatenate(A1::TTOperator, A2::TTOperator) -> TTOperator

Join two tensor trains into one with `nsites(tt1) + nsites(tt2)` cores: the cores of `tt1`
followed by the cores of `tt2`. The last rank of the first argument must equal
the first rank of the second; for standard boundary ranks of 1 the result
represents the tensor (Kronecker) product.
"""
function concatenate(tt1::AbstractTTVector, tt2::AbstractTTVector)
    if tt1.ranks[end] != tt2.ranks[1]
        throw(ArgumentError("The final rank of the first TTVector must equal the initial rank of the second TTVector."))
    end

    ttv_vec = vcat(tt1.cores, tt2.cores)
    ttv_dims = (tt1.dims..., tt2.dims...)
    ttv_rks = vcat(tt1.ranks[1:(end - 1)], tt2.ranks)

    return TTVector{eltype(tt1), length(ttv_dims)}(ttv_vec, ttv_dims, ttv_rks; orthogonality = _joined_orthogonality(tt1, tt2))
end


function concatenate(tt1::AbstractTTOperator, tt2::AbstractTTOperator)
    if tt1.ranks[end] != tt2.ranks[1]
        throw(ArgumentError("The final rank of the first TTOperator must equal the initial rank of the second TTOperator."))
    end

    tto_vec = vcat(tt1.cores, tt2.cores)
    row_dims = (tt1.row_dims..., tt2.row_dims...)
    col_dims = (tt1.col_dims..., tt2.col_dims...)
    tto_rks = vcat(tt1.ranks[1:(end - 1)], tt2.ranks)

    return TTOperator{eltype(tt1), length(row_dims)}(tto_vec, row_dims, col_dims, tto_rks; orthogonality = _joined_orthogonality(tt1, tt2))
end

# Make `cores[k]` right-orthonormal and multiply the triangular factor into
# `cores[k - 1]`. The bond between them shrinks to at most `n * r_right`.
function _orthogonalize_right!(cores::Vector, k)
    n, rl, rr = size(cores[k])
    F = lq(reshape(permutedims(cores[k], (2, 1, 3)), rl, n * rr))
    Q = Matrix(F.Q)
    r = size(Q, 1)
    cores[k] = permutedims(reshape(Q, r, n, rr), (2, 1, 3))
    prev = cores[k - 1]
    cores[k - 1] = reshape(reshape(prev, :, rl) * F.L[:, 1:r], size(prev, 1), size(prev, 2), r)
    return cores
end

# Make `cores[k]` left-orthonormal and multiply the triangular factor into
# `cores[k + 1]`. The bond between them shrinks to at most `n * r_left`.
function _orthogonalize_left!(cores::Vector, k)
    n, rl, rr = size(cores[k])
    F = qr(reshape(cores[k], n * rl, rr))
    Q = Matrix(F.Q)
    r = size(Q, 2)
    cores[k] = reshape(Q, n, rl, r)
    next = cores[k + 1]
    moved = F.R * reshape(permutedims(next, (2, 1, 3)), rr, :)
    cores[k + 1] = permutedims(reshape(moved, r, size(next, 1), size(next, 3)), (2, 1, 3))
    return cores
end

# Smallest rank whose discarded singular-value tail has Frobenius norm ≤ δ,
# capped at max_bond and floored at 1.
function _frob_trunc_rank(s::AbstractVector{<:Real}, δ::Real, max_bond::Int)
    r = length(s)
    if δ > 0
        tail = zero(float(eltype(s)))
        while r > 1 && sqrt(tail + abs2(s[r])) ≤ δ
            tail += abs2(s[r])
            r -= 1
        end
    end
    return max(min(r, max_bond), 1)
end

# Rank kept on one of the d − 1 bonds of a TT whose orthogonality center holds
# the singular values `s` (so ‖s‖ is the norm of the TT). Truncating every bond
# this way changes the TT by at most trunc_tol·‖s‖ in Frobenius norm.
function _trunc_rank(s::AbstractVector{<:Real}, trunc_tol::Real, d::Integer, max_bond::Integer)
    δ = trunc_tol > 0 ? trunc_tol * norm(s) / sqrt(max(d - 1, 1)) : 0.0
    return _frob_trunc_rank(s, δ, Int(min(max_bond, typemax(Int))))
end

# Relative Frobenius norm ‖s[r+1:end]‖ / ‖s‖ of the singular values a rank-r truncation discards.
_discarded_weight(s::AbstractVector{<:Real}, r::Integer) =
    r < length(s) ? norm(view(s, (r + 1):length(s))) / norm(s) : zero(float(eltype(s)))

# SVD of `A` truncated with `_trunc_rank`; the fourth value is the discarded weight.
function _truncated_svd(A::AbstractMatrix, trunc_tol::Real, d::Integer, max_bond::Integer)
    F = svd(A)
    r = _trunc_rank(F.S, trunc_tol, d, max_bond)
    return F.U[:, 1:r], Diagonal(F.S[1:r]), F.Vt[1:r, :], _discarded_weight(F.S, r)
end

# One TT-rounding pass (Oseledets 2011): a right-to-left orthogonalization via
# `orthogonalize`, then a left-to-right sweep of truncated SVDs with the
# orthogonality center carried along, so every SVD sees the state's true
# Schmidt values. `select` maps a singular-value vector to the rank to keep.
# Mutates the *contents* of x's containers (not just the field bindings) so
# wrappers sharing them — e.g. `QTTVector` — observe the update.
function _tt_truncate_sweep!(x::AbstractTTVector{T, N}, select::F) where {T <: Number, N, F}
    d = nsites(x)
    y = orthogonalize(x; i = 1)
    for k in 1:d
        x.cores[k] = y.cores[k]
    end
    x.ranks .= y.ranks
    _set_orthogonality!(x, 1, 1)
    d == 1 && return x
    for k in 1:(d - 1)
        C = x.cores[k]                          # center core, (n, r_l, r_r)
        n, rl, rr = size(C)
        F_svd = svd(reshape(permutedims(C, (2, 1, 3)), rl * n, rr))
        r_new = min(select(F_svd.S), length(F_svd.S))
        U = F_svd.U[:, 1:r_new]
        x.cores[k] = permutedims(reshape(U, rl, n, r_new), (2, 1, 3))
        x.ranks[k + 1] = r_new
        SV = Diagonal(F_svd.S[1:r_new]) * F_svd.Vt[1:r_new, :]
        B = x.cores[k + 1]                      # (n₂, r_r, r₃)
        n2 = size(B, 1)
        r3 = size(B, 3)
        Bnew = reshape(SV * reshape(permutedims(B, (2, 1, 3)), rr, n2 * r3), r_new, n2, r3)
        x.cores[k + 1] = permutedims(Bnew, (2, 1, 3))
        _set_orthogonality!(x, k + 1, k + 1)
    end
    return x
end

"""
    tt_round!(x::TTVector; trunc_tol=0.0, max_bond=typemax(Int))

Truncate the TT ranks of `x` in place with the TT-rounding algorithm of
Oseledets (2011): one right-to-left orthogonalization sweep followed by one
left-to-right truncating SVD sweep, O(d·n·r³) in total.

`trunc_tol` is a relative Frobenius tolerance for the whole tensor: each of the
`d − 1` bonds discards a singular-value tail of norm at most
`trunc_tol·‖x‖/√(d−1)`, so `‖x − round(x)‖ ≤ trunc_tol·‖x‖`. `max_bond`
additionally caps every bond dimension. The result is left-canonical with the
orthogonality center on the last core.
"""
function tt_round!(x::AbstractTTVector{T, N}; trunc_tol::Real = 0.0, max_bond::Int = typemax(Int)) where {T <: Number, N}
    return _tt_truncate_sweep!(x, s -> _trunc_rank(s, trunc_tol, nsites(x), max_bond))
end

"""
    tt_round(x::TTVector; trunc_tol=0.0, max_bond=typemax(Int))

Non-mutating variant of [`tt_round!`](@ref).
"""
tt_round(x::AbstractTTVector; kwargs...) = tt_round!(copy(x); kwargs...)

"""
    tt_compress!(ψ::TTVector, max_bond::Int; trunc_tol=0.0, sweeps=1, verbosity=1)

Compress `ψ` in place to bond dimension at most `max_bond` with TT rounding
(see [`tt_round!`](@ref)); `trunc_tol` has the same meaning as there. A single
sweep already gives the quasi-optimal rounding; `sweeps > 1` repeats the pass.
`verbosity ≥ 2` logs one line per pass.
"""
function tt_compress!(ψ::AbstractTTVector{T, N}, max_bond::Int; trunc_tol::Real = 0.0, sweeps::Int = 1, verbosity::Int = 1) where {T <: Number, N}
    sweeps ≥ 1 || throw(ArgumentError("`sweeps` must be ≥ 1; got $sweeps"))
    for sw in 1:sweeps
        verbosity ≥ 2 && @info "TT compress: sweep $sw"
        _tt_truncate_sweep!(ψ, s -> _trunc_rank(s, trunc_tol, nsites(ψ), max_bond))
    end
    return ψ
end
