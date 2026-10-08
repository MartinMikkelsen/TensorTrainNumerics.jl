using Random
using LinearAlgebra
using IterativeSolvers
using TensorOperations
import Base: isempty, eltype, copy, complex
import KrylovKit: orthogonalize

"""
    AbstractTTvector

Supertype of tensor-train vectors: [`TTvector`](@ref) and the QTT wrapper
[`QTTvector`](@ref).
"""
abstract type AbstractTTvector end
"""
    TTvector{T<:Number, M}

A tensor in tensor-train (TT) format, also called a matrix product state:

    x[i₁, …, i_M] = G₁[i₁, :, :] * G₂[i₂, :, :] * ⋯ * G_M[i_M, :, :]

where core `Gₖ` has size `(nₖ, rₖ₋₁, rₖ)` and `r₀ = r_M = 1`.

# Fields
- `ttv_vec::Vector{Array{T,3}}`: the cores, each of size `(ttv_dims[k], ttv_rks[k], ttv_rks[k+1])`.
- `ttv_dims::NTuple{M,Int64}`: physical dimensions `(n₁, …, n_M)`.
- `ttv_rks::Vector{Int64}`: TT ranks `(r₀, r₁, …, r_M)`, of length `M + 1`.
- `orthogonality::Vector{Int64}`: the interval `[left, right]`, with
  `1 ≤ left ≤ right ≤ M`. The cores `k < left` are left-orthogonal and the cores
  `k > right` are right-orthogonal; nothing is recorded about the cores inside
  the interval. `[c, c]` is an orthogonality center at core `c`, and `[1, M]`
  records nothing.

The fields cannot be reassigned, but in-place operations such as
[`tt_round!`](@ref) and [`add!`](@ref) replace the contents of `ttv_vec`,
`ttv_rks`, and `orthogonality`; any other object built on the same vectors (such
as a [`QTTvector`](@ref) wrapping this `TTvector`) sees the change.

The number of cores (sites) is the type parameter `M`, returned by `nsites(x)`.

# Constructor

    TTvector{T, M}(cores, dims, ranks; orthogonality = (1, M))

A `Vector{Int64}` passed as `orthogonality` is stored without copying. The
constructor checks the orthogonality interval but not that the cores, `dims`,
and `ranks` are consistent with each other or that the cores have the recorded
orthogonality. Use constructors such as [`ttv_decomp`](@ref), [`rand_tt`](@ref),
or [`zeros_tt`](@ref) to build valid instances.
"""
struct TTvector{T <: Number, M} <: AbstractTTvector
    ttv_vec::Vector{Array{T, 3}}
    ttv_dims::NTuple{M, Int64}
    ttv_rks::Vector{Int64}
    orthogonality::Vector{Int64}
    function TTvector{T, M}(ttv_vec, ttv_dims, ttv_rks; orthogonality = (1, M)) where {T <: Number, M}
        return new{T, M}(ttv_vec, ttv_dims, ttv_rks, _orthogonality_storage(orthogonality, M))
    end
end

TTvector(ttv_vec::Vector{Array{T, 3}}, ttv_dims::NTuple{M, Int64}, ttv_rks; kwargs...) where {T <: Number, M} =
    TTvector{T, M}(ttv_vec, ttv_dims, ttv_rks; kwargs...)

Base.eltype(::TTvector{T, N}) where {T <: Number, N} = T


"""
    AbstractTToperator

Supertype of tensor-train operators: [`TToperator`](@ref) and the QTT wrapper
[`QTToperator`](@ref).
"""
abstract type AbstractTToperator end
"""
    TToperator{T<:Number, M}

A linear operator in tensor-train format, also called a matrix product operator:

    A[(i₁, …, i_M), (j₁, …, j_M)] = A₁[i₁, j₁, :, :] * ⋯ * A_M[i_M, j_M, :, :]

where core `Aₖ` has size `(nₖ, nₖ, rₖ₋₁, rₖ)` (output index, input index, left
rank, right rank) and `r₀ = r_M = 1`.

# Fields
- `tto_vec::Vector{Array{T,4}}`: the cores.
- `tto_dims::NTuple{M,Int64}`: physical dimensions `(n₁, …, n_M)`, shared by input and output.
- `tto_rks::Vector{Int64}`: TT ranks `(r₀, r₁, …, r_M)`, of length `M + 1`.
- `orthogonality::Vector{Int64}`: the orthogonality interval `[left, right]`,
  with the same meaning as for [`TTvector`](@ref).

The number of cores (sites) is the type parameter `M`, returned by `nsites(A)`.

# Constructor

    TToperator{T, M}(cores, dims, ranks; orthogonality = (1, M))

A `Vector{Int64}` passed as `orthogonality` is stored without copying. The
constructor checks the orthogonality interval but not that the other fields are
consistent with each other.
"""
struct TToperator{T <: Number, M} <: AbstractTToperator
    tto_vec::Vector{Array{T, 4}}
    tto_dims::NTuple{M, Int64}
    tto_rks::Vector{Int64}
    orthogonality::Vector{Int64}
    function TToperator{T, M}(tto_vec, tto_dims, tto_rks; orthogonality = (1, M)) where {T <: Number, M}
        return new{T, M}(tto_vec, tto_dims, tto_rks, _orthogonality_storage(orthogonality, M))
    end
end

TToperator(tto_vec::Vector{Array{T, 4}}, tto_dims::NTuple{M, Int64}, tto_rks; kwargs...) where {T <: Number, M} =
    TToperator{T, M}(tto_vec, tto_dims, tto_rks; kwargs...)

Base.eltype(::TToperator{T, M}) where {T, M} = T

"""
    nsites(x::Union{AbstractTTvector, AbstractTToperator}) -> Int

Number of cores (sites) of the tensor train `x`.
"""
function nsites end

# `getfield` because `getproperty` calls `nsites`.
nsites(x::AbstractTTvector) = length(getfield(x, :ttv_dims))
nsites(A::AbstractTToperator) = length(getfield(A, :tto_dims))

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
function _orthogonality(x::Union{AbstractTTvector, AbstractTToperator})
    o = getfield(x, :orthogonality)
    return o[1], o[2]
end

# Overwrite the entries of the interval; objects sharing it with `x` see the update.
function _set_orthogonality!(x::Union{AbstractTTvector, AbstractTToperator}, left::Integer, right::Integer)
    o = getfield(x, :orthogonality)
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

# Interval holding the guarantees of per-core flags (`1` left-orthogonal, `-1`
# right-orthogonal, `0` neither): the leading run of `1`s and the trailing run
# of `-1`s, shortened so that the interval contains at least one core.
function _flags_to_orthogonality(ot::AbstractVector, M)
    Base.require_one_based_indexing(ot)
    length(ot) == M || throw(ArgumentError("expected $M orthogonality flags; got $(length(ot))"))
    nleft = something(findfirst(!=(1), ot), M + 1) - 1
    nright = M - something(findlast(!=(-1), ot), 0)
    right = max(M - nright, 1)
    left = min(nleft + 1, right)
    return left, right
end

function _orthogonality_to_flags(x)
    left, right = _orthogonality(x)
    return [k < left ? 1 : k > right ? -1 : 0 for k in 1:nsites(x)]
end

# Deprecated: `N` (the number of cores) and the per-core orthogonality flags
# `ttv_ot`/`tto_ot` are no longer fields.
function Base.getproperty(x::Union{AbstractTTvector, AbstractTToperator}, name::Symbol)
    if name === :N
        Base.depwarn("the `N` field is deprecated, use `nsites(x)`.", :getproperty)
        return nsites(x)
    elseif name === :ttv_ot || name === :tto_ot
        Base.depwarn("the `$name` field is deprecated, use the `orthogonality` interval.", :getproperty)
        return _orthogonality_to_flags(x)
    end
    return getfield(x, name)
end

# Deprecated: constructors taking per-core orthogonality flags, optionally
# preceded by the number of cores.
const _FLAGS_DEPWARN = "passing per-core orthogonality flags is deprecated, use the `orthogonality = (left, right)` keyword."

function TTvector{T, M}(ttv_vec, ttv_dims, ttv_rks, ot) where {T <: Number, M}
    Base.depwarn("`TTvector`: " * _FLAGS_DEPWARN, :TTvector)
    return TTvector{T, M}(ttv_vec, ttv_dims, ttv_rks; orthogonality = _flags_to_orthogonality(ot, M))
end

TTvector(ttv_vec::Vector{Array{T, 3}}, ttv_dims::NTuple{M, Int64}, ttv_rks, ot) where {T <: Number, M} =
    TTvector{T, M}(ttv_vec, ttv_dims, ttv_rks, ot)

function TTvector{T, M}(N::Integer, ttv_vec, ttv_dims, ttv_rks, ot) where {T <: Number, M}
    Base.depwarn("`TTvector{T, M}(N, cores, dims, rks, ot)` is deprecated, omit `N`.", :TTvector)
    return TTvector{T, M}(ttv_vec, ttv_dims, ttv_rks, ot)
end

function TTvector(N::Integer, ttv_vec::Vector{Array{T, 3}}, ttv_dims::NTuple{M, Int64}, ttv_rks, ot) where {T <: Number, M}
    Base.depwarn("`TTvector(N, cores, dims, rks, ot)` is deprecated, omit `N`.", :TTvector)
    return TTvector{T, M}(ttv_vec, ttv_dims, ttv_rks, ot)
end

function TToperator{T, M}(tto_vec, tto_dims, tto_rks, ot) where {T <: Number, M}
    Base.depwarn("`TToperator`: " * _FLAGS_DEPWARN, :TToperator)
    return TToperator{T, M}(tto_vec, tto_dims, tto_rks; orthogonality = _flags_to_orthogonality(ot, M))
end

TToperator(tto_vec::Vector{Array{T, 4}}, tto_dims::NTuple{M, Int64}, tto_rks, ot) where {T <: Number, M} =
    TToperator{T, M}(tto_vec, tto_dims, tto_rks, ot)

function TToperator{T, M}(N::Integer, tto_vec, tto_dims, tto_rks, ot) where {T <: Number, M}
    Base.depwarn("`TToperator{T, M}(N, cores, dims, rks, ot)` is deprecated, omit `N`.", :TToperator)
    return TToperator{T, M}(tto_vec, tto_dims, tto_rks, ot)
end

function TToperator(N::Integer, tto_vec::Vector{Array{T, 4}}, tto_dims::NTuple{M, Int64}, tto_rks, ot) where {T <: Number, M}
    Base.depwarn("`TToperator(N, cores, dims, rks, ot)` is deprecated, omit `N`.", :TToperator)
    return TToperator{T, M}(tto_vec, tto_dims, tto_rks, ot)
end

function Base.complex(A::TToperator{T, M}) where {T, M}
    return TToperator{Complex{real(T)}, M}(
        complex.(A.tto_vec), A.tto_dims, copy(A.tto_rks); orthogonality = copy(A.orthogonality)
    )
end

function Base.complex(v::TTvector{T, M}) where {T, M}
    return TTvector{Complex{real(T)}, M}(
        complex.(v.ttv_vec), v.ttv_dims, copy(v.ttv_rks); orthogonality = copy(v.orthogonality)
    )
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
    rand_tt([T=Float64,] dims, rks; normalize=false, orthogonal=false)
    rand_tt(dims, rmax::Int; normalize=false, orthogonal=false)

Generate a random [`TTvector`](@ref) with element type `T`, physical dimensions
`dims`, and TT ranks `rks` (a vector of length `length(dims) + 1` with
`rks[1] == rks[end] == 1`). Core entries are drawn from `randn`.

With an integer `rmax`, every interior rank is set to `rmax`, reduced where the
dimensions force a smaller rank (see [`r_and_d_to_rks`](@ref)).

# Keyword arguments
- `normalize::Bool=false`: scale core `k` by `1/√(dims[k]·rks[k+1])`.
- `orthogonal::Bool=false`: when `normalize` is also `true`, replace every core by
  a right-orthogonal core from a QR factorization. Has no effect otherwise.
"""
function rand_tt(dims, rks; kwargs...)
    return rand_tt(Float64, dims, rks; kwargs...)
end

function rand_tt(::Type{T}, dims, rks; normalize = false, orthogonal = false, normalise = nothing) where {T}
    if normalise !== nothing
        Base.depwarn("the `normalise` keyword of `rand_tt` is deprecated, use `normalize`.", :rand_tt)
        normalize = normalise
    end
    y = zeros_tt(T, dims, rks)
    @inbounds for i in eachindex(y.ttv_vec)
        y.ttv_vec[i] = randn(T, dims[i], rks[i], rks[i + 1])
        if normalize
            y.ttv_vec[i] *= 1 / sqrt(dims[i] * rks[i + 1])
            if orthogonal
                q, _ = qr(reshape(permutedims(y.ttv_vec[i], (1, 3, 2)), dims[i] * rks[i + 1], rks[i]))
                y.ttv_vec[i] = permutedims(reshape(Matrix(q), dims[i], rks[i + 1], rks[i]), (1, 3, 2))
            end
        end
    end
    return y
end

function rand_tt(dims, rmax::Int; kwargs...)
    d = length(dims)
    rks = rmax * ones(Int, d + 1)
    rks = r_and_d_to_rks(rks, dims; rmax = rmax)
    return rand_tt(dims, rks; kwargs...)
end

"""
    rand_tt(x_tt::TTvector{T,N}; ε=convert(T,1e-5)) -> TTvector{T,N}

Generate a random tensor train (TT) vector by adding Gaussian noise to the input TT vector `x_tt`.

# Arguments
- `x_tt::TTvector{T,N}`: The input TT vector to which noise will be added.
- `ε`: The standard deviation of the Gaussian noise to be added. Default is `1e-5` converted to type `T`.

# Returns
- `TTvector{T,N}`: A new TT vector with added Gaussian noise and independent mutable storage.
"""
function rand_tt(x_tt::TTvector{T, N}; ε = convert(T, 1.0e-5)) where {T, N}
    tt_vec = copy(x_tt.ttv_vec)
    for i in eachindex(x_tt.ttv_vec)
        tt_vec[i] += ε * randn(x_tt.ttv_dims[i], x_tt.ttv_rks[i], x_tt.ttv_rks[i + 1])
    end
    return TTvector{T, N}(tt_vec, x_tt.ttv_dims, copy(x_tt.ttv_rks))
end

"""
    Base.copy(x_tt::TTvector{T,N}) where {T<:Number,N}

Create a deep copy of a `TTvector` object.

# Arguments
- `x_tt::TTvector{T,N}`: The `TTvector` object to be copied, where `T` is a subtype of `Number` and `N` is the dimensionality.

# Returns
- A new `TTvector` object that is a deep copy of `x_tt`.
"""
Base.copy(x_tt::TTvector{T, N}) where {T <: Number, N} =
    TTvector{T, N}(copy.(x_tt.ttv_vec), x_tt.ttv_dims, copy(x_tt.ttv_rks); orthogonality = copy(x_tt.orthogonality))

"""
    ttv_decomp(tensor::Array; index=1, tol=1e-12) -> TTvector

Decompose a dense `tensor` into a [`TTvector`](@ref) with the TT-SVD
(hierarchical SVD) algorithm of Oseledets (2011); see also Schollwöck (2011).

The cores `k < index` are left-orthogonal, the cores `k > index` are
right-orthogonal, and core `index` carries the norm. At every SVD, singular
values smaller than `tol` are discarded; `tol` is an absolute threshold, not
relative to the norm of `tensor`.

* Oseledets, I. V. (2011). Tensor-train decomposition. *SIAM Journal on Scientific Computing*, 33(5), 2295-2317.
* Schollwöck, U. (2011). The density-matrix renormalization group in the age of matrix product states. *Annals of Physics*, 326(1), 96-192.
"""
function ttv_decomp(tensor::Array{T, d}; index = 1, tol = 1.0e-12) where {T <: Number, d}
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

    # Define the return value as a TTvector
    return TTvector{T, d}(ttv_vec, dims, rks; orthogonality = (index, index))
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
    d = nsites(x_tt)
    # Progressive contraction: P holds the partial contraction of cores 1:k as a
    # (prod(dims[1:k]), r_k) matrix with the first physical index fastest, so the
    # final reshape matches Julia's column-major tensor layout. O(d·n·r²·prod)
    # instead of one O(d·r²) chain per entry.
    P = x_tt.ttv_vec[1][:, 1, :]
    for k in 2:d
        G = x_tt.ttv_vec[k]
        nk = size(G, 1)
        rk = size(G, 3)
        Pnew = Array{T, 3}(undef, size(P, 1), nk, rk)
        for s in 1:nk
            Pnew[:, s, :] = P * G[s, :, :]
        end
        P = reshape(Pnew, size(Pnew, 1) * nk, rk)
    end
    return reshape(P[:, 1], x_tt.ttv_dims)
end

"""
    tto_to_ttv(A::TToperator{T,N}) where {T<:Number,N}

Convert a `TToperator` to a `TTvector`.

# Arguments
- `A::TToperator{T,N}`: The TToperator to be converted. `T` is the element type, and `N` is the number of dimensions.

# Returns
- `TTvector{T,N}`: The resulting TTvector.

# Details
This function takes a `TToperator` and converts it into a `TTvector`. It reshapes the cores of the `TToperator` and constructs a `TTvector` with the appropriate dimensions and ranks.
The result owns its cores, ranks, and orthogonality flags; mutating it does not change `A`.

"""
function tto_to_ttv(A::TToperator{T, N}) where {T <: Number, N}
    d = nsites(A)
    xtt_vec = Array{Array{T, 3}, 1}(undef, d)
    A_rks = A.tto_rks
    for i in eachindex(xtt_vec)
        xtt_vec[i] = reshape(copy(A.tto_vec[i]), A.tto_dims[i]^2, A_rks[i], A_rks[i + 1])
    end
    return TTvector{T, N}(xtt_vec, A.tto_dims .^ 2, copy(A.tto_rks); orthogonality = copy(A.orthogonality))
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
The result owns its cores, ranks, and orthogonality flags; mutating it does not change `x`.
"""
function ttv_to_tto(x::TTvector{T, N}) where {T <: Number, N}
    @assert(isqrt.(x.ttv_dims) .^ 2 == x.ttv_dims, DimensionMismatch)
    d = nsites(x)
    Att_vec = Array{Array{T, 4}, 1}(undef, d)
    x_rks = x.ttv_rks
    A_dims = isqrt.(x.ttv_dims)
    for i in eachindex(A_dims)
        Att_vec[i] = reshape(copy(x.ttv_vec[i]), A_dims[i], A_dims[i], x_rks[i], x_rks[i + 1])
    end
    return TToperator{T, N}(Att_vec, A_dims, copy(x.ttv_rks); orthogonality = copy(x.orthogonality))
end

"""
    tto_decomp(tensor::Array; index=1) -> TToperator

Decompose a dense operator into a [`TToperator`](@ref) with the TT-SVD algorithm.
`tensor` has `2d` indices ordered `[i₁, …, i_d, j₁, …, j_d]` (output indices
first, then input indices). The index pairs are interleaved to
`[(i₁, j₁), …, (i_d, j_d)]` and decomposed with [`ttv_decomp`](@ref) using its
default tolerance.
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
    index_sorted = vec(Transpose(reshape(1:(2 * d), :, 2)))
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
    return TToperator{T, d}(tto_vec, tto_dims, rks; orthogonality = ttv.orthogonality)
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
    d = nsites(tto)
    # Fuse each core's (i, j) pair, contract progressively as a TT vector, then
    # split the fused axes back and sort them into [i_1,…,i_d, j_1,…,j_d].
    fused = ttv_to_tensor(tto_to_ttv(tto))
    pairs = ntuple(i -> tto.tto_dims[cld(i, 2)], 2 * d)
    split = reshape(fused, pairs)
    perm = (ntuple(k -> 2k - 1, d)..., ntuple(k -> 2k, d)...)
    return permutedims(split, perm)
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
    increase_ranks(x_tt::TTvector{T,N}, max_bond::Int; rks=vcat(1, max_bond*ones(Int, length(x_tt.ttv_dims)-1), 1), noise=0.0) where {T<:Number, N}

Increase the bond ranks of a Tensor Train (TT) vector `x_tt` up to `max_bond`,
padding the new rank dimensions with `noise`-scaled random orthogonal values.
With `noise=0` this is exact (zero-padding); with `noise>0` it enriches the TT
so a fixed-rank solver (e.g. ALS) has room to develop higher-rank structure.

# Arguments
- `x_tt::TTvector{T,N}`: The input TT vector.
- `max_bond::Int`: The maximum bond dimension to increase the ranks to.
- `rks`: Optional. A vector specifying the target ranks. Defaults to `1` at the boundaries and `max_bond` in between.
- `noise::Float64`: Optional. The noise level added to the new rank dimensions. Defaults to `0.0` (exact zero-padding).

# Returns
- `TTvector{T,N}`: A new TT vector with increased ranks.
"""
function increase_ranks(x_tt::TTvector{T, N}, max_bond::Int; rks = vcat(1, max_bond * ones(Int, length(x_tt.ttv_dims) - 1), 1), noise = 0.0) where {T <: Number, N}
    d = nsites(x_tt)
    vec_out = Array{Array{T}}(undef, d)
    @assert(max_bond > maximum(x_tt.ttv_rks), "New bond dimension too low")
    rks = r_and_d_to_rks(rks, x_tt.ttv_dims; rmax = max_bond)
    for i in 1:d
        vec_out[i] = increase_ranks_noise(x_tt.ttv_vec[i], rks[i], rks[i + 1], noise)
    end
    return TTvector{T, N}(vec_out, x_tt.ttv_dims, rks)
end

# Deprecated: renamed to `increase_ranks` (the `ϵ_wn` keyword is now `noise`).
function tt_up_rks(x, max_bond::Int; ϵ_wn = 0.0, kwargs...)
    Base.depwarn("`tt_up_rks` is deprecated, use `increase_ranks` (the `ϵ_wn` keyword is now `noise`).", :tt_up_rks)
    return increase_ranks(x, max_bond; noise = ϵ_wn, kwargs...)
end

"""
    orthogonalize(x_tt::TTvector{T,N}; i=1::Int) where {T<:Number, N}

Orthogonalizes the given Tensor Train (TT) vector `x_tt` with respect to the `i`-th core. The orthogonalization process involves QR and LQ decompositions so that the cores left of `i` are left-orthogonal and the cores right of `i` are right-orthogonal.

# Arguments
- `x_tt::TTvector{T,N}`: The input TT vector to be orthogonalized.
- `i::Int=1`: The core index with respect to which the orthogonalization is performed. Defaults to 1.

# Returns
- `y_tt`: The orthogonalized TT vector.

"""
function orthogonalize(x_tt::TTvector{T, N}; i = 1::Int) where {T <: Number, N}
    d = nsites(x_tt)
    @assert(1 ≤ i ≤ d, DimensionMismatch("Impossible orthogonalization"))
    y_rks = r_and_d_to_rks(x_tt.ttv_rks, x_tt.ttv_dims)
    y_tt = zeros_tt(T, x_tt.ttv_dims, y_rks)
    FR = ones(T, 1, 1)
    yleft_temp = zeros(T, maximum(x_tt.ttv_rks), maximum(x_tt.ttv_dims), maximum(x_tt.ttv_rks))
    for j in 1:(i - 1)
        @tensoropt((βⱼ₋₁, αⱼ), yleft_temp[1:y_tt.ttv_rks[j], 1:x_tt.ttv_dims[j], 1:x_tt.ttv_rks[j + 1]][αⱼ₋₁, iⱼ, αⱼ] = FR[αⱼ₋₁, βⱼ₋₁] * x_tt.ttv_vec[j][iⱼ, βⱼ₋₁, αⱼ])
        F = qr(reshape(yleft_temp[1:y_tt.ttv_rks[j], 1:x_tt.ttv_dims[j], 1:x_tt.ttv_rks[j + 1]], x_tt.ttv_dims[j] * y_tt.ttv_rks[j], :))
        y_tt.ttv_rks[j + 1] = size(Matrix(F.Q), 2)
        y_tt.ttv_vec[j] = permutedims(reshape(Matrix(F.Q), y_tt.ttv_rks[j], x_tt.ttv_dims[j], y_tt.ttv_rks[j + 1]), [2 1 3])
        FR = F.R[1:y_tt.ttv_rks[j + 1], :]
    end
    FL = ones(T, 1, 1)
    (i < nsites(x_tt)) && (yright_temp = zeros(T, maximum(x_tt.ttv_rks), maximum(y_tt.ttv_rks), maximum(x_tt.ttv_dims)))
    for j in d:-1:(i + 1)
        yright_temp = zeros(T, x_tt.ttv_rks[j], y_tt.ttv_rks[j + 1], x_tt.ttv_dims[j])
        @tensoropt((αⱼ₋₁, αⱼ), yright_temp[1:x_tt.ttv_rks[j], 1:y_tt.ttv_rks[j + 1], 1:x_tt.ttv_dims[j]][αⱼ₋₁, βⱼ, iⱼ] = x_tt.ttv_vec[j][iⱼ, αⱼ₋₁, αⱼ] * FL[αⱼ, βⱼ])
        F = lq(reshape(yright_temp[1:x_tt.ttv_rks[j], 1:y_tt.ttv_rks[j + 1], 1:x_tt.ttv_dims[j]], x_tt.ttv_rks[j], :))
        y_tt.ttv_rks[j] = size(Matrix(F.Q), 1)
        y_tt.ttv_vec[j] = permutedims(reshape(Matrix(F.Q), y_tt.ttv_rks[j], y_tt.ttv_rks[j + 1], x_tt.ttv_dims[j]), [3 1 2])
        FL = F.L[:, 1:y_tt.ttv_rks[j]]
    end
    _set_orthogonality!(y_tt, i, i)
    y_tt.ttv_vec[i] = zeros(T, y_tt.ttv_dims[i], y_tt.ttv_rks[i], y_tt.ttv_rks[i + 1])
    @simd for k in 1:x_tt.ttv_dims[i]
        y_tt.ttv_vec[i][k, :, :] = FR * x_tt.ttv_vec[i][k, :, :] * FL
    end
    return y_tt
end

"""
    entanglement_entropy(ψ::TTvector; base=exp(1.0))

Compute the von Neumann entanglement entropy across every bond of an MPS.

The returned vector has length `nsites(ψ) - 1`; entry `k` is the entropy of the
bipartition `1:k | k+1:N`. The input state is not mutated. Use `base = 2`
to return entropy in bits.
"""
function entanglement_entropy(ψ::TTvector; base::Real = exp(1.0))
    @assert base > 0 && base != 1 "base must be positive and not equal to 1"

    N = nsites(ψ)
    entropy = zeros(Float64, max(N - 1, 0))
    N <= 1 && return entropy
    logscale = log(base)

    canonical = orthogonalize(ψ; i = 1)
    cores = [permutedims(copy(core), (2, 1, 3)) for core in canonical.ttv_vec]

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

# Deprecated: renamed to `entanglement_entropy`.
function entanglemententropy(args...; kwargs...)
    Base.depwarn("`entanglemententropy` is deprecated, use `entanglement_entropy`.", :entanglemententropy)
    return entanglement_entropy(args...; kwargs...)
end

function _orthogonality_description(x)
    left, right = _orthogonality(x)
    left == right && return "center @ site $left"
    (left, right) == (1, nsites(x)) && return "none"
    return "center within sites $left:$right"
end

function Base.show(io::IO, tt::TTvector{T, N}) where {T <: Number, N}
    return print(io, "MPS{$T}($(nsites(tt)) sites)")
end

function Base.show(io::IO, tto::TToperator{T, N}) where {T <: Number, N}
    return print(io, "MPO{$T}($(nsites(tto)) sites)")
end

function Base.show(io::IO, ::MIME"text/plain", tt::TTvector{T, N}) where {T <: Number, N}
    println(io, "MPS{$T} with $(nsites(tt)) sites")
    println(io, "  Physical dims : $(tt.ttv_dims)")
    println(io, "  Bond dims     : $(tt.ttv_rks)")
    return print(io, "  Orthogonality : $(_orthogonality_description(tt))")
end

function Base.show(io::IO, ::MIME"text/plain", tto::TToperator{T, N}) where {T <: Number, N}
    println(io, "MPO{$T} with $(nsites(tto)) sites")
    println(io, "  Physical dims : $(tto.tto_dims)")
    println(io, "  Bond dims     : $(tto.tto_rks)")
    return print(io, "  Orthogonality : $(_orthogonality_description(tto))")
end

"""
    visualize(tt)

Print an ASCII bond diagram of the TT structure to stdout.
"""
function visualize(tt::TTvector)
    N = nsites(tt)
    dims = collect(tt.ttv_dims)
    ranks = tt.ttv_rks
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

function visualize(tt::TToperator)
    N = nsites(tt)
    dims = collect(tt.tto_dims)
    ranks = tt.tto_rks
    total_length = 0
    positions_C = Int[]
    line1 = ""
    for i in 1:N
        seg = i == 1 ? " $(ranks[i])-- • --$(ranks[i + 1])" : "-- • --$(ranks[i + 1])"
        line1 *= seg
        push!(positions_C, total_length + findfirst(isequal('•'), seg))
        total_length += length(seg)
    end
    function dim_line()
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
    println(dim_line()); println(vline); println(line1); println(vline)
    return println(dim_line())
end

"""
    matricize(qtt::TTvector, core::Int) -> Vector

Evaluate a binary (QTT) tensor train on the coarse grid spanned by its first
`core` bits, returning `2^core` values in big-endian order (bit 1 is the most
significant, matching `tuple_to_index`). The remaining sites are fixed at
physical index 1 (bit value 0). For `core == nsites(qtt)` this is the full grid
vector, identical to [`qtt_to_vector`](@ref).

The contraction is progressive — O(d·r²·2^core) — and never materializes the
full tensor.
"""
function matricize(qtt::TTvector{T}, core::Int)::Vector{T} where {T <: Number}
    d = nsites(qtt)
    @assert 1 ≤ core ≤ d "core must be in 1:$(d)"
    @assert all(==(2), qtt.ttv_dims) "matricize expects binary (QTT) physical dimensions"

    # Contract the trailing cores at physical index 1 into a boundary vector.
    v = ones(T, 1)
    for k in d:-1:(core + 1)
        v = qtt.ttv_vec[k][1, :, :] * v
    end

    # Progressive contraction over the first `core` cores, appending each bit
    # as the least-significant index (big-endian, as in `qtt_to_vector`).
    P = qtt.ttv_vec[1][:, 1, :]
    for k in 2:core
        G = qtt.ttv_vec[k]
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
    concatenate(tt1::TTvector, tt2::TTvector) -> TTvector
    concatenate(A1::TToperator, A2::TToperator) -> TToperator

Join two tensor trains into one with `nsites(tt1) + nsites(tt2)` cores: the cores of `tt1`
followed by the cores of `tt2`. The last rank of the first argument must equal
the first rank of the second; for standard boundary ranks of 1 the result
represents the tensor (Kronecker) product.
"""
function concatenate(tt1::TTvector, tt2::TTvector)
    if tt1.ttv_rks[end] != tt2.ttv_rks[1]
        throw(ArgumentError("The final rank of the first TTvector must equal the initial rank of the second TTvector."))
    end

    ttv_vec = vcat(tt1.ttv_vec, tt2.ttv_vec)
    ttv_dims = (tt1.ttv_dims..., tt2.ttv_dims...)
    ttv_rks = vcat(tt1.ttv_rks[1:(end - 1)], tt2.ttv_rks)

    return TTvector{eltype(tt1), length(ttv_dims)}(ttv_vec, ttv_dims, ttv_rks; orthogonality = _joined_orthogonality(tt1, tt2))
end


function concatenate(tt1::TToperator, tt2::TToperator)
    if tt1.tto_rks[end] != tt2.tto_rks[1]
        throw(ArgumentError("The final rank of the first TToperator must equal the initial rank of the second TToperator."))
    end

    tto_vec = vcat(tt1.tto_vec, tt2.tto_vec)
    tto_dims = (tt1.tto_dims..., tt2.tto_dims...)
    tto_rks = vcat(tt1.tto_rks[1:(end - 1)], tt2.tto_rks)

    return TToperator{eltype(tt1), length(tto_dims)}(tto_vec, tto_dims, tto_rks; orthogonality = _joined_orthogonality(tt1, tt2))
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
# wrappers sharing them — e.g. `QTTvector` — observe the update.
function _tt_truncate_sweep!(x::TTvector{T, N}, select::F) where {T <: Number, N, F}
    d = nsites(x)
    y = orthogonalize(x; i = 1)
    for k in 1:d
        x.ttv_vec[k] = y.ttv_vec[k]
    end
    x.ttv_rks .= y.ttv_rks
    _set_orthogonality!(x, 1, 1)
    d == 1 && return x
    for k in 1:(d - 1)
        C = x.ttv_vec[k]                          # center core, (n, r_l, r_r)
        n, rl, rr = size(C)
        F_svd = svd(reshape(permutedims(C, (2, 1, 3)), rl * n, rr))
        r_new = min(select(F_svd.S), length(F_svd.S))
        U = F_svd.U[:, 1:r_new]
        x.ttv_vec[k] = permutedims(reshape(U, rl, n, r_new), (2, 1, 3))
        x.ttv_rks[k + 1] = r_new
        SV = Diagonal(F_svd.S[1:r_new]) * F_svd.Vt[1:r_new, :]
        B = x.ttv_vec[k + 1]                      # (n₂, r_r, r₃)
        n2 = size(B, 1)
        r3 = size(B, 3)
        Bnew = reshape(SV * reshape(permutedims(B, (2, 1, 3)), rr, n2 * r3), r_new, n2, r3)
        x.ttv_vec[k + 1] = permutedims(Bnew, (2, 1, 3))
        _set_orthogonality!(x, k + 1, k + 1)
    end
    return x
end

"""
    tt_round!(x::TTvector; trunc_tol=0.0, max_bond=typemax(Int))

Truncate the TT ranks of `x` in place with the TT-rounding algorithm of
Oseledets (2011): one right-to-left orthogonalization sweep followed by one
left-to-right truncating SVD sweep, O(d·n·r³) in total.

`trunc_tol` is a relative Frobenius tolerance for the whole tensor: each of the
`d − 1` bonds discards a singular-value tail of norm at most
`trunc_tol·‖x‖/√(d−1)`, so `‖x − round(x)‖ ≤ trunc_tol·‖x‖`. `max_bond`
additionally caps every bond dimension. The result is left-canonical with the
orthogonality center on the last core.
"""
function tt_round!(x::TTvector{T, N}; trunc_tol::Real = 0.0, max_bond::Int = typemax(Int)) where {T <: Number, N}
    return _tt_truncate_sweep!(x, s -> _trunc_rank(s, trunc_tol, nsites(x), max_bond))
end

"""
    tt_round(x::TTvector; trunc_tol=0.0, max_bond=typemax(Int))

Non-mutating variant of [`tt_round!`](@ref).
"""
tt_round(x::TTvector; kwargs...) = tt_round!(copy(x); kwargs...)

"""
    tt_compress!(ψ::TTvector, max_bond::Int; trunc_tol=0.0, sweeps=1, verbosity=1)

Compress `ψ` in place to bond dimension at most `max_bond` with TT rounding
(see [`tt_round!`](@ref)); `trunc_tol` has the same meaning as there. A single
sweep already gives the quasi-optimal rounding; `sweeps > 1` repeats the pass.
`verbosity ≥ 2` logs one line per pass.
"""
function tt_compress!(ψ::TTvector{T, N}, max_bond::Int; trunc_tol::Real = 0.0, sweeps::Int = 1, verbosity::Int = 1) where {T <: Number, N}
    sweeps ≥ 1 || throw(ArgumentError("`sweeps` must be ≥ 1; got $sweeps"))
    for sw in 1:sweeps
        verbosity ≥ 2 && @info "TT compress: sweep $sw"
        _tt_truncate_sweep!(ψ, s -> _trunc_rank(s, trunc_tol, nsites(ψ), max_bond))
    end
    return ψ
end
