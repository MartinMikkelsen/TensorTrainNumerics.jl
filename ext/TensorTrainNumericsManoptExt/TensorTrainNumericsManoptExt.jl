module TensorTrainNumericsManoptExt

using ManifoldsBase
using Manopt
using TensorTrainNumerics

struct TTVectorSpace{T <: Real, N} <: ManifoldsBase.AbstractManifold{ManifoldsBase.ℝ}
    dims::NTuple{N, Int64}
    ranks::Vector{Int64}
end

function TensorTrainNumerics.ttvector_manifold(x::TTVector{T, N}) where {T <: Real, N}
    return TTVectorSpace{T, N}(x.dims, copy(x.ranks))
end

_copy_ttvector!(dst::TTVector, src::TTVector) = TensorTrainNumerics._overwrite!(dst, src)

ManifoldsBase.representation_size(M::TTVectorSpace) = M.dims
ManifoldsBase.default_retraction_method(::TTVectorSpace) = ManifoldsBase.ProjectionRetraction()
ManifoldsBase.default_retraction_method(M::TTVectorSpace, ::Type) =
    ManifoldsBase.default_retraction_method(M)
Manopt.max_stepsize(::TTVectorSpace) = Inf

function ManifoldsBase.allocate_result(
        ::TTVectorSpace, ::typeof(ManifoldsBase.zero_vector), p::TTVector
    )
    return zeros_tt(eltype(p), p.dims, p.ranks)
end

function ManifoldsBase.copy(M::TTVectorSpace, p::TTVector)
    return TensorTrainNumerics.copy(p)
end

function ManifoldsBase.copyto!(::TTVectorSpace, q::TTVector, p::TTVector)
    return _copy_ttvector!(q, TensorTrainNumerics.copy(p))
end

function ManifoldsBase.copyto!(::TTVectorSpace, Y::TTVector, ::TTVector, X::TTVector)
    return _copy_ttvector!(Y, TensorTrainNumerics.copy(X))
end

function ManifoldsBase.zero_vector!(::TTVectorSpace, X::TTVector, ::TTVector)
    for core in X.cores
        fill!(core, zero(eltype(core)))
    end
    return X
end

function ManifoldsBase.inner(::TTVectorSpace, ::TTVector, X::TTVector, Y::TTVector)
    return real(TensorTrainNumerics.dot(X, Y))
end

function ManifoldsBase.norm(M::TTVectorSpace, p::TTVector, X::TTVector)
    return sqrt(max(ManifoldsBase.inner(M, p, X, X), zero(eltype(X))))
end

function ManifoldsBase.distance(M::TTVectorSpace, p::TTVector, q::TTVector)
    return ManifoldsBase.norm(M, p, p - q)
end

function ManifoldsBase.retract_project!(::TTVectorSpace, q::TTVector, p::TTVector, X::TTVector)
    return _copy_ttvector!(q, orthogonalize(p + X))
end

function ManifoldsBase.retract_project_fused!(
        ::TTVectorSpace, q::TTVector, p::TTVector, X::TTVector, t::Number
    )
    return _copy_ttvector!(q, orthogonalize(p + t * X))
end

end
