module TensorTrainNumericsManoptExt

using ManifoldsBase
using Manopt
using TensorTrainNumerics

struct TTVectorSpace{T <: Real, N} <: ManifoldsBase.AbstractManifold{ManifoldsBase.ℝ}
    dims::NTuple{N, Int64}
    ranks::Vector{Int64}
end

function TensorTrainNumerics.ttvector_manifold(x::TTvector{T, N}) where {T <: Real, N}
    return TTVectorSpace{T, N}(x.ttv_dims, copy(x.ttv_rks))
end

function _copy_ttvector!(dst::TTvector, src::TTvector)
    dst.N = src.N
    dst.ttv_vec = src.ttv_vec
    dst.ttv_dims = src.ttv_dims
    dst.ttv_rks = src.ttv_rks
    dst.ttv_ot = src.ttv_ot
    return dst
end

ManifoldsBase.representation_size(M::TTVectorSpace) = M.dims
ManifoldsBase.default_retraction_method(::TTVectorSpace) = ManifoldsBase.ProjectionRetraction()
ManifoldsBase.default_retraction_method(M::TTVectorSpace, ::Type) =
    ManifoldsBase.default_retraction_method(M)
Manopt.max_stepsize(::TTVectorSpace) = Inf

function ManifoldsBase.allocate_result(
        ::TTVectorSpace, ::typeof(ManifoldsBase.zero_vector), p::TTvector
    )
    return zeros_tt(eltype(p), p.ttv_dims, p.ttv_rks)
end

function ManifoldsBase.copy(M::TTVectorSpace, p::TTvector)
    return TensorTrainNumerics.copy(p)
end

function ManifoldsBase.copyto!(::TTVectorSpace, q::TTvector, p::TTvector)
    return _copy_ttvector!(q, TensorTrainNumerics.copy(p))
end

function ManifoldsBase.copyto!(::TTVectorSpace, Y::TTvector, ::TTvector, X::TTvector)
    return _copy_ttvector!(Y, TensorTrainNumerics.copy(X))
end

function ManifoldsBase.zero_vector!(::TTVectorSpace, X::TTvector, ::TTvector)
    for core in X.ttv_vec
        fill!(core, zero(eltype(core)))
    end
    return X
end

function ManifoldsBase.inner(::TTVectorSpace, ::TTvector, X::TTvector, Y::TTvector)
    return real(TensorTrainNumerics.dot(X, Y))
end

function ManifoldsBase.norm(M::TTVectorSpace, p::TTvector, X::TTvector)
    return sqrt(max(ManifoldsBase.inner(M, p, X, X), zero(eltype(X))))
end

function ManifoldsBase.distance(M::TTVectorSpace, p::TTvector, q::TTvector)
    return ManifoldsBase.norm(M, p, p - q)
end

function ManifoldsBase.retract_project!(::TTVectorSpace, q::TTvector, p::TTvector, X::TTvector)
    return _copy_ttvector!(q, orthogonalize(p + X))
end

function ManifoldsBase.retract_project_fused!(
        ::TTVectorSpace, q::TTvector, p::TTvector, X::TTvector, t::Number
    )
    return _copy_ttvector!(q, orthogonalize(p + t * X))
end

end
