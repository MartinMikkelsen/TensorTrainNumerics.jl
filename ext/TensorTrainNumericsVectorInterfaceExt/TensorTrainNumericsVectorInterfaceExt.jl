module TensorTrainNumericsVectorInterfaceExt
using TensorTrainNumerics
using VectorInterface
import Base: zero

const _RankBoundedTTVector = TensorTrainNumerics._RankBoundedTTVector

_round(r::TTVector) = orthogonalize(r)

function _max_bond(y::_RankBoundedTTVector, x::_RankBoundedTTVector)
    y.max_bond == x.max_bond || throw(ArgumentError("cannot combine rank-bounded TT vectors with different bounds"))
    return y.max_bond
end

function _bound(r::TTVector, max_bond::Int)
    return _RankBoundedTTVector(tt_compress!(r, max_bond), max_bond)
end

const _overwrite! = TensorTrainNumerics._overwrite!

function _overwrite_bounded!(destination::_RankBoundedTTVector, source::TTVector)
    bounded = tt_compress!(source, destination.max_bond)
    _overwrite!(destination.tt, bounded)
    return destination
end

function _promoted_add(y::TTVector, x::TTVector, α::Number, β::Number)
    S = promote_type(eltype(y), eltype(x), typeof(α), typeof(β))
    return _round(convert(S, β) * y + convert(S, α) * x)
end

function VectorInterface.add(a::TTVector, b::TTVector)
    return _round(a + b)
end
function VectorInterface.add(a::TTOperator, b::TTOperator)
    return a + b
end

function VectorInterface.add(a::TTVector, b::TTVector, α::Number)
    return _round(a + α * b)
end
function VectorInterface.add(a::TTVector, b::TTVector, α::Number, β::Number)
    return _round(β * a + α * b)
end

function VectorInterface.add!(y::TTVector, x::TTVector)
    _overwrite!(y, y + x)
    return _round(y)
end
function VectorInterface.add!(y::TTVector{T, N}, x::TTVector{T, N}, α::Number, β::Number) where {T, N}
    αT = convert(T, α); βT = convert(T, β)
    _overwrite!(y, βT * y + αT * x)
    return _round(y)
end

function VectorInterface.add!!(y::TTVector, x::TTVector)
    return if promote_type(eltype(y), eltype(x)) <: eltype(y)
        VectorInterface.add!(y, x)
    else
        _round(y + x)
    end
end
function VectorInterface.add!!(y::TTVector, x::TTVector, α::Number)
    return if promote_type(eltype(y), eltype(x), typeof(α)) <: eltype(y)
        VectorInterface.add!(y, x, α, one(eltype(y)))
    else
        _promoted_add(y, x, α, one(eltype(y)))
    end
end
function VectorInterface.add!!(y::TTVector, x::TTVector, α::Number, β::Number)
    return if promote_type(eltype(y), eltype(x), typeof(α), typeof(β)) <: eltype(y)
        VectorInterface.add!(y, x, α, β)
    else
        _promoted_add(y, x, α, β)
    end
end

function VectorInterface.scale(x::TTVector, α::Number)
    return orthogonalize(α * x)
end
function VectorInterface.scale!(x::TTVector{T}, α::Number) where {T}
    αT = convert(T, α)
    i = TensorTrainNumerics._scale_site(x)
    @. x.ttv_vec[i] = αT * x.ttv_vec[i]
    return x
end
function VectorInterface.scale!!(x::TTVector{T}, α::Number) where {T}
    S = promote_type(T, typeof(α))
    return S === T ? orthogonalize(VectorInterface.scale!(x, α)) : orthogonalize(VectorInterface.scale(x, α))
end
function VectorInterface.scale!!(y::TTVector, x::TTVector, α::Number)
    S = promote_type(eltype(y), eltype(x), typeof(α))
    if S === eltype(y)
        return _round(_overwrite!(y, VectorInterface.scale(x, α)))
    else
        return VectorInterface.scale(x, α)
    end
end

function VectorInterface.zerovector(x::_RankBoundedTTVector, ::Type{S}) where {S <: Number}
    z = zeros_tt(S, x.tt.ttv_dims, copy(x.tt.ttv_rks))
    return _RankBoundedTTVector(z, x.max_bond)
end
function VectorInterface.zerovector!(x::_RankBoundedTTVector)
    VectorInterface.zerovector!(x.tt)
    return x
end
function VectorInterface.zerovector!!(x::_RankBoundedTTVector)
    VectorInterface.zerovector!!(x.tt)
    return x
end

function VectorInterface.scale(x::_RankBoundedTTVector, α::Number)
    return _RankBoundedTTVector(VectorInterface.scale(x.tt, α), x.max_bond)
end
function VectorInterface.scale!(x::_RankBoundedTTVector, α::Number)
    VectorInterface.scale!(x.tt, α)
    return x
end
function VectorInterface.scale!!(x::_RankBoundedTTVector, α::Number)
    return _RankBoundedTTVector(VectorInterface.scale!!(x.tt, α), x.max_bond)
end
function VectorInterface.scale!(y::_RankBoundedTTVector, x::_RankBoundedTTVector, α::Number)
    _max_bond(y, x)
    return _overwrite_bounded!(y, VectorInterface.scale(x.tt, α))
end
function VectorInterface.scale!!(y::_RankBoundedTTVector, x::_RankBoundedTTVector, α::Number)
    max_bond = _max_bond(y, x)
    return _RankBoundedTTVector(VectorInterface.scale!!(y.tt, x.tt, α), max_bond)
end

function VectorInterface.add(
        y::_RankBoundedTTVector, x::_RankBoundedTTVector,
        α::Number, β::Number
    )
    max_bond = _max_bond(y, x)
    return _bound(VectorInterface.add(y.tt, x.tt, α, β), max_bond)
end
function VectorInterface.add!(
        y::_RankBoundedTTVector, x::_RankBoundedTTVector,
        α::Number, β::Number
    )
    _max_bond(y, x)
    return _overwrite_bounded!(y, VectorInterface.add(y.tt, x.tt, α, β))
end
function VectorInterface.add!!(
        y::_RankBoundedTTVector, x::_RankBoundedTTVector,
        α::Number, β::Number
    )
    max_bond = _max_bond(y, x)
    return _bound(VectorInterface.add!!(y.tt, x.tt, α, β), max_bond)
end

function VectorInterface.zerovector(a::TTVector, ::Type{S}) where {S <: Number}
    return zeros_tt(S, a.ttv_dims, copy(a.ttv_rks))
end
function VectorInterface.zerovector(a::TTOperator, ::Type{S}) where {S <: Number}
    return zeros_tto(S, a.tto_dims, copy(a.tto_rks))
end
function VectorInterface.zerovector!(a::TTVector)
    for core in a.ttv_vec
        fill!(core, zero(eltype(core)))
    end
    return a
end
function VectorInterface.zerovector!!(a::TTVector)
    return VectorInterface.zerovector!(a)
end

VectorInterface.length(a::TTVector) = prod(a.ttv_dims)
VectorInterface.length(a::TTOperator) = prod(a.tto_dims)^2

zero(a::TTVector) = zeros_tt(eltype(a), a.ttv_dims, a.ttv_rks)

function VectorInterface.inner(a::TTVector, b::TTVector)
    return TensorTrainNumerics.dot(a, b)
end
function VectorInterface.inner(a::_RankBoundedTTVector, b::_RankBoundedTTVector)
    _max_bond(a, b)
    return VectorInterface.inner(a.tt, b.tt)
end

VectorInterface.norm(a::_RankBoundedTTVector) = norm(a.tt)

VectorInterface.scalartype(a::TTVector) = eltype(a)
VectorInterface.scalartype(a::TTOperator) = eltype(a)
VectorInterface.scalartype(::Type{<:TTVector{T}}) where {T} = T
VectorInterface.scalartype(::Type{<:TTOperator{T}}) where {T} = T
function VectorInterface.scalartype(::Type{<:_RankBoundedTTVector{V}}) where {V}
    return VectorInterface.scalartype(V)
end


end
