module TensorTrainNumericsVectorInterfaceExt
using TensorTrainNumerics
using VectorInterface
import Base: zero

const _RankBoundedTTvector = TensorTrainNumerics._RankBoundedTTvector

_round(r::TTvector) = orthogonalize(r)

function _max_bond(y::_RankBoundedTTvector, x::_RankBoundedTTvector)
    y.max_bond == x.max_bond || throw(ArgumentError("cannot combine rank-bounded TT vectors with different bounds"))
    return y.max_bond
end

function _bound(r::TTvector, max_bond::Int)
    return _RankBoundedTTvector(tt_compress!(r, max_bond), max_bond)
end

function _overwrite!(destination::TTvector, source::TTvector)
    destination.ttv_vec = source.ttv_vec
    destination.ttv_rks = source.ttv_rks
    destination.ttv_dims = source.ttv_dims
    destination.ttv_ot = source.ttv_ot
    return destination
end

function _overwrite_bounded!(destination::_RankBoundedTTvector, source::TTvector)
    bounded = tt_compress!(source, destination.max_bond)
    _overwrite!(destination.tt, bounded)
    return destination
end

function _promoted_add(y::TTvector, x::TTvector, α::Number, β::Number)
    S = promote_type(eltype(y), eltype(x), typeof(α), typeof(β))
    return _round(convert(S, β) * y + convert(S, α) * x)
end

function VectorInterface.add(a::TTvector, b::TTvector)
    return _round(a + b)
end
function VectorInterface.add(a::TToperator, b::TToperator)
    return orthogonalize(a + b)
end

function VectorInterface.add(a::TTvector, b::TTvector, α::Number)
    return _round(a + α * b)
end
function VectorInterface.add(a::TTvector, b::TTvector, α::Number, β::Number)
    return _round(β * a + α * b)
end

function VectorInterface.add!(y::TTvector, x::TTvector)
    r = y + x
    y.ttv_vec = r.ttv_vec; y.ttv_rks = r.ttv_rks; y.ttv_dims = r.ttv_dims; y.ttv_ot = r.ttv_ot
    return _round(y)
end
function VectorInterface.add!(y::TTvector{T, N}, x::TTvector{T, N}, α::Number, β::Number) where {T, N}
    αT = convert(T, α); βT = convert(T, β)
    r = βT * y + αT * x
    y.ttv_vec = r.ttv_vec; y.ttv_rks = r.ttv_rks; y.ttv_dims = r.ttv_dims; y.ttv_ot = r.ttv_ot
    return _round(y)
end

function VectorInterface.add!!(y::TTvector, x::TTvector)
    return if promote_type(eltype(y), eltype(x)) <: eltype(y)
        VectorInterface.add!(y, x)
    else
        _round(y + x)
    end
end
function VectorInterface.add!!(y::TTvector, x::TTvector, α::Number)
    return if promote_type(eltype(y), eltype(x), typeof(α)) <: eltype(y)
        VectorInterface.add!(y, x, α, one(eltype(y)))
    else
        _promoted_add(y, x, α, one(eltype(y)))
    end
end
function VectorInterface.add!!(y::TTvector, x::TTvector, α::Number, β::Number)
    return if promote_type(eltype(y), eltype(x), typeof(α), typeof(β)) <: eltype(y)
        VectorInterface.add!(y, x, α, β)
    else
        _promoted_add(y, x, α, β)
    end
end

function VectorInterface.scale(x::TTvector, α::Number)
    return orthogonalize(α * x)
end
function VectorInterface.scale!(x::TTvector{T}, α::Number) where {T}
    αT = convert(T, α)
    i = findfirst(==(0), x.ttv_ot); i === nothing && (i = 1)
    @. x.ttv_vec[i] = αT * x.ttv_vec[i]
    return x
end
function VectorInterface.scale!!(x::TTvector{T}, α::Number) where {T}
    S = promote_type(T, typeof(α))
    return S === T ? orthogonalize(VectorInterface.scale!(x, α)) : orthogonalize(VectorInterface.scale(x, α))
end
function VectorInterface.scale!!(y::TTvector, x::TTvector, α::Number)
    r = VectorInterface.scale(x, α)
    y.ttv_vec = r.ttv_vec; y.ttv_rks = r.ttv_rks; y.ttv_dims = r.ttv_dims; y.ttv_ot = r.ttv_ot
    return orthogonalize(y)
end

function VectorInterface.zerovector(x::_RankBoundedTTvector, ::Type{S}) where {S <: Number}
    z = zeros_tt(S, x.tt.ttv_dims, copy(x.tt.ttv_rks))
    return _RankBoundedTTvector(z, x.max_bond)
end
function VectorInterface.zerovector!(x::_RankBoundedTTvector)
    VectorInterface.zerovector!(x.tt)
    return x
end
function VectorInterface.zerovector!!(x::_RankBoundedTTvector)
    VectorInterface.zerovector!!(x.tt)
    return x
end

function VectorInterface.scale(x::_RankBoundedTTvector, α::Number)
    return _RankBoundedTTvector(VectorInterface.scale(x.tt, α), x.max_bond)
end
function VectorInterface.scale!(x::_RankBoundedTTvector, α::Number)
    VectorInterface.scale!(x.tt, α)
    return x
end
function VectorInterface.scale!!(x::_RankBoundedTTvector, α::Number)
    return _RankBoundedTTvector(VectorInterface.scale!!(x.tt, α), x.max_bond)
end
function VectorInterface.scale!(y::_RankBoundedTTvector, x::_RankBoundedTTvector, α::Number)
    _max_bond(y, x)
    return _overwrite_bounded!(y, VectorInterface.scale(x.tt, α))
end
function VectorInterface.scale!!(y::_RankBoundedTTvector, x::_RankBoundedTTvector, α::Number)
    max_bond = _max_bond(y, x)
    return _RankBoundedTTvector(VectorInterface.scale!!(y.tt, x.tt, α), max_bond)
end

function VectorInterface.add(
        y::_RankBoundedTTvector, x::_RankBoundedTTvector,
        α::Number, β::Number
    )
    max_bond = _max_bond(y, x)
    return _bound(VectorInterface.add(y.tt, x.tt, α, β), max_bond)
end
function VectorInterface.add!(
        y::_RankBoundedTTvector, x::_RankBoundedTTvector,
        α::Number, β::Number
    )
    _max_bond(y, x)
    return _overwrite_bounded!(y, VectorInterface.add(y.tt, x.tt, α, β))
end
function VectorInterface.add!!(
        y::_RankBoundedTTvector, x::_RankBoundedTTvector,
        α::Number, β::Number
    )
    max_bond = _max_bond(y, x)
    return _bound(VectorInterface.add!!(y.tt, x.tt, α, β), max_bond)
end

function VectorInterface.zerovector(a::TTvector)
    return zeros_tt(eltype(a), a.ttv_dims, a.ttv_rks)
end
function VectorInterface.zerovector(a::TToperator)
    return zeros_tto(eltype(a), a.tto_dims, copy(a.tto_rks))
end
function VectorInterface.zerovector!(a::TTvector)
    for core in a.ttv_vec
        fill!(core, zero(eltype(core)))
    end
    return a
end
function VectorInterface.zerovector!!(a::TTvector)
    return VectorInterface.zerovector!(a)
end

VectorInterface.length(a::TTvector) = prod(a.ttv_dims)
VectorInterface.length(a::TToperator) = prod(a.tto_dims)^2

zero(a::TTvector) = zeros_tt(eltype(a), a.ttv_dims, a.ttv_rks)

function VectorInterface.inner(a::TTvector, b::TTvector)
    return TensorTrainNumerics.dot(a, b)
end
function VectorInterface.inner(a::_RankBoundedTTvector, b::_RankBoundedTTvector)
    _max_bond(a, b)
    return VectorInterface.inner(a.tt, b.tt)
end

VectorInterface.norm(a::_RankBoundedTTvector) = norm(a.tt)

VectorInterface.scalartype(a::TTvector) = eltype(a)
VectorInterface.scalartype(a::TToperator) = eltype(a)
VectorInterface.scalartype(::Type{<:TTvector{T}}) where {T} = T
function VectorInterface.scalartype(::Type{<:_RankBoundedTTvector{V}}) where {V}
    return VectorInterface.scalartype(V)
end


end
