module TensorTrainNumericsVectorInterfaceExt
using TensorTrainNumerics
using VectorInterface
import Base: zero, copy

# Optional rank rounding for the rank-growing `add` operations. When the parent
# package's `KRYLOV_ROUND_RANK` is > 0 (set only during a Krylov solve), truncate
# the sum back to that bond dimension; otherwise just orthogonalize, the default,
# exact behavior used by Manopt and everything else. This keeps Krylov basis and
# iterate updates from accumulating rank through repeated vector additions.
function _round(r::TTvector)
    rk = TensorTrainNumerics.KRYLOV_ROUND_RANK[]
    return rk > 0 ? tt_compress!(r, rk) : orthogonalize(r)
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
        _round(y + α * x)
    end
end
function VectorInterface.add!!(y::TTvector, x::TTvector, α::Number, β::Number)
    return if promote_type(eltype(y), eltype(x), typeof(α), typeof(β)) <: eltype(y)
        VectorInterface.add!(y, x, α, β)
    else
        r = _round(β * y + α * x)
        y.ttv_vec = r.ttv_vec; y.ttv_rks = r.ttv_rks; y.ttv_dims = r.ttv_dims; y.ttv_ot = r.ttv_ot
        y
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

function VectorInterface.zerovector(a::TTvector)
    return zeros_tt(eltype(a), a.ttv_dims, a.ttv_rks)
end
function VectorInterface.zerovector(a::TToperator)
    return zeros_tt(eltype(a), a.tto_dims, a.tto_rks)
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
VectorInterface.length(a::TToperator) = prod(a.tto_dims)

zero(a::TTvector) = zeros_tt(eltype(a), a.ttv_dims, a.ttv_rks)

function VectorInterface.inner(a::TTvector, b::TTvector)
    return TensorTrainNumerics.dot(a, b)
end

VectorInterface.scalartype(a::TTvector) = eltype(a)
VectorInterface.scalartype(a::TToperator) = eltype(a)


end
