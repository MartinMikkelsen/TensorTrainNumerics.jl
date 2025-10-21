module TensorTrainNumericsVectorInterfaceExt
using TensorTrainNumerics
using VectorInterface
import Base: zero, copy

function VectorInterface.add(a::TTvector, b::TTvector)
    orthogonalize(a + b)
end
function VectorInterface.add(a::TToperator, b::TToperator)
    orthogonalize(a + b)
end

function VectorInterface.add(a::TTvector, b::TTvector, α::Number)
    orthogonalize(a + α*b)
end
function VectorInterface.add(a::TTvector, b::TTvector, α::Number, β::Number)
    orthogonalize(β*a + α*b)
end

function VectorInterface.add!(y::TTvector, x::TTvector)
    r = y + x
    y.ttv_vec = r.ttv_vec; y.ttv_rks = r.ttv_rks; y.ttv_dims = r.ttv_dims; y.ttv_ot = r.ttv_ot
    orthogonalize(y)
end
function VectorInterface.add!(y::TTvector{T,N}, x::TTvector{T,N}, α::Number, β::Number) where {T,N}
    αT = convert(T, α); βT = convert(T, β)
    r = βT*y + αT*x
    y.ttv_vec = r.ttv_vec; y.ttv_rks = r.ttv_rks; y.ttv_dims = r.ttv_dims; y.ttv_ot = r.ttv_ot
    orthogonalize(y)
end

function VectorInterface.add!!(y::TTvector, x::TTvector)
    if promote_type(eltype(y), eltype(x)) <: eltype(y)
        VectorInterface.add!(y, x)
    else
        orthogonalize(y + x)
    end
end
function VectorInterface.add!!(y::TTvector, x::TTvector, α::Number, β::Number)
    if promote_type(eltype(y), eltype(x), typeof(α), typeof(β)) <: eltype(y)
        VectorInterface.add!(y, x, α, β)
    else
        r = orthogonalize(β*y + α*x)
        y.ttv_vec = r.ttv_vec; y.ttv_rks = r.ttv_rks; y.ttv_dims = r.ttv_dims; y.ttv_ot = r.ttv_ot
        y
    end
end

function VectorInterface.scale(x::TTvector, α::Number)
    orthogonalize(α * x)
end
function VectorInterface.scale!(x::TTvector{T}, α::Number) where {T}
    αT = convert(T, α)
    i = findfirst(==(0), x.ttv_ot); i === nothing && (i = 1)
    @. x.ttv_vec[i] = αT * x.ttv_vec[i]
    x
end
function VectorInterface.scale!!(x::TTvector{T}, α::Number) where {T}
    S = promote_type(T, typeof(α))
    S === T ? orthogonalize(VectorInterface.scale!(x, α)) : orthogonalize(VectorInterface.scale(x, α))
end
function VectorInterface.scale!!(y::TTvector, x::TTvector, α::Number)
    r = VectorInterface.scale(x, α)
    y.ttv_vec = r.ttv_vec; y.ttv_rks = r.ttv_rks; y.ttv_dims = r.ttv_dims; y.ttv_ot = r.ttv_ot
    orthogonalize(y)
end

function VectorInterface.zerovector(a::TTvector)
    zeros_tt(eltype(a), a.ttv_dims, a.ttv_rks)
end
function VectorInterface.zerovector(a::TToperator)
    zeros_tt(eltype(a), a.tto_dims, a.tto_rks)
end

VectorInterface.length(a::TTvector)   = prod(a.ttv_dims)
VectorInterface.length(a::TToperator) = prod(a.tto_dims)

zero(a::TTvector) = zeros_tt(eltype(a), a.ttv_dims, a.ttv_rks)

function VectorInterface.inner(a::TTvector, b::TTvector)
    real(TensorTrainNumerics.dot(a, b))
end

VectorInterface.scalartype(a::TTvector)   = eltype(a)
VectorInterface.scalartype(a::TToperator) = eltype(a)


end
