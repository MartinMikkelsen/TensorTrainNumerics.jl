module TensorTrainNumericsVectorInterfaceExt
using TensorTrainNumerics
using VectorInterface
import Base: zero, copy
import LinearAlgebra: dot, norm


function VectorInterface.add(a::TTvector, b::TTvector)
    return orthogonalize(a + b)
end
function VectorInterface.add(a::TToperator, b::TToperator)
    return orthogonalize(a + b)
end
function VectorInterface.add(a::TTvector, b::TTvector, α::Number)
    return orthogonalize(a + b * α)
end
function VectorInterface.add(a::TTvector, b::TTvector, α::Number, β::Number)
    return orthogonalize(β * a + α * b)
end

function VectorInterface.add!(y::TTvector, x::TTvector)
    result = y + x
    y.ttv_vec = result.ttv_vec
    y.ttv_rks = result.ttv_rks
    y.ttv_dims = result.ttv_dims
    y.ttv_ot = result.ttv_ot
    return orthogonalize(y)
end

function VectorInterface.add!(y::TTvector{T, N}, x::TTvector{T, N}, α::Number, β::Number) where {T, N}
    αT = convert(T, α)
    βT = convert(T, β)
    result = αT * y + βT * x
    y.ttv_vec = result.ttv_vec
    y.ttv_rks = result.ttv_rks
    y.ttv_dims = result.ttv_dims
    y.ttv_ot = result.ttv_ot
    return orthogonalize(y)
end


function VectorInterface.add!!(y::TTvector, x::TTvector, α::Number, β::Number)
    if VectorInterface.promote_add(y, x, α, β) <: scalartype(y)
        return VectorInterface.add!(y, x, α, β)
    else
        return orthogonalize(α * y + β * x)
    end
end

function VectorInterface.scale(x::TTvector, α::Number)
    return orthogonalize(α * x)
end

function VectorInterface.scale!(x::TTvector, α::Number)
    i = findfirst(isequal(0), x.ttv_ot)
    x.ttv_vec[i] .= α .* x.ttv_vec[i]
    return x
end

function VectorInterface.scale!!(x::TTvector{T, N}, α::Number) where {T, N}
    T2 = typejoin(T, typeof(α))
    return if T2 === T
        orthogonalize(VectorInterface.scale!(x, α))
    else
        orthogonalize(VectorInterface.scale(x, α))
    end
end

function VectorInterface.scale!!(y::TTvector, x::TTvector, α::Number)
    result = VectorInterface.scale(x, α)
    y.ttv_vec = result.ttv_vec
    y.ttv_rks = result.ttv_rks
    y.ttv_dims = result.ttv_dims
    y.ttv_ot = result.ttv_ot
    return orthogonalize(y)
end

function VectorInterface.zerovector(a::TTvector)
    T = eltype(a)
    return zeros_tt(T, a.ttv_dims, a.ttv_rks)
end

function VectorInterface.zerovector(a::TToperator)
    return zeros_tt(eltype(a), a.tto_dims, a.tto_rks)
end


VectorInterface.length(a::TTvector) = prod(a.ttv_dims)

VectorInterface.length(a::TToperator) = prod(a.tto_dims)

zero(a::TTvector) = zeros_tt(eltype(a), a.ttv_dims, a.ttv_rks)

function VectorInterface.inner(a::TTvector, b::TTvector)
    return TensorTrainNumerics.dot(a, b)
end

function VectorInterface.scalartype(a::TTvector)
    return eltype(a)
end

function VectorInterface.scalartype(a::TToperator)
    return eltype(a)
end

function Base.copy(x::TTvector)
    return TTvector(
        x.N,
        copy.(x.ttv_vec),
        copy(x.ttv_dims),
        copy(x.ttv_rks),
        copy(x.ttv_ot),
    )
end

end
