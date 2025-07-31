module TensorTrainNumericsVectorInterfaceExt
using TensorTrainNumerics
using VectorInterface
using Base.Threads
import Base: zero, copy, +
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

function VectorInterface.scale(x::TTvector, α::Number)
    return orthogonalize(α * x)
end

function VectorInterface.zerovector(a::TTvector)
    T = eltype(a)
    return zeros_tt(T, a.ttv_dims, a.ttv_rks)
end

function VectorInterface.zerovector(a::TToperator)
    return zeros_tt(eltype(a), a.tto_dims, a.tto_rks)
end

function VectorInterface.length(a::TTvector)
    return length(a.ttv_dims)
end
function VectorInterface.length(a::TToperator)
    return length(a.tto_dims)
end

zero(a::TTvector) = zeros_tt(eltype(a), a.ttv_dims, a.ttv_rks)

function VectorInterface.inner(a::TTvector, b::TTvector)
    return TensorTrainNumerics.dot(a, b)
end


function LinearAlgebra.norm(a::TTvector)
    v = TensorTrainNumerics.dot(a, a)
    return sqrt(max(v, 0.0))
end
function VectorInterface.scalartype(a::TTvector)
    return eltype(a)
end

function VectorInterface.scalartype(a::TToperator)
    return eltype(a)
end

end
