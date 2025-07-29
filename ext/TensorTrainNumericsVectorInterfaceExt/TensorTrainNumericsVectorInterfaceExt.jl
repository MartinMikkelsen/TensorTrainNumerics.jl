module TensorTrainNumericsVectorInterfaceExt
using TensorTrainNumerics
using VectorInterface

function VectorInterface.add(a::TTvector, b::TTvector)
    return a + b
end

function VectorInterface.add(a::TToperator, b::TToperator)
    return a + b
end

function VectorInterface.add(a::TTvector, b::TTvector, c::Number)
    return a + b * c
end


function VectorInterface.inner(a::TTvector, b::TTvector)
    return TensorTrainNumerics.dot(a, b)
end

function VectorInterface.scale(a::TTvector, c::Number)
    return a * c
end

function VectorInterface.scale(a::TToperator, c::Number)
    return a * c
end

function VectorInterface.zerovector(a::TTvector)
    return zeros_tt(eltype(a), a.ttv_dims, a.ttv_rks)
end

function VectorInterface.zerovector(a::TTvector, T::Type{<:Number})
    return zeros_tt(T, a.ttv_dims, a.ttv_rks)
end

function VectorInterface.zerovector(a::TToperator, T::Type{<:Number})
    return zeros_tt(T, a.tto_dims, a.tto_rks)
end

function VectorInterface.zerovector(a::TToperator)
    return zeros_tt(eltype(a), a.tto_dims, a.tto_rks)
end

function VectorInterface.scalartype(a::TTvector)
    return eltype(a)
end
function VectorInterface.scalartype(a::TToperator)
    return eltype(a)
end

function VectorInterface.scalartype(::Type{TTvector{T, N}}) where {T, N}
    return T
end
function VectorInterface.scalartype(::Type{TToperator{T, N}}) where {T, N}
    return T
end



end
