module TTVectorInterfaceEXT

using TensorTrainNumerics
import VectorInterface: VectorInterface
import LinearAlgebra: dot, norm as l2norm
import Base: isempty

function VectorInterface.add(x::TTvector, y::TTvector)
    return x + y
end

function add!(x::TTvector, y::TTvector)
    return x .= x+y
end

function VectorInterface.add(x::TTvector, y::TTvector, c::Number)
    return x+y*c
end

function VectorInterface.add!(x::TTvector, y::TTvector, c::Number)
    return x .= x + y * c  # Reuse your `+` and `*` operators
end

function VectorInterface.add(a::TTvector, b::TTvector, α::Number, β::Number)
  return a * β + b * α
end
function VectorInterface.add!(a::TTvector, b::TTvector, α::Number, β::Number)
  a .= a .* β .+ b .* α
  return a
end

function VectorInterface.inner(a::TTvector, b::TTvector)
  return TensorTrainNumerics.dot(a, b)
end

function VectorInterface.scalartype(a::TTvector)
  return eltype(a)
end

function VectorInterface.scale(a::TTvector, α::Number)
  return a * α
end
function VectorInterface.scale!(a::TTvector, α::Number)
  a .= a * α
  return a
end

function zerovector(v::TTvector{T,M}) where {T<:Number,M}
    return zeros_tt(T, v.ttv_dims, v.ttv_rks) 
end

function VectorInterface.zerovector!(a::TTvector)
    return a .= zeros_tt(eltype(a), a.ttv_dims, a.ttv_rks)  
end

# Remove the previous ambiguous method and replace with a more specific one
function VectorInterface.zerovector(x::TTvector{T,N}, ::Type{S}) where {T,N,S<:Number}
    return zeros_tt(S, x.ttv_dims, x.ttv_rks)
end

# Overload zerovector for tuples of custom types (e.g., (TTvector, Float64, ...))
function VectorInterface.zerovector(x::Tuple, ::Type{T}) where {T}
    return map(e -> VectorInterface.zerovector(e, T), x)
end

# Optionally, for single-argument tuple fallback
function VectorInterface.zerovector(x::Tuple)
    return map(VectorInterface.zerovector, x)
end

# --- TToperator VectorInterface methods ---

function VectorInterface.add(x::TToperator, y::TToperator)
    return x + y
end

function VectorInterface.add!(x::TToperator, y::TToperator)
    x .= x + y
    return x
end

function VectorInterface.add(x::TToperator, y::TToperator, c::Number)
    return x + y * c
end

function VectorInterface.add!(x::TToperator, y::TToperator, c::Number)
    x .= x + y * c
    return x
end

function VectorInterface.inner(a::TToperator, b::TToperator)
    return dot(a, b)
end

function VectorInterface.scalartype(a::TToperator)
    return eltype(a)
end

function VectorInterface.scale(a::TToperator, α::Number)
    return a * α
end

function VectorInterface.scale!(a::TToperator, α::Number)
    a .= a * α
    return a
end

function VectorInterface.zerovector(op::TToperator{T,N}) where {T<:Number,N}
    return zeros_tto(T, op.tto_dims, op.tto_rks)
end

function VectorInterface.zerovector!(a::TToperator)
    a .= zeros_tto(eltype(a), a.tto_dims, a.tto_rks)
    return a
end

function isempty(x::TTvector)
    # A TTvector is empty if any core is empty or has zero size
    any(isempty, x.ttv_vec)
end

function norm(x::TTvector, p::Real=2)
    if p == 2
        return sqrt(dot(x, x))
    else
        throw(ArgumentError("Only p=2 (Euclidean) norm is supported for TTvector. Requested p=$p."))
    end
end

end