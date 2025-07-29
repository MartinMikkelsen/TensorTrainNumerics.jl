module TensorTrainNumericsVectorInterfaceExt
using TensorTrainNumerics
using VectorInterface
using Base.Threads
import Base: zero, copy, +
import LinearAlgebra: dot

# Helper for type promotion
is_concrete_number_type(T) = isconcretetype(T) && T <: Number
function concrete_promote_type(types...)
    T = promote_type(types...)
    if !is_concrete_number_type(T)
        @warn "Fallback to Float64 for promote_type($(types...)) => $T"
        return Float64
    end
    return T
end

# Addition of TTvectors with concrete type promotion
function +(x::TTvector{T1, N}, y::TTvector{T2, N}) where {T1 <: Number, T2 <: Number, N}
    T = concrete_promote_type(T1, T2)
    xT = convert(TTvector{T, N}, x)
    yT = convert(TTvector{T, N}, y)
    d = xT.N
    ttv_vec = Array{Array{T, 3}, 1}(undef, d)
    rks = xT.ttv_rks + yT.ttv_rks
    rks[1] = 1
    rks[d + 1] = 1
    @threads for k in 1:d
        ttv_vec[k] = zeros(T, xT.ttv_dims[k], rks[k], rks[k + 1])
    end
    @inbounds begin
        ttv_vec[1][:, :, 1:xT.ttv_rks[2]] = xT.ttv_vec[1]
        ttv_vec[1][:, :, (xT.ttv_rks[2] + 1):rks[2]] = yT.ttv_vec[1]
        @threads for k in 2:(d - 1)
            ttv_vec[k][:, 1:xT.ttv_rks[k], 1:xT.ttv_rks[k + 1]] = xT.ttv_vec[k]
            ttv_vec[k][:, (xT.ttv_rks[k] + 1):rks[k], (xT.ttv_rks[k + 1] + 1):rks[k + 1]] = yT.ttv_vec[k]
        end
        ttv_vec[d][:, 1:xT.ttv_rks[d], 1] = xT.ttv_vec[d]
        ttv_vec[d][:, (xT.ttv_rks[d] + 1):rks[d], 1] = yT.ttv_vec[d]
    end
    return TTvector{T, N}(d, ttv_vec, xT.ttv_dims, rks, zeros(Int64, d))
end

function Base.convert(::Type{TTvector{T, N}}, x::TTvector{S, N}) where {T, S, N}
    if !is_concrete_number_type(T)
        error("Cannot convert TTvector to non-concrete element type $T")
    end
    new_tt_vec = [convert(Array{T,3}, core) for core in x.ttv_vec]
    return TTvector{T, N}(x.N, new_tt_vec, x.ttv_dims, x.ttv_rks, x.ttv_ot)
end

# --- VectorInterface ADD ---
function VectorInterface.add(a::TTvector, b::TTvector)
    return a + b
end
function VectorInterface.add(a::TToperator, b::TToperator)
    return a + b
end
function VectorInterface.add(a::TTvector, b::TTvector, α::Number)
    return a + b * α
end
function VectorInterface.add(a::TTvector, b::TTvector, α::Number, β::Number)
    return β * a + α * b
end

# In-place versions: just assign the new result (no broadcasting!)
function VectorInterface.add!(a::TTvector, b::TTvector)
    a_new = a + b
    # Copy a_new data into a (destructive)
    a.ttv_vec = a_new.ttv_vec
    a.ttv_rks = a_new.ttv_rks
    a.ttv_dims = a_new.ttv_dims
    a.ttv_ot = a_new.ttv_ot
    return a
end
function VectorInterface.add!(a::TTvector, b::TTvector, α::Number)
    a_new = a + b * α
    a.ttv_vec = a_new.ttv_vec
    a.ttv_rks = a_new.ttv_rks
    a.ttv_dims = a_new.ttv_dims
    a.ttv_ot = a_new.ttv_ot
    return a
end
function VectorInterface.add!(a::TTvector, b::TTvector, α::Number, β::Number)
    a_new = β * a + α * b
    a.ttv_vec = a_new.ttv_vec
    a.ttv_rks = a_new.ttv_rks
    a.ttv_dims = a_new.ttv_dims
    a.ttv_ot = a_new.ttv_ot
    return a
end

function VectorInterface.add!!(a::TTvector{T1,N}, b::TTvector{T2,N}) where {T1,T2,N}
    T = concrete_promote_type(T1, T2)
    if T === T1
        VectorInterface.add!(a, b)
    else
        VectorInterface.add(a, b)
    end
end
function VectorInterface.add!!(a::TTvector{T1,N}, b::TTvector{T2,N}, α::Number) where {T1,T2,N}
    T = concrete_promote_type(T1, T2, typeof(α))
    if T === T1
        VectorInterface.add!(a, b, α)
    else
        VectorInterface.add(a, b, α)
    end
end
function VectorInterface.add!!(a::TTvector{T1,N}, b::TTvector{T2,N}, α::Number, β::Number) where {T1,T2,N}
    T = concrete_promote_type(T1, T2, typeof(α), typeof(β))
    if T === T1
        VectorInterface.add!(a, b, α, β)
    else
        VectorInterface.add(a, b, α, β)
    end
end

# --- VectorInterface SCALE ---
function VectorInterface.scale(a::TTvector, α::Number)
    return α * a
end
function VectorInterface.scale!(a::TTvector, α::Number)
    a_new = α * a
    a.ttv_vec = a_new.ttv_vec
    a.ttv_rks = a_new.ttv_rks
    a.ttv_dims = a_new.ttv_dims
    a.ttv_ot = a_new.ttv_ot
    return a
end
function VectorInterface.scale!!(a::TTvector{T,N}, α::Number) where {T,N}
    T2 = concrete_promote_type(T, typeof(α))
    if T2 === T
        VectorInterface.scale!(a, α)
    else
        VectorInterface.scale(a, α)
    end
end

# scale! for dest, src, α
function VectorInterface.scale!(dest::TTvector, src::TTvector, α::Number)
    dest_new = src * α
    dest.ttv_vec = dest_new.ttv_vec
    dest.ttv_rks = dest_new.ttv_rks
    dest.ttv_dims = dest_new.ttv_dims
    dest.ttv_ot = dest_new.ttv_ot
    return dest
end
function VectorInterface.scale!!(dest::TTvector{T1,N}, src::TTvector{T2,N}, α::Number) where {T1,T2,N}
    T = concrete_promote_type(T1, T2, typeof(α))
    if T === T1
        VectorInterface.scale!(dest, src, α)
    else
        VectorInterface.scale(src, α)
    end
    return dest
end

# --- zerovector (safe fallback to Float64 if eltype is abstract) ---
function VectorInterface.zerovector(a::TTvector)
    T = eltype(a)
    if !is_concrete_number_type(T)
        @warn "Promoting zerovector to Float64 from abstract type $T"
        T = Float64
    end
    return zeros_tt(T, a.ttv_dims, a.ttv_rks)
end
function VectorInterface.zerovector(a::TTvector, T::Type{<:Number})
    if !is_concrete_number_type(T)
        @warn "Promoting zerovector to Float64 from abstract type $T"
        T = Float64
    end
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

function VectorInterface.length(a::TTvector)
    return length(a.ttv_dims)
end
function VectorInterface.length(a::TToperator)
    return length(a.tto_dims)
end

function similar(a::TTvector)
    return zeros_tt(eltype(a), a.ttv_dims, a.ttv_rks)
end
function similar(a::TToperator)
    return zeros_tt(eltype(a), a.tto_dims, a.tto_rks)
end

zero(a::TTvector) = zeros_tt(eltype(a), a.ttv_dims, a.ttv_rks)

function VectorInterface.inner(a::TTvector, b::TTvector)
    return TensorTrainNumerics.dot(a, b)
end

function copy(x::TTvector{T, N}) where {T, N}
    new_N = deepcopy(x.N)
    new_tt_vec = deepcopy(x.ttv_vec)
    new_dims = deepcopy(x.ttv_dims)
    new_rks = deepcopy(x.ttv_rks)
    new_ot = deepcopy(x.ttv_ot)
    return TTvector{T, N}(new_N, new_tt_vec, new_dims, new_rks, new_ot)
end
function copy(x::TToperator{T, N}) where {T, N}
    new_N = deepcopy(x.N)
    new_tt_vec = deepcopy(x.tto_vec)
    new_dims = deepcopy(x.tto_dims)
    new_rks = deepcopy(x.tto_rks)
    new_ot = deepcopy(x.tto_ot)
    return TToperator{T, N}(new_N, new_tt_vec, new_dims, new_rks, new_ot)
end

end
