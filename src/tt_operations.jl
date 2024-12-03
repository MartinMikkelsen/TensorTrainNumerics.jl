using Base.Threads
using TensorOperations
import Base.+
import Base.-
import Base.*
import Base./
import LinearAlgebra.dot

"""
Addition of two TTvector
"""
function +(x::TTvector{T,N},y::TTvector{T,N}) where {T<:Number,N}
    @assert x.ttv_dims == y.ttv_dims "Incompatible dimensions"
    d = x.N
    ttv_vec = Array{Array{T,3},1}(undef,d)
    rks = x.ttv_rks + y.ttv_rks
    rks[1] = 1
    rks[d+1] = 1
    #initialize ttv_vec
    @threads for k in 1:d
        ttv_vec[k] = zeros(T,x.ttv_dims[k],rks[k],rks[k+1])
    end
    @inbounds begin
        #first core 
        ttv_vec[1][:,:,1:x.ttv_rks[2]] = x.ttv_vec[1]
        ttv_vec[1][:,:,(x.ttv_rks[2]+1):rks[2]] = y.ttv_vec[1]
        #2nd to end-1 cores
        @threads for k in 2:(d-1)
            ttv_vec[k][:,1:x.ttv_rks[k],1:x.ttv_rks[k+1]] = x.ttv_vec[k]
            ttv_vec[k][:,(x.ttv_rks[k]+1):rks[k],(x.ttv_rks[k+1]+1):rks[k+1]] = y.ttv_vec[k]
        end
        #last core
        ttv_vec[d][:,1:x.ttv_rks[d],1] = x.ttv_vec[d]
        ttv_vec[d][:,(x.ttv_rks[d]+1):rks[d],1] = y.ttv_vec[d]
        end
    return TTvector{T,N}(d,ttv_vec,x.ttv_dims,rks,zeros(Int64,d))
end

"""
Addition of two TToperators
"""
function +(x::TToperator{T,N},y::TToperator{T,N}) where {T<:Number,N}
    @assert x.tto_dims == y.tto_dims "Incompatible dimensions"
    d = x.N
    tto_vec = Array{Array{T,4},1}(undef,d)
    rks = x.tto_rks + y.tto_rks
    rks[1] = 1
    rks[d+1] = 1
    #initialize tto_vec
    @threads for k in 1:d
        tto_vec[k] = zeros(T,x.tto_dims[k],x.tto_dims[k],rks[k],rks[k+1])
    end
    @inbounds begin
        #first core 
        tto_vec[1][:,:,:,1:x.tto_rks[1+1]] = x.tto_vec[1]
        tto_vec[1][:,:,:,(x.tto_rks[2]+1):rks[2]] = y.tto_vec[1]
        #2nd to end-1 cores
        @threads for k in 2:(d-1)
            tto_vec[k][:,:,1:x.tto_rks[k],1:x.tto_rks[k+1]] = x.tto_vec[k]
            tto_vec[k][:,:,(x.tto_rks[k]+1):rks[k],(x.tto_rks[k+1]+1):rks[k+1]] = y.tto_vec[k]
        end
        #last core
        tto_vec[d][:,:,1:x.tto_rks[d],1] = x.tto_vec[d]
        tto_vec[d][:,:,(x.tto_rks[d]+1):rks[d],1] = y.tto_vec[d]
    end
    return TToperator{T,N}(d,tto_vec,x.tto_dims,rks,zeros(Int64,d))
end


#matrix vector multiplication in TT format
function *(A::TToperator{T,N},v::TTvector{T,N}) where {T<:Number,N}
    @assert A.tto_dims==v.ttv_dims "Incompatible dimensions"
    y = zeros_tt(T,A.tto_dims,A.tto_rks.*v.ttv_rks)
    @inbounds begin @simd for k in 1:v.N
        yvec_temp = reshape(y.ttv_vec[k], (y.ttv_dims[k], A.tto_rks[k], v.ttv_rks[k], A.tto_rks[k+1], v.ttv_rks[k+1]))
        @tensoropt((νₖ₋₁,νₖ), yvec_temp[iₖ,αₖ₋₁,νₖ₋₁,αₖ,νₖ] = A.tto_vec[k][iₖ,jₖ,αₖ₋₁,αₖ]*v.ttv_vec[k][jₖ,νₖ₋₁,νₖ])
    end end
    return y
end

#matrix matrix multiplication in TT format
function *(A::TToperator{T,N},B::TToperator{T,N}) where {T<:Number,N}
    @assert A.tto_dims==B.tto_dims "Incompatible dimensions"
    d = A.N
    A_rks = A.tto_rks #R_0, ..., R_d
    B_rks = B.tto_rks #r_0, ..., r_d
    Y = [zeros(T,A.tto_dims[k], A.tto_dims[k], A_rks[k]*B_rks[k], A_rks[k+1]*B_rks[k+1]) for k in eachindex(A.tto_dims)]
    @inbounds @simd for k in eachindex(Y)
		M_temp = reshape(Y[k], A.tto_dims[k], A.tto_dims[k], A_rks[k],B_rks[k], A_rks[k+1],B_rks[k+1])
        @simd for jₖ in size(M_temp,2)
            @simd for iₖ in size(M_temp,1)
                @tensor M_temp[iₖ,jₖ,αₖ₋₁,βₖ₋₁,αₖ,βₖ] = A.tto_vec[k][iₖ,z,αₖ₋₁,αₖ]*B.tto_vec[k][z,jₖ,βₖ₋₁,βₖ]
            end
        end
    end
    return TToperator{T,N}(d,Y,A.tto_dims,A.tto_rks.*B.tto_rks,zeros(Int64,d))
end

*(A::TToperator{T,N},B...) where {T,N} = *(A,*(B...))

function *(A::Array{TTvector{T,N},1},x::Vector{T}) where {T,N}
    out = x[1]*A[1]
    for i in 2:length(A)
        out = out + x[i]*A[i]
    end
    return out
end

#dot returns the dot product of two TTvector
function dot(A::TTvector{T,N},B::TTvector{T,N}) where {T<:Number,N}
    @assert A.ttv_dims==B.ttv_dims "TT dimensions are not compatible"
    A_rks = A.ttv_rks
    B_rks = B.ttv_rks
	out = zeros(T,maximum(A_rks),maximum(B_rks))
    out[1,1] = one(T)
    @inbounds for k in eachindex(A.ttv_dims)
        M = @view(out[1:A_rks[k+1],1:B_rks[k+1]])
		@tensor M[a,b] = A.ttv_vec[k][z,α,a]*(B.ttv_vec[k][z,β,b]*out[1:A_rks[k],1:B_rks[k]][α,β]) #size R^A_{k} × R^B_{k} 
    end
    return out[1,1]::T
end

"""
`dot_par(x_tt,y_tt)' returns the dot product of `x_tt` and `y_tt` in a parallelized algorithm
"""
function dot_par(A::TTvector{T,N},B::TTvector{T,N}) where {T<:Number,N}
    @assert A.ttv_dims==B.ttv_dims "TT dimensions are not compatible"
    d = length(A.ttv_dims)
    Y = Array{Array{T,2},1}(undef,d)
    A_rks = A.ttv_rks
    B_rks = B.ttv_rks
	C = zeros(T,maximum(A_rks.*B_rks))
    @threads for k in 1:d
		M = zeros(T,A_rks[k],B_rks[k],A_rks[k+1],B_rks[k+1])
		@tensor M[a,b,c,d] = A.ttv_vec[k][z,a,c]*B.ttv_vec[k][z,b,d] #size R^A_{k-1} ×  R^B_{k-1} × R^A_{k} × R^B_{k} 
		Y[k] = reshape(M, A_rks[k]*B_rks[k], A_rks[k+1]*B_rks[k+1])
    end
    @inbounds C[1:length(Y[d])] = Y[d][:]
    for k in d-1:-1:1
        @inbounds C[1:size(Y[k],1)] = Y[k]*C[1:size(Y[k],2)]
    end
    return C[1]::T
end

function *(a::S,A::TTvector{R,N}) where {S<:Number,R<:Number,N}
    T = typejoin(typeof(a),R)
    if iszero(a)
        return zeros_tt(T,A.ttv_dims,ones(Int64,A.N+1))
    else
        i = findfirst(isequal(0),A.ttv_ot)
        X = copy(A.ttv_vec)
        X[i] = a*X[i]
        return TTvector{T,N}(A.N,X,A.ttv_dims,A.ttv_rks,A.ttv_ot)
    end
end

function *(a::S,A::TToperator{R,N}) where {S<:Number,R<:Number,N}
    i = findfirst(isequal(0),A.tto_ot)
    T = typejoin(typeof(a),R)
    X = copy(A.tto_vec)
    X[i] = a*X[i]
    return TToperator{T,N}(A.N,X,A.tto_dims,A.tto_rks,A.tto_ot)
end

function -(A::TTvector{T,N},B::TTvector{T,N}) where {T<:Number,N}
    return *(-1.0,B)+A
end

function -(A::TToperator{T,N},B::TToperator{T,N}) where {T<:Number,N}
    return *(-1.0,B)+A
end

function /(A::TTvector,a)
    return 1/a*A
end

"""
returns the matrix x y' in the TTO format
"""
function outer_product(x::TTvector{T,N},y::TTvector{T,N}) where {T<:Number,N}
    Y = [zeros(T,x.ttv_dims[k], x.ttv_dims[k], x.ttv_rks[k]*y.ttv_rks[k], x.ttv_rks[k+1]*y.ttv_rks[k+1]) for k in eachindex(x.ttv_dims)]
    @inbounds @simd for k in eachindex(Y)
		M_temp = reshape(Y[k], x.ttv_dims[k], x.ttv_dims[k], x.ttv_rks[k], y.ttv_rks[k], x.ttv_rks[k+1],y.ttv_rks[k+1])
        @simd for jₖ in size(M_temp,2)
            @simd for iₖ in size(M_temp,1)
                @tensor M_temp[iₖ,jₖ,αₖ₋₁,βₖ₋₁,αₖ,βₖ] = x.ttv_vec[k][iₖ,αₖ₋₁,αₖ]*conj(y.ttv_vec[k][jₖ,βₖ₋₁,βₖ])
            end
        end
    end
    return TToperator{T,N}(x.N,Y,x.ttv_dims,x.ttv_rks.*y.ttv_rks,zeros(Int64,x.N))
end

"""
    concatenate(tt::TTvector{T, N}, other::Union{TTvector{T}, Vector{Array{T, 3}}}, overwrite::Bool=false) where {T, N}

Concatenates a `TTvector` with another `TTvector` or a vector of 3-dimensional arrays.

# Arguments
- `tt::TTvector{T, N}`: The original TTvector to be concatenated.
- `other::Union{TTvector{T}, Vector{Array{T, 3}}}`: The TTvector or vector of 3-dimensional arrays to concatenate with `tt`.
- `overwrite::Bool=false`: If `true`, the original `tt` is modified in place. Otherwise, a copy of `tt` is made before concatenation.

# Returns
- `TTvector{T, N_new}`: A new TTvector with concatenated cores, updated dimensions, ranks, and orthogonality indicators.

# Throws
- `DimensionMismatch`: If the ranks or dimensions of `tt` and `other` do not match.
- `ArgumentError`: If `other` is not of type `TTvector` or `Vector{Array{T, 3}}`.
"""
function concatenate(tt::TTvector{T, N}, other::Union{TTvector{T}, Vector{Array{T, 3}}}, overwrite::Bool=false) where {T, N}
    # Copy the TTvector if overwrite is false
    tt_base = overwrite ? tt : copy(tt)
    
    if other isa TTvector{T}
        # Check rank compatibility
        if last(tt_base.ttv_rks) != first(other.ttv_rks)
            throw(DimensionMismatch("Ranks do not match!"))
        end
        # Concatenate cores
        new_vec = vcat(tt_base.ttv_vec, other.ttv_vec)
        
        # Update dimensions
        new_dims = (tt_base.ttv_dims..., other.ttv_dims...)
        
        # Update ranks, excluding the first rank of `other`
        new_rks = vcat(tt_base.ttv_rks[1:end-1], other.ttv_rks)
        
        # Update orthogonality indicators
        new_ot = vcat(tt_base.ttv_ot, other.ttv_ot)
    elseif other isa Vector{Array{T, 3}}
        # Ensure all cores are 3-dimensional tensors
        if any(ndims(core) != 3 for core in other)
            throw(DimensionMismatch("List elements must be 3-dimensional arrays"))
        end
        
        # Check rank continuity across appended cores
        if any(size(other[i], 3) != size(other[i + 1], 2) for i in 1:length(other) - 1)
            throw(DimensionMismatch("Ranks in the provided cores list do not match"))
        end
        if last(tt_base.ttv_rks) != size(other[1], 2)
            throw(DimensionMismatch("Ranks do not match between `tt` and `other`"))
        end
        
        # Concatenate cores
        new_vec = vcat(tt_base.ttv_vec, other)
        
        # Update dimensions
        new_dims = (tt_base.ttv_dims..., [size(core, 1) for core in other]...)
        
        # Update ranks
        new_rks = vcat(tt_base.ttv_rks[1:end-1], [size(core, 2) for core in other], size(other[end], 3))
        
        # Assuming orthogonality indicators are zeros for new cores
        new_ot = vcat(tt_base.ttv_ot, zeros(Int, length(other)))
    else
        throw(ArgumentError("Invalid type for `other`. Must be TTvector or Vector{Array{T, 3}}"))
    end

    # Compute the new order
    N_new = length(new_dims)
    # Ensure that N_new is an Int64
    @assert N_new isa Int64
    
    # Create a new TTvector with updated parameters
	tt_new = TTvector{T, N_new}(N_new, new_vec, Tuple(new_dims), new_rks, new_ot)
    
    return tt_new
end