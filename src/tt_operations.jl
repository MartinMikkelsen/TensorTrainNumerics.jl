using Base.Threads
using TensorOperations
import Base.+
import Base.-
import Base.*
import Base./
using LinearAlgebra

"""
    +(x::TTvector{T,N}, y::TTvector{T,N}) where {T<:Number, N}

Add two TTvector objects `x` and `y` of the same type and dimension.

# Arguments
- `x::TTvector{T,N}`: The first TTvector object.
- `y::TTvector{T,N}`: The second TTvector object.

# Returns
- `TTvector{T,N}`: A new TTvector object representing the sum of `x` and `y`.

# Throws
- `AssertionError`: If the dimensions of `x` and `y` are not compatible.

# Details
This function performs the addition of two TTvector objects by combining their tensor train cores. It initializes a new tensor train with the appropriate dimensions and ranks, then fills in the cores by combining the corresponding cores from `x` and `y`.

The addition is performed as follows:
1. The first core is combined by concatenating the cores of `x` and `y` along the third dimension.
2. The intermediate cores (from the second to the second-to-last) are combined by concatenating the cores of `x` and `y` along both the second and third dimensions.
3. The last core is combined by concatenating the cores of `x` and `y` along the second dimension.

The resulting TTvector object has the combined ranks and dimensions of the input TTvector objects.
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
    +(x::TToperator{T,N}, y::TToperator{T,N}) where {T<:Number, N}

Adds two `TToperator` objects `x` and `y` of the same type `T` and dimension `N`.

# Arguments
- `x::TToperator{T,N}`: The first `TToperator` object.
- `y::TToperator{T,N}`: The second `TToperator` object.

# Returns
- `TToperator{T,N}`: A new `TToperator` object representing the sum of `x` and `y`.

# Throws
- `AssertionError`: If the dimensions of `x` and `y` are not compatible.

# Details
The function performs the following steps:
1. Asserts that the dimensions of `x` and `y` are compatible.
2. Initializes a new tensor train operator vector `tto_vec` with the appropriate dimensions and ranks.
3. Uses multi-threading to initialize the cores of `tto_vec`.
4. Copies the cores of `x` and `y` into the appropriate positions in `tto_vec`.

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

"""
    *(A::TToperator{T,N}, v::TTvector{T,N}) where {T<:Number, N}

Multiplies a tensor-train operator `A` by a tensor-train vector `v`.

# Arguments
- `A::TToperator{T,N}`: A tensor-train operator of type `T` and order `N`.
- `v::TTvector{T,N}`: A tensor-train vector of type `T` and order `N`.

# Returns
- `y::TTvector{T,N}`: The resulting tensor-train vector after multiplication.

# Details
- Asserts that the dimensions of `A` and `v` are compatible.
- Initializes a zero tensor-train vector `y` with appropriate dimensions and ranks.
- Performs the multiplication using a loop over the tensor-train cores and tensor contraction.

"""
function *(A::TToperator{T,N},v::TTvector{T,N}) where {T<:Number,N}
    @assert A.tto_dims==v.ttv_dims "Incompatible dimensions"
    y = zeros_tt(T,A.tto_dims,A.tto_rks.*v.ttv_rks)
    @inbounds begin @simd for k in 1:v.N
        yvec_temp = reshape(y.ttv_vec[k], (y.ttv_dims[k], A.tto_rks[k], v.ttv_rks[k], A.tto_rks[k+1], v.ttv_rks[k+1]))
        @tensoropt((νₖ₋₁,νₖ), yvec_temp[iₖ,αₖ₋₁,νₖ₋₁,αₖ,νₖ] = A.tto_vec[k][iₖ,jₖ,αₖ₋₁,αₖ]*v.ttv_vec[k][jₖ,νₖ₋₁,νₖ])
    end end
    return y
end

"""
    *(A::TToperator{T,N}, B::TToperator{T,N}) where {T<:Number, N}

Multiply two TToperators `A` and `B` of the same type `T` and dimension `N`.

# Arguments
- `A::TToperator{T,N}`: The first TToperator.
- `B::TToperator{T,N}`: The second TToperator.

# Returns
- `TToperator{T,N}`: The resulting TToperator after multiplication.

# Preconditions
- `A.tto_dims == B.tto_dims`: The dimensions of `A` and `B` must be compatible.

# Description
This function performs the multiplication of two TToperators by iterating over their tensor train cores and performing tensor contractions. The resulting TToperator has updated ranks and tensor cores.

"""
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

"""
    dot(A::TTvector{T,N}, B::TTvector{T,N}) where {T<:Number, N}

Compute the dot product of two TTvector objects `A` and `B`.

# Arguments
- `A::TTvector{T,N}`: The first TTvector.
- `B::TTvector{T,N}`: The second TTvector.

# Returns
- `T`: The dot product of the two TTvector objects.

# Preconditions
- `A.ttv_dims == B.ttv_dims`: The TT dimensions of `A` and `B` must be compatible.

# Notes
- The function uses tensor operations to compute the dot product.
- The `@inbounds` macro is used to skip array bounds checking for performance.
- The `@tensor` macro is used for tensor contractions.

"""
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
    outer_product(x::TTvector{T,N}, y::TTvector{T,N}) where {T<:Number, N}

Compute the outer product of two TTvectors `x` and `y`.

# Arguments
- `x::TTvector{T,N}`: The first TTvector.
- `y::TTvector{T,N}`: The second TTvector.

# Returns
- `TToperator{T,N}`: The resulting TToperator from the outer product of `x` and `y`.

# Details
This function computes the outer product of two TTvectors `x` and `y`, resulting in a TToperator. The function initializes an array `Y` to store the intermediate results, and then iterates over each element to compute the outer product using tensor contractions. The resulting TToperator is constructed from the computed array `Y`.

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

"""
    TTdiag(x::TTvector{T,M}) where {T<:Number,M}

Construct the diagonal TT‐matrix whose diagonal entries come from the TT‐vector `x`.
Returns a `TToperator{T,M}` with each core of size (n_i, n_i, r_i, r_{i+1}).

This matches MATLAB’s `[tm] = diag(tt)` in TT‐Toolbox.
"""
function TTdiag(x::TTvector{T,M}) where {T<:Number,M}
    d      = x.N                              # number of dimensions (cores)
    dims   = x.ttv_dims                       # (n₁, n₂, …, n_d)
    rks    = x.ttv_rks                        # (r₀=1, r₁, …, r_d=1)
    cores  = x.ttv_vec                        # Vector of length d, each core is Array{T,3} sized (n_i, r_i, r_{i+1})

    new_rks = copy(rks)                       
    new_ot  = zeros(Int, d)
    new_cores = Vector{Array{T,4}}(undef, d)

    for i in 1:d
        ni  = dims[i]
        ri  = rks[i]
        rip = rks[i+1]
        C = cores[i]
        D = zeros(T, ni, ni, ri, rip)
        @inbounds for s1 in 1:ri
            for s2 in 1:rip
                v = @view C[:, s1, s2]    
                for j in 1:ni
                    D[j, j, s1, s2] = v[j]
                end
            end
        end
        new_cores[i] = D
    end

    return TToperator{T,M}(d, new_cores, dims, new_rks, new_ot)
end


function permute(x::TTvector{T,N}, order::Vector{Int}, eps::Real) where {T<:Number,N}
    d = x.N
    cores = [copy(c) for c in x.ttv_vec]
    dims = collect(x.ttv_dims)
    rks = copy(x.ttv_rks)
    idx = invperm(order)
    ϵ = eps / d^1.5

    for kk in d:-1:3
        r_k, n_k, r_kp1 = rks[kk], dims[kk], rks[kk+1]
        M = reshape(cores[kk], (r_k, n_k*r_kp1))'
        F = qr(M)
        Q = Matrix(F.Q)
        R = F.R
        tr = min(r_k, n_k*r_kp1)
        cores[kk] = reshape(transpose(Q[:,1:tr]), (tr, n_k, r_kp1))
        rks[kk] = tr
        rkm1, n_km1 = rks[kk-1], dims[kk-1]
        Mprev = reshape(cores[kk-1], (rkm1*n_km1, r_k))
        cores[kk-1] = reshape(Mprev * R', (rkm1, n_km1, tr))
    end

    k = 1
    while true
        nk = k
        while nk < d && idx[nk] < idx[nk+1]
            nk += 1
        end
        if nk == d
            break
        end

        for kk in k:(nk-1)
            r_k, n_k, r_kp1 = rks[kk], dims[kk], rks[kk+1]
            M = reshape(cores[kk], (r_k*n_k, r_kp1))
            F = qr(M)
            Q = Matrix(F.Q)
            R = F.R
            tr = min(r_k*n_k, r_kp1)
            cores[kk] = reshape(Q[:,1:tr], (r_k, n_k, tr))
            rks[kk+1] = tr
            r_kp2 = rks[kk+2]
            Mnext = reshape(cores[kk+1], (r_kp1, dims[kk+1]*r_kp2))
            cores[kk+1] = reshape(R * Mnext, (tr, dims[kk+1], r_kp2))
        end

        k = nk
        r_k, n_k, r_kp1, n_kp1, r_kp2 = rks[k], dims[k], rks[k+1], dims[k+1], rks[k+2]
        C = reshape(reshape(cores[k], (r_k*n_k, r_kp1)) * reshape(cores[k+1], (r_kp1, n_kp1*r_kp2)),
                    (r_k, n_k, n_kp1, r_kp2))
        C = permutedims(C, (1,3,2,4))
        M = reshape(C, (r_k * n_kp1, n_k * r_kp2))
        F = svd(M; full = false)
        s = F.S
        thresh = norm(s) * ϵ
        rnew = count(x->x > thresh, s)
        rnew = max(rnew, 1)
        U, V = F.U, F.Vt'
        tmp = U * Diagonal(s)
        cores[k]   = reshape(tmp[:,1:rnew], (r_k, n_kp1, rnew))
        cores[k+1] = reshape(V[:,1:rnew]', (rnew, n_k, r_kp2))
        rks[k+1] = rnew
        idx[k], idx[k+1] = idx[k+1], idx[k]
        dims[k], dims[k+1] = dims[k+1], dims[k]
        k = max(k-1, 1)
    end

    return TTvector{T,N}(d, cores, Tuple(dims), rks, zeros(Int, N))
end

function hadamard(a::TTvector{T,N}, b::TTvector{T,N}, eps::Real) where {T,N}
    a.N == b.N || error("TTvector dimensions must match")
    a.ttv_dims == b.ttv_dims || error("TTvector mode sizes must match")
    d = a.N
    result_cores = Vector{Array{T,3}}(undef, d)
    for k in 1:d
        # Each core: (n_k, r_k, r_{k+1})
        core_a = a.ttv_vec[k]
        core_b = b.ttv_vec[k]
        # Hadamard product in TT: kron over ranks, elementwise over n_k
        result_cores[k] = Array{T,3}(undef, a.ttv_dims[k], a.ttv_rks[k]*b.ttv_rks[k], a.ttv_rks[k+1]*b.ttv_rks[k+1])
        for i in 1:a.ttv_dims[k]
            result_cores[k][i, :, :] = kron(core_a[i, :, :], core_b[i, :, :])
        end
    end
    result_rks = a.ttv_rks .* b.ttv_rks
    return TTvector{T,N}(d, result_cores, a.ttv_dims, result_rks, zeros(Int, d))
end

