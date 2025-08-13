using Base.Threads
using TensorOperations
import Base.+
import Base.-
import Base.*
import Base./
import Base.kron
import Base.⊗
using LinearAlgebra
import LinearAlgebra: norm

function +(x::TTvector{T, N}, y::TTvector{T, N}) where {T <: Number, N}
    @assert x.ttv_dims == y.ttv_dims "Incompatible dimensions"
    d = x.N
    ttv_vec = Array{Array{T, 3}, 1}(undef, d)
    rks = x.ttv_rks + y.ttv_rks
    rks[1] = 1
    rks[d + 1] = 1
    #initialize ttv_vec
    @threads for k in 1:d
        ttv_vec[k] = zeros(T, x.ttv_dims[k], rks[k], rks[k + 1])
    end
    @inbounds begin
        #first core
        ttv_vec[1][:, :, 1:x.ttv_rks[2]] = x.ttv_vec[1]
        ttv_vec[1][:, :, (x.ttv_rks[2] + 1):rks[2]] = y.ttv_vec[1]
        #2nd to end-1 cores
        @threads for k in 2:(d - 1)
            ttv_vec[k][:, 1:x.ttv_rks[k], 1:x.ttv_rks[k + 1]] = x.ttv_vec[k]
            ttv_vec[k][:, (x.ttv_rks[k] + 1):rks[k], (x.ttv_rks[k + 1] + 1):rks[k + 1]] = y.ttv_vec[k]
        end
        #last core
        ttv_vec[d][:, 1:x.ttv_rks[d], 1] = x.ttv_vec[d]
        ttv_vec[d][:, (x.ttv_rks[d] + 1):rks[d], 1] = y.ttv_vec[d]
    end
    return TTvector{T, N}(d, ttv_vec, x.ttv_dims, rks, zeros(Int64, d))
end

function add!(x::TTvector{T, N}, y::TTvector{T, N}) where {T <: Number, N}
    @assert x.ttv_dims == y.ttv_dims "Incompatible dimensions"
    d = x.N
    ttv_vec = Array{Array{T, 3}, 1}(undef, d)
    rks = x.ttv_rks + y.ttv_rks
    rks[1] = 1
    rks[d + 1] = 1
    #initialize ttv_vec
    @threads for k in 1:d
        ttv_vec[k] = zeros(T, x.ttv_dims[k], rks[k], rks[k + 1])
    end
    @inbounds begin
        #first core
        ttv_vec[1][:, :, 1:x.ttv_rks[2]] = x.ttv_vec[1]
        ttv_vec[1][:, :, (x.ttv_rks[2] + 1):rks[2]] = y.ttv_vec[1]
        #2nd to end-1 cores
        @threads for k in 2:(d - 1)
            ttv_vec[k][:, 1:x.ttv_rks[k], 1:x.ttv_rks[k + 1]] = x.ttv_vec[k]
            ttv_vec[k][:, (x.ttv_rks[k] + 1):rks[k], (x.ttv_rks[k + 1] + 1):rks[k + 1]] = y.ttv_vec[k]
        end
        #last core
        ttv_vec[d][:, 1:x.ttv_rks[d], 1] = x.ttv_vec[d]
        ttv_vec[d][:, (x.ttv_rks[d] + 1):rks[d], 1] = y.ttv_vec[d]
    end
    # Overwrite x fields
    x.ttv_vec = ttv_vec
    x.ttv_rks = rks
    x.ttv_ot = zeros(Int64, d)
    return x
end


function +(x::TToperator{T, N}, y::TToperator{T, N}) where {T <: Number, N}
    @assert x.tto_dims == y.tto_dims "Incompatible dimensions"
    d = x.N
    tto_vec = Array{Array{T, 4}, 1}(undef, d)
    rks = x.tto_rks + y.tto_rks
    rks[1] = 1
    rks[d + 1] = 1
    #initialize tto_vec
    @threads for k in 1:d
        tto_vec[k] = zeros(T, x.tto_dims[k], x.tto_dims[k], rks[k], rks[k + 1])
    end
    @inbounds begin
        #first core
        tto_vec[1][:, :, :, 1:x.tto_rks[1 + 1]] = x.tto_vec[1]
        tto_vec[1][:, :, :, (x.tto_rks[2] + 1):rks[2]] = y.tto_vec[1]
        #2nd to end-1 cores
        @threads for k in 2:(d - 1)
            tto_vec[k][:, :, 1:x.tto_rks[k], 1:x.tto_rks[k + 1]] = x.tto_vec[k]
            tto_vec[k][:, :, (x.tto_rks[k] + 1):rks[k], (x.tto_rks[k + 1] + 1):rks[k + 1]] = y.tto_vec[k]
        end
        #last core
        tto_vec[d][:, :, 1:x.tto_rks[d], 1] = x.tto_vec[d]
        tto_vec[d][:, :, (x.tto_rks[d] + 1):rks[d], 1] = y.tto_vec[d]
    end
    return TToperator{T, N}(d, tto_vec, x.tto_dims, rks, zeros(Int64, d))
end

function *(A::TToperator{T, N}, v::TTvector{T, N}) where {T <: Number, N}
    @assert A.tto_dims == v.ttv_dims "Incompatible dimensions"
    y = zeros_tt(T, A.tto_dims, A.tto_rks .* v.ttv_rks)
    @inbounds begin
        @simd for k in 1:v.N
            yvec_temp = reshape(y.ttv_vec[k], (y.ttv_dims[k], A.tto_rks[k], v.ttv_rks[k], A.tto_rks[k + 1], v.ttv_rks[k + 1]))
            @tensoropt((νₖ₋₁, νₖ), yvec_temp[iₖ, αₖ₋₁, νₖ₋₁, αₖ, νₖ] = A.tto_vec[k][iₖ, jₖ, αₖ₋₁, αₖ] * v.ttv_vec[k][jₖ, νₖ₋₁, νₖ])
        end
    end
    return y
end

function (A::TToperator{T, N})(x::TTvector{T, N}) where {T, N}
    return A * x
end

function (A::TToperator{T, N})(x::TTvector{T, N}, ::Val{S}) where {T, N, S}
    return A(x)
end

function *(A::TToperator{T, N}, B::TToperator{T, N}) where {T <: Number, N}
    @assert A.tto_dims == B.tto_dims "Incompatible dimensions"
    d = A.N
    A_rks = A.tto_rks #R_0, ..., R_d
    B_rks = B.tto_rks #r_0, ..., r_d
    Y = [zeros(T, A.tto_dims[k], A.tto_dims[k], A_rks[k] * B_rks[k], A_rks[k + 1] * B_rks[k + 1]) for k in eachindex(A.tto_dims)]
    @inbounds @simd for k in eachindex(Y)
        M_temp = reshape(Y[k], A.tto_dims[k], A.tto_dims[k], A_rks[k], B_rks[k], A_rks[k + 1], B_rks[k + 1])
        @simd for jₖ in size(M_temp, 2)
            @simd for iₖ in size(M_temp, 1)
                @tensor M_temp[iₖ, jₖ, αₖ₋₁, βₖ₋₁, αₖ, βₖ] = A.tto_vec[k][iₖ, z, αₖ₋₁, αₖ] * B.tto_vec[k][z, jₖ, βₖ₋₁, βₖ]
            end
        end
    end
    return TToperator{T, N}(d, Y, A.tto_dims, A.tto_rks .* B.tto_rks, zeros(Int64, d))
end

function *(A::Array{TTvector{T, N}, 1}, x::Vector{T}) where {T, N}
    out = x[1] * A[1]
    for i in 2:length(A)
        out = out + x[i] * A[i]
    end
    return out
end


function dot(A::TTvector{T, N}, B::TTvector{T, N}) where {T <: Number, N}
    @assert A.ttv_dims == B.ttv_dims "TT dimensions are not compatible"
    A_rks = A.ttv_rks
    B_rks = B.ttv_rks
    out = zeros(T, maximum(A_rks), maximum(B_rks))
    out[1, 1] = one(T)
    @inbounds for k in eachindex(A.ttv_dims)
        M = @view(out[1:A_rks[k + 1], 1:B_rks[k + 1]])
        @tensor M[a, b] = A.ttv_vec[k][z, α, a] * (B.ttv_vec[k][z, β, b] * out[1:A_rks[k], 1:B_rks[k]][α, β]) #size R^A_{k} × R^B_{k}
    end
    return out[1, 1]::T
end

function dot_par(A::TTvector{T, N}, B::TTvector{T, N}) where {T <: Number, N}
    @assert A.ttv_dims == B.ttv_dims "TT dimensions are not compatible"
    d = length(A.ttv_dims)
    Y = Array{Array{T, 2}, 1}(undef, d)
    A_rks = A.ttv_rks
    B_rks = B.ttv_rks
    C = zeros(T, maximum(A_rks .* B_rks))
    @threads for k in 1:d
        M = zeros(T, A_rks[k], B_rks[k], A_rks[k + 1], B_rks[k + 1])
        @tensor M[a, b, c, d] = A.ttv_vec[k][z, a, c] * B.ttv_vec[k][z, b, d] #size R^A_{k-1} ×  R^B_{k-1} × R^A_{k} × R^B_{k}
        Y[k] = reshape(M, A_rks[k] * B_rks[k], A_rks[k + 1] * B_rks[k + 1])
    end
    @inbounds C[1:length(Y[d])] = Y[d][:]
    for k in (d - 1):-1:1
        @inbounds C[1:size(Y[k], 1)] = Y[k] * C[1:size(Y[k], 2)]
    end
    return C[1]::T
end

function *(a::S, A::TTvector{R,N}) where {S<:Number,R<:Number,N}
    T  = promote_type(S,R)
    aT = convert(T, a)
    if iszero(aT)
        return zeros_tt(T, A.ttv_dims, A.ttv_rks)
    end
    i = findfirst(==(0), A.ttv_ot); i === nothing && (i = 1)
    X = copy(A.ttv_vec)
    X[i] = aT * X[i]
    if T != R
        for k in eachindex(X)
            X[k] = convert.(T, X[k])
        end
    end
    return TTvector{T,N}(A.N, X, A.ttv_dims, A.ttv_rks, A.ttv_ot)
end

# TToperator scaling
function *(a::S, A::TToperator{R,N}) where {S<:Number,R<:Number,N}
    T  = promote_type(S,R)
    aT = convert(T, a)
    if iszero(aT)
        return zeros_tt(T, A.tto_dims, A.tto_rks)
    end
    i = findfirst(==(0), A.tto_ot); i === nothing && (i = 1)
    X = copy(A.tto_vec)
    X[i] = aT * X[i]
    if T != R
        for k in eachindex(X)
            X[k] = convert.(T, X[k])
        end
    end
    return TToperator{T,N}(A.N, X, A.tto_dims, A.tto_rks, A.tto_ot)
end


function Base.:*(A::TTvector{T, N}, a::S) where {T <: Number, S <: Number, N}
    return a * A
end

function -(A::TTvector{T, N}, B::TTvector{T, N}) where {T <: Number, N}
    return *(-1.0, B) + A
end

function -(A::TToperator{T, N}, B::TToperator{T, N}) where {T <: Number, N}
    return *(-1.0, B) + A
end

function /(A::TTvector, a)
    return 1 / a * A
end

function outer_product(x::TTvector{T, N}, y::TTvector{T, N}) where {T <: Number, N}
    Y = [zeros(T, x.ttv_dims[k], x.ttv_dims[k], x.ttv_rks[k] * y.ttv_rks[k], x.ttv_rks[k + 1] * y.ttv_rks[k + 1]) for k in eachindex(x.ttv_dims)]
    @inbounds @simd for k in eachindex(Y)
        M_temp = reshape(Y[k], x.ttv_dims[k], x.ttv_dims[k], x.ttv_rks[k], y.ttv_rks[k], x.ttv_rks[k + 1], y.ttv_rks[k + 1])
        @simd for jₖ in size(M_temp, 2)
            @simd for iₖ in size(M_temp, 1)
                @tensor M_temp[iₖ, jₖ, αₖ₋₁, βₖ₋₁, αₖ, βₖ] = x.ttv_vec[k][iₖ, αₖ₋₁, αₖ] * conj(y.ttv_vec[k][jₖ, βₖ₋₁, βₖ])
            end
        end
    end
    return TToperator{T, N}(x.N, Y, x.ttv_dims, x.ttv_rks .* y.ttv_rks, zeros(Int64, x.N))
end


function concatenate(tt::TTvector{T, N}, other::Union{TTvector{T}, Vector{Array{T, 3}}}, overwrite::Bool = false) where {T, N}
    tt_base = overwrite ? tt : copy(tt)
    if other isa TTvector{T}
        if last(tt_base.ttv_rks) != first(other.ttv_rks)
            throw(DimensionMismatch("Ranks do not match!"))
        end
        new_vec = vcat(tt_base.ttv_vec, other.ttv_vec)

        new_dims = (tt_base.ttv_dims..., other.ttv_dims...)

        new_rks = vcat(tt_base.ttv_rks[1:(end - 1)], other.ttv_rks)

        new_ot = vcat(tt_base.ttv_ot, other.ttv_ot)
    elseif other isa Vector{Array{T, 3}}
        if any(ndims(core) != 3 for core in other)
            throw(DimensionMismatch("List elements must be 3-dimensional arrays"))
        end

        if any(size(other[i], 3) != size(other[i + 1], 2) for i in 1:(length(other) - 1))
            throw(DimensionMismatch("Ranks in the provided cores list do not match"))
        end
        if last(tt_base.ttv_rks) != size(other[1], 2)
            throw(DimensionMismatch("Ranks do not match between `tt` and `other`"))
        end

        new_vec = vcat(tt_base.ttv_vec, other)
        new_dims = (tt_base.ttv_dims..., [size(core, 1) for core in other]...)
        new_rks = vcat(tt_base.ttv_rks[1:(end - 1)], [size(core, 2) for core in other], size(other[end], 3))

        new_ot = vcat(tt_base.ttv_ot, zeros(Int, length(other)))
    else
        throw(ArgumentError("Invalid type for `other`. Must be TTvector or Vector{Array{T, 3}}"))
    end

    N_new = length(new_dims)
    @assert N_new isa Int64

    tt_new = TTvector{T, N_new}(N_new, new_vec, Tuple(new_dims), new_rks, new_ot)

    return tt_new
end

function TTdiag(x::TTvector{T, M}) where {T <: Number, M}
    d = x.N                              # number of dimensions (cores)
    dims = x.ttv_dims                       # (n₁, n₂, …, n_d)
    rks = x.ttv_rks                        # (r₀=1, r₁, …, r_d=1)
    cores = x.ttv_vec                        # Vector of length d, each core is Array{T,3} sized (n_i, r_i, r_{i+1})

    new_rks = copy(rks)
    new_ot = zeros(Int, d)
    new_cores = Vector{Array{T, 4}}(undef, d)

    for i in 1:d
        ni = dims[i]
        ri = rks[i]
        rip = rks[i + 1]
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

    return TToperator{T, M}(d, new_cores, dims, new_rks, new_ot)
end


function permute(x::TTvector{T, N}, order::Vector{Int}, eps::Real) where {T <: Number, N}
    d = x.N
    cores = [copy(c) for c in x.ttv_vec]
    dims = collect(x.ttv_dims)
    rks = copy(x.ttv_rks)
    idx = invperm(order)
    ϵ = eps / d^1.5

    for kk in d:-1:3
        r_k, n_k, r_kp1 = rks[kk], dims[kk], rks[kk + 1]
        M = reshape(cores[kk], (r_k, n_k * r_kp1))'
        F = qr(M)
        Q = Matrix(F.Q)
        R = F.R
        tr = min(r_k, n_k * r_kp1)
        cores[kk] = reshape(transpose(Q[:, 1:tr]), (tr, n_k, r_kp1))
        rks[kk] = tr
        rkm1, n_km1 = rks[kk - 1], dims[kk - 1]
        Mprev = reshape(cores[kk - 1], (rkm1 * n_km1, r_k))
        cores[kk - 1] = reshape(Mprev * R', (rkm1, n_km1, tr))
    end

    k = 1
    while true
        nk = k
        while nk < d && idx[nk] < idx[nk + 1]
            nk += 1
        end
        if nk == d
            break
        end

        for kk in k:(nk - 1)
            r_k, n_k, r_kp1 = rks[kk], dims[kk], rks[kk + 1]
            M = reshape(cores[kk], (r_k * n_k, r_kp1))
            F = qr(M)
            Q = Matrix(F.Q)
            R = F.R
            tr = min(r_k * n_k, r_kp1)
            cores[kk] = reshape(Q[:, 1:tr], (r_k, n_k, tr))
            rks[kk + 1] = tr
            r_kp2 = rks[kk + 2]
            Mnext = reshape(cores[kk + 1], (r_kp1, dims[kk + 1] * r_kp2))
            cores[kk + 1] = reshape(R * Mnext, (tr, dims[kk + 1], r_kp2))
        end

        k = nk
        r_k, n_k, r_kp1, n_kp1, r_kp2 = rks[k], dims[k], rks[k + 1], dims[k + 1], rks[k + 2]
        C = reshape(
            reshape(cores[k], (r_k * n_k, r_kp1)) * reshape(cores[k + 1], (r_kp1, n_kp1 * r_kp2)),
            (r_k, n_k, n_kp1, r_kp2)
        )
        C = permutedims(C, (1, 3, 2, 4))
        M = reshape(C, (r_k * n_kp1, n_k * r_kp2))
        F = svd(M; full = false)
        s = F.S
        thresh = maximum(s) * ϵ
        rnew = count(x -> x > thresh, s)
        rnew = max(rnew, 1)
        U, V = F.U, F.Vt'
        tmp = U * Diagonal(s)
        cores[k] = reshape(tmp[:, 1:rnew], (r_k, n_kp1, rnew))
        cores[k + 1] = reshape(V[:, 1:rnew]', (rnew, n_k, r_kp2))
        rks[k + 1] = rnew
        idx[k], idx[k + 1] = idx[k + 1], idx[k]
        dims[k], dims[k + 1] = dims[k + 1], dims[k]
        k = max(k - 1, 1)
    end

    return TTvector{T, N}(d, cores, Tuple(dims), rks, zeros(Int, N))
end

function hadamard(x::TTvector{T, N}, y::TTvector{T, N}) where {T <: Number, N}
    @assert x.ttv_dims == y.ttv_dims "Incompatible TT dimensions"
    d = x.N
    ttv_vec = Vector{Array{T, 3}}(undef, d)
    dims = x.ttv_dims
    rks = [x.ttv_rks[k] * y.ttv_rks[k] for k in 1:(d + 1)]

    for k in 1:d
        n = dims[k]
        rx1, rx2 = x.ttv_rks[k], x.ttv_rks[k + 1]
        ry1, ry2 = y.ttv_rks[k], y.ttv_rks[k + 1]
        core = zeros(T, n, rx1 * ry1, rx2 * ry2)
        for s in 1:n
            core[s, :, :] = kron(x.ttv_vec[k][s, :, :], y.ttv_vec[k][s, :, :])
        end
        ttv_vec[k] = core
    end
    return TTvector{T, N}(d, ttv_vec, dims, rks, zeros(Int64, d))
end

⊕(x::TTvector{T, N}, y::TTvector{T, N}) where {T <: Number, N} = hadamard(x, y)

function kron(A::TToperator{T, d1}, B::TToperator{T, d2}) where {T, d1, d2}
    cores = vcat(A.tto_vec, B.tto_vec)
    dims = (A.tto_dims..., B.tto_dims...)
    rks = vcat(A.tto_rks[1:(end - 1)], B.tto_rks)
    ot = vcat(A.tto_ot, B.tto_ot)
    return TToperator{T, d1 + d2}(d1 + d2, cores, dims, rks, ot)
end

⊗(A::TToperator{T, d1}, B::TToperator{T, d2}) where {T, d1, d2} = kron(A, B)

function kron(a::TTvector{T, d1}, b::TTvector{T, d2}) where {T, d1, d2}
    return TTvector{T, d1 + d2}(
        d1 + d2,
        vcat(a.ttv_vec, b.ttv_vec),
        (a.ttv_dims..., b.ttv_dims...),
        vcat(a.ttv_rks[1:(end - 1)], b.ttv_rks),
        vcat(a.ttv_ot, b.ttv_ot)
    )
end

⊗(a::TTvector{T, d1}, b::TTvector{T, d2}) where {T, d1, d2} = kron(a, b)

function euclidean_distance(a::TTvector{T, N}, b::TTvector{T, N}) where {T <: Number, N}
    @assert a.ttv_dims == b.ttv_dims "TT dimensions must match"
    return sqrt(dot(a, a) - 2 * real(dot(b, a)) + dot(b, b))
end

function euclidean_distance_normalized(a::TTvector{T, N}, b::TTvector{T, N}) where {T <: Number, N}
    @assert a.ttv_dims == b.ttv_dims "TT dimensions must match"
    return sqrt(1.0 + dot(a, a) / dot(b, b) - 2.0 * real(dot(b, a)) / dot(b, b))
end

function norm(a::TTvector{T,N}) where {T<:Number,N}
    s = dot(a, a)                
    v = real(s)                  
    v = v < 0 ? zero(v) : v      
    return sqrt(v)
end