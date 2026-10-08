using Base.Threads
using TensorOperations
import Base: +, -, *, /, kron
using LinearAlgebra
# `dot` and `norm` extend the LinearAlgebra generics (and are re-exported), so
# `using TensorTrainNumerics` alongside `using LinearAlgebra` causes no name clash.
import LinearAlgebra: norm, dot

"""
    _convert_eltype(T, x)

Return `x` with its cores converted to element type `T` (`x` itself if already `T`).
"""
function _convert_eltype(::Type{T}, x::TTVector{S, N}) where {T <: Number, S <: Number, N}
    T === S && return x
    return TTVector{T, N}([convert(Array{T, 3}, c) for c in x.ttv_vec], x.ttv_dims, copy(x.ttv_rks); orthogonality = copy(x.orthogonality))
end

function _convert_eltype(::Type{T}, A::TTOperator{S, N}) where {T <: Number, S <: Number, N}
    T === S && return A
    return TTOperator{T, N}([convert(Array{T, 4}, c) for c in A.tto_vec], A.tto_dims, copy(A.tto_rks); orthogonality = copy(A.orthogonality))
end

"""
Adds two TTVectors and returns a new TTVector.
"""
function +(x::TTVector{T, N}, y::TTVector{T, N}) where {T <: Number, N}
    @assert x.ttv_dims == y.ttv_dims "Incompatible dimensions"
    d = nsites(x)
    if d == 1
        return TTVector{T, N}([x.ttv_vec[1] + y.ttv_vec[1]], x.ttv_dims, [1, 1])
    end
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
    return TTVector{T, N}(ttv_vec, x.ttv_dims, rks)
end

"""
    add!(x::TTVector, y::TTVector) -> x

Overwrite `x` with `x + y`. The ranks of the result are the sums of the ranks
of `x` and `y`; no truncation is performed (see [`tt_round!`](@ref)).
"""
function add!(x::TTVector{T, N}, y::TTVector{T, N}) where {T <: Number, N}
    return _overwrite!(x, x + y)
end

# Make `x` represent `src` by replacing the contents of its core list, ranks, and
# orthogonality interval; objects sharing these vectors with `x` see the update. The
# number of cores is fixed by the type, so only the entries change. The core
# arrays themselves are shared with `src`.
function _overwrite!(x::TTVector, src::TTVector)
    x.ttv_dims == src.ttv_dims || throw(DimensionMismatch("cannot overwrite a TTVector with dimensions $(x.ttv_dims) by one with dimensions $(src.ttv_dims)"))
    copyto!(x.ttv_vec, src.ttv_vec)
    copyto!(x.ttv_rks, src.ttv_rks)
    copyto!(x.orthogonality, src.orthogonality)
    return x
end

"""
Adds two TTOperators and returns a new TTOperator.
"""
function +(x::TTOperator{T, N}, y::TTOperator{T, N}) where {T <: Number, N}
    @assert x.tto_dims == y.tto_dims "Incompatible dimensions"
    d = nsites(x)
    if d == 1
        return TTOperator{T, N}([x.tto_vec[1] + y.tto_vec[1]], x.tto_dims, [1, 1])
    end
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
    return TTOperator{T, N}(tto_vec, x.tto_dims, rks)
end

"""
Contracts the TTOperator A with the TTVector x.
"""
function *(A::TTOperator{T, N}, v::TTVector{T, N}) where {T <: Number, N}
    @assert A.tto_dims == v.ttv_dims "Incompatible dimensions"
    y = zeros_tt(T, A.tto_dims, A.tto_rks .* v.ttv_rks)
    begin
        @inbounds for k in 1:nsites(v)
            yvec_temp = reshape(y.ttv_vec[k], (y.ttv_dims[k], A.tto_rks[k], v.ttv_rks[k], A.tto_rks[k + 1], v.ttv_rks[k + 1]))
            @tensoropt((νₖ₋₁, νₖ), yvec_temp[iₖ, αₖ₋₁, νₖ₋₁, αₖ, νₖ] = A.tto_vec[k][iₖ, jₖ, αₖ₋₁, αₖ] * v.ttv_vec[k][jₖ, νₖ₋₁, νₖ])
        end
    end
    return y
end

"""
Contracts a rectangular TTOperator with one additional output site against a TTVector.
"""
function *(A::TTOperator{T, M}, v::TTVector{T, N}) where {T <: Number, M, N}
    @assert M == N + 1 "Rectangular TTOperator must have one additional output site"
    singleton_sites = findall(k -> size(A.tto_vec[k], 2) == 1, 1:M)
    @assert length(singleton_sites) == 1 "Rectangular TTOperator must have exactly one singleton input site"
    singleton_site = only(singleton_sites)
    input_dims = ntuple(k -> size(A.tto_vec[k < singleton_site ? k : k + 1], 2), N)
    @assert input_dims == v.ttv_dims "Incompatible input dimensions"
    @assert v.ttv_rks[end] == 1 "Input TTVector must have a closed right boundary rank"

    out_dims = ntuple(k -> size(A.tto_vec[k], 1), M)
    out_rks = Vector{Int64}(undef, M + 1)
    @inbounds for boundary in 0:M
        consumed_inputs = boundary - (boundary >= singleton_site ? 1 : 0)
        out_rks[boundary + 1] = A.tto_rks[boundary + 1] * v.ttv_rks[consumed_inputs + 1]
    end
    y = zeros_tt(T, out_dims, out_rks)

    @inbounds for k in 1:M
        if k == singleton_site
            consumed_inputs = k - 1
            νdim = v.ttv_rks[consumed_inputs + 1]
            yvec_temp = reshape(y.ttv_vec[k], (out_dims[k], A.tto_rks[k], νdim, A.tto_rks[k + 1], νdim))
            for i in 1:out_dims[k], αₖ₋₁ in 1:A.tto_rks[k], αₖ in 1:A.tto_rks[k + 1], ν in 1:νdim
                yvec_temp[i, αₖ₋₁, ν, αₖ, ν] = A.tto_vec[k][i, 1, αₖ₋₁, αₖ]
            end
        else
            input_site = k < singleton_site ? k : k - 1
            yvec_temp = reshape(y.ttv_vec[k], (out_dims[k], A.tto_rks[k], v.ttv_rks[input_site], A.tto_rks[k + 1], v.ttv_rks[input_site + 1]))
            @tensoropt((νₖ₋₁, νₖ), yvec_temp[iₖ, αₖ₋₁, νₖ₋₁, αₖ, νₖ] = A.tto_vec[k][iₖ, jₖ, αₖ₋₁, αₖ] * v.ttv_vec[input_site][jₖ, νₖ₋₁, νₖ])
        end
    end
    return y
end


function (A::TTOperator{T, N})(x::TTVector{T, N}) where {T, N}
    return A * x
end

# KrylovKit's calling convention for maps that also provide their adjoint.
(A::TTOperator{T, N})(x::TTVector{T, N}, ::Val{false}) where {T, N} = A * x
(A::TTOperator{T, N})(x::TTVector{T, N}, ::Val{true}) where {T, N} = adjoint(A) * x

"""
    adjoint(A::TTOperator) -> TTOperator
    A'

Conjugate transpose of `A`: every core has its output and input indices swapped
and its entries conjugated. The ranks are unchanged.
"""
function Base.adjoint(A::TTOperator{T, N}) where {T, N}
    cores = [conj(permutedims(c, (2, 1, 3, 4))) for c in A.tto_vec]
    return TTOperator{T, N}(cores, A.tto_dims, copy(A.tto_rks); orthogonality = copy(A.orthogonality))
end

"""
Multiplies two TTOperators and returns a new TTOperator.
"""
function *(A::TTOperator{T, N}, B::TTOperator{T, N}) where {T <: Number, N}
    @assert A.tto_dims == B.tto_dims "Incompatible dimensions"
    d = nsites(A)
    A_rks = A.tto_rks #R_0, ..., R_d
    B_rks = B.tto_rks #r_0, ..., r_d
    Y = [zeros(T, A.tto_dims[k], A.tto_dims[k], A_rks[k] * B_rks[k], A_rks[k + 1] * B_rks[k + 1]) for k in eachindex(A.tto_dims)]
    @inbounds for k in eachindex(Y)
        M_temp = reshape(Y[k], A.tto_dims[k], A.tto_dims[k], A_rks[k], B_rks[k], A_rks[k + 1], B_rks[k + 1])
        @tensor M_temp[iₖ, jₖ, αₖ₋₁, βₖ₋₁, αₖ, βₖ] = A.tto_vec[k][iₖ, z, αₖ₋₁, αₖ] * B.tto_vec[k][z, jₖ, βₖ₋₁, βₖ]
    end
    return TTOperator{T, N}(Y, A.tto_dims, A.tto_rks .* B.tto_rks)
end

"""
    ⨝(A::TTOperator, B::TTOperator)
    A ⨝ B

Inner core product of two TT operators (the `⋈` of the QTT literature; Julia's
parser rejects U+22C8, so the visually identical U+2A1D `⨝` is used). At each
site the physical TT blocks are combined by a tensor (Kronecker) product, growing
the physical dimension from `dᴬ` to `dᴬ·dᴮ`, while the bond indices combine
multiplicatively:

    (A ⨝ B)ₖ[(iᴬiᴮ), (jᴬjᴮ), (aᴬaᴮ), (bᴬbᴮ)] = Aₖ[iᴬ,jᴬ,aᴬ,bᴬ] · Bₖ[iᴮ,jᴮ,aᴮ,bᴮ].

The result is an operator on the tensor-product physical space, interleaving the
two operators site-by-site, with ranks `A.tto_rks .* B.tto_rks`. For separable
(rank-1) operators this is exactly the textbook inner core product; in general it
is its rank-stable extension to arbitrary TT operators. Unlike [`kron`](@ref),
which concatenates the cores of the two operators (sequential bit ordering), `⨝`
keeps the same number of sites and interleaves them — usually the lower-rank
ordering for multivariate operators.

The dual operation — physical blocks composed by matrix product, bonds tensored
(the *outer* core product `•`) — is ordinary operator multiplication `A * B`.
"""
function ⨝(A::TTOperator{T, N}, B::TTOperator{T, N}) where {T <: Number, N}
    @assert nsites(A) == nsites(B) "Inner core product requires operators with the same number of cores"
    d = nsites(A)
    Y = Vector{Array{T, 4}}(undef, d)
    @inbounds for k in 1:d
        Ak = A.tto_vec[k]
        Bk = B.tto_vec[k]
        nAo, nAi, rAl, rAr = size(Ak)
        nBo, nBi, rBl, rBr = size(Bk)
        # B-fastest broadcast on every axis so the single reshape produces the
        # Kronecker (A-major, B-minor) layout on physical and bond axes alike.
        T8 = reshape(Bk, nBo, 1, nBi, 1, rBl, 1, rBr, 1) .*
            reshape(Ak, 1, nAo, 1, nAi, 1, rAl, 1, rAr)
        Y[k] = reshape(T8, nBo * nAo, nBi * nAi, rBl * rAl, rBr * rAr)
    end
    dims = ntuple(k -> A.tto_dims[k] * B.tto_dims[k], N)
    rks = A.tto_rks .* B.tto_rks
    return TTOperator{T, N}(Y, dims, rks)
end

"""
    ∙(A::TTOperator, B::TTOperator)
    A ∙ B

Outer core product of two TT operators. The physical TT blocks are composed by matrix product while the bond
indices are tensored — i.e. ordinary operator composition, with ranks combining
as `A.tto_rks .* B.tto_rks`. 
"""
∙(A::TTOperator{T, N}, B::TTOperator{T, N}) where {T <: Number, N} = A * B

function *(A::Array{TTVector{T, N}, 1}, x::Vector{T}) where {T, N}
    out = x[1] * A[1]
    for i in 2:length(A)
        out = out + x[i] * A[i]
    end
    return out
end

"""
Computes the dot product of two TTVectors and returns a scalar.
"""
function dot(A::TTVector{T, N}, B::TTVector{T, N}) where {T <: Number, N}
    @assert A.ttv_dims == B.ttv_dims "TT dimensions are not compatible"
    A_rks = A.ttv_rks
    B_rks = B.ttv_rks
    out = zeros(T, maximum(A_rks), maximum(B_rks))
    out[1, 1] = one(T)
    @inbounds for k in eachindex(A.ttv_dims)
        M = @view(out[1:A_rks[k + 1], 1:B_rks[k + 1]])
        @tensoropt M[a, b] = conj(A.ttv_vec[k][z, α, a]) * (B.ttv_vec[k][z, β, b] * out[1:A_rks[k], 1:B_rks[k]][α, β]) #size R^A_{k} × R^B_{k}
    end
    return out[1, 1]::T
end


# Core that carries a scalar factor. It lies inside the orthogonality interval,
# so scaling it keeps the recorded orthogonality valid.
_scale_site(x) = _orthogonality(x)[1]

"""
Multiplies a TTVector by a scalar and returns a new TTVector.
"""
function *(a::S, A::TTVector{R, N}) where {S <: Number, R <: Number, N}
    T = promote_type(S, R)
    aT = convert(T, a)
    if iszero(aT)
        return zeros_tt(T, A.ttv_dims, copy(A.ttv_rks))
    end
    i = _scale_site(A)
    X = [Array{T, 3}(c) for c in A.ttv_vec]   # promote and copy cores before scaling
    X[i] = aT * X[i]
    return TTVector{T, N}(X, A.ttv_dims, copy(A.ttv_rks); orthogonality = copy(A.orthogonality))
end

"""
Multiplies a TTOperator by a scalar and returns a new TTOperator.
"""
function *(a::S, A::TTOperator{R, N}) where {S <: Number, R <: Number, N}
    T = promote_type(S, R)
    aT = convert(T, a)
    if iszero(aT)
        return zeros_tto(T, A.tto_dims, copy(A.tto_rks))
    end
    i = _scale_site(A)
    X = [Array{T, 4}(c) for c in A.tto_vec]   # promote and copy cores before scaling
    X[i] = aT * X[i]
    return TTOperator{T, N}(X, A.tto_dims, copy(A.tto_rks); orthogonality = copy(A.orthogonality))
end

Base.:*(A::TTVector{T, N}, a::S) where {T <: Number, S <: Number, N} = a * A

-(A::TTVector{T, N}) where {T <: Number, N} = (-one(T)) * A
-(A::TTOperator{T, N}) where {T <: Number, N} = (-one(T)) * A

function -(A::TTVector{T, N}, B::TTVector{T, N}) where {T <: Number, N}
    return A + (-one(T)) * B
end

function -(A::TTOperator{T, N}, B::TTOperator{T, N}) where {T <: Number, N}
    return A + (-one(T)) * B
end

# Mixed element types promote to a common type and dispatch to the same-type methods.
function +(x::TTVector{T1, N}, y::TTVector{T2, N}) where {T1 <: Number, T2 <: Number, N}
    T = promote_type(T1, T2)
    return _convert_eltype(T, x) + _convert_eltype(T, y)
end

function +(x::TTOperator{T1, N}, y::TTOperator{T2, N}) where {T1 <: Number, T2 <: Number, N}
    T = promote_type(T1, T2)
    return _convert_eltype(T, x) + _convert_eltype(T, y)
end

function -(x::TTVector{T1, N}, y::TTVector{T2, N}) where {T1 <: Number, T2 <: Number, N}
    T = promote_type(T1, T2)
    return _convert_eltype(T, x) - _convert_eltype(T, y)
end

function -(x::TTOperator{T1, N}, y::TTOperator{T2, N}) where {T1 <: Number, T2 <: Number, N}
    T = promote_type(T1, T2)
    return _convert_eltype(T, x) - _convert_eltype(T, y)
end

function *(A::TTOperator{T1, N}, v::TTVector{T2, N}) where {T1 <: Number, T2 <: Number, N}
    T = promote_type(T1, T2)
    return _convert_eltype(T, A) * _convert_eltype(T, v)
end

function *(A::TTOperator{T1, N}, B::TTOperator{T2, N}) where {T1 <: Number, T2 <: Number, N}
    T = promote_type(T1, T2)
    return _convert_eltype(T, A) * _convert_eltype(T, B)
end

function dot(A::TTVector{T1, N}, B::TTVector{T2, N}) where {T1 <: Number, T2 <: Number, N}
    T = promote_type(T1, T2)
    return dot(_convert_eltype(T, A), _convert_eltype(T, B))
end

function /(A::TTVector, a)
    return 1 / a * A
end

"""
    outer_product(x::TTVector, y::TTVector) -> TTOperator

Return the rank-one operator `x y†` as a [`TTOperator`](@ref), with entries
`x[i] * conj(y[j])`. Its TT ranks are the products of the ranks of `x` and `y`.
"""
function outer_product(x::TTVector{T, N}, y::TTVector{T, N}) where {T <: Number, N}
    Y = [zeros(T, x.ttv_dims[k], x.ttv_dims[k], x.ttv_rks[k] * y.ttv_rks[k], x.ttv_rks[k + 1] * y.ttv_rks[k + 1]) for k in eachindex(x.ttv_dims)]
    @inbounds for k in eachindex(Y)
        M_temp = reshape(Y[k], x.ttv_dims[k], x.ttv_dims[k], x.ttv_rks[k], y.ttv_rks[k], x.ttv_rks[k + 1], y.ttv_rks[k + 1])
        @tensor M_temp[iₖ, jₖ, αₖ₋₁, βₖ₋₁, αₖ, βₖ] = x.ttv_vec[k][iₖ, αₖ₋₁, αₖ] * conj(y.ttv_vec[k][jₖ, βₖ₋₁, βₖ])
    end
    return TTOperator{T, N}(Y, x.ttv_dims, x.ttv_rks .* y.ttv_rks)
end


"""
Creates a diagonal TTOperator from a TTVector.
"""
function ttv_to_diag_tto(x::TTVector{T, M}) where {T <: Number, M}
    d = nsites(x)                              # number of dimensions (cores)
    dims = x.ttv_dims                       # (n₁, n₂, …, n_d)
    rks = x.ttv_rks                        # (r₀=1, r₁, …, r_d=1)
    cores = x.ttv_vec                        # Vector of length d, each core is Array{T,3} sized (n_i, r_i, r_{i+1})

    new_rks = copy(rks)
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

    return TTOperator{T, M}(new_cores, dims, new_rks)
end

"""
Computes the Hadamard product (element-wise multiplication) of two TTVectors and returns a new TTVector.
"""
function hadamard(x::TTVector{T, N}, y::TTVector{T, N}) where {T <: Number, N}
    @assert x.ttv_dims == y.ttv_dims "Incompatible TT dimensions"
    d = nsites(x)
    ttv_vec = Vector{Array{T, 3}}(undef, d)
    dims = x.ttv_dims
    rks = [x.ttv_rks[k] * y.ttv_rks[k] for k in 1:(d + 1)]

    @inbounds for k in 1:d
        n = dims[k]
        rx1, rx2 = x.ttv_rks[k], x.ttv_rks[k + 1]
        ry1, ry2 = y.ttv_rks[k], y.ttv_rks[k + 1]
        core = zeros(T, n, rx1 * ry1, rx2 * ry2)
        for s in 1:n
            core[s, :, :] = kron(x.ttv_vec[k][s, :, :], y.ttv_vec[k][s, :, :])
        end
        ttv_vec[k] = core
    end
    return TTVector{T, N}(ttv_vec, dims, rks)
end

"""
    x ⊕ y

Elementwise (Hadamard) product of two `TTVector`s; identical to [`hadamard`](@ref).
"""
⊕(x::TTVector{T, N}, y::TTVector{T, N}) where {T <: Number, N} = hadamard(x, y)

# Swap cores j and j + 1 of a chain through a truncated SVD (rule of `_trunc_rank`
# for a TT with d cores). The orthogonality center of the chain must be on one
# of the two cores, so that the singular values are those of the whole chain
# across this bond; it ends on core j.
function _ttm_swap!(
        cores::Vector{Array{T, 3}}, j::Int;
        trunc_tol::Real = 0.0, max_bond::Int = typemax(Int), d::Int
    ) where {T}
    A = cores[j]         # (dA, rL, rM)
    B = cores[j + 1]     # (dB, rM, rR)
    dA, rL, _ = size(A)
    dB, _, rR = size(B)
    # Contract shared bond: C[sA, sB, m, n] = Σ_a A[sA,m,a]*B[sB,a,n], shape (dA,dB,rL,rR)
    @tensor C[sA, sB, m, n] := A[sA, m, a] * B[sB, a, n]
    # Permute to (rL, dB, dA, rR) and flatten: rows=(m,σB), cols=(σA,n)
    mat = reshape(permutedims(C, (3, 2, 1, 4)), rL * dB, dA * rR)
    U, S, Vt = _truncated_svd(mat, trunc_tol, d, max_bond)
    r = size(U, 2)
    cores[j] = permutedims(reshape(U * S, rL, dB, r), (2, 1, 3))    # (dB, rL, r)
    cores[j + 1] = permutedims(reshape(Vt, r, dA, rR), (2, 1, 3))   # (dA, r, rR)
    return cores
end

# Merge cores p and p + 1, which carry the same physical index, into their
# elementwise product on that index.
function _ttm_contract!(cores::Vector{Array{T, 3}}, p::Int) where {T}
    A = cores[p]         # (d, rL, rM)
    B = cores[p + 1]     # (d, rM, rR)
    d_phys, rL = size(A, 1), size(A, 2)
    rR = size(B, 3)
    Pi = zeros(T, d_phys, rL, rR)
    for s in 1:d_phys
        mul!(view(Pi, s, :, :), view(A, s, :, :), view(B, s, :, :))
    end
    cores[p] = Pi
    deleteat!(cores, p + 1)
    return cores
end

"""
    hadamard_ttm(x::TTVector, y::TTVector; trunc_tol=1e-14, max_bond=typemax(Int)) -> TTVector

Elementwise (Hadamard) product of `x` and `y` by tensor train multiplication
(Michailidis, Fenton & Kiffner, arXiv:2410.19747). The cores of `x` followed by
the reversed cores of `y` form a chain of `2d` cores. The cores of `y` are moved
through the chain by swaps of adjacent cores until each meets the core of `x`
with the same physical index, and the two are then merged. Each swap is a
truncated SVD, so the full product with ranks `rˣ·rʸ` is never formed.

The chain is kept in mixed canonical form with its orthogonality center on the
pair being swapped, which makes each truncation optimal for the whole chain.
`trunc_tol` and `max_bond` are applied to every swap with the rule of
[`tt_round!`](@ref), relative to the norm of the chain. That norm is at most
`‖x‖·‖y‖`, so the error of the result is of the order of
`trunc_tol·‖x‖·‖y‖`, which can exceed `trunc_tol·‖x∘y‖`.

`max_bond` limits every bond of the intermediate chain, and those can need a
larger rank than the bonds of the product. A cap that the product itself would
fit under can therefore still cause an error; [`hadamard`](@ref) followed by
[`tt_round!`](@ref) truncates the product only.

The result has its orthogonality center on the first core.
"""
function hadamard_ttm(
        x::TTVector{T, N}, y::TTVector{T, N};
        trunc_tol::Real = 1.0e-14,
        max_bond::Int = typemax(Int)
    ) where {T <: Number, N}
    @assert x.ttv_dims == y.ttv_dims "Incompatible TT dimensions"
    d = nsites(x)

    cores = Vector{Array{T, 3}}(undef, 2d)
    for k in 1:d
        cores[k] = copy(x.ttv_vec[k])
    end
    for k in 1:d
        cores[d + k] = permutedims(y.ttv_vec[d + 1 - k], (1, 3, 2))
    end
    # Mixed canonical form with the orthogonality center on core d.
    for k in 1:(d - 1)
        _orthogonalize_left!(cores, k)
    end
    for k in (2d):-1:(d + 1)
        _orthogonalize_right!(cores, k)
    end
    # Site p of the product is formed from core p and the core of `y` that the
    # swaps bring to position p + 1. Before the swaps the center is on core
    # p + 1 (on core d for p = d) and is moved to core d; each swap then leaves
    # it on the left core of its pair, and the merge leaves it on core p.
    for p in d:-1:1
        for k in (p + 1):(d - 1)
            _orthogonalize_left!(cores, k)
        end
        for j in d:-1:(p + 1)
            _ttm_swap!(cores, j; trunc_tol, max_bond, d)
        end
        _ttm_contract!(cores, p)
    end
    rks = [1; [size(c, 3) for c in cores]]
    return TTVector{T, N}(cores, x.ttv_dims, rks; orthogonality = (1, 1))
end

"""
Computes the Kronecker product of two TTOperators and returns a new TTOperator.
"""
function kron(A::TTOperator{T, d1}, B::TTOperator{T, d2}) where {T, d1, d2}
    d = vcat(A.tto_vec, B.tto_vec)
    dims = (A.tto_dims..., B.tto_dims...)
    rks = vcat(A.tto_rks[1:(end - 1)], B.tto_rks)
    return TTOperator{T, d1 + d2}(d, dims, rks; orthogonality = _joined_orthogonality(A, B))
end

"""
    A ⊗ B

Kronecker product of two `TTOperator`s or two `TTVector`s; identical to
[`kron`](@ref). The result has the cores of `A` followed by the cores of `B`.
"""
⊗(A::TTOperator{T, d1}, B::TTOperator{T, d2}) where {T, d1, d2} = kron(A, B)

"""
Computes the Kronecker product of two TTVectors and returns a new TTVector.
"""
function kron(a::TTVector{T, d1}, b::TTVector{T, d2}) where {T, d1, d2}
    return TTVector{T, d1 + d2}(
        vcat(a.ttv_vec, b.ttv_vec),
        (a.ttv_dims..., b.ttv_dims...),
        vcat(a.ttv_rks[1:(end - 1)], b.ttv_rks);
        orthogonality = _joined_orthogonality(a, b)
    )
end

⊗(a::TTVector{T, d1}, b::TTVector{T, d2}) where {T, d1, d2} = kron(a, b)

"""
    euclidean_distance(a::TTVector, b::TTVector) -> Real

Return `‖a − b‖`, computed from the inner products as
`√(⟨a,a⟩ − 2 Re⟨b,a⟩ + ⟨b,b⟩)` (clamped at zero) without forming `a − b`.
Because of cancellation, distances below about `√eps · max(‖a‖, ‖b‖)` are not
resolved; use `norm(a - b)` when small differences matter.
"""
function euclidean_distance(a::TTVector{T, N}, b::TTVector{T, N}) where {T <: Number, N}
    @assert a.ttv_dims == b.ttv_dims "TT dimensions must match"
    return sqrt(max(real(dot(a, a) - 2 * real(dot(b, a)) + dot(b, b)), zero(real(T))))
end

"""
    euclidean_distance_normalized(a::TTVector, b::TTVector)

Return the relative distance `‖a − b‖ / ‖b‖`, computed from inner products as
`√(1 + ⟨a,a⟩/⟨b,b⟩ − 2 Re⟨b,a⟩/⟨b,b⟩)`. For complex element types the result is
a complex number. The same cancellation limit as [`euclidean_distance`](@ref)
applies.
"""
function euclidean_distance_normalized(a::TTVector{T, N}, b::TTVector{T, N}) where {T <: Number, N}
    @assert a.ttv_dims == b.ttv_dims "TT dimensions must match"
    return sqrt(1.0 + dot(a, a) / dot(b, b) - 2.0 * real(dot(b, a)) / dot(b, b))
end

"""
Computes the norm of a TTVector.
"""
function norm(a::TTVector{T, N}) where {T <: Number, N}
    s = TensorTrainNumerics.dot(a, a)
    v = real(s)
    v = v < 0 ? zero(v) : v
    return sqrt(v)
end
