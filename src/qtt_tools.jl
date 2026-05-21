using LinearAlgebra

function index_to_point(t; L = 1.0)
    d = length(t)
    return sum(2.0^(d - i) * (t[i] - 1) / (2^d - 1) for i in 1:d)
end

function tuple_to_index(t)
    d = length(t)
    return sum(2^(d - i) * (t[i] - 1) for i in 1:d) + 1
end

function function_to_tensor(f, d; a = 0.0, b = 1.0)
    out = zeros(ntuple(x -> 2, d))
    for t in CartesianIndices(out)
        out[t] = f(index_to_point(Tuple(t); L = b - a))
    end
    return out
end

function tensor_to_grid(tensor)
    T = eltype(tensor)
    out = Vector{T}(undef, length(tensor))
    @inbounds for t in CartesianIndices(tensor)
        out[tuple_to_index(Tuple(t))] = tensor[t]
    end
    return out
end

"""
Converts a univariate function `f` into its Quantized Tensor Train (QTT) representation.
"""
function function_to_qtt(f, d; a = 0.0, b = 1.0)
    tensor = function_to_tensor(f, d; a = a, b = b)
    return ttv_decomp(tensor)
end

"""
Converts a quantized tensor train (QTT) vector `qtt` into a function representation.
"""
function qtt_to_function(qtt::TTvector{T, d}) where {T <: Number, d}
    return qtt_to_vector(qtt)
end

function qtt_to_vector(qtt::TTvector{T}) where {T}
    d = qtt.N
    P = qtt.ttv_vec[1][:, 1, :]
    for k in 2:d
        G = qtt.ttv_vec[k]
        n_prev = size(P, 1)
        P_new = similar(P, 2 * n_prev, size(G, 3))
        @views begin
            P_new[1:2:end, :] .= P * G[1, :, :]
            P_new[2:2:end, :] .= P * G[2, :, :]
        end
        P = P_new
    end
    return vec(P)
end

function function_to_qtt_uniform(f, d::Int)
    N = 2^d
    y = [f(n / N) for n in 0:(N - 1)]
    A = zeros(eltype(y), ntuple(_ -> 2, d)...)
    @inbounds for n in 0:(N - 1)
        bits = (digits(n, base = 2, pad = d)) .+ 1
        A[CartesianIndex(Tuple(bits))] = y[n + 1]
    end
    return ttv_decomp(A)
end

"""
Constructs a Quantized Tensor Train (QTT) representation a polynomial with given coefficients
over a uniform grid in the interval `[a, b]` with `2^d` points.
"""
function qtt_polynom(coef, d; a = 0.0, b = 1.0)
    p = length(coef)
    h = (b - a) / (2^d - 1)
    out = zeros_tt(2, d, p; r_and_d = false)
    φ(x, s) = sum(coef[k + 1] * x^(k - s) * binomial(k, s) for k in s:(p - 1))
    t₁ = a
    out.ttv_vec[1][1, 1, :] = [φ(t₁, k) for k in 0:(p - 1)]
    t₁ = a + h * 2^(d - 1) #convention : coarsest first
    out.ttv_vec[1][2, 1, :] = [φ(t₁, k) for k in 0:(p - 1)]
    @fastmath for k in 2:(d - 1)
        for j in 0:(p - 1)
            out.ttv_vec[k][1, j + 1, j + 1] = 1.0
            for i in 0:(p - 1)
                tₖ = h * 2^(d - k)
                out.ttv_vec[k][2, i + 1, j + 1] = binomial(i, i - j) * tₖ^(i - j)
            end
        end
    end
    out.ttv_vec[d][1, 1, 1] = 1.0
    td = h
    out.ttv_vec[d][2, :, 1] = [td^k for k in 0:(p - 1)]
    return out
end

"""
Constructs a Quantized Tensor Train (QTT) representation of cos(λπx)
over a uniform grid in the interval `[a, b]` with `2^d` points.
"""
function qtt_cos(d; a = 0.0, b = 1.0, λ = 1.0)
    out = zeros_tt(2, d, 2)
    h = (b - a) / (2^d - 1)
    t₁ = a
    out.ttv_vec[1][1, 1, :] = [cos(λ * π * t₁); -sin(λ * π * t₁)]
    t₁ = a + h * 2^(d - 1) #convention : coarsest first
    out.ttv_vec[1][2, 1, :] = [cos(λ * π * t₁); -sin(λ * π * t₁)]
    @fastmath for k in 2:(d - 1)
        out.ttv_vec[k][1, :, :] = [1 0;0 1]
        tₖ = h * 2^(d - k)
        out.ttv_vec[k][2, :, :] = [cos(λ * π * tₖ) -sin(λ * π * tₖ); sin(λ * π * tₖ) cos(λ * π * tₖ)]
    end
    out.ttv_vec[d][1, 1, 1] = 1.0
    td = h
    out.ttv_vec[d][2, :, 1] = [cos(λ * π * td); sin(λ * π * td)]
    return out
end

"""
Constructs a Quantized Tensor Train (QTT) representation of sin(λπx)
over a uniform grid in the interval `[a, b]` with `2^d` points.
"""
function qtt_sin(d; a = 0.0, b = 1.0, λ = 1.0)
    out = zeros_tt(2, d, 2)
    h = (b - a) / (2^d - 1)
    t₁ = a
    out.ttv_vec[1][1, 1, :] = [sin(λ * π * t₁); cos(λ * π * t₁)]
    t₁ = a + h * 2^(d - 1) #convention : coarsest first
    out.ttv_vec[1][2, 1, :] = [sin(λ * π * t₁); cos(λ * π * t₁)]
    @fastmath for k in 2:(d - 1)
        out.ttv_vec[k][1, :, :] = [1 0;0 1]
        tₖ = h * 2^(d - k)
        out.ttv_vec[k][2, :, :] = [cos(λ * π * tₖ) -sin(λ * π * tₖ); sin(λ * π * tₖ) cos(λ * π * tₖ)]
    end
    out.ttv_vec[d][1, 1, 1] = 1.0
    td = h
    out.ttv_vec[d][2, :, 1] = [cos(λ * π * td); sin(λ * π * td)]
    return out
end

"""
Constructs a Quantized Tensor Train (QTT) representation of the exponential function
over a uniform grid in the interval `[a, b]` with `2^d` points.
"""
function qtt_exp(d; a = 0.0, b = 1.0, α = 1.0, β = 0.0)
    out = zeros_tt(2, d, 1)
    h = (b - a) / (2^d - 1)
    t₁ = a
    out.ttv_vec[1][1, 1, 1] = exp(α * t₁ + β)
    t₁ = a + h * 2^(d - 1)
    out.ttv_vec[1][2, 1, 1] = exp(α * t₁ + β)
    @fastmath for k in 2:(d - 1)
        tₖ = h * 2^(d - k)
        out.ttv_vec[k][1, 1, 1] = 1.0
        out.ttv_vec[k][2, 1, 1] = exp(α * tₖ)
    end
    out.ttv_vec[d][1, 1, 1] = 1.0
    td = h
    out.ttv_vec[d][2, 1, 1] = exp(α * td)
    return out
end

"""
Converts a quantics tensor train operator (`TToperator`) into its full matrix representation.
"""
function qtto_to_matrix(Aqtto::TToperator{T, d}) where {T, d}
    A = zeros(2^d, 2^d)
    A_tensor = tto_to_tensor(Aqtto)
    @inbounds for t in CartesianIndices(A_tensor)
        A[tuple_to_index(Tuple(t)[1:d]), tuple_to_index(Tuple(t)[(d + 1):end])] = A_tensor[t]
    end
    return A
end

function qtt_basis_vector(d, pos::Int, val::Number = 1.0)
    out = zeros_tt(2, d, 1)
    bits = reverse(digits(pos - 1, base = 2, pad = d))
    @inbounds for k in 1:d
        out.ttv_vec[k][:, 1, 1] .= 0.0
        out.ttv_vec[k][bits[k] + 1, 1, 1] = val
        val = 1.0
    end
    return out
end

"""
Constructs a Quantized Tensor Train (QTT) representation of the Chebyshev polynomial of degree `n` over `2^d` Chebyshev-Lobatto nodes.

# Details
- The function uses the Gauss-Chebyshev-Lobatto nodes, shifted to the interval [0, 1].
"""
function qtt_chebyshev(n, d)
    out = zeros_tt(2, d, 2)
    N = 2^d
    x_nodes, _ = gauss_chebyshev_lobatto(N; shifted = true)
    θ = acos.(clamp.(2 .* x_nodes .- 1, -1.0, 1.0))
    out.ttv_vec[1][1, 1, :] = [cos(n * θ[1]); -sin(n * θ[1])]
    out.ttv_vec[1][2, 1, :] = [cos(n * θ[2^(d - 1) + 1]); -sin(n * θ[2^(d - 1) + 1])]
    @fastmath for k in 2:(d - 1)
        out.ttv_vec[k][1, :, :] .= [1.0 0.0; 0.0 1.0]
        idx = 2^(d - k) + 1
        out.ttv_vec[k][2, :, :] .= [cos(n * θ[idx]) -sin(n * θ[idx]);sin(n * θ[idx])  cos(n * θ[idx])]
    end
    out.ttv_vec[d][1, :, 1] .= [1.0, 0.0]
    out.ttv_vec[d][2, :, 1] .= [cos(n * θ[2]), sin(n * θ[2])]

    return out
end

function qtt_trapezoidal(d; a = 0.0, b = 1.0)
    out = zeros_tt(2, d, 1)
    h = (b - a) / (2^d - 1)

    out.ttv_vec[1][1, 1, 1] = 1.0
    out.ttv_vec[1][2, 1, 1] = 1.0

    @inbounds for k in 2:(d - 1)
        out.ttv_vec[k][1, 1, 1] = 1.0
        out.ttv_vec[k][2, 1, 1] = 1.0
    end

    out.ttv_vec[d][1, 1, 1] = 1.0
    out.ttv_vec[d][2, 1, 1] = 1.0

    return h * out
end

"""
    to_qtt(tt, split_dims; threshold=0.0)

Convert a `TTvector` to QTT format by splitting each core's physical dimension via SVD.

`split_dims[i]` is a list of integers whose product equals `tt.ttv_dims[i]`, specifying
how to factor that core. The **first** entry is the coarsest (most significant) dimension,
consistent with the rest of the package's QTT convention.

An optional `threshold` (relative to the largest singular value) controls rank truncation.
"""
function to_qtt(
        tt::TTvector{T, N}, split_dims::Vector{Vector{Int}};
        threshold::Float64 = 0.0
    ) where {T <: Number, N}
    @assert length(split_dims) == N "split_dims must have one entry per TT core"
    for i in 1:N
        @assert prod(split_dims[i]) == tt.ttv_dims[i] "prod(split_dims[$i]) must equal $(tt.ttv_dims[i])"
    end

    qtt_cores = Vector{Array{T, 3}}()
    new_rks = Int[1]
    new_dims = Int[]

    for i in 1:N
        # Work in (r_l, n, r_r) layout for easy reshaping
        core = permutedims(tt.ttv_vec[i], (2, 1, 3))
        rank_prev = new_rks[end]
        rank_next = tt.ttv_rks[i + 1]
        remaining = tt.ttv_dims[i]

        for j in 1:(length(split_dims[i]) - 1)
            split_size = split_dims[i][j]
            remaining = div(remaining, split_size)

            # BIG-ENDIAN split: split_size is COARSE (outer), remaining is FINE (inner).
            # reshape (r_l, n, r_r) → (r_l, remaining, split_size, r_r) in column-major
            # gives n_0 = fine_0 + coarse_0 * remaining ≡ coarse_0 * remaining + fine_0 ✓
            core = reshape(core, (rank_prev, remaining, split_size, rank_next))
            core = permutedims(core, (1, 3, 2, 4))   # (r_l, split_size, remaining, r_r)
            M = reshape(core, (rank_prev * split_size, remaining * rank_next))

            F = svd(M; full = false)
            U, S, Vt = F.U, F.S, F.Vt
            if threshold > 0.0
                keep = findall(S ./ S[1] .> threshold)
                U, S, Vt = U[:, keep], S[keep], Vt[keep, :]
            end
            new_rank = length(S)

            # Left QTT core: reshape U → (r_l, split_size, new_rank), permute → (split_size, r_l, new_rank)
            push!(qtt_cores, permutedims(reshape(U, (rank_prev, split_size, new_rank)), (2, 1, 3)))
            push!(new_rks, new_rank)
            push!(new_dims, split_size)

            core = reshape(Diagonal(S) * Vt, (new_rank, remaining, rank_next))
            rank_prev = new_rank
        end

        # Last (or only) QTT core for this TT core: core is (r_l, remaining, r_r)
        push!(qtt_cores, permutedims(core, (2, 1, 3)))
        push!(new_rks, rank_next)
        push!(new_dims, remaining)
    end

    N_new = length(qtt_cores)
    return TTvector{T, N_new}(N_new, qtt_cores, Tuple(new_dims), new_rks, zeros(Int64, N_new))
end

"""
    to_ttv(qtt, merge_numbers)

Convert a QTT (or any `TTvector`) back to TT format by contracting consecutive cores.

`merge_numbers[i]` is the number of consecutive QTT cores to merge into TT core `i`.
`sum(merge_numbers)` must equal the total number of QTT cores.

Physical dimensions are merged using the same BIG-ENDIAN convention as `to_qtt`:
the earlier (coarser) core provides the more significant bits.
"""
function to_ttv(qtt::TTvector{T, M}, merge_numbers::Vector{Int}) where {T <: Number, M}
    @assert sum(merge_numbers) == M "merge_numbers must sum to $(M) (the number of QTT cores)"

    tt_cores = Vector{Array{T, 3}}()
    k = 1

    for count in merge_numbers
        # Work in (r_l, n, r_r) layout
        core = permutedims(qtt.ttv_vec[k], (2, 1, 3))   # (r_l, n1, r_mid)

        for j in (k + 1):(k + count - 1)
            G2 = permutedims(qtt.ttv_vec[j], (2, 1, 3))  # (r_mid, n2, r_r)
            r_l, n1, r_mid = size(core, 1), size(core, 2), size(core, 3)
            n2, r_r = size(G2, 2), size(G2, 3)

            # Contract core × G2 over r_mid, then BIG-ENDIAN merge of physical dims.
            # G1_mat: (r_l*n1, r_mid)  — r_l fast (column-major)
            # G2_mat: (r_mid, n2*r_r)  — n2 fast
            G1_mat = reshape(core, (r_l * n1, r_mid))
            G2_mat = reshape(G2, (r_mid, n2 * r_r))
            Mc = G1_mat * G2_mat                          # (r_l*n1, n2*r_r)

            # Reshape to (r_l, n1, n2, r_r), permute to (r_l, n2, n1, r_r),
            # then reshape to (r_l, n1*n2, r_r) → BIG-ENDIAN merged dim ✓
            Mc_r = reshape(Mc, (r_l, n1, n2, r_r))
            core = reshape(permutedims(Mc_r, (1, 3, 2, 4)), (r_l, n1 * n2, r_r))
        end

        # Convert back to (n, r_l, r_r) TTvector format
        push!(tt_cores, permutedims(core, (2, 1, 3)))
        k += count
    end

    N_new = length(tt_cores)
    new_dims = ntuple(i -> size(tt_cores[i], 1), N_new)
    new_rks = vcat([size(c, 2) for c in tt_cores], size(tt_cores[end], 3))
    return TTvector{T, N_new}(N_new, tt_cores, new_dims, new_rks, zeros(Int64, N_new))
end

"""
A Quantized Tensor Train vector with explicit multi-dimensional ordering metadata.

Identical TT fields as `TTvector` plus:
- `n_dims`: number of spatial dimensions
- `bits_per_dim`: bits per dimension (total sites = n_dims × bits_per_dim)
- `ordering`: `:interleaved` or `:serial`
"""
mutable struct QTTvector{T <: Number, M} <: AbstractTTvector
    N::Int64
    ttv_vec::Vector{Array{T, 3}}
    ttv_dims::NTuple{M, Int64}
    ttv_rks::Vector{Int64}
    ttv_ot::Vector{Int64}
    n_dims::Int
    bits_per_dim::Int
    ordering::Symbol
end

"""
A Quantized Tensor Train operator with explicit multi-dimensional ordering metadata.
"""
struct QTToperator{T <: Number, M} <: AbstractTToperator
    N::Int64
    tto_vec::Array{Array{T, 4}, 1}
    tto_dims::NTuple{M, Int64}
    tto_rks::Array{Int64, 1}
    tto_ot::Array{Int64, 1}
    n_dims::Int
    bits_per_dim::Int
    ordering::Symbol
end

Base.eltype(::QTTvector{T, M}) where {T, M} = T
Base.eltype(::QTToperator{T, M}) where {T, M} = T

"""
    QTTvector(ttv::TTvector{T, M}, n_dims::Int, bits_per_dim::Int, ordering::Symbol)

Wrap a `TTvector` as a `QTTvector` by specifying multi-dimensional QTT metadata.

# Arguments
- `ttv::TTvector`: The underlying TT vector to wrap
- `n_dims::Int`: Number of spatial dimensions
- `bits_per_dim::Int`: Number of bits per dimension (N = n_dims × bits_per_dim)
- `ordering::Symbol`: Either `:interleaved` or `:serial`

All physical dimensions in `ttv` must be 2.
"""
function QTTvector(ttv::TTvector{T, M}, n_dims::Int, bits_per_dim::Int, ordering::Symbol) where {T, M}
    @assert n_dims * bits_per_dim == ttv.N "n_dims * bits_per_dim must equal ttv.N (got $(n_dims)*$(bits_per_dim)=$(n_dims*bits_per_dim) ≠ $(ttv.N))"
    @assert all(==(2), ttv.ttv_dims) "All physical dimensions must be 2 for QTT (got $(ttv.ttv_dims))"
    @assert ordering ∈ (:interleaved, :serial) "ordering must be :interleaved or :serial (got $ordering)"
    QTTvector{T, M}(ttv.N, ttv.ttv_vec, ttv.ttv_dims, ttv.ttv_rks, ttv.ttv_ot, n_dims, bits_per_dim, ordering)
end

"""
    QTToperator(tto::TToperator{T, M}, n_dims::Int, bits_per_dim::Int, ordering::Symbol)

Wrap a `TToperator` as a `QTToperator` by specifying multi-dimensional QTT metadata.

# Arguments
- `tto::TToperator`: The underlying TT operator to wrap
- `n_dims::Int`: Number of spatial dimensions
- `bits_per_dim::Int`: Number of bits per dimension (N = n_dims × bits_per_dim)
- `ordering::Symbol`: Either `:interleaved` or `:serial`

All physical dimensions in `tto` must be 2.
"""
function QTToperator(tto::TToperator{T, M}, n_dims::Int, bits_per_dim::Int, ordering::Symbol) where {T, M}
    @assert n_dims * bits_per_dim == tto.N "n_dims * bits_per_dim must equal tto.N (got $(n_dims)*$(bits_per_dim)=$(n_dims*bits_per_dim) ≠ $(tto.N))"
    @assert all(==(2), tto.tto_dims) "All physical dimensions must be 2 for QTT (got $(tto.tto_dims))"
    @assert ordering ∈ (:interleaved, :serial) "ordering must be :interleaved or :serial (got $ordering)"
    QTToperator{T, M}(tto.N, tto.tto_vec, tto.tto_dims, tto.tto_rks, tto.tto_ot, n_dims, bits_per_dim, ordering)
end

"""
    TTvector(q::QTTvector{T, M})

Strip QTT metadata to recover the underlying `TTvector`.
"""
TTvector(q::QTTvector{T, M}) where {T, M} =
    TTvector{T, M}(q.N, q.ttv_vec, q.ttv_dims, q.ttv_rks, q.ttv_ot)

"""
    TToperator(q::QTToperator{T, M})

Strip QTT metadata to recover the underlying `TToperator`.
"""
TToperator(q::QTToperator{T, M}) where {T, M} =
    TToperator{T, M}(q.N, q.tto_vec, q.tto_dims, q.tto_rks, q.tto_ot)

"""
    check_compat(a::QTTvector, b::QTTvector)

Verify that two `QTTvector`s have compatible QTT metadata (n_dims, bits_per_dim, ordering).

Throws AssertionError if incompatible.
"""
function check_compat(a::QTTvector, b::QTTvector)
    @assert a.n_dims == b.n_dims "QTTvector n_dims mismatch: $(a.n_dims) ≠ $(b.n_dims)"
    @assert a.bits_per_dim == b.bits_per_dim "QTTvector bits_per_dim mismatch: $(a.bits_per_dim) ≠ $(b.bits_per_dim)"
    @assert a.ordering == b.ordering "QTTvector ordering mismatch: $(a.ordering) ≠ $(b.ordering)"
end

"""
    check_compat(A::QTToperator, ψ::QTTvector)

Verify that a `QTToperator` and `QTTvector` have compatible QTT metadata.

Throws AssertionError if incompatible.
"""
function check_compat(A::QTToperator, ψ::QTTvector)
    @assert A.n_dims == ψ.n_dims "QTToperator/QTTvector n_dims mismatch: $(A.n_dims) ≠ $(ψ.n_dims)"
    @assert A.bits_per_dim == ψ.bits_per_dim "QTToperator/QTTvector bits_per_dim mismatch: $(A.bits_per_dim) ≠ $(ψ.bits_per_dim)"
    @assert A.ordering == ψ.ordering "QTToperator/QTTvector ordering mismatch: $(A.ordering) ≠ $(ψ.ordering)"
end

"""
    check_compat(::TTvector, ::TTvector)

No-op for plain TTvectors (always compatible).
"""
check_compat(::TTvector, ::TTvector) = nothing

"""
    check_compat(::TToperator, ::TTvector)

No-op for plain TToperator/TTvector pairs (always compatible).
"""
check_compat(::TToperator, ::TTvector) = nothing

function check_compat(A::QTToperator, B::QTToperator)
    @assert A.n_dims == B.n_dims "QTToperator n_dims mismatch: $(A.n_dims) ≠ $(B.n_dims)"
    @assert A.bits_per_dim == B.bits_per_dim "QTToperator bits_per_dim mismatch: $(A.bits_per_dim) ≠ $(B.bits_per_dim)"
    @assert A.ordering == B.ordering "QTToperator ordering mismatch: $(A.ordering) ≠ $(B.ordering)"
end

function orthogonalize(q::QTTvector; i::Int = 1)
    QTTvector(orthogonalize(TTvector(q); i = i), q.n_dims, q.bits_per_dim, q.ordering)
end

function Base.copy(q::QTTvector)
    QTTvector(copy(TTvector(q)), q.n_dims, q.bits_per_dim, q.ordering)
end

function Base.complex(q::QTTvector)
    QTTvector(complex(TTvector(q)), q.n_dims, q.bits_per_dim, q.ordering)
end

function Base.:+(a::QTTvector, b::QTTvector)
    check_compat(a, b)
    QTTvector(TTvector(a) + TTvector(b), a.n_dims, a.bits_per_dim, a.ordering)
end

function Base.:-(a::QTTvector, b::QTTvector)
    check_compat(a, b)
    QTTvector(TTvector(a) - TTvector(b), a.n_dims, a.bits_per_dim, a.ordering)
end

function Base.:*(α::Number, q::QTTvector)
    QTTvector(α * TTvector(q), q.n_dims, q.bits_per_dim, q.ordering)
end

function Base.:*(q::QTTvector, α::Number)
    QTTvector(TTvector(q) * α, q.n_dims, q.bits_per_dim, q.ordering)
end

function hadamard(a::QTTvector, b::QTTvector)
    check_compat(a, b)
    QTTvector(hadamard(TTvector(a), TTvector(b)), a.n_dims, a.bits_per_dim, a.ordering)
end

function LinearAlgebra.dot(a::QTTvector, b::QTTvector)
    check_compat(a, b)
    dot(TTvector(a), TTvector(b))
end

function dot(a::QTTvector, b::QTTvector)
    check_compat(a, b)
    dot(TTvector(a), TTvector(b))
end

function LinearAlgebra.norm(q::QTTvector)
    norm(TTvector(q))
end

function Base.copy(A::QTToperator{T, M}) where {T, M}
    tto = TToperator(A)
    tto_copy = TToperator{T, M}(tto.N, copy.(tto.tto_vec), tto.tto_dims, copy(tto.tto_rks), copy(tto.tto_ot))
    QTToperator(tto_copy, A.n_dims, A.bits_per_dim, A.ordering)
end

function Base.:+(A::QTToperator, B::QTToperator)
    check_compat(A, B)
    QTToperator(TToperator(A) + TToperator(B), A.n_dims, A.bits_per_dim, A.ordering)
end

function Base.:*(α::Number, A::QTToperator)
    QTToperator(α * TToperator(A), A.n_dims, A.bits_per_dim, A.ordering)
end

function Base.:*(A::QTToperator, ψ::QTTvector)
    check_compat(A, ψ)
    QTTvector(TToperator(A) * TTvector(ψ), ψ.n_dims, ψ.bits_per_dim, ψ.ordering)
end

function Base.:*(A::TToperator, q::QTTvector)
    A * TTvector(q)
end

function Base.:*(A::QTToperator, v::TTvector)
    TToperator(A) * v
end

function Base.:-(a::QTTvector, b::TTvector)
    TTvector(a) - b
end

function Base.:-(a::TTvector, b::QTTvector)
    a - TTvector(b)
end

function Base.:+(a::QTTvector, b::TTvector)
    TTvector(a) + b
end

function Base.:+(a::TTvector, b::QTTvector)
    a + TTvector(b)
end

function Base.:/(q::QTTvector, α::Number)
    QTTvector(TTvector(q) / α, q.n_dims, q.bits_per_dim, q.ordering)
end

dot(a::TTvector, b::QTTvector) = dot(a, TTvector(b))
dot(a::QTTvector, b::TTvector) = dot(TTvector(a), b)

LinearAlgebra.dot(a::TTvector, b::QTTvector) = dot(a, TTvector(b))
LinearAlgebra.dot(a::QTTvector, b::TTvector) = dot(TTvector(a), b)

function Base.:-(A::TToperator, B::QTToperator)
    A - TToperator(B)
end

function Base.:-(A::QTToperator, B::TToperator)
    TToperator(A) - B  # TToperator(q::QTToperator) strips metadata
end

function Base.:+(A::TToperator, B::QTToperator)
    A + TToperator(B)
end

function Base.:+(A::QTToperator, B::TToperator)
    TToperator(A) + B  # TToperator(q::QTToperator) strips metadata
end

"""
    _swap_adjacent_sites(A, B; threshold=0.0)

Swap the physical indices of two adjacent TT cores `A` (site k) and `B` (site k+1).

Both cores follow the `(phys_dim, left_rank, right_rank)` layout. The two-site
tensor is contracted, the physical indices are transposed, and the result is
re-factorized via a (possibly truncated) SVD.

Returns `(new_A, new_B)` with the swapped cores.
"""
function _swap_adjacent_sites(A::AbstractArray{T, 3}, B::AbstractArray{T, 3};
        threshold::Real = 0.0) where {T}
    d1, rl, rm = size(A)   # (phys_dim=2, left_rank, mid_rank)
    d2, _rm, rr = size(B)  # (phys_dim=2, mid_rank, right_rank)

    C = zeros(T, d1, d2, rl, rr)
    for σ1 in 1:d1, σ2 in 1:d2, l in 1:rl, r in 1:rr
        for m in 1:rm
            C[σ1, σ2, l, r] += A[σ1, l, m] * B[σ2, m, r]
        end
    end

    C_for_svd = permutedims(C, (2, 3, 1, 4))  # (d2, rl, d1, rr) = (σ2, l, σ1, r)

    M = reshape(C_for_svd, d2 * rl, d1 * rr)
    F = svd(M)

    sv = F.S
    r_new = if threshold > 0
        max(1, sum(sv .> threshold * sv[1]))
    else
        length(sv)
    end

    U = F.U[:, 1:r_new]           # (d2*rl, r_new)
    S = Diagonal(sv[1:r_new])
    Vt = F.Vt[1:r_new, :]        # (r_new, d1*rr)

    new_A = reshape(U, d2, rl, r_new)
    SV = S * Vt
    new_B = permutedims(reshape(SV, r_new, d1, rr), (2, 1, 3))

    return new_A, new_B
end

"""
    _bubble_sort_swaps(perm)

Given a permutation vector `perm` (1-based, 0-based values), return the list of
adjacent swap positions (1-based) that bubble-sort `perm` into ascending order.

Each returned index `k` means "swap positions k and k+1".
"""
function _bubble_sort_swaps(perm)
    p = copy(perm)
    swaps = Int[]
    n = length(p)
    for i in 1:n
        for j in 1:(n - i)
            if p[j] > p[j + 1]
                p[j], p[j + 1] = p[j + 1], p[j]
                push!(swaps, j)
            end
        end
    end
    return swaps
end

"""
    reorder(q::QTTvector, new_ordering::Symbol; threshold=0.0)

Convert a `QTTvector` between `:interleaved` and `:serial` orderings by performing
a sequence of adjacent site swaps (sorting-network style, via bubble sort).

Each adjacent swap contracts two neighboring cores, permutes their physical indices,
and re-factorizes via SVD. The optional `threshold` (relative to the largest singular
value) controls rank truncation during each SVD step.

Returns a new `QTTvector` with `ordering == new_ordering`.
"""
function reorder(q::QTTvector, new_ordering::Symbol; threshold::Real = 0.0)
    @assert new_ordering ∈ (:interleaved, :serial) "ordering must be :interleaved or :serial"
    q.ordering == new_ordering && return copy(q)

    n_dims = q.n_dims
    bits_per_dim = q.bits_per_dim
    N = q.N

    perm = zeros(Int, N)
    if q.ordering == :serial && new_ordering == :interleaved

        for d in 1:n_dims, b in 0:(bits_per_dim - 1)
            src = (d - 1) * bits_per_dim + b
            tgt = b * n_dims + (d - 1)
            perm[src + 1] = tgt
        end
    else  # :interleaved → :serial
        # Interleaved site b*n_dims + (d-1)  →  serial position (d-1)*bits_per_dim + b
        for d in 1:n_dims, b in 0:(bits_per_dim - 1)
            src = b * n_dims + (d - 1)
            tgt = (d - 1) * bits_per_dim + b
            perm[src + 1] = tgt
        end
    end

    swaps = _bubble_sort_swaps(perm)

    # Apply swaps to a mutable copy of the cores
    cores = deepcopy(q.ttv_vec)
    for k in swaps
        new_k, new_kp1 = _swap_adjacent_sites(cores[k], cores[k + 1]; threshold = threshold)
        cores[k] = new_k
        cores[k + 1] = new_kp1
    end

    rks = ones(Int, N + 1)
    for k in 1:N
        rks[k + 1] = size(cores[k], 3)
    end
    dims = ntuple(_ -> 2, N)
    new_ttv = TTvector{eltype(q), N}(N, cores, dims, rks, zeros(Int, N))
    return QTTvector(new_ttv, n_dims, bits_per_dim, new_ordering)
end

"""
    tt_compress!(q::QTTvector, max_bond::Int; kwargs...)

In-place compression of a `QTTvector`, preserving QTT metadata.

Delegates to the underlying `TTvector` compression via shared array mutation.
"""
function tt_compress!(q::QTTvector, max_bond::Int; kwargs...)
    tt_compress!(TTvector(q), max_bond; kwargs...)
    return q
end

"""
    tt_up_rks(q::QTTvector, rk_max::Int; kwargs...)

Increase the bond dimension of a `QTTvector`, preserving QTT metadata.
"""
function tt_up_rks(q::QTTvector, rk_max::Int; kwargs...)
    QTTvector(tt_up_rks(TTvector(q), rk_max; kwargs...), q.n_dims, q.bits_per_dim, q.ordering)
end

"""
    function_to_qttv(f, n_dims, bits_per_dim; ordering=:interleaved, a=0.0, b=1.0)

Evaluate an `n_dims`-dimensional function `f` on a uniform grid with `2^bits_per_dim` points
per dimension and return a `QTTvector`. `f` receives an `n_dims`-length coordinate vector.

Grid points: `a + i * (b - a) / (2^bits_per_dim - 1)` for `i = 0, ..., 2^bits_per_dim - 1`.
"""
function function_to_qttv(f, n_dims::Int, bits_per_dim::Int;
        ordering::Symbol = :interleaved, a::Real = 0.0, b::Real = 1.0)
    N = n_dims * bits_per_dim
    n_pts = 2^bits_per_dim
    h = (b - a) / (n_pts - 1)

    tensor = zeros(ntuple(_ -> 2, N))
    grid_idx = zeros(Int, n_dims)
    coords = zeros(n_dims)

    for idx in CartesianIndices(tensor)
        bits = Tuple(idx)
        fill!(grid_idx, 0)
        for site in 1:N
            bit_val = bits[site] - 1
            if ordering == :interleaved
                dim = ((site - 1) % n_dims) + 1
                level = (site - 1) ÷ n_dims
            else
                dim = ((site - 1) ÷ bits_per_dim) + 1
                level = (site - 1) % bits_per_dim
            end
            grid_idx[dim] += bit_val * 2^(bits_per_dim - 1 - level)
        end
        for d in 1:n_dims
            coords[d] = a + grid_idx[d] * h
        end
        tensor[idx] = f(coords)
    end

    ttv = ttv_decomp(tensor)
    return QTTvector(ttv, n_dims, bits_per_dim, ordering)
end

"""
    _swap_adjacent_sites_op(A, B; threshold=0.0)

Swap the physical indices of two adjacent TToperator cores `A` (site k) and `B` (site k+1).

Both cores follow the `(phys_dim, phys_dim, left_rank, right_rank)` layout. The two-site
tensor is contracted, the physical indices (both row and column) are transposed, and the
result is re-factorized via a (possibly truncated) SVD.

Returns `(new_A, new_B)` with the swapped cores.
"""
function _swap_adjacent_sites_op(A::AbstractArray{T, 4}, B::AbstractArray{T, 4};
        threshold::Real = 0.0) where {T}
    d1, _, rl, rm = size(A)   # (phys, phys, left_rank, mid_rank)
    d2, _, _rm, rr = size(B)  # (phys, phys, mid_rank, right_rank)

    C = zeros(T, d1, d1, d2, d2, rl, rr)
    for i1 in 1:d1, j1 in 1:d1, i2 in 1:d2, j2 in 1:d2, l in 1:rl, r in 1:rr
        for m in 1:rm
            C[i1, j1, i2, j2, l, r] += A[i1, j1, l, m] * B[i2, j2, m, r]
        end
    end

    C_for_svd = permutedims(C, (3, 4, 5, 1, 2, 6)) 
    M = reshape(C_for_svd, d2 * d2 * rl, d1 * d1 * rr)
    F = svd(M)

    sv = F.S
    r_new = if threshold > 0
        max(1, sum(sv .> threshold * sv[1]))
    else
        length(sv)
    end

    U = F.U[:, 1:r_new]           # (d2*d2*rl, r_new)
    SV = Diagonal(sv[1:r_new]) * F.Vt[1:r_new, :]  # (r_new, d1*d1*rr)

    new_A = reshape(U, d2, d2, rl, r_new)

    new_B = permutedims(reshape(SV, r_new, d1, d1, rr), (2, 3, 1, 4))

    return new_A, new_B
end

"""
    reorder(A::QTToperator, new_ordering::Symbol; threshold=0.0)

Convert a `QTToperator` between `:interleaved` and `:serial` orderings by performing
a sequence of adjacent site swaps (sorting-network style, via bubble sort).

Returns a new `QTToperator` with `ordering == new_ordering`.
"""
function reorder(A::QTToperator, new_ordering::Symbol; threshold::Real = 0.0)
    @assert new_ordering ∈ (:interleaved, :serial) "ordering must be :interleaved or :serial"
    A.ordering == new_ordering && return copy(A)

    n_dims = A.n_dims
    bits_per_dim = A.bits_per_dim
    N = A.N

    # Build the same permutation as for QTTvector reorder
    perm = zeros(Int, N)
    if A.ordering == :serial && new_ordering == :interleaved
        for d in 1:n_dims, b in 0:(bits_per_dim - 1)
            src = (d - 1) * bits_per_dim + b
            tgt = b * n_dims + (d - 1)
            perm[src + 1] = tgt
        end
    else  # :interleaved → :serial
        for d in 1:n_dims, b in 0:(bits_per_dim - 1)
            src = b * n_dims + (d - 1)
            tgt = (d - 1) * bits_per_dim + b
            perm[src + 1] = tgt
        end
    end

    swaps = _bubble_sort_swaps(perm)

    cores = deepcopy(A.tto_vec)
    for k in swaps
        new_k, new_kp1 = _swap_adjacent_sites_op(cores[k], cores[k + 1]; threshold = threshold)
        cores[k] = new_k
        cores[k + 1] = new_kp1
    end

    rks = ones(Int, N + 1)
    for k in 1:N
        rks[k + 1] = size(cores[k], 4)
    end
    dims = ntuple(_ -> 2, N)
    new_tto = TToperator{eltype(A), N}(N, cores, dims, rks, zeros(Int, N))
    return QTToperator(new_tto, n_dims, bits_per_dim, new_ordering)
end

"""
    qttv_to_array(q::QTTvector)

Contract the TT chain and return an `n_dims`-dimensional array of size `2^bits_per_dim`
per dimension, with values ordered on the uniform grid (index 1 = leftmost grid point).
"""
function qttv_to_array(q::QTTvector)
    N = q.N
    n_dims = q.n_dims
    bits_per_dim = q.bits_per_dim
    ordering = q.ordering
    n_pts = 2^bits_per_dim

    full_tensor = ttv_to_tensor(TTvector(q))
    out = zeros(eltype(full_tensor), ntuple(_ -> n_pts, n_dims))
    grid_idx = zeros(Int, n_dims)

    for idx in CartesianIndices(full_tensor)
        bits = Tuple(idx)
        fill!(grid_idx, 0)
        for site in 1:N
            bit_val = bits[site] - 1
            if ordering == :interleaved
                dim = ((site - 1) % n_dims) + 1
                level = (site - 1) ÷ n_dims
            else
                dim = ((site - 1) ÷ bits_per_dim) + 1
                level = (site - 1) % bits_per_dim
            end
            grid_idx[dim] += bit_val * 2^(bits_per_dim - 1 - level)
        end
        out[CartesianIndex(ntuple(d -> grid_idx[d] + 1, n_dims))] = full_tensor[idx]
    end

    return out
end
