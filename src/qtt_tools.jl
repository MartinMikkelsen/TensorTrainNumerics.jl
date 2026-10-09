using LinearAlgebra

"""
    gauss_chebyshev_lobatto(n; shifted=true) -> (x, w)

Return the `n` Chebyshev–Lobatto nodes `x[j+1] = cos(πj/(n−1))`, `j = 0, …, n−1`
(in decreasing order), and the Gauss–Chebyshev–Lobatto weights for the weight
function `1/√(1−x²)`: `π/(n−1)`, halved at both endpoints. With `shifted = true`
the nodes are mapped from `[−1, 1]` to `[0, 1]` and the weights are halved.
"""
function gauss_chebyshev_lobatto(n; shifted = true)
    x = [cos(π * j / (n - 1)) for j in 0:(n - 1)]
    w = π / (n - 1) * ones(n)
    w[1] /= 2
    w[end] /= 2
    if shifted
        x .= (x .+ 1) ./ 2
        w .= w ./ 2
    end
    return x, w
end

# With a single site there is nothing to chain: the one core holds the values at
# the two grid points (or the 2 × 2 matrix of an operator) directly.
_single_site_qtt(values) = TTVector{eltype(values), 1}([reshape(collect(values), 2, 1, 1)], (2,), [1, 1])
_single_site_qtto(M) = TTOperator{eltype(M), 1}([reshape(collect(M), 2, 2, 1, 1)], (2,), [1, 1])

"""
    index_to_point(t; L=1.0) -> Float64

Map a tuple of 1-based QTT indices `t = (t₁, …, t_d)`, with `t₁` the most
significant bit, to the grid point `L·j/(2^d − 1)` in `[0, L]`, where
`j = Σₖ 2^(d−k)(tₖ − 1)`.
"""
function index_to_point(t; L = 1.0)
    d = length(t)
    return L * sum(2.0^(d - i) * (t[i] - 1) for i in 1:d) / (2^d - 1)
end

"""
    tuple_to_index(t) -> Int

Map a tuple of 1-based QTT indices `t = (t₁, …, t_d)`, with `t₁` the most
significant bit, to the 1-based linear index `1 + Σₖ 2^(d−k)(tₖ − 1)`.
"""
function tuple_to_index(t)
    d = length(t)
    return sum(2^(d - i) * (t[i] - 1) for i in 1:d) + 1
end

"""
    function_to_tensor(f, d; a=0.0, b=1.0) -> Array{Float64,d}

Evaluate `f` on the `2^d` grid points `x_j = a + j(b − a)/(2^d − 1)`,
`j = 0, …, 2^d − 1`, which include both endpoints, and return the values as a
`2 × ⋯ × 2` array indexed by the bits of `j` (most significant first).
"""
function function_to_tensor(f, d; a = 0.0, b = 1.0)
    out = zeros(ntuple(x -> 2, d))
    for t in CartesianIndices(out)
        out[t] = f(a + index_to_point(Tuple(t); L = b - a))
    end
    return out
end

"""
    tensor_to_grid(tensor) -> Vector

Flatten a `2 × ⋯ × 2` array to a vector of length `2^d`, reading the array
indices as the bits of the position with the first index most significant
(the ordering of [`tuple_to_index`](@ref)).
"""
function tensor_to_grid(tensor)
    T = eltype(tensor)
    out = Vector{T}(undef, length(tensor))
    @inbounds for t in CartesianIndices(tensor)
        out[tuple_to_index(Tuple(t))] = tensor[t]
    end
    return out
end

"""
    function_to_qtt(f, d; a=0.0, b=1.0) -> TTVector

Sample the univariate function `f` on the `2^d` uniform grid points of
`[a, b]` (both endpoints included; see [`function_to_tensor`](@ref)) and
compress the samples with [`tt_decomp`](@ref). This is the grid used by
[`qtt_polynomial`](@ref), [`qtt_sin`](@ref), and the other closed-form QTT
constructors. The dense `2^d` samples are formed first, so this is practical
only for moderate `d`; [`tt_cross`](@ref) avoids that.
"""
function function_to_qtt(f, d; a = 0.0, b = 1.0)
    tensor = function_to_tensor(f, d; a = a, b = b)
    return tt_decomp(tensor)
end

"""
    qtt_to_function(qtt::TTVector) -> Vector

Return the `2^d` entries of the binary QTT `qtt` as a vector; identical to
[`qtt_to_vector`](@ref).
"""
function qtt_to_function(qtt::AbstractTTVector{T, d}) where {T <: Number, d}
    return qtt_to_vector(qtt)
end

"""
    qtt_to_vector(qtt::TTVector) -> Vector

Return the `2^d` entries of the binary QTT `qtt` as a vector, with site 1 the
most significant bit. The contraction is progressive and never forms the
`2 × ⋯ × 2` tensor.
"""
function qtt_to_vector(qtt::AbstractTTVector{T}) where {T}
    d = nsites(qtt)
    P = qtt.cores[1][:, 1, :]
    for k in 2:d
        G = qtt.cores[k]
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

"""
    function_to_qtt_uniform(f, d::Int) -> TTVector

Sample `f` at the periodic grid `x_n = n/2^d`, `n = 0, …, 2^d − 1` (the right
endpoint `1` is excluded), and compress the samples with [`tt_decomp`](@ref).

Site 1 of the result holds the *least* significant bit of `n`, which is the
input layout expected by [`fourier_qtto`](@ref). The other QTT constructors and
[`qtt_to_vector`](@ref) put the most significant bit first; apply
[`reverse_qtt_bits`](@ref) to convert between the two.
"""
function function_to_qtt_uniform(f, d::Int)
    N = 2^d
    y = [f(n / N) for n in 0:(N - 1)]
    A = zeros(eltype(y), ntuple(_ -> 2, d)...)
    @inbounds for n in 0:(N - 1)
        bits = (digits(n, base = 2, pad = d)) .+ 1
        A[CartesianIndex(Tuple(bits))] = y[n + 1]
    end
    return tt_decomp(A)
end

"""
Constructs a Quantized Tensor Train (QTT) representation a polynomial with given coefficients
over a uniform grid in the interval `[a, b]` with `2^d` points.
"""
function qtt_polynomial(coef, d; a = 0.0, b = 1.0)
    d == 1 && return _single_site_qtt([evalpoly(t, coef) for t in (a, b)])
    p = length(coef)
    h = (b - a) / (2^d - 1)
    out = zeros_tt(2, d, p; admissible = false)
    φ(x, s) = sum(coef[k + 1] * x^(k - s) * binomial(k, s) for k in s:(p - 1))
    t₁ = a
    out.cores[1][1, 1, :] = [φ(t₁, k) for k in 0:(p - 1)]
    t₁ = a + h * 2^(d - 1) #convention : coarsest first
    out.cores[1][2, 1, :] = [φ(t₁, k) for k in 0:(p - 1)]
    @fastmath for k in 2:(d - 1)
        for j in 0:(p - 1)
            out.cores[k][1, j + 1, j + 1] = 1.0
            for i in 0:(p - 1)
                tₖ = h * 2^(d - k)
                out.cores[k][2, i + 1, j + 1] = binomial(i, i - j) * tₖ^(i - j)
            end
        end
    end
    out.cores[d][1, 1, 1] = 1.0
    td = h
    out.cores[d][2, :, 1] = [td^k for k in 0:(p - 1)]
    return out
end

"""
Constructs a Quantized Tensor Train (QTT) representation of cos(λπx)
over a uniform grid in the interval `[a, b]` with `2^d` points.
"""
function qtt_cos(d; a = 0.0, b = 1.0, λ = 1.0)
    d == 1 && return _single_site_qtt([cos(λ * π * t) for t in (a, b)])
    out = zeros_tt(2, d, 2)
    h = (b - a) / (2^d - 1)
    t₁ = a
    out.cores[1][1, 1, :] = [cos(λ * π * t₁); -sin(λ * π * t₁)]
    t₁ = a + h * 2^(d - 1) #convention : coarsest first
    out.cores[1][2, 1, :] = [cos(λ * π * t₁); -sin(λ * π * t₁)]
    @fastmath for k in 2:(d - 1)
        out.cores[k][1, :, :] = [1 0;0 1]
        tₖ = h * 2^(d - k)
        out.cores[k][2, :, :] = [cos(λ * π * tₖ) -sin(λ * π * tₖ); sin(λ * π * tₖ) cos(λ * π * tₖ)]
    end
    out.cores[d][1, 1, 1] = 1.0
    td = h
    out.cores[d][2, :, 1] = [cos(λ * π * td); sin(λ * π * td)]
    return out
end

"""
Constructs a Quantized Tensor Train (QTT) representation of sin(λπx)
over a uniform grid in the interval `[a, b]` with `2^d` points.
"""
function qtt_sin(d; a = 0.0, b = 1.0, λ = 1.0)
    d == 1 && return _single_site_qtt([sin(λ * π * t) for t in (a, b)])
    out = zeros_tt(2, d, 2)
    h = (b - a) / (2^d - 1)
    t₁ = a
    out.cores[1][1, 1, :] = [sin(λ * π * t₁); cos(λ * π * t₁)]
    t₁ = a + h * 2^(d - 1) #convention : coarsest first
    out.cores[1][2, 1, :] = [sin(λ * π * t₁); cos(λ * π * t₁)]
    @fastmath for k in 2:(d - 1)
        out.cores[k][1, :, :] = [1 0;0 1]
        tₖ = h * 2^(d - k)
        out.cores[k][2, :, :] = [cos(λ * π * tₖ) -sin(λ * π * tₖ); sin(λ * π * tₖ) cos(λ * π * tₖ)]
    end
    out.cores[d][1, 1, 1] = 1.0
    td = h
    out.cores[d][2, :, 1] = [cos(λ * π * td); sin(λ * π * td)]
    return out
end

"""
Constructs a Quantized Tensor Train (QTT) representation of the exponential function
over a uniform grid in the interval `[a, b]` with `2^d` points.
"""
function qtt_exp(d; a = 0.0, b = 1.0, α = 1.0, β = 0.0)
    d == 1 && return _single_site_qtt([exp(α * t + β) for t in (a, b)])
    out = zeros_tt(2, d, 1)
    h = (b - a) / (2^d - 1)
    t₁ = a
    out.cores[1][1, 1, 1] = exp(α * t₁ + β)
    t₁ = a + h * 2^(d - 1)
    out.cores[1][2, 1, 1] = exp(α * t₁ + β)
    @fastmath for k in 2:(d - 1)
        tₖ = h * 2^(d - k)
        out.cores[k][1, 1, 1] = 1.0
        out.cores[k][2, 1, 1] = exp(α * tₖ)
    end
    out.cores[d][1, 1, 1] = 1.0
    td = h
    out.cores[d][2, 1, 1] = exp(α * td)
    return out
end

"""
Converts a quantics tensor train operator (`TTOperator`) into its full matrix representation.
"""
function qtto_to_matrix(Aqtto::AbstractTTOperator{T, d}) where {T, d}
    A = zeros(T, 2^d, 2^d)
    A_tensor = tto_to_tensor(Aqtto)
    @inbounds for t in CartesianIndices(A_tensor)
        A[tuple_to_index(Tuple(t)[1:d]), tuple_to_index(Tuple(t)[(d + 1):end])] = A_tensor[t]
    end
    return A
end

"""
    qtt_basis_vector(d, pos::Int, val=1.0) -> TTVector

Return the rank-1 QTT with `d` sites that is `val` at the 1-based position `pos`
(site 1 the most significant bit) and zero elsewhere.
"""
function qtt_basis_vector(d, pos::Int, val::Number = 1.0)
    out = zeros_tt(2, d, 1)
    bits = reverse(digits(pos - 1, base = 2, pad = d))
    @inbounds for k in 1:d
        out.cores[k][:, 1, 1] .= 0.0
        out.cores[k][bits[k] + 1, 1, 1] = val
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
    N = 2^d
    x_nodes, _ = gauss_chebyshev_lobatto(N; shifted = true)
    d == 1 && return _single_site_qtt(cos.(n .* acos.(clamp.(2 .* x_nodes .- 1, -1.0, 1.0))))
    out = zeros_tt(2, d, 2)
    θ = acos.(clamp.(2 .* x_nodes .- 1, -1.0, 1.0))
    out.cores[1][1, 1, :] = [cos(n * θ[1]); -sin(n * θ[1])]
    out.cores[1][2, 1, :] = [cos(n * θ[2^(d - 1) + 1]); -sin(n * θ[2^(d - 1) + 1])]
    @fastmath for k in 2:(d - 1)
        out.cores[k][1, :, :] .= [1.0 0.0; 0.0 1.0]
        idx = 2^(d - k) + 1
        out.cores[k][2, :, :] .= [cos(n * θ[idx]) -sin(n * θ[idx]);sin(n * θ[idx])  cos(n * θ[idx])]
    end
    out.cores[d][1, :, 1] .= [1.0, 0.0]
    out.cores[d][2, :, 1] .= [cos(n * θ[2]), sin(n * θ[2])]

    return out
end

"""
    qtt_trapezoidal(d; a=0.0, b=1.0) -> TTVector

Return the weights of the composite trapezoidal rule on the `2^d` uniform grid
points of `[a, b]` (the grid of [`function_to_qtt`](@ref)) as a QTT:
`h = (b − a)/(2^d − 1)` at interior points and `h/2` at both endpoints. Then
`dot(qtt_trapezoidal(d; a, b), u)` approximates `∫ₐᵇ u(x) dx`. The weights have
TT rank at most 3.
"""
function qtt_trapezoidal(d; a = 0.0, b = 1.0)
    h = (b - a) / (2^d - 1)
    endpoints = qtt_basis_vector(d, 1) + qtt_basis_vector(d, 2^d)
    return tt_round!(h * ones_tt(2, d) - (h / 2) * endpoints)
end

"""
    to_qtt(tt, split_dims; threshold=0.0)

Convert a `TTVector` to QTT format by splitting each core's physical dimension via SVD.

`split_dims[i]` is a list of integers whose product equals `tt.dims[i]`, specifying
how to factor that core. The **first** entry is the coarsest (most significant) dimension,
consistent with the rest of the package's QTT convention.

An optional `threshold` (relative to the largest singular value) controls rank truncation.
"""
function to_qtt(
        tt::TTVector{T, N}, split_dims::Vector{Vector{Int}};
        threshold::Float64 = 0.0
    ) where {T <: Number, N}
    @assert length(split_dims) == N "split_dims must have one entry per TT core"
    for i in 1:N
        @assert prod(split_dims[i]) == tt.dims[i] "prod(split_dims[$i]) must equal $(tt.dims[i])"
    end

    qtt_cores = Vector{Array{T, 3}}()
    new_rks = Int[1]
    new_dims = Int[]

    for i in 1:N
        # Work in (r_l, n, r_r) layout for easy reshaping
        core = permutedims(tt.cores[i], (2, 1, 3))
        rank_prev = new_rks[end]
        rank_next = tt.ranks[i + 1]
        remaining = tt.dims[i]

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
    return TTVector{T, N_new}(qtt_cores, Tuple(new_dims), new_rks)
end

"""
    to_ttv(qtt, merge_numbers)

Convert a QTT (or any `TTVector`) back to TT format by contracting consecutive cores.

`merge_numbers[i]` is the number of consecutive QTT cores to merge into TT core `i`.
`sum(merge_numbers)` must equal the total number of QTT cores.

Physical dimensions are merged using the same BIG-ENDIAN convention as `to_qtt`:
the earlier (coarser) core provides the more significant bits.
"""
function to_ttv(qtt::AbstractTTVector{T, M}, merge_numbers::Vector{Int}) where {T <: Number, M}
    @assert sum(merge_numbers) == M "merge_numbers must sum to $(M) (the number of QTT cores)"

    tt_cores = Vector{Array{T, 3}}()
    k = 1

    for count in merge_numbers
        # Work in (r_l, n, r_r) layout
        core = permutedims(qtt.cores[k], (2, 1, 3))   # (r_l, n1, r_mid)

        for j in (k + 1):(k + count - 1)
            G2 = permutedims(qtt.cores[j], (2, 1, 3))  # (r_mid, n2, r_r)
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

        # Convert back to (n, r_l, r_r) TTVector format
        push!(tt_cores, permutedims(core, (2, 1, 3)))
        k += count
    end

    N_new = length(tt_cores)
    new_dims = ntuple(i -> size(tt_cores[i], 1), N_new)
    new_rks = vcat([size(c, 2) for c in tt_cores], size(tt_cores[end], 3))
    return TTVector{T, N_new}(tt_cores, new_dims, new_rks)
end

"""
A Quantized Tensor Train vector with explicit multi-dimensional ordering metadata.

Identical TT fields as `TTVector` plus:
- `n_dims`: number of spatial dimensions
- `bits_per_dim`: bits per dimension (total sites = n_dims × bits_per_dim)
- `ordering`: `:interleaved` or `:serial`
"""
struct QTTVector{T <: Number, M} <: AbstractTTVector{T, M}
    cores::Vector{Array{T, 3}}
    dims::NTuple{M, Int64}
    ranks::Vector{Int64}
    orthogonality::Vector{Int64}
    n_dims::Int
    bits_per_dim::Int
    ordering::Symbol
    function QTTVector{T, M}(cores, dims, ranks, orthogonality, n_dims, bits_per_dim, ordering) where {T <: Number, M}
        return new{T, M}(cores, dims, ranks, _orthogonality_storage(orthogonality, M), n_dims, bits_per_dim, ordering)
    end
end

"""
A Quantized Tensor Train operator with explicit multi-dimensional ordering metadata.
"""
struct QTTOperator{T <: Number, M} <: AbstractTTOperator{T, M}
    cores::Vector{Array{T, 4}}
    row_dims::NTuple{M, Int64}
    col_dims::NTuple{M, Int64}
    ranks::Vector{Int64}
    orthogonality::Vector{Int64}
    n_dims::Int
    bits_per_dim::Int
    ordering::Symbol
    function QTTOperator{T, M}(cores, row_dims, col_dims, ranks, orthogonality, n_dims, bits_per_dim, ordering) where {T <: Number, M}
        return new{T, M}(cores, row_dims, col_dims, ranks, _orthogonality_storage(orthogonality, M), n_dims, bits_per_dim, ordering)
    end
end


function Base.show(io::IO, q::QTTVector{T, M}) where {T, M}
    return print(io, "QTT-MPS{$T}($(nsites(q)) sites, $(q.n_dims)d×$(q.bits_per_dim)bits, $(q.ordering))")
end

function Base.show(io::IO, ::MIME"text/plain", q::QTTVector{T, M}) where {T, M}
    println(io, "QTT-MPS{$T} with $(nsites(q)) sites")
    println(io, "  Dimensions    : $(q.n_dims)d × $(q.bits_per_dim) bits/dim  ($(q.n_dims * 2^q.bits_per_dim) grid points per dim)")
    println(io, "  Ordering      : $(q.ordering)")
    println(io, "  Physical dims : $(q.dims)")
    println(io, "  Bond dims     : $(q.ranks)")
    return print(io, "  Orthogonality : $(_orthogonality_description(q))")
end

function Base.show(io::IO, A::QTTOperator{T, M}) where {T, M}
    return print(io, "QTT-MPO{$T}($(nsites(A)) sites, $(A.n_dims)d×$(A.bits_per_dim)bits, $(A.ordering))")
end

function Base.show(io::IO, ::MIME"text/plain", A::QTTOperator{T, M}) where {T, M}
    println(io, "QTT-MPO{$T} with $(nsites(A)) sites")
    println(io, "  Dimensions    : $(A.n_dims)d × $(A.bits_per_dim) bits/dim  ($(A.n_dims * 2^A.bits_per_dim) grid points per dim)")
    println(io, "  Ordering      : $(A.ordering)")
    println(io, "  Physical dims : $(_dims_description(A))")
    println(io, "  Bond dims     : $(A.ranks)")
    return print(io, "  Orthogonality : $(_orthogonality_description(A))")
end

"""
    QTTVector(ttv::TTVector{T, M}, n_dims::Int, bits_per_dim::Int, ordering::Symbol)

Wrap a `TTVector` as a `QTTVector` by specifying multi-dimensional QTT metadata.

# Arguments
- `ttv::TTVector`: The underlying TT vector to wrap
- `n_dims::Int`: Number of spatial dimensions
- `bits_per_dim::Int`: Number of bits per dimension (N = n_dims × bits_per_dim)
- `ordering::Symbol`: Either `:interleaved` or `:serial`

All physical dimensions in `ttv` must be 2.
"""
function QTTVector(ttv::TTVector{T, M}, n_dims::Int, bits_per_dim::Int, ordering::Symbol) where {T, M}
    @assert n_dims * bits_per_dim == nsites(ttv) "n_dims * bits_per_dim must equal nsites(ttv) (got $(n_dims)*$(bits_per_dim)=$(n_dims * bits_per_dim) ≠ $(nsites(ttv)))"
    @assert all(==(2), ttv.dims) "All physical dimensions must be 2 for QTT (got $(ttv.dims))"
    @assert ordering ∈ (:interleaved, :serial) "ordering must be :interleaved or :serial (got $ordering)"
    return QTTVector{T, M}(ttv.cores, ttv.dims, ttv.ranks, ttv.orthogonality, n_dims, bits_per_dim, ordering)
end

"""
    QTTOperator(tto::TTOperator{T, M}, n_dims::Int, bits_per_dim::Int, ordering::Symbol)

Wrap a `TTOperator` as a `QTTOperator` by specifying multi-dimensional QTT metadata.

# Arguments
- `tto::TTOperator`: The underlying TT operator to wrap
- `n_dims::Int`: Number of spatial dimensions
- `bits_per_dim::Int`: Number of bits per dimension (N = n_dims × bits_per_dim)
- `ordering::Symbol`: Either `:interleaved` or `:serial`

All physical dimensions in `tto` must be 2.
"""
function QTTOperator(tto::TTOperator{T, M}, n_dims::Int, bits_per_dim::Int, ordering::Symbol) where {T, M}
    @assert n_dims * bits_per_dim == nsites(tto) "n_dims * bits_per_dim must equal nsites(tto) (got $(n_dims)*$(bits_per_dim)=$(n_dims * bits_per_dim) ≠ $(nsites(tto)))"
    @assert all(==(2), tto.row_dims) && all(==(2), tto.col_dims) "All physical dimensions must be 2 for QTT (got $(_dims_description(tto)))"
    @assert ordering ∈ (:interleaved, :serial) "ordering must be :interleaved or :serial (got $ordering)"
    return QTTOperator{T, M}(tto.cores, tto.row_dims, tto.col_dims, tto.ranks, tto.orthogonality, n_dims, bits_per_dim, ordering)
end

"""
    TTVector(q::QTTVector{T, M})

Strip QTT metadata to recover the underlying `TTVector`.
"""
TTVector(q::QTTVector{T, M}) where {T, M} =
    TTVector{T, M}(q.cores, q.dims, q.ranks; orthogonality = q.orthogonality)

"""
    TTOperator(q::QTTOperator{T, M})

Strip QTT metadata to recover the underlying `TTOperator`.
"""
TTOperator(q::QTTOperator{T, M}) where {T, M} =
    TTOperator{T, M}(q.cores, q.row_dims, q.col_dims, q.ranks; orthogonality = q.orthogonality)

"""
    check_compat(a::QTTVector, b::QTTVector)

Verify that two `QTTVector`s have compatible QTT metadata (n_dims, bits_per_dim, ordering).

Throws AssertionError if incompatible.
"""
function check_compat(a::QTTVector, b::QTTVector)
    @assert a.n_dims == b.n_dims "QTTVector n_dims mismatch: $(a.n_dims) ≠ $(b.n_dims)"
    @assert a.bits_per_dim == b.bits_per_dim "QTTVector bits_per_dim mismatch: $(a.bits_per_dim) ≠ $(b.bits_per_dim)"
    return @assert a.ordering == b.ordering "QTTVector ordering mismatch: $(a.ordering) ≠ $(b.ordering)"
end

"""
    check_compat(A::QTTOperator, ψ::QTTVector)

Verify that a `QTTOperator` and `QTTVector` have compatible QTT metadata.

Throws AssertionError if incompatible.
"""
function check_compat(A::QTTOperator, ψ::QTTVector)
    @assert A.n_dims == ψ.n_dims "QTTOperator/QTTVector n_dims mismatch: $(A.n_dims) ≠ $(ψ.n_dims)"
    @assert A.bits_per_dim == ψ.bits_per_dim "QTTOperator/QTTVector bits_per_dim mismatch: $(A.bits_per_dim) ≠ $(ψ.bits_per_dim)"
    return @assert A.ordering == ψ.ordering "QTTOperator/QTTVector ordering mismatch: $(A.ordering) ≠ $(ψ.ordering)"
end

"""
    check_compat(::TTVector, ::TTVector)

No-op for plain TTVectors (always compatible).
"""
check_compat(::TTVector, ::TTVector) = nothing

"""
    check_compat(::TTOperator, ::TTVector)

No-op for plain TTOperator/TTVector pairs (always compatible).
"""
check_compat(::TTOperator, ::TTVector) = nothing

function check_compat(A::QTTOperator, B::QTTOperator)
    @assert A.n_dims == B.n_dims "QTTOperator n_dims mismatch: $(A.n_dims) ≠ $(B.n_dims)"
    @assert A.bits_per_dim == B.bits_per_dim "QTTOperator bits_per_dim mismatch: $(A.bits_per_dim) ≠ $(B.bits_per_dim)"
    return @assert A.ordering == B.ordering "QTTOperator ordering mismatch: $(A.ordering) ≠ $(B.ordering)"
end

"""
    _swap_adjacent_sites(A, B; threshold=0.0)

Swap the physical indices of two adjacent TT cores `A` (site k) and `B` (site k+1).

Both cores follow the `(phys_dim, left_rank, right_rank)` layout. The two-site
tensor is contracted, the physical indices are transposed, and the result is
re-factorized via a (possibly truncated) SVD.

Returns `(new_A, new_B)` with the swapped cores.
"""
function _swap_adjacent_sites(
        A::AbstractArray{T, 3}, B::AbstractArray{T, 3};
        threshold::Real = 0.0
    ) where {T}
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
    reorder(q::QTTVector, new_ordering::Symbol; threshold=0.0)

Convert a `QTTVector` between `:interleaved` and `:serial` orderings by performing
a sequence of adjacent site swaps (sorting-network style, via bubble sort).

Each adjacent swap contracts two neighboring cores, permutes their physical indices,
and re-factorizes via SVD. The optional `threshold` (relative to the largest singular
value) controls rank truncation during each SVD step.

Returns a new `QTTVector` with `ordering == new_ordering`.
"""
function reorder(q::QTTVector, new_ordering::Symbol; threshold::Real = 0.0)
    @assert new_ordering ∈ (:interleaved, :serial) "ordering must be :interleaved or :serial"
    q.ordering == new_ordering && return copy(q)

    n_dims = q.n_dims
    bits_per_dim = q.bits_per_dim
    N = nsites(q)

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
    cores = deepcopy(q.cores)
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
    new_ttv = TTVector{eltype(q), N}(cores, dims, rks)
    return QTTVector(new_ttv, n_dims, bits_per_dim, new_ordering)
end

_rewrap(q::QTTVector, x::TTVector) = QTTVector(x, q.n_dims, q.bits_per_dim, q.ordering)
_rewrap(q::QTTVector, A::TTOperator) = QTTOperator(A, q.n_dims, q.bits_per_dim, q.ordering)
_rewrap(Q::QTTOperator, A::TTOperator) = QTTOperator(A, Q.n_dims, Q.bits_per_dim, Q.ordering)

function _rewrap(a::QTTVector, b::QTTVector, x::TTVector)
    check_compat(a, b)
    return _rewrap(a, x)
end

function _rewrap(A::QTTOperator, b::QTTVector, x::TTVector)
    check_compat(A, b)
    return _rewrap(b, x)
end

function _rewrap(A::QTTOperator, B::QTTOperator, X::TTOperator)
    check_compat(A, B)
    return _rewrap(A, X)
end

_check_qtt(a::Union{QTTOperator, QTTVector}, b::QTTVector) = check_compat(a, b)
_check_qtt(A::QTTOperator, B::QTTOperator) = check_compat(A, B)

"""
    function_to_qttv(f, n_dims, bits_per_dim; ordering=:interleaved, a=0.0, b=1.0)

Evaluate an `n_dims`-dimensional function `f` on a uniform grid with `2^bits_per_dim` points
per dimension and return a `QTTVector`. `f` receives an `n_dims`-length coordinate vector.

Grid points: `a + i * (b - a) / (2^bits_per_dim - 1)` for `i = 0, ..., 2^bits_per_dim - 1`.
"""
function function_to_qttv(
        f, n_dims::Int, bits_per_dim::Int;
        ordering::Symbol = :interleaved, a::Real = 0.0, b::Real = 1.0
    )
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

    ttv = tt_decomp(tensor)
    return QTTVector(ttv, n_dims, bits_per_dim, ordering)
end

"""
    _swap_adjacent_sites_op(A, B; threshold=0.0)

Swap the physical indices of two adjacent TTOperator cores `A` (site k) and `B` (site k+1).

Both cores follow the `(phys_dim, phys_dim, left_rank, right_rank)` layout. The two-site
tensor is contracted, the physical indices (both row and column) are transposed, and the
result is re-factorized via a (possibly truncated) SVD.

Returns `(new_A, new_B)` with the swapped cores.
"""
function _swap_adjacent_sites_op(
        A::AbstractArray{T, 4}, B::AbstractArray{T, 4};
        threshold::Real = 0.0
    ) where {T}
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
    reorder(A::QTTOperator, new_ordering::Symbol; threshold=0.0)

Convert a `QTTOperator` between `:interleaved` and `:serial` orderings by performing
a sequence of adjacent site swaps (sorting-network style, via bubble sort).

Returns a new `QTTOperator` with `ordering == new_ordering`.
"""
function reorder(A::QTTOperator, new_ordering::Symbol; threshold::Real = 0.0)
    @assert new_ordering ∈ (:interleaved, :serial) "ordering must be :interleaved or :serial"
    A.ordering == new_ordering && return copy(A)

    n_dims = A.n_dims
    bits_per_dim = A.bits_per_dim
    N = nsites(A)

    # Build the same permutation as for QTTVector reorder
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

    cores = deepcopy(A.cores)
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
    new_tto = TTOperator{eltype(A), N}(cores, dims, rks)
    return QTTOperator(new_tto, n_dims, bits_per_dim, new_ordering)
end

"""
    qttv_to_array(q::QTTVector)

Contract the TT chain and return an `n_dims`-dimensional array of size `2^bits_per_dim`
per dimension, with values ordered on the uniform grid (index 1 = leftmost grid point).
"""
function qttv_to_array(q::QTTVector)
    N = nsites(q)
    n_dims = q.n_dims
    bits_per_dim = q.bits_per_dim
    ordering = q.ordering
    n_pts = 2^bits_per_dim

    full_tensor = tt_to_tensor(TTVector(q))
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
