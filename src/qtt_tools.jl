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
Converts a quantum tensor train operator (`TToperator`) into its full matrix representation.
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

function qtt_simpson(d; a = 0.0, b = 1.0)
    N = 2^d
    h = (b - a) / (N - 1)
    tensors = [zeros_tt(2, d, 1) for _ in 1:N]
    @inbounds for i in 0:(N - 1)
        weight = (i == 0 || i == N - 1) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        bits = reverse(digits(i, base = 2, pad = d)) .+ 1

        for k in 1:d
            fill!(tensors[i + 1].ttv_vec[k], 0.0)
            tensors[i + 1].ttv_vec[k][bits[k], 1, 1] = 1.0
        end

        tensors[i + 1] = weight * tensors[i + 1]
    end

    simpson_tt = tensors[1]
    for j in 2:N
        simpson_tt += tensors[j]
    end

    return (h / 3) * simpson_tt
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
