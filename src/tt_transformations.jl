struct LagrangePolynomials{T}
    grid::Vector{T}  # nodes c_j ∈ [0,1]
    w::Vector{T}     # weights
end

function cheb_lobatto_grid(K::Int)::LagrangePolynomials{Float64}
    c = 0.5 .* (1 .- cospi.((0:K) ./ K))
    w = [j == 0 || j == K ? 0.5 : 1.0 for j in 0:K]
    w .= w .* ((-1.0) .^ (0:K))
    return LagrangePolynomials{Float64}(c, w)
end

@inline function lagrange_eval(P::LagrangePolynomials{T}, α::Int, x::T)::T where {T}
    xα = P.grid[α + 1]
    if isapprox(x, xα; atol = 1.0e-14, rtol = 0)
        return one(T)
    end
    num = P.w[α + 1] / (x - xα)
    denom = zero(T)
    @inbounds for j in 0:(length(P.grid) - 1)
        denom += P.w[j + 1] / (x - P.grid[j + 1])
    end
    return num / denom
end

@inline function qft_core_entry(
        P::LagrangePolynomials{Float64},
        α::Int, β::Int, σ::Int, τ::Int; sign::Float64 = -1.0
    )
    cβ = P.grid[β + 1]
    x = 0.5 * (σ + cβ)
    return lagrange_eval(P, α, x) * cispi(sign * (σ + cβ) * τ)
end

"""
    fourier_qtto(d; sign=-1.0, K=25, normalize=true) -> TToperator{ComplexF64}

Discrete Fourier transform on `2^d` points as a QTT operator, built with the
interpolative construction of Chen and Lindsey (arXiv:2404.03182):

    y_k = s · Σₙ x_n exp(sign · 2πi · k n / 2^d),   s = 2^(−d/2) if `normalize`, else 1.

The operator reverses the bit order: the input must have its *least*
significant bit of `n` on site 1 (as produced by
[`function_to_qtt_uniform`](@ref)), and the output has the *most* significant
bit of `k` on site 1 (as read by [`qtt_to_vector`](@ref) and
[`matricize`](@ref)). Every interior bond has rank `K + 1`; larger `K` gives a
more accurate transform.
"""
function fourier_qtto(d::Int; sign::Float64 = -1.0, K::Int = 25, normalize::Bool = true)
    @assert d ≥ 1
    # On one bit the transform is exactly [1 1; 1 −1] for either sign.
    d == 1 && return _single_site_qtto(ComplexF64[1 1; 1 -1] .* (normalize ? inv(sqrt(2.0)) : 1.0))
    P = cheb_lobatto_grid(K)
    r = K + 1

    A = Array{ComplexF64}(undef, 2, 2, r, r)
    @inbounds for α in 0:K, β in 0:K, σ in 0:1, τ in 0:1
        A[σ + 1, τ + 1, α + 1, β + 1] = qft_core_entry(P, α, β, σ, τ; sign = sign)
    end

    AL = Array{ComplexF64}(undef, 2, 2, 1, r)
    @inbounds for β in 1:r, σ in 1:2, τ in 1:2
        s = zero(ComplexF64)
        for α in 1:r
            s += A[σ, τ, α, β]
        end
        AL[σ, τ, 1, β] = s
    end

    AR = Array{ComplexF64}(undef, 2, 2, r, 1)
    @inbounds for α in 1:r, σ in 1:2, τ in 1:2
        AR[σ, τ, α, 1] = A[σ, τ, α, 1]
    end

    cores = Vector{Array{ComplexF64, 4}}(undef, d)
    cores[1] = AL
    for k in 2:(d - 1)
        cores[k] = A
    end
    cores[d] = AR

    if normalize
        cores[1] .*= inv(sqrt(ComplexF64(2.0^d)))
    end

    dims = ntuple(_ -> 2, d)
    rks = vcat(1, fill(r, d - 1), 1)
    ot = zeros(Int, d)
    return TToperator{ComplexF64, d}(d, cores, dims, rks, ot)
end

"""
    reverse_qtt_bits(x::TTvector) -> TTvector

Reverse the order of the sites of `x`. For a binary QTT this reverses the bit
order of the position index, converting between most-significant-bit-first and
least-significant-bit-first layouts.
"""
function reverse_qtt_bits(x::TTvector{T, d}) where {T, d}
    new_vecs = reverse(copy.(x.ttv_vec))
    new_vecs = map(c -> permutedims(c, (1, 3, 2)), new_vecs)
    new_dims = reverse(x.ttv_dims)
    new_ot = reverse(x.ttv_ot)
    new_rks = [1; reverse(x.ttv_rks[2:(end - 1)]); 1]
    return TTvector{T, d}(x.N, new_vecs, new_dims, new_rks, new_ot)
end
