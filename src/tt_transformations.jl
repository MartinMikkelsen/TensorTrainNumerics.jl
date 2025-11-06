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

@inline function laplace_core_entry(
        P::LagrangePolynomials{Float64},
        α::Int, β::Int, σ::Int, τ::Int;
        scale::Float64 = 1.0
    )
    cβ = P.grid[β + 1]
    x = 0.5 * (σ + cβ)
    return lagrange_eval(P, α, x) * exp(-scale * (σ + cβ) * τ)
end

function laplace_qtto(d::Int,K::Int,scale::Float64)

    @assert d ≥ 1
    P = cheb_lobatto_grid(K)
    r = K + 1

    A = Array{Float64}(undef, 2, 2, r, r)
    @inbounds for α in 0:K, β in 0:K, σ in 0:1, τ in 0:1
        A[σ + 1, τ + 1, α + 1, β + 1] = laplace_core_entry(P, α, β, σ, τ; scale)
    end

    AL = Array{Float64}(undef, 2, 2, 1, r)
    @inbounds for β in 1:r, σ in 1:2, τ in 1:2
        s = zero(ComplexF64)
        for α in 1:r
            s += A[σ, τ, α, β]
        end
        AL[σ, τ, 1, β] = s
    end

    AR = Array{Float64}(undef, 2, 2, r, 1)
    @inbounds for α in 1:r, σ in 1:2, τ in 1:2
        AR[σ, τ, α, 1] = A[σ, τ, α, 1]
    end

    cores = Vector{Array{Float64, 4}}(undef, d)
    cores[1] = AL
    for k in 2:(d - 1)
        cores[k] = A
    end
    cores[d] = AR

    dims = ntuple(_ -> 2, d)
    rks = vcat(1, fill(r, d - 1), 1)
    ot = zeros(Int, d)
    return TToperator{Float64, d}(d, cores, dims, rks, ot)
end


"""
Reference: https://arxiv.org/pdf/2404.03182
"""
function fourier_qtto(d::Int; sign::Float64 = -1.0, K::Int = 25, normalize::Bool = true)
    @assert d ≥ 1
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


function reverse_qtt_bits(x::TTvector{T, d}) where {T, d}
    new_vecs = reverse(copy.(x.ttv_vec))
    new_vecs = map(c -> permutedims(c, (1, 3, 2)), new_vecs)
    new_dims = reverse(x.ttv_dims)
    new_ot = reverse(x.ttv_ot)
    new_rks = [1; reverse(x.ttv_rks[2:(end - 1)]); 1]
    return TTvector{T, d}(x.N, new_vecs, new_dims, new_rks, new_ot)
end
