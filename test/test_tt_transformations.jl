using Test
using TensorTrainNumerics

import TensorTrainNumerics: cheb_lobatto_grid, lagrange_eval, qft_core_entry

@testset "Spikes" begin

    function function_to_qtt_uniform_msb(f, d::Int)
        N = 2^d
        y = [f(n / N) for n in 0:(N - 1)]
        A = zeros(eltype(y), ntuple(_ -> 2, d)...)
        @inbounds for n in 0:(N - 1)
            bits = (digits(n, base = 2, pad = d)) .+ 1
            A[CartesianIndex(Tuple(bits))] = y[n + 1]
        end
        return ttv_decomp(A)
    end

    d = 10
    N = 2^d
    K = 50
    sign = -1.0
    normalize = true

    Random.seed!(1234)
    r = 12
    coeffs = randn(r) .+ 1im * randn(r)

    f(x) = sum(coeffs .* cispi.(2 .* (0:(r - 1)) .* x))

    F = fourier_qtto(d; K = K, sign = -1.0, normalize = true)
    x_qtt = function_to_qtt_uniform_msb(f, d)
    y_qtt = F * x_qtt

    spec = matricize((y_qtt), d)
    scale = sqrt(N)

    @test norm(spec[1:r] .- scale .* coeffs) / (scale * norm(coeffs)) < 1.0e-8
    @test norm(spec[(r + 1):end]) / norm(spec) < 1.0e-10

end

@testset "cheb_lobatto_grid" begin
	K = 4
	P = cheb_lobatto_grid(K)
	# Check grid length and type
	@test length(P.grid) == K + 1
	@test eltype(P.grid) == Float64
	# Check weights length and type
	@test length(P.w) == K + 1
	@test eltype(P.w) == Float64
	# Check grid endpoints
	@test isapprox(P.grid[1], 0.0; atol=1e-14)
	@test isapprox(P.grid[end], 1.0; atol=1e-14)
	# Check weights at endpoints
	@test isapprox(abs(P.w[1]), 0.5; atol=1e-14)
	@test isapprox(abs(P.w[end]), 0.5; atol=1e-14)
	# Check alternating sign of weights
	for j in 1:(K+1)
		@test P.w[j] ≈ ((j == 1 || j == K+1) ? 0.5 : 1.0) * (-1.0)^(j-1)
	end
end

@testset "lagrange_eval" begin
	# Use a small K for simplicity
	K = 4
	P = cheb_lobatto_grid(K)
	xs = P.grid

	# Test that lagrange_eval(P, α, xα) ≈ 1 and lagrange_eval(P, β, xα) ≈ 0 for β ≠ α
	for α in 0:K
		xα = xs[α + 1]
		# At node, should be 1
		@test isapprox(lagrange_eval(P, α, xα), 1.0; atol=1e-12)
		# At other nodes, should be 0
		for β in 0:K
			if β != α
				xβ = xs[β + 1]
				@test isapprox(lagrange_eval(P, α, xβ), 0.0; atol=1e-12)
			end
		end
	end

	# Test partition of unity: sum_{α} lagrange_eval(P, α, x) ≈ 1 for arbitrary x in [0,1]
	for x in range(0, stop=1, length=10)
		s = sum(lagrange_eval(P, α, x) for α in 0:K)
		@test isapprox(s, 1.0; atol=1e-12)
	end

	# Test that lagrange_eval returns a Float64
	@test eltype([lagrange_eval(P, α, 0.3) for α in 0:K]) == Float64
end

@testset "qft_core_entry" begin
	K = 4
	P = cheb_lobatto_grid(K)
	sign = -1.0

	# Test that output is ComplexF64 and has expected magnitude for simple cases
	for α in 0:K, β in 0:K, σ in 0:1, τ in 0:1
		val = qft_core_entry(P, α, β, σ, τ; sign=sign)
		@test typeof(val) == ComplexF64
		# For τ == 0, cispi(0) == 1, so value should be real and equal to lagrange_eval
		if τ == 0
			expected = lagrange_eval(P, α, 0.5 * (σ + P.grid[β + 1]))
			@test isapprox(real(val), expected; atol=1e-12)
			@test isapprox(imag(val), 0.0; atol=1e-12)
		end
	end

	# Test that for α == β, σ == 0, τ == 0, the value is lagrange_eval at cβ/2
	for α in 0:K
		β = α
		σ = 0
		τ = 0
		cβ = P.grid[β + 1]
		expected = lagrange_eval(P, α, 0.5 * (σ + cβ))
		val = qft_core_entry(P, α, β, σ, τ; sign=sign)
		@test isapprox(val, expected; atol=1e-12)
	end

	# Test that magnitude is ≤ 1 for all valid indices (since lagrange_eval is a Lagrange basis and cispi is unit modulus)
	for α in 0:K, β in 0:K, σ in 0:1, τ in 0:1
		val = qft_core_entry(P, α, β, σ, τ; sign=sign)
		@test abs(val) ≤ 1.0 + 1e-12
	end
end


