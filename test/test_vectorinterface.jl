using Test
using LinearAlgebra
using KrylovKit
using TensorTrainNumerics
using VectorInterface

@testset "Complex TT dot and norm" begin
    d = 4
    u = complex(qtt_sin(d, λ = π))
    w = (1im) * complex(id_tto(d)) * u
    w_dense = qtt_to_vector(w)

    @test isapprox(TensorTrainNumerics.dot(w, w), LinearAlgebra.dot(w_dense, w_dense); atol = 1.0e-12)
    @test isapprox(norm(w), norm(w_dense); atol = 1.0e-12)
end

@testset "Complex Krylov exponential action" begin
    d = 4
    u₀ = complex(qtt_sin(d, λ = π))
    A = (1im) * complex(id_tto(d))

    y, info = expintegrator(A, 0.7, u₀)

    expected = exp(0.7im) .* qtt_to_vector(u₀)
    actual = qtt_to_vector(y)

    @test info.converged == 1
    @test norm(actual - expected) / norm(expected) < 1.0e-12
end

@testset "VectorInterface add!! promotes when destination eltype is too narrow" begin
    d = 3
    y = qtt_sin(d)
    x = complex(qtt_cos(d))
    α = 0.25 + 0.5im
    β = 1.5

    z1 = VectorInterface.add!!(copy(y), x, α)
    expected1 = qtt_to_vector(y) + α * qtt_to_vector(x)
    @test z1 isa TTvector{ComplexF64}
    @test norm(qtt_to_vector(z1) - expected1) / norm(expected1) < 1.0e-12

    z2 = VectorInterface.add!!(copy(y), x, α, β)
    expected2 = β * qtt_to_vector(y) + α * qtt_to_vector(x)
    @test z2 isa TTvector{ComplexF64}
    @test norm(qtt_to_vector(z2) - expected2) / norm(expected2) < 1.0e-12
end
