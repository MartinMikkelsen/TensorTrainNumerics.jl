using Test
using LinearAlgebra
using KrylovKit
using TensorTrainNumerics

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
