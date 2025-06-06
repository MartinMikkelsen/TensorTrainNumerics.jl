using Test
using LinearAlgebra

@testset "tdvp_step" begin
    H = [0.0 1.0; -1.0 0.0]
    A = tto_decomp(H)
    x0 = ttv_decomp([1.0, 0.0])
    x1 = tdvp_step(A, x0, 1.0)
    result = vec(ttv_to_tensor(x1))
    expected = exp(H) * [1.0, 0.0]
    @test isapprox(result, expected; atol=1e-8)
end

