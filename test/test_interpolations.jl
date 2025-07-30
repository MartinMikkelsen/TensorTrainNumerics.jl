using Test
import TensorTrainNumerics: A_L, A_C, A_R, gauss_chebyshev_lobatto

@testset "Node Generation Tests" begin
    @testset "Chebyshev Lobatto Nodes" begin
        nodes = chebyshev_lobatto_nodes(4)
        @test length(nodes) == 5
        @test nodes[1] == 1.0
        @test nodes[end] == 0.0
    end

    @testset "Equally Spaced Nodes" begin
        nodes = equally_spaced_nodes(4)
        @test length(nodes) == 5
        @test nodes[1] == 0.0
        @test nodes[end] == 1.0
    end


end

@testset "Lagrange Basis Tests" begin
    nodes = chebyshev_lobatto_nodes(4)
    
    @testset "Lagrange Basis for Scalar x" begin
        x = 0.5
        for j in 0:4
            basis = lagrange_basis(nodes, x, j)
            @test isfinite(basis)
        end
    end
end
@testset "Node Generation Functions" begin
    @test chebyshev_lobatto_nodes(4) ≈ [1.0, 0.8535533905932737, 0.5, 0.14644660940672627, 0.0]
    @test equally_spaced_nodes(4) == [0.0, 0.25, 0.5, 0.75, 1.0]
end

@testset "Lagrange Basis Functions" begin
    nodes = chebyshev_lobatto_nodes(4)
    x = 0.5
    @test lagrange_basis(nodes, x, 2) ≈ 1.0
    x_vec = [0.0, 0.25, 0.5, 0.75, 1.0]
    @test lagrange_basis(nodes, x_vec, 2) ≈ [-0.0, 0.37499999999999994, 1.0, 0.37499999999999994, 0.0]
end


# Import the function to test

@testset "A_L function" begin
    # Test with a simple function and known nodes
    f(x) = x^2
    nodes = [0.0, 0.5, 1.0]
    A = A_L(f, nodes)
    @test size(A) == (1, 2, 1, 3)
    # Check values for σ = 0
    expected0 = f.(0.5 .* (0 .+ nodes))
    @test A[1, 1, 1, :] ≈ expected0
    # Check values for σ = 1
    expected1 = f.(0.5 .* (1 .+ nodes))
    @test A[1, 2, 1, :] ≈ expected1

    # Test with custom interval
    A2 = A_L(f, nodes, 2.0, 4.0)
    expected0b = f.(0.5 .* (0 .+ nodes) .* (4.0 - 2.0) .+ 2.0)
    expected1b = f.(0.5 .* (1 .+ nodes) .* (4.0 - 2.0) .+ 2.0)
    @test A2[1, 1, 1, :] ≈ expected0b
    @test A2[1, 2, 1, :] ≈ expected1b

    # Test with a different function
    g(x) = sin(x)
    A3 = A_L(g, nodes)
    @test A3[1, 1, 1, :] ≈ g.(0.5 .* (0 .+ nodes))
    @test A3[1, 2, 1, :] ≈ g.(0.5 .* (1 .+ nodes))
end
@testset "A_C function" begin
    # Test with 3 nodes (simple case)
    nodes = [0.0, 0.5, 1.0]
    A = A_C(nodes)
    @test size(A) == (3, 2, 1, 3)

    # For σ = 0, x = 0.5 * (0 .+ nodes) = nodes * 0.5
    x0 = 0.5 .* nodes
    for α in 1:3
        expected = lagrange_basis(nodes, x0, α - 1)
        @test A[α, 1, 1, :] ≈ expected
    end

    # For σ = 1, x = 0.5 * (1 .+ nodes)
    x1 = 0.5 .* (1 .+ nodes)
    for α in 1:3
        expected = lagrange_basis(nodes, x1, α - 1)
        @test A[α, 2, 1, :] ≈ expected
    end

    # Test with 2 nodes (edge case)
    nodes2 = [0.0, 1.0]
    A2 = A_C(nodes2)
    @test size(A2) == (2, 2, 1, 2)
    x0_2 = 0.5 .* nodes2
    x1_2 = 0.5 .* (1 .+ nodes2)
    for α in 1:2
        @test A2[α, 1, 1, :] ≈ lagrange_basis(nodes2, x0_2, α - 1)
        @test A2[α, 2, 1, :] ≈ lagrange_basis(nodes2, x1_2, α - 1)
    end

    # Test with non-uniform nodes
    nodes3 = [0.0, 0.3, 0.9]
    A3 = A_C(nodes3)
    @test size(A3) == (3, 2, 1, 3)
    x0_3 = 0.5 .* nodes3
    x1_3 = 0.5 .* (1 .+ nodes3)
    for α in 1:3
        @test A3[α, 1, 1, :] ≈ lagrange_basis(nodes3, x0_3, α - 1)
        @test A3[α, 2, 1, :] ≈ lagrange_basis(nodes3, x1_3, α - 1)
    end
end

@testset "A_R function" begin
    # Test with 3 nodes (simple case)
    nodes = [0.0, 0.5, 1.0]
    A = A_R(nodes)
    @test size(A) == (3, 2, 1, 1)

    # For σ = 0, x = 0.0
    x0 = 0.0
    for α in 1:3
        expected = lagrange_basis(nodes, x0, α - 1)
        @test A[α, 1, 1, 1] ≈ expected
    end

    # For σ = 1, x = 0.5 * 1 = 0.5
    x1 = 0.5
    for α in 1:3
        expected = lagrange_basis(nodes, x1, α - 1)
        @test A[α, 2, 1, 1] ≈ expected
    end

    # Test with 2 nodes (edge case)
    nodes2 = [0.0, 1.0]
    A2 = A_R(nodes2)
    @test size(A2) == (2, 2, 1, 1)
    for α in 1:2
        @test A2[α, 1, 1, 1] ≈ lagrange_basis(nodes2, 0.0, α - 1)
        @test A2[α, 2, 1, 1] ≈ lagrange_basis(nodes2, 0.5, α - 1)
    end

    # Test with non-uniform nodes
    nodes3 = [0.0, 0.3, 0.9]
    A3 = A_R(nodes3)
    @test size(A3) == (3, 2, 1, 1)
    for α in 1:3
        @test A3[α, 1, 1, 1] ≈ lagrange_basis(nodes3, 0.0, α - 1)
        @test A3[α, 2, 1, 1] ≈ lagrange_basis(nodes3, 0.5, α - 1)
    end
end

@testset "integrating_qtt" begin
    f1(x) = 1.0
    d = 15
    N = 100
    approx1 = integrating_qtt(f1, d, N,method="Interpolating")
    @test isapprox(approx1, 1.0; atol=1e-6)

    f2(x) = x
    approx2 = integrating_qtt(f2, d, N,method="Interpolating")
    @test isapprox(approx2, 0.5; atol=1e-3)

    f3(x) = x^2
    approx3 = integrating_qtt(f3, d, N,method="Interpolating")
    @test isapprox(approx3, 1/3; atol=1e-3)

    f4(x) = sin(π * x)
    approx4 = integrating_qtt(f4, d, N,method="Interpolating")
    @test isapprox(approx4, 2/π; atol=1e-6)

    f5(x) = cos(x)
    approx5 = integrating_qtt(f5, d, N)
    @test isapprox(approx5, sin(1); atol=1e-4)

    f6(x) = sin(x)
    approx6 = integrating_qtt(f6, d, N)
    @test isapprox(approx6, 1 - cos(1); atol=1e-4)

    f7(x) = exp(x)
    approx7 = integrating_qtt(f7, d, N, method="Interpolating")
    @test isapprox(approx7, exp(1) - 1; atol=1e-4)
end