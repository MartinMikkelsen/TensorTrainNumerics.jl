using Test

include("../src/tt_interpolations.jl")

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

    @testset "Legendre Nodes" begin
        nodes = legendre_nodes(4)
        @test length(nodes) == 4
        @test all(nodes .>= 0.0) && all(nodes .<= 1.0)
    end

    @testset "Get Nodes" begin
        @test length(get_nodes(4, "chebyshev")) == 5
        @test length(get_nodes(4, "equally_spaced")) == 5
        @test length(get_nodes(4, "legendre")) == 4
        @test_throws ErrorException get_nodes(4, "unknown")
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
    @test legendre_nodes(4) ≈ [0.06943184420297371, 0.33000947820757187, 0.6699905217924281, 0.9305681557970262]
end

@testset "Lagrange Basis Functions" begin
    nodes = chebyshev_lobatto_nodes(4)
    x = 0.5
    @test lagrange_basis(nodes, x, 2) ≈ 1.0
    x_vec = [0.0, 0.25, 0.5, 0.75, 1.0]
    @test lagrange_basis(nodes, x_vec, 2) ≈ [-0.0, 0.37499999999999994, 1.0, 0.37499999999999994, 0.0]
end
