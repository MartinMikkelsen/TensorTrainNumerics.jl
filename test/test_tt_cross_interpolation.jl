using Test

@testset "tt_cross_interpolation" begin

    @testset "_evaluate_on_domain" begin
        domain = [collect(1.0:3.0), collect(2.0:4.0)]
        indices = [1 1; 2 2; 3 3]
        f(x) = sum(x, dims = 2)
        result = TensorTrainNumerics._evaluate_on_domain(f, domain, indices)
        @test length(result) == 3
        @test result[1] ≈ 3.0
    end

    @testset "_cap_ranks!" begin
        Rs = [1, 4, 4, 1]
        Is = [3, 3, 3]
        TensorTrainNumerics._cap_ranks!(Rs, Is, 5)
        @test maximum(Rs) <= 5
        @test Rs[1] == 1
        @test Rs[end] == 1
    end

    @testset "_evaluate_tt" begin
        cores = [randn(2, 1, 2), randn(2, 2, 1)]
        indices = [1 1; 2 2]
        result = TensorTrainNumerics._evaluate_tt(cores, indices, 2)
        @test length(result) == 2
        @test eltype(result) <: Real
    end

    @testset "tt_cross with 1D function" begin
        f(x) = sin.(x[:, 1])
        domain = [collect(0.0:0.1:1.0)]
        tt = tt_cross(f, domain, ranks_tt = 2, max_iter = 5, verbose = false)
        @test tt.N == 1
        @test length(tt.N) == 1
    end

    @testset "tt_cross with 2D function" begin
        f(x) = x[:, 1] .* x[:, 2]
        domain = [collect(0.0:0.5:2.0), collect(0.0:0.5:2.0)]
        tt = tt_cross(f, domain, ranks_tt = 2, max_iter = 3, verbose = false)
        @test tt.N == 2
        @test tt.ttv_dims == (5, 5)
    end

    @testset "tt_cross with tuple dims" begin
        f(x) = sin.(x[:, 1])
        tt = tt_cross(f, (5,), ranks_tt = 2, max_iter = 2, verbose = false)
        @test tt.N == 1
    end

    @testset "tt_cross with vector dims" begin
        f(x) = sum(x, dims = 2)
        tt = tt_cross(f, [3, 3], ranks_tt = 2, max_iter = 2, verbose = false)
        @test tt.N == 2
    end

    @testset "tt_cross convergence" begin
        f(x) = x[:, 1]
        domain = [collect(1.0:10.0)]
        tt = tt_cross(f, domain, ranks_tt = 1, eps = 1.0e-3, max_iter = 10, verbose = false)
        @test tt.N == 1
    end

    @testset "tt_cross with no kickrank" begin
        f(x) = sum(x, dims = 2)
        tt = tt_cross(f, [4, 4], ranks_tt = 2, kickrank = nothing, max_iter = 2, verbose = false)
        @test tt.N == 2
    end
end

@testset "tt_integrate" begin
    @testset "tt_integrate 1D function" begin
        f(x) = x[:, 1] .^ 2
        bounds = [(0.0, 1.0)]
        result = tt_integrate(f, bounds, nquad = 10, max_iter = 3, verbose = false)
        @test isa(result, Real)
        @test 0.2 < result < 0.4
    end

    @testset "tt_integrate 2D function" begin
        f(x) = x[:, 1] .* x[:, 2]
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        result = tt_integrate(f, bounds, nquad = 10, max_iter = 3, verbose = false)
        @test isa(result, Real)
        @test 0.2 < result < 0.3
    end

    @testset "tt_integrate with custom nquad" begin
        f(x) = sum(x, dims = 2)
        bounds = [(1.0, 2.0), (1.0, 2.0)]
        result = tt_integrate(f, bounds, nquad = 15, max_iter = 2, verbose = false)
        @test isa(result, Real)
    end

    @testset "tt_integrate constant function" begin
        f(x) = ones(size(x, 1))
        bounds = [(0.0, 1.0)]
        result = tt_integrate(f, bounds, nquad = 10, max_iter = 3, verbose = false)
        @test 0.9 < result < 1.1
    end

    @testset "tt_integrate with different bounds" begin
        f(x) = x[:, 1]
        bounds = [(2.0, 4.0)]
        result = tt_integrate(f, bounds, nquad = 10, max_iter = 3, verbose = false)
        @test isa(result, Real)
        @test 5.0 < result < 7.0
    end
end
