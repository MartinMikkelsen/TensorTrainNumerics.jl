using Test
using TensorTrainNumerics

include("test_interpolations.jl")
include("test_tt_tools.jl")
include("test_als.jl")
include("test_mals.jl")
include("test_dmrg.jl")

@testset "TensorTrainNumerics.jl" begin

    @testset "Test Interpolations" begin
        try
            include("test_interpolations.jl")
        catch e
            @test false
            println("Error in test_interpolations.jl: $e")
        end
    end

    @testset "Test TT Tools" begin
        try
            include("test_tt_tools.jl")
        catch e
            @test false
            println("Error in test_tt_tools.jl: $e")
        end
    end

    @testset "Test ALS" begin
        try
            include("test_als.jl")
        catch e
            @test false
            println("Error in test_als.jl: $e")
        end
    end

    @testset "Test MALS" begin
        try
            include("test_mals.jl")
        catch e
            @test false
            println("Error in test_mals.jl: $e")
        end
    end

    @testset "Test DMRG" begin
        try
            include("test_dmrg.jl")
        catch e
            @test false
            println("Error in test_dmrg.jl: $e")
        end
    end

end