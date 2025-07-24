using Test
using LinearAlgebra
import TensorTrainNumerics


@testset "toeplitz_to_qtto" begin
    # Minimal mock for zeros_tto to allow testing
    struct MockTTO
        tto_vec::Vector
    end
    function zeros_tto(args...)
        # For this test, we only care about the shapes
        d = args[2]
        tto_vec = []
        # First core: (2,2,1,3)
        push!(tto_vec, zeros(2,2,1,3))
        # Middle cores: (2,2,3,3)
        for _ in 2:d-1
            push!(tto_vec, zeros(2,2,3,3))
        end
        # Last core: (2,2,3,1)
        push!(tto_vec, zeros(2,2,3,1))
        return MockTTO(tto_vec)
    end

    # Patch the function in the module namespace for testing
    @eval TensorTrainNumerics begin
        import ..Main.zeros_tto
    end

    # Test for d = 3
    d = 3
    α, β, γ = 2.0, -1.0, 0.5
    tto = TensorTrainNumerics.toeplitz_to_qtto(α, β, γ, d)
    @test length(tto.tto_vec) == d
    @test size(tto.tto_vec[1]) == (2,2,1,3)
    @test size(tto.tto_vec[2]) == (2,2,3,3)
    @test size(tto.tto_vec[3]) == (2,2,3,1)

    # Check that the first core is filled as expected for a simple case
    # For α=1, β=0, γ=0, the first core should be identity in the first slice
    tto_id = TensorTrainNumerics.toeplitz_to_qtto(1, 0, 0, d)
    @test tto_id.tto_vec[1][1,1,1,1] ≈ 1.0
    @test tto_id.tto_vec[1][2,2,1,1] ≈ 1.0
    @test tto_id.tto_vec[1][1,2,1,1] ≈ 0.0
    @test tto_id.tto_vec[1][2,1,1,1] ≈ 0.0

    # Check that the last core is filled as expected for a simple case
    # For α=1, β=0, γ=0, the last core should be α*id in the first slice
    @test tto_id.tto_vec[d][1,1,1,1] ≈ 1.0
    @test tto_id.tto_vec[d][2,2,1,1] ≈ 1.0
    @test tto_id.tto_vec[d][1,2,1,1] ≈ 0.0
    @test tto_id.tto_vec[d][2,1,1,1] ≈ 0.0
end

@testset "shift, ∇, Δ" begin
    # Test for d = 3
    d = 3

    # shift should call toeplitz_to_qtto(0,1,0,d)
    tto_shift = TensorTrainNumerics.shift(d)
    tto_ref = TensorTrainNumerics.toeplitz_to_qtto(0,1,0,d)
    @test all(size.(tto_shift.tto_vec) .== size.(tto_ref.tto_vec))
    @test all(tto_shift.tto_vec[1] .== tto_ref.tto_vec[1])
    @test all(tto_shift.tto_vec[2] .== tto_ref.tto_vec[2])
    @test all(tto_shift.tto_vec[3] .== tto_ref.tto_vec[3])

    # ∇ should call toeplitz_to_qtto(1,-1,0,d)
    tto_grad = TensorTrainNumerics.∇(d)
    tto_ref = TensorTrainNumerics.toeplitz_to_qtto(1,-1,0,d)
    @test all(size.(tto_grad.tto_vec) .== size.(tto_ref.tto_vec))
    @test all(tto_grad.tto_vec[1] .== tto_ref.tto_vec[1])
    @test all(tto_grad.tto_vec[2] .== tto_ref.tto_vec[2])
    @test all(tto_grad.tto_vec[3] .== tto_ref.tto_vec[3])

    # Δ should call toeplitz_to_qtto(2,-1,-1,d)
    tto_lap = TensorTrainNumerics.Δ(d)
    tto_ref = TensorTrainNumerics.toeplitz_to_qtto(2,-1,-1,d)
    @test all(size.(tto_lap.tto_vec) .== size.(tto_ref.tto_vec))
    @test all(tto_lap.tto_vec[1] .== tto_ref.tto_vec[1])
    @test all(tto_lap.tto_vec[2] .== tto_ref.tto_vec[2])
    @test all(tto_lap.tto_vec[3] .== tto_ref.tto_vec[3])
end

@testset "qtto_prolongation" begin
    # Minimal mock for zeros_tto to allow testing
    struct MockTTOProlong
        tto_vec::Vector
    end
    function zeros_tto(::Type{Float64}, dims::NTuple{N,Int}, shapes::Vector{Int}) where N
        # Each core: (2,2,2,2)
        tto_vec = [zeros(2,2,2,2) for _ in 1:N]
        return MockTTOProlong(tto_vec)
    end

    # Patch the function in the module namespace for testing
    @eval TensorTrainNumerics begin
        import ..Main.zeros_tto
    end

    d = 3
    tto = TensorTrainNumerics.qtto_prolongation(d)
    @test length(tto.tto_vec) == d
    for j in 1:d
        core = tto.tto_vec[j]
        @test size(core) == (2,2,2,2)
        # Check the specific entries set to 1.0
        @test core[1,1,1,1] ≈ 1.0
        @test core[1,1,2,2] ≈ 1.0
        @test core[1,2,2,1] ≈ 1.0
        @test core[2,2,1,2] ≈ 1.0
        # All other entries should be zero
        for i1 in 1:2, i2 in 1:2, i3 in 1:2, i4 in 1:2
            if !((i1,i2,i3,i4) in [(1,1,1,1), (1,1,2,2), (1,2,2,1), (2,2,1,2)])
                @test core[i1,i2,i3,i4] ≈ 0.0
            end
        end
    end
end
