using Test
using TensorTrainNumerics  # Assuming your package is named this way
using LinearAlgebra
import TensorTrainNumerics: updateH_mals!
@testset "updateH_mals! basic functionality" begin
    # Set up small random tensors for testing
    T = Float64
    x_vec = rand(T, 2, 3, 4)
    A_vec = rand(T, 3, 2, 2, 5)
    Hi = rand(T, 5, 3, 4, 6, 7)
    Him = zeros(T, 2, 3, 6, 2, 7)

    # Call the function
    updateH_mals!(x_vec, A_vec, Hi, Him)

    # Check that Him has been modified (not all zeros)
    @test !all(Him .== 0.0)

    # Check that the shape is preserved
    @test size(Him) == (2, 3, 6, 2, 7)

    # Check type
    @test eltype(Him) == T
end

@testset "updateH_mals! with known input" begin
    # Use small, simple tensors for deterministic output
    x_vec = ones(Float64, 1, 1, 1)
    A_vec = ones(Float64, 1, 1, 1, 1)
    Hi = ones(Float64, 1, 1, 1, 1, 1)
    Him = zeros(Float64, 1, 1, 1, 1, 1)

    updateH_mals!(x_vec, A_vec, Hi, Him)

    # The result should be 1.0 in the only entry
    @test Him[1,1,1,1,1] ≈ 1.0
end