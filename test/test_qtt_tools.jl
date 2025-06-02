using Test

# Test for index_to_point
@testset "index_to_point" begin
    # 1D case
    @test isapprox(index_to_point((1,)), 0.0)
    @test isapprox(index_to_point((2,)), 1.0)
    # 2D case
    @test isapprox(index_to_point((1,1)), 0.0)
    @test isapprox(index_to_point((2,2)), 1.0)
    # 3D case
    @test isapprox(index_to_point((1,1,1)), 0.0)
    @test isapprox(index_to_point((2,2,2)), 1.0)
end

# Test for tuple_to_index
@testset "tuple_to_index" begin
    # 1D case
    @test tuple_to_index((1,)) == 1
    @test tuple_to_index((2,)) == 2
    # 2D case
    @test tuple_to_index((1,1)) == 1
    @test tuple_to_index((1,2)) == 2
    @test tuple_to_index((2,1)) == 3
    @test tuple_to_index((2,2)) == 4
    # 3D case
    @test tuple_to_index((1,1,1)) == 1
    @test tuple_to_index((1,1,2)) == 2
    @test tuple_to_index((1,2,1)) == 3
    @test tuple_to_index((2,1,1)) == 5
    @test tuple_to_index((2,2,2)) == 8
end
# Tests for function_to_tensor
@testset "function_to_tensor" begin
    # 1D: f(x) = x
    f1(x) = x
    tensor1 = function_to_tensor(f1, 1; a=0.0, b=1.0)
    @test size(tensor1) == (2,)
    @test isapprox(tensor1[1], 0.0)
    @test isapprox(tensor1[2], 1.0)

    # 2D: f(x) = x
    f2(x) = x
    tensor2 = function_to_tensor(f2, 2; a=0.0, b=1.0)
    @test size(tensor2) == (2,2)
    @test isapprox(tensor2[1,1], 0.0)
    @test isapprox(tensor2[2,2], 1.0)

    # 2D: f(x) = x^2
    f3(x) = x^2
    tensor3 = function_to_tensor(f3, 2; a=0.0, b=1.0)
    @test isapprox(tensor3[1,1], 0.0)
    @test isapprox(tensor3[2,2], 1.0)
end

# Tests for tensor_to_grid
@testset "tensor_to_grid" begin
    # 1D
    tensor1 = [10.0, 20.0]
    grid1 = tensor_to_grid(tensor1)
    @test grid1 == [10.0, 20.0]

    # 2D
    tensor2 = [1.0 2.0; 3.0 4.0]
    grid2 = tensor_to_grid(tensor2)
    # tuple_to_index((1,1))=1, (1,2)=2, (2,1)=3, (2,2)=4
    @test grid2 == [1.0, 2.0, 3.0, 4.0]

    tensor3 = reshape(1:8, 2,2,2)
    grid3 = tensor_to_grid(tensor3)
    # Build expected output using tuple_to_index
    expected3 = zeros(8)
    for t in CartesianIndices(tensor3)
        expected3[tuple_to_index(Tuple(t))] = tensor3[t]
    end
    @test grid3 == expected3
end



@testset "qtt_polynom" begin
    coef = [0.0, 1.0]  # p(x) = x
    d = 3
    tt = qtt_polynom(coef, d; a=0.0, b=1.0)
    # Check types and shapes
    @test hasproperty(tt, :ttv_vec)
    @test length(tt.ttv_vec) == d
    @test size(tt.ttv_vec[1]) == (2, 1, 2)
    @test size(tt.ttv_vec[d]) == (2, 2, 1)
    # Check that the first core is filled as expected for x=0 and x=1
    @test isapprox((tt.ttv_vec[1])[1, 1, 2], 1.0)  # not 0.0
    @test isapprox(tt.ttv_vec[1][2,1,2], 1.0)  # φ(1,1) for x^1
end

@testset "qtt_cos" begin
    d = 3
    λ = 1.0
    a = 0.0
    b = 1.0
    tt = qtt_cos(d; a=a, b=b, λ=λ)
    # Check structure
    @test hasproperty(tt, :ttv_vec)
    @test length(tt.ttv_vec) == d
    @test size(tt.ttv_vec[1]) == (2, 1, 2)
    @test size(tt.ttv_vec[d]) == (2, 2, 1)
    # Check first core values at x=a
    t₁ = a
    @test isapprox(tt.ttv_vec[1][1,1,1], cos(λ*π*t₁))
    @test isapprox(tt.ttv_vec[1][1,1,2], -sin(λ*π*t₁))
end

@testset "qtt_sin" begin
    d = 3
    λ = 1.0
    a = 0.0
    b = 1.0
    tt = qtt_sin(d; a=a, b=b, λ=λ)
    # Check structure
    @test hasproperty(tt, :ttv_vec)
    @test length(tt.ttv_vec) == d
    @test size(tt.ttv_vec[1]) == (2, 1, 2)
    @test size(tt.ttv_vec[d]) == (2, 2, 1)
    # Check first core values at x=a
    t₁ = a
    @test isapprox(tt.ttv_vec[1][1,1,1], sin(λ*π*t₁))
    @test isapprox(tt.ttv_vec[1][1,1,2], cos(λ*π*t₁))
end

@testset "qtt_exp" begin
    d = 3
    α = 2.0
    β = 0.5
    a = 0.0
    b = 1.0
    tt = qtt_exp(d; a=a, b=b, α=α, β=β)
    # Check structure
    @test hasproperty(tt, :ttv_vec)
    @test length(tt.ttv_vec) == d
    @test size(tt.ttv_vec[1]) == (2, 1, 1)
    @test size(tt.ttv_vec[d]) == (2, 1, 1)
    # Check first core values
    h = (b - a) / (2^d - 1)
    t₁ = a
    @test isapprox(tt.ttv_vec[1][1,1,1], exp(α * t₁ + β))
    t₁ = a + h * 2^(d-1)
    @test isapprox(tt.ttv_vec[1][2,1,1], exp(α * t₁ + β))
end
