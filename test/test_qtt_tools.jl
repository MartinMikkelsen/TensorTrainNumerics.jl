using Test



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

@testset "index_to_point" begin
    # For d=3, t = (1,1,1) should map to 0.0
    t = (1, 1, 1)
    @test isapprox(index_to_point(t; L=1.0), 0.0)
    # For d=3, t = (2,2,2) should map to 1.0
    t = (2, 2, 2)
    @test isapprox(index_to_point(t; L=1.0), 1.0)
    # For d=2, t = (1,2)
    t = (1, 2)
    @test isapprox(index_to_point(t; L=1.0), 1/3)
end

@testset "tuple_to_index" begin
    # For d=3, t = (1,1,1) should map to 1
    t = (1, 1, 1)
    @test tuple_to_index(t) == 1
    # For d=3, t = (2,2,2) should map to 8
    t = (2, 2, 2)
    @test tuple_to_index(t) == 8
    # For d=2, t = (2,1)
    t = (2, 1)
    @test tuple_to_index(t) == 3
end

@testset "function_to_tensor" begin
    f(x) = 2x
    d = 2
    T = function_to_tensor(f, d; a=0.0, b=1.0)
    @test size(T) == (2, 2)
    xs = collect(range(0.0, stop=1.0, length=4))
    expected = reshape(f.(xs), 2, 2)
    @test isapprox.(T, expected) |> all
end

@testset "tensor_to_grid" begin
    T = [1.0 2.0; 3.0 4.0]
    v = tensor_to_grid(T)
    @test v == [1.0, 3.0, 2.0, 4.0]
    # Test with 3D tensor
    T3 = reshape(1:8, 2, 2, 2)
    v3 = tensor_to_grid(T3)
    @test v3 == collect(1:8)
end

@testset "x^2" begin
    h(x) = (x^2)
    d = 8
    Q = function_to_qtt(h, d; a=-1, b=1)
    A = qtt_to_function(Q)
    xs = collect(range(-1.0, 1.0, length=2^d))
    expected = h.(xs)
    @test isapprox.(A, expected) |> all
end 

@testset "Weird function" begin
    h(x) = sin(x^3.5) + cos(x^3.5) + x^1.3
    d = 8
    Q = function_to_qtt(h, d; a=0.0, b=7.3)
    A = qtt_to_function(Q)
    xs = collect(range(0.0, 7.3, length=2^d))
    expected = h.(xs)
    @test isapprox.(A, expected) |> all
end 

@testset "Polynomial function" begin
    h(x) = x^4 - 2x^3 + x^2 - x + 1
    d = 8
    Q = function_to_qtt(h, d; a=-2.0, b=2.0)
    A = qtt_to_function(Q)
    xs = collect(range(-2.0, 2.0, length=2^d))
    expected = h.(xs)
    @test isapprox.(A, expected) |> all
end 

@testset "Exponential function" begin
    h(x) = exp(-x^2) + x^3
    d = 8
    Q = function_to_qtt(h, d; a=-1.0, b=1.0)
    A = qtt_to_function(Q)
    xs = collect(range(-1.0, 1.0, length=2^d))
    expected = h.(xs)
    @test isapprox.(A, expected) |> all
end 

@testset "Logarithmic function" begin
    h(x) = log(x + 1) + x^0.5
    d = 8
    Q = function_to_qtt(h, d; a=0.0, b=5.0)
    A = qtt_to_function(Q)
    xs = collect(range(0.0, 5.0, length=2^d))
    expected = h.(xs)
    @test isapprox.(A, expected; atol=1e-6) |> all
end 

@testset "Absolute value function" begin
    h(x) = abs(x^3 - x^2 + x - 1)
    d = 8
    Q = function_to_qtt(h, d; a=-2.0, b=2.0)
    A = qtt_to_function(Q)
    xs = collect(range(-2.0, 2.0, length=2^d))
    expected = h.(xs)
    @test isapprox.(A, expected) |> all
end 