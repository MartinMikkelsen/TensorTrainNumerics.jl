using Test
using LinearAlgebra
using TensorTrainNumerics
using InterpolativeQTT

@testset "interpolative_qtt builds 1D TTvector" begin
    bits = 8
    u = interpolative_qtt(x -> sin(2π * x), bits;
        degree = 6,
        tolerance = 1.0e-12,
        maxbonddim = 32,
    )

    @test u isa TTvector
    @test u.N == bits
    @test all(==(2), u.ttv_dims)
    @test maximum(u.ttv_rks) <= 32
end

@testset "invert_interpolative_qtt recovers Chebyshev-Lobatto values" begin
    bits = 10
    degree = 6
    q = 1
    f(x) = sin(2π * x)
    u = interpolative_qtt(f, bits;
        degree = degree,
        tolerance = 1.0e-12,
        maxbonddim = 32,
    )

    tables = invert_interpolative_qtt(u; degree = degree, q = q)

    @test length(tables) == bits - q
    finest = last(tables)
    @test size(finest) == (2^(bits - q), degree + 1)

    P = InterpolativeQTT.getChebyshevGrid(degree)
    max_error = 0.0
    for interval in axes(finest, 1), β in 0:degree
        x = (interval - 1 + P.grid[β + 1]) / 2^(bits - q)
        max_error = max(max_error, abs(finest[interval, β + 1] - f(x)))
    end

    @test max_error < 5.0e-5
end

@testset "project_nonlinearity builds coefficient QTT through interpolation" begin
    bits = 10
    degree = 10
    f(x) = sin(2π * x)
    u = interpolative_qtt(f, bits;
        degree = degree,
        tolerance = 1.0e-12,
        maxbonddim = 48,
    )

    coeff = project_nonlinearity(u, z -> z^2;
        degree = degree,
        tolerance = 1.0e-10,
        maxbonddim = 48,
        q = 1,
    )

    values = real.(qtt_to_function(coeff))
    expected = [f((i - 1) / 2^bits)^2 for i in 1:(2^bits)]
    relerr = norm(values - expected) / max(norm(expected), eps(Float64))

    @test coeff isa TTvector
    @test coeff.N == bits
    @test maximum(coeff.ttv_rks) <= 48
    @test relerr < 1.0e-5
end
