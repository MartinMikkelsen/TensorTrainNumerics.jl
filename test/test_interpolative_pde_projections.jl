using Test
using LinearAlgebra
using TensorTrainNumerics
using InterpolativeQTT

function _dyadic_values(f, bits::Int)
    return [f((i - 1) / 2^bits) for i in 1:(2^bits)]
end

function _relative_error(values, expected)
    return norm(values - expected) / max(norm(expected), eps(Float64))
end

@testset "Allen-Cahn reaction projects through InterpolativeQTT" begin
    bits = 10
    degree = 12
    ufun(x) = 0.8 * sin(2π * x)
    reaction(z) = z - z^3

    u = interpolative_qtt(ufun, bits;
        degree = degree,
        tolerance = 1.0e-12,
        maxbonddim = 64,
    )
    coeff = project_nonlinearity(u, reaction;
        degree = degree,
        tolerance = 1.0e-11,
        maxbonddim = 64,
        q = 1,
    )

    values = real.(qtt_to_function(coeff))
    expected = _dyadic_values(x -> reaction(ufun(x)), bits)

    @test coeff isa TTvector
    @test maximum(coeff.ttv_rks) <= 64
    @test _relative_error(values, expected) < 5.0e-5
end

@testset "GPE density projects through InterpolativeQTT" begin
    bits = 10
    degree = 12
    ψfun(x) = exp(-24 * (x - 0.5)^2) * (1 + 0.1 * cos(2π * x))

    ψ = interpolative_qtt(ψfun, bits;
        degree = degree,
        tolerance = 1.0e-12,
        maxbonddim = 64,
    )
    density = project_nonlinearity(ψ, abs2;
        degree = degree,
        tolerance = 1.0e-11,
        maxbonddim = 64,
        q = 1,
    )

    values = real.(qtt_to_function(density))
    expected = _dyadic_values(x -> abs2(ψfun(x)), bits)

    @test density isa TTvector
    @test maximum(density.ttv_rks) <= 64
    @test _relative_error(values, expected) < 5.0e-5
end

@testset "KdV transport product projects from two InterpolativeQTT fields" begin
    bits = 10
    degree = 12
    ufun(x) = sin(2π * x)
    uxfun(x) = 2π * cos(2π * x)

    u = interpolative_qtt(ufun, bits;
        degree = degree,
        tolerance = 1.0e-12,
        maxbonddim = 64,
    )
    ux = interpolative_qtt(uxfun, bits;
        degree = degree,
        tolerance = 1.0e-12,
        maxbonddim = 64,
    )
    transport = project_nonlinearity((u, ux), (z, dz) -> z * dz;
        degree = degree,
        tolerance = 1.0e-11,
        maxbonddim = 64,
        q = 1,
    )

    values = real.(qtt_to_function(transport))
    expected = _dyadic_values(x -> ufun(x) * uxfun(x), bits)

    @test transport isa TTvector
    @test maximum(transport.ttv_rks) <= 64
    @test _relative_error(values, expected) < 5.0e-5
end
