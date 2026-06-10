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

function _dyadic_grid_values_2d(f, bits::Int)
    n = 2^bits
    return [f((i - 1) / n, (j - 1) / n) for i in 1:n, j in 1:n]
end

function _decode_mv_interval(row::Int, ndims::Int, levels::Int)
    phys = 2^ndims
    r = row - 1
    sites = Vector{Int}(undef, levels)
    for level in levels:-1:1
        sites[level] = r % phys
        r ÷= phys
    end

    intervals = zeros(Int, ndims)
    for level in 1:levels
        for dim in 1:ndims
            bit = (sites[level] >> (dim - 1)) & 1
            intervals[dim] += bit * 2^(levels - level)
        end
    end
    return intervals
end

function _decode_mv_cheb(col::Int, ndims::Int, degree::Int)
    base = degree + 1
    c = col - 1
    β = Vector{Int}(undef, ndims)
    for dim in 1:ndims
        β[dim] = c % base
        c ÷= base
    end
    return β
end

@testset "interpolative_qttv builds representation-safe 2D QTTvectors" begin
    bits = 4
    degree = 4
    f(x, y) = 10x + y

    q_interleaved = interpolative_qttv(f, 2, bits;
        ordering = :interleaved,
        degree = degree,
        tolerance = 1.0e-12,
        maxbonddim = 32,
    )
    q_serial = interpolative_qttv(f, 2, bits;
        ordering = :serial,
        degree = degree,
        tolerance = 1.0e-12,
        maxbonddim = 32,
    )

    @test q_interleaved isa QTTvector
    @test q_interleaved.ordering == :interleaved
    @test q_interleaved.n_dims == 2
    @test q_interleaved.bits_per_dim == bits
    @test q_serial.ordering == :serial

    expected = _dyadic_grid_values_2d(f, bits)
    @test norm(qttv_to_array(q_interleaved) - expected) / norm(expected) < 1.0e-8
    @test norm(qttv_to_array(q_serial) - expected) / norm(expected) < 1.0e-8
end

@testset "invert_interpolative_qtt supports 2D serial and interleaved QTTvectors" begin
    bits = 5
    degree = 4
    qdrop = 1
    f(x, y) = 10x + y

    P = InterpolativeQTT.getChebyshevGrid(degree)
    for ordering in (:interleaved, :serial)
        u = interpolative_qttv(f, 2, bits;
            ordering = ordering,
            degree = degree,
            tolerance = 1.0e-12,
            maxbonddim = 64,
        )
        tables = invert_interpolative_qtt(u; degree = degree, q = qdrop)
        finest = last(tables)
        levels = bits - qdrop

        @test size(finest) == (2^(2 * levels), (degree + 1)^2)

        max_error = 0.0
        for row in (1, 3, 12, size(finest, 1)), col in (1, 2, degree + 2, size(finest, 2))
            interval = _decode_mv_interval(row, 2, levels)
            β = _decode_mv_cheb(col, 2, degree)
            x = (interval[1] + P.grid[β[1] + 1]) / 2^levels
            y = (interval[2] + P.grid[β[2] + 1]) / 2^levels
            max_error = max(max_error, abs(finest[row, col] - f(x, y)))
        end
        @test max_error < 1.0e-10
    end
end

@testset "adaptive mode resolves localized features" begin
    bits = 7
    Ldom = 25.0
    soliton(x) = 0.5 * sech(0.5 * (x - 9.0))^2

    # I_q alone: adaptive construction must be spectrally accurate where the
    # single-scale degree-8 interpolant fails (~7e-2) on a localized bump.
    u_adapt = interpolative_qtt(soliton, bits;
        degree = 8,
        tolerance = 1.0e-12,
        maxbonddim = 64,
        a = 0.0,
        b = Ldom,
        mode = :adaptive,
        adaptive_tolerance = 1.0e-10,
    )
    dyadic = [soliton(Ldom * (i - 1) / 2^bits) for i in 1:(2^bits)]
    @test norm(real.(qtt_to_function(u_adapt)) - dyadic) / norm(dyadic) < 1.0e-6

    # Full R_q -> identity -> I_q roundtrip: now limited only by the R_q
    # stage-1 linear-interpolation floor O(u''·4^-bits), not by the degree.
    u0 = function_to_qtt(soliton, bits; b = Ldom)
    c_single = project_nonlinearity(u0, identity;
        degree = 8, tolerance = 1.0e-12, maxbonddim = 64, q = 1,
        a = 0.0, b = Ldom,
    )
    c_adapt = project_nonlinearity(u0, identity;
        degree = 8, tolerance = 1.0e-12, maxbonddim = 64, q = 1,
        a = 0.0, b = Ldom,
        mode = :adaptive,
        adaptive_tolerance = 1.0e-10,
    )
    err_single = norm(c_single - u0) / norm(u0)
    err_adapt = norm(c_adapt - u0) / norm(u0)
    @test err_adapt < 2.0e-2
    @test err_adapt < err_single / 3
end

@testset "project_nonlinearity respects maxbonddim for serial QTTvectors" begin
    bits = 5
    degree = 8
    cap = 12
    f(x, y) = 0.8 * exp(-12 * ((x - 0.45)^2 + (y - 0.3)^2)) + 0.2 * sin(2pi * x) * cos(pi * y)

    u = interpolative_qttv(f, 2, bits;
        ordering = :serial,
        degree = degree,
        tolerance = 1.0e-12,
        maxbonddim = 64,
    )
    coeff = project_nonlinearity(u, z -> z^2;
        degree = degree,
        tolerance = 1.0e-10,
        maxbonddim = cap,
        q = 1,
    )

    expected = _dyadic_grid_values_2d((x, y) -> f(x, y)^2, bits)

    @test maximum(coeff.ttv_rks) <= cap
    @test norm(qttv_to_array(coeff) - expected) / norm(expected) < 5.0e-2
end

@testset "project_nonlinearity uses multivariate InterpolativeQTT for 2D QTTvectors" begin
    bits = 5
    degree = 4
    ufun(x, y) = 0.25 + 0.5x - 0.2y
    reaction(z) = z^2 - z

    u = interpolative_qttv(ufun, 2, bits;
        ordering = :serial,
        degree = degree,
        tolerance = 1.0e-12,
        maxbonddim = 64,
    )
    coeff = project_nonlinearity(u, reaction;
        degree = degree,
        tolerance = 1.0e-10,
        maxbonddim = 64,
        q = 1,
    )

    expected = _dyadic_grid_values_2d((x, y) -> reaction(ufun(x, y)), bits)

    @test coeff isa QTTvector
    @test coeff.ordering == :serial
    @test coeff.n_dims == 2
    @test coeff.bits_per_dim == bits
    @test norm(qttv_to_array(coeff) - expected) / norm(expected) < 1.0e-10
end
