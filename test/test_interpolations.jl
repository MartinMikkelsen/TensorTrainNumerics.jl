using Test
using TensorTrainNumerics
using InterpolativeQTT
import TensorCrossInterpolation as TCI

@testset "to_ttvector — 1D structural" begin
    f(x) = sin(2π * x)
    numbits = 8
    degree = 4

    tt_tci = interpolatesinglescale(f, 0.0, 1.0, numbits, degree)
    @test tt_tci isa TCI.TensorTrain

    tt = to_ttvector(tt_tci)
    @test tt isa TTVector
    @test nsites(tt) == numbits
    @test tt.ranks[1] == 1
    @test tt.ranks[end] == 1
    @test all(d -> d == 2, tt.dims)
    for k in 1:nsites(tt)
        @test size(tt.cores[k], 2) == tt.ranks[k]
        @test size(tt.cores[k], 3) == tt.ranks[k + 1]
    end
end

@testset "to_ttvector — 1D value correctness" begin
    f(x) = sin(2π * x)
    numbits = 8
    degree = 4

    tt_tci = interpolatesinglescale(f, 0.0, 1.0, numbits, degree)
    tt = to_ttvector(tt_tci)

    # Both representations should evaluate to the same values at random binary indices
    full = ttv_to_tensor(tt)
    for idx in ([1, 1, 1, 1, 1, 1, 1, 1], [2, 1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1, 2])
        @test full[idx...] ≈ tt_tci(idx) atol = 1.0e-14
    end
end

@testset "to_ttvector — 2D fused" begin
    f(x, y) = sin(2π * x) * cos(2π * y)
    numbits = 6
    degree = 3

    tt_tci = interpolatesinglescale(f, (0.0, 0.0), (1.0, 1.0), numbits, degree)
    tt = to_ttvector(tt_tci)

    @test tt isa TTVector
    @test nsites(tt) == numbits
    @test tt.ranks[1] == 1
    @test tt.ranks[end] == 1
    # Fused: each site has phys_dim = 2^2 = 4
    @test all(d -> d == 4, tt.dims)
    for k in 1:nsites(tt)
        @test size(tt.cores[k], 2) == tt.ranks[k]
        @test size(tt.cores[k], 3) == tt.ranks[k + 1]
    end
end
