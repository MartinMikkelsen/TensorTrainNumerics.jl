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
    @test tt isa TTvector
    @test tt.N == numbits
    @test tt.ttv_rks[1] == 1
    @test tt.ttv_rks[end] == 1
    @test all(d -> d == 2, tt.ttv_dims)
    for k in 1:tt.N
        @test size(tt.ttv_vec[k], 2) == tt.ttv_rks[k]
        @test size(tt.ttv_vec[k], 3) == tt.ttv_rks[k + 1]
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

    @test tt isa TTvector
    @test tt.N == numbits
    @test tt.ttv_rks[1] == 1
    @test tt.ttv_rks[end] == 1
    # Fused: each site has phys_dim = 2^2 = 4
    @test all(d -> d == 4, tt.ttv_dims)
    for k in 1:tt.N
        @test size(tt.ttv_vec[k], 2) == tt.ttv_rks[k]
        @test size(tt.ttv_vec[k], 3) == tt.ttv_rks[k + 1]
    end
end

@testset "to_tci_tensortrain — roundtrip values" begin
    f(x, y) = sin(2π * x) * cos(2π * y)
    numbits = 5
    degree = 3

    tt_tci = interpolatesinglescale(f, (0.0, 0.0), (1.0, 1.0), numbits, degree)
    tt_ttn = to_ttvector(tt_tci)
    tt_roundtrip = to_tci_tensortrain(tt_ttn)

    @test tt_roundtrip isa TCI.TensorTrain
    @test TCI.sitedims(tt_roundtrip) == TCI.sitedims(tt_tci)

    for idx in ([1, 1, 1, 1, 1], [2, 1, 2, 1, 2], [1, 2, 1, 2, 1])
        @test tt_roundtrip(idx) ≈ tt_tci(idx) atol = 1.0e-14
    end
end

@testset "TCI fused to serial QTT and back preserves values" begin
    f(t, x, y) = exp(-t) * sin(π * x) * cos(π * y)
    numbits = 4
    degree = 3

    tt_tci = interpolatesinglescale(
        f,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        numbits,
        degree;
        tolerance = 1.0e-12
    )

    tt_fused = to_ttvector(tt_tci)
    tt_split = to_qtt(tt_fused, [[2, 2, 2] for _ in 1:numbits]; threshold = 1.0e-14)
    q_interleaved = QTTvector(tt_split, 3, numbits, :interleaved)
    q_serial = reorder(q_interleaved, :serial; threshold = 1.0e-14)

    q_interleaved_roundtrip = reorder(q_serial, :interleaved; threshold = 1.0e-14)
    tt_fused_roundtrip = to_ttv(TTvector(q_interleaved_roundtrip), fill(3, numbits))
    tt_tci_roundtrip = to_tci_tensortrain(tt_fused_roundtrip)

    for idx in ([1, 1, 1, 1], [2, 1, 2, 1], [1, 2, 1, 2])
        @test tt_tci_roundtrip(idx) ≈ tt_tci(idx) atol = 1.0e-10
    end
end
