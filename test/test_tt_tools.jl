include("../src/tt_tools.jl")

using Test

@testset "TT constructors and properties" begin
    # Test TTvector constructor
    N = 3
    vec = [randn(2, 1, 2), randn(2, 2, 2), randn(2, 2, 1)]
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    ot = [0, 0, 0]
    tt = TTvector{Float64, 3}(N, vec, dims, rks, ot)
    @test tt.N == N
    @test tt.ttv_vec == vec
    @test tt.ttv_dims == dims
    @test tt.ttv_rks == rks
    @test tt.ttv_ot == ot

    # Test TToperator constructor
    N = 3
    vec = [randn(2, 2, 1, 2), randn(2, 2, 2, 2), randn(2, 2, 2, 1)]
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    ot = [0, 0, 0]
    tto = TToperator{Float64, 3}(N, vec, dims, rks, ot)
    @test tto.N == N
    @test tto.tto_vec == vec
    @test tto.tto_dims == dims
    @test tto.tto_rks == rks
    @test tto.tto_ot == ot

    # Test complex conversion
    tt_complex = complex(tt)
    @test eltype(tt_complex) == Complex{Float64}
    @test all(eltype.(tt_complex.ttv_vec) .== Complex{Float64})

    tto_complex = complex(tto)
    @test eltype(tto_complex) == Complex{Float64}
    @test all(eltype.(tto_complex.tto_vec) .== Complex{Float64})

    # Test QTTvector and is_qtt
    qtt_vec = [randn(2, 1, 2), randn(2, 2, 2), randn(2, 2, 1)]
    qtt_rks = [1, 2, 2, 1]
    qtt_ot = [0, 0, 0]
    qtt = QTTvector(qtt_vec, qtt_rks, qtt_ot)
    @test is_qtt(qtt)

    # Test QTToperator and is_qtt_operator
    qtt_op_vec = [randn(2, 2, 1, 2), randn(2, 2, 2, 2), randn(2, 2, 2, 1)]
    qtt_op_rks = [1, 2, 2, 1]
    qtt_op_ot = [0, 0, 0]
    qtt_op = QTToperator(qtt_op_vec, qtt_op_rks, qtt_op_ot)
    @test is_qtt_operator(qtt_op)
end

@testset "TT constructors and properties" begin
    # Test TTvector constructor
    N = 3
    vec = [randn(2, 1, 2), randn(2, 2, 2), randn(2, 2, 1)]
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    ot = [0, 0, 0]
    tt = TTvector{Float64, 3}(N, vec, dims, rks, ot)
    @test tt.N == N
    @test tt.ttv_vec == vec
    @test tt.ttv_dims == dims
    @test tt.ttv_rks == rks
    @test tt.ttv_ot == ot

    # Test TToperator constructor
    N = 3
    vec = [randn(2, 2, 1, 2), randn(2, 2, 2, 2), randn(2, 2, 2, 1)]
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    ot = [0, 0, 0]
    tto = TToperator{Float64, 3}(N, vec, dims, rks, ot)
    @test tto.N == N
    @test tto.tto_vec == vec
    @test tto.tto_dims == dims
    @test tto.tto_rks == rks
    @test tto.tto_ot == ot

    # Test complex conversion
    tt_complex = complex(tt)
    @test eltype(tt_complex) == Complex{Float64}
    @test all(eltype.(tt_complex.ttv_vec) .== Complex{Float64})

    tto_complex = complex(tto)
    @test eltype(tto_complex) == Complex{Float64}
    @test all(eltype.(tto_complex.tto_vec) .== Complex{Float64})

    # Test QTTvector and is_qtt
    qtt_vec = [randn(2, 1, 2), randn(2, 2, 2), randn(2, 2, 1)]
    qtt_rks = [1, 2, 2, 1]
    qtt_ot = [0, 0, 0]
    qtt = QTTvector(qtt_vec, qtt_rks, qtt_ot)
    @test is_qtt(qtt)

    # Test QTToperator and is_qtt_operator
    qtt_op_vec = [randn(2, 2, 1, 2), randn(2, 2, 2, 2), randn(2, 2, 2, 1)]
    qtt_op_rks = [1, 2, 2, 1]
    qtt_op_ot = [0, 0, 0]
    qtt_op = QTToperator(qtt_op_vec, qtt_op_rks, qtt_op_ot)
    @test is_qtt_operator(qtt_op)
end

@testset "TTvector and TToperator functions" begin
    # Test zeros_tt
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    zero_tt = zeros_tt(dims, rks)
    @test all(x -> all(==(0), x), zero_tt.ttv_vec)

    # Test ones_tt
    ones_tt = ones_tt(dims)
    @test all(x -> all(==(1), x), ones_tt.ttv_vec)

    # Test zeros_tto
    zero_tto = zeros_tto(dims, rks)
    @test all(x -> all(==(0), x), zero_tto.tto_vec)

    # Test rand_tt
    rand_tt_vec = rand_tt(dims, rks)
    @test all(x -> eltype(x) == Float64, rand_tt_vec.ttv_vec)

    # Test rand_tto
    rand_tto_vec = rand_tto(dims, 2)
    @test all(x -> eltype(x) == Float64, rand_tto_vec.tto_vec)

    # Test ttv_decomp and ttv_to_tensor
    tensor = randn(2, 2, 2)
    tt_decomp = ttv_decomp(tensor)
    tensor_reconstructed = ttv_to_tensor(tt_decomp)
    @test isapprox(tensor, tensor_reconstructed, atol=1e-12)

    # Test tto_decomp and tto_to_tensor
    tensor_op = randn(2, 2, 2, 2, 2, 2)
    tto_decomp = tto_decomp(tensor_op)
    tensor_op_reconstructed = tto_to_tensor(tto_decomp)
    @test isapprox(tensor_op, tensor_op_reconstructed, atol=1e-12)

    # Test tto_to_ttv and ttv_to_tto
    ttv_from_tto = tto_to_ttv(tto)
    tto_from_ttv = ttv_to_tto(ttv_from_tto)
    @test tto_from_ttv == tto

    # Test visualize functions
    @test visualize(tt) === nothing
    @test visualize(tto) === nothing
end
