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
    dims1 = (2, 2)
    rks1 = [1, 2, 1]
    tt1 = rand_tt(dims1, rks1)
    dims2 = (2, 2)
    rks2 = [1, 2, 1]
    tt2 = rand_tt(dims2, rks2)
    tt_concat = concatenate(tt1, tt2)
    @test tt_concat.N == tt1.N + tt2.N
    @test tt_concat.ttv_dims == (tt1.ttv_dims..., tt2.ttv_dims...)
    @test tt_concat.ttv_rks == vcat(tt1.ttv_rks[1:end-1], tt2.ttv_rks)
    @test tt_concat.ttv_ot == vcat(tt1.ttv_ot, tt2.ttv_ot)

    # Test concatenate function for TToperator
    dims_op1 = (2, 2)
    tto1 = rand_tto(dims_op1, 3)
    dims_op2 = (2, 2)
    tto2 = rand_tto(dims_op2, 3)
    tto_concat = concatenate(tto1, tto2)
    @test tto_concat.N == tto1.N + tto2.N
    @test tto_concat.tto_dims == (tto1.tto_dims..., tto2.tto_dims...)
    @test tto_concat.tto_rks == vcat(tto1.tto_rks[1:end-1], tto2.tto_rks)
    @test tto_concat.tto_ot == vcat(tto1.tto_ot, tto2.tto_ot)

    # Test concatenate function
    dims1 = (2, 2)
    rks1 = [1, 2, 1]
    tt1 = rand_tt(dims1, rks1)
    dims2 = (2, 2)
    rks2 = [1, 2, 1]
    tt2 = rand_tt(dims2, rks2)
    tt_concat = concatenate(tt1, tt2)
    @test tt_concat.N == tt1.N + tt2.N
    @test tt_concat.ttv_dims == (tt1.ttv_dims..., tt2.ttv_dims...)
    @test tt_concat.ttv_rks == vcat(tt1.ttv_rks[1:end-1], tt2.ttv_rks)
    @test tt_concat.ttv_ot == vcat(tt1.ttv_ot, tt2.ttv_ot)
end
