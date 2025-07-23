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
end

@testset "Matricize function for TTvector" begin
    # Create a simple TTvector for testing
    N = 3
    dims = (2, 3, 4)
    
    # Create cores with simple values
    r1 = 2
    r2 = 2
    
    # Use simple values to make verification easier
    core1 = ones(Float64, 2, 1, r1)  # dim1 × rank0 × rank1
    core2 = ones(Float64, 3, r1, r2)  # dim2 × rank1 × rank2
    core3 = ones(Float64, 4, r2, 1)  # dim3 × rank2 × rank3
    
    vec = [core1, core2, core3]
    rks = [1, r1, r2, 1]
    ot = [0, 0, 0]
    
    tt = TTvector{Float64, 3}(N, vec, dims, rks, ot)
    
    # Test basic functionality
    result = matricize(tt)
    
    # Check dimensions - should be (prod(dims),) since final rank is 1
    @test size(result) == (prod(dims),)
    
    # For this simple case of all ones, all elements should be r1*r2 = 4
    @test all(result .== 4.0)
    
    # Test error handling with incorrect core size
    bad_core1 = randn(3, 1, r1)  # Incorrect first dimension
    bad_vec = [bad_core1, core2, core3]
    bad_tt = TTvector{Float64, 3}(N, bad_vec, dims, rks, ot)
    
    @test_throws ErrorException matricize(bad_tt)
end

@testset "Base.eltype and Base.complex for TTvector and TToperator" begin
    # Test eltype for TTvector
    N = 2
    vec = [randn(2, 1, 2), randn(2, 2, 1)]
    dims = (2, 2)
    rks = [1, 2, 1]
    ot = [0, 0]
    tt = TTvector{Float64, 2}(N, vec, dims, rks, ot)
    @test eltype(tt) == Float64

    # Test eltype for TToperator
    op_vec = [randn(2, 2, 1, 2), randn(2, 2, 2, 1)]
    op_dims = (2, 2)
    op_rks = [1, 2, 1]
    op_ot = [0, 0]
    tto = TToperator{Float64, 2}(N, op_vec, op_dims, op_rks, op_ot)
    @test eltype(tto) == Float64

    # Test Base.complex for TTvector
    tt_c = TTvector{ComplexF64, 2}(tt.N, [complex.(core) for core in tt.ttv_vec], tt.ttv_dims, tt.ttv_rks, tt.ttv_ot)
    @test typeof(tt_c) == TTvector{ComplexF64, 2}
    @test tt_c.N == tt.N
    @test tt_c.ttv_dims == tt.ttv_dims
    @test tt_c.ttv_rks == tt.ttv_rks
    @test tt_c.ttv_ot == tt.ttv_ot
    @test all(eltype(core) == ComplexF64 for core in tt_c.ttv_vec)

    # Test Base.complex for TToperator
    tto_c = TToperator{ComplexF64, 2}(tto.N, [complex.(core) for core in tto.tto_vec], tto.tto_dims, tto.tto_rks, tto.tto_ot)
    @test typeof(tto_c) == TToperator{ComplexF64, 2}
    @test tto_c.N == tto.N
    @test tto_c.tto_dims == tto.tto_dims
    @test tto_c.tto_rks == tto.tto_rks
    @test tto_c.tto_ot == tto.tto_ot
    @test all(eltype(core) == ComplexF64 for core in tto_c.tto_vec)
end

