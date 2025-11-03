using Test
using TensorTrainNumerics
import TensorTrainNumerics: rand_orthogonal
using LinearAlgebra

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
    @test tt_concat.ttv_rks == vcat(tt1.ttv_rks[1:(end - 1)], tt2.ttv_rks)
    @test tt_concat.ttv_ot == vcat(tt1.ttv_ot, tt2.ttv_ot)

    # Test concatenate function for TToperator
    dims_op1 = (2, 2)
    tto1 = rand_tto(dims_op1, 3)
    dims_op2 = (2, 2)
    tto2 = rand_tto(dims_op2, 3)
    tto_concat = concatenate(tto1, tto2)
    @test tto_concat.N == tto1.N + tto2.N
    @test tto_concat.tto_dims == (tto1.tto_dims..., tto2.tto_dims...)
    @test tto_concat.tto_rks == vcat(tto1.tto_rks[1:(end - 1)], tto2.tto_rks)
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
    @test tt_concat.ttv_rks == vcat(tt1.ttv_rks[1:(end - 1)], tt2.ttv_rks)
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
    @test tt_concat.ttv_rks == vcat(tt1.ttv_rks[1:(end - 1)], tt2.ttv_rks)
    @test tt_concat.ttv_ot == vcat(tt1.ttv_ot, tt2.ttv_ot)

    # Test concatenate function for TToperator
    dims_op1 = (2, 2)
    tto1 = rand_tto(dims_op1, 3)
    dims_op2 = (2, 2)
    tto2 = rand_tto(dims_op2, 3)
    tto_concat = concatenate(tto1, tto2)
    @test tto_concat.N == tto1.N + tto2.N
    @test tto_concat.tto_dims == (tto1.tto_dims..., tto2.tto_dims...)
    @test tto_concat.tto_rks == vcat(tto1.tto_rks[1:(end - 1)], tto2.tto_rks)
    @test tto_concat.tto_ot == vcat(tto1.tto_ot, tto2.tto_ot)
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


@testset "Base.eltype for TTvector" begin
    # Test with Float64
    N = 2
    vec = [randn(2, 1, 2), randn(2, 2, 1)]
    dims = (2, 2)
    rks = [1, 2, 1]
    ot = [0, 0]
    tt_float = TTvector{Float64, 2}(N, vec, dims, rks, ot)
    @test eltype(tt_float) == Float64

    # Test with Int
    vec_int = [rand(Int, 2, 1, 2), rand(Int, 2, 2, 1)]
    tt_int = TTvector{Int, 2}(N, vec_int, dims, rks, ot)
    @test eltype(tt_int) == Int

    # Test with ComplexF64
    vec_c = [randn(ComplexF64, 2, 1, 2), randn(ComplexF64, 2, 2, 1)]
    tt_c = TTvector{ComplexF64, 2}(N, vec_c, dims, rks, ot)
    @test eltype(tt_c) == ComplexF64
end

@testset "ttv decomp" begin

    # test if ttv matches https://scfp.jinguo-group.science/chap2-linalg/tensor-network.html

    tensor = ones(Float64, fill(2, 10)...)

    struct MPS{T}
        tensors::Vector{Array{T, 3}}
    end

    function truncated_svd(current_tensor::AbstractArray, largest_rank::Int, atol)
        U, S, V = svd(current_tensor)
        r = min(largest_rank, sum(S .> atol))
        S_truncated = Diagonal(S[1:r])
        U_truncated = U[:, 1:r]
        V_truncated = V[:, 1:r]
        return U_truncated, S_truncated, V_truncated, r
    end

    function tensor_train_decomposition(tensor::AbstractArray, largest_rank::Int; atol = 1.0e-6)
        dims = size(tensor)
        n = length(dims)

        # Initialize the cores of the TT decomposition
        tensors = Array{Float64, 3}[]

        # Reshape the tensor into a matrix
        rpre = 1
        current_tensor = reshape(tensor, dims[1], :)

        # Perform SVD for each core except the last one
        for i in 1:(n - 1)
            # Truncate to the specified rank
            U_truncated, S_truncated, V_truncated, r = truncated_svd(current_tensor, largest_rank, atol)

            # Middle cores have shape (largest_rank, dims[i], r)
            push!(tensors, reshape(U_truncated, (rpre, dims[i], r)))

            # Prepare the tensor for the next iteration
            current_tensor = S_truncated * V_truncated'

            # Reshape for the next SVD
            current_tensor = reshape(current_tensor, r * dims[i + 1], :)
            rpre = r
        end

        # Add the last core
        push!(tensors, reshape(current_tensor, (rpre, dims[n], 1)))

        return MPS(tensors)
    end

    A = tensor_train_decomposition(tensor, 10)


    B = ttv_decomp(tensor)

    size(A.tensors) == size(B.ttv_vec)

    for i in 1:length(B.ttv_vec)
        @test isapprox(abs.(reshape(reverse(A.tensors)[i], 2, 1, 1)), abs.((B.ttv_vec)[i]), rtol = 1.0e-10)
    end

    tensor_rand = rand(Float64, fill(2, 10)...)

    A_rand = tensor_train_decomposition(tensor, 10)


    B_rand = ttv_decomp(tensor)

    size(A_rand.tensors) == size(B_rand.ttv_vec)

    for i in 1:length(B_rand.ttv_vec)
        @test isapprox(abs.(reshape(reverse(A_rand.tensors)[i], 2, 1, 1)), abs.((B_rand.ttv_vec)[i]), rtol = 1.0e-10)
    end

end

@testset "random TT" begin

    dims = (1, 2, 2, 1)

    max_r = 2

    A = rand_tt(dims, max_r; normalise = true, orthogonal = true)

    @test maximum(A.ttv_rks) == max_r
    @test A.N == length(dims)
    @test isapprox(norm(A), 1.0, atol = 1.0e-10)
    @test A.ttv_ot == [0, 0, 0, 0]
end

@testset "Copy" begin

    dims = (1, 2, 3, 4, 5, 1)
    A = rand_tt(dims, 5; normalise = true, orthogonal = true)
    B = copy(A)
    @test A.ttv_dims == B.ttv_dims
    @test A.ttv_ot == B.ttv_ot
    @test A.ttv_rks == B.ttv_rks
    @test A.ttv_vec == B.ttv_vec

end

@testset "rand_orthogonal behavior" begin
    using Random
    rng_seed = 12345
    Random.seed!(rng_seed)

    n = 5; m = 5
    A = rand_orthogonal(n, m)
    @test size(A) == (n, m)
    @test eltype(A) == Float64
    @test isapprox(Matrix(A' * A), Matrix{Float64}(I, m, m); atol = 1.0e-12)
    @test isapprox(Matrix(A * A'), Matrix{Float64}(I, n, n); atol = 1.0e-12)

    n = 7; m = 3
    Random.seed!(rng_seed)
    A = rand_orthogonal(n, m)
    @test size(A) == (n, m)
    @test isapprox(Matrix(A' * A), Matrix{Float64}(I, m, m); atol = 1.0e-12)

    n = 3; m = 7
    Random.seed!(rng_seed)
    A = rand_orthogonal(n, m)
    @test size(A) == (n, m)
    @test isapprox(Matrix(A * A'), Matrix{Float64}(I, n, n); atol = 1.0e-12)

    n = 6; m = 4
    B = rand_orthogonal(n, m; T = Float32)
    @test size(B) == (n, m)
    @test eltype(B) == Float32
    @test isapprox(Matrix(B' * B), Matrix{Float32}(I, m, m); atol = 1.0e-6)
end
