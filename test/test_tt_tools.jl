using Test
using TensorTrainNumerics
import TensorTrainNumerics: rand_orthogonal, tto_to_ttv, ttv_to_tto
using LinearAlgebra
using Random

@testset "Complex TT conversion is idempotent and owns metadata" begin
    x = rand_tt((2, 3), [1, 2, 1])
    xc = complex(x)
    xcc = complex(xc)
    @test xcc isa TTVector{ComplexF64}
    @test ttv_to_tensor(xcc) == complex.(ttv_to_tensor(x))
    @test xc.ttv_rks !== x.ttv_rks
    @test xc.orthogonality !== x.orthogonality

    A = rand_tto((2, 3), 2)
    Ac = complex(A)
    Acc = complex(Ac)
    @test Acc isa TTOperator{ComplexF64}
    @test tto_to_tensor(Acc) == complex.(tto_to_tensor(A))
    @test Ac.tto_rks !== A.tto_rks
    @test Ac.orthogonality !== A.orthogonality
end

@testset "TT constructors and properties" begin
    # Test TTVector constructor
    N = 3
    vec = [randn(2, 1, 2), randn(2, 2, 2), randn(2, 2, 1)]
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    tt = TTVector{Float64, 3}(vec, dims, rks)
    @test nsites(tt) == N
    @test tt.ttv_vec == vec
    @test tt.ttv_dims == dims
    @test tt.ttv_rks == rks
    @test tt.orthogonality == [1, nsites(tt)]

    # Test TTOperator constructor
    N = 3
    vec = [randn(2, 2, 1, 2), randn(2, 2, 2, 2), randn(2, 2, 2, 1)]
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    tto = TTOperator{Float64, 3}(vec, dims, rks)
    @test nsites(tto) == N
    @test tto.tto_vec == vec
    @test tto.tto_dims == dims
    @test tto.tto_rks == rks
    @test tto.orthogonality == [1, nsites(tto)]

end

@testset "TT constructors and properties" begin
    # Test TTVector constructor
    N = 3
    vec = [randn(2, 1, 2), randn(2, 2, 2), randn(2, 2, 1)]
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    tt = TTVector{Float64, 3}(vec, dims, rks)
    @test nsites(tt) == N
    @test tt.ttv_vec == vec
    @test tt.ttv_dims == dims
    @test tt.ttv_rks == rks
    @test tt.orthogonality == [1, nsites(tt)]

    # Test TTOperator constructor
    N = 3
    vec = [randn(2, 2, 1, 2), randn(2, 2, 2, 2), randn(2, 2, 2, 1)]
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    tto = TTOperator{Float64, 3}(vec, dims, rks)
    @test nsites(tto) == N
    @test tto.tto_vec == vec
    @test tto.tto_dims == dims
    @test tto.tto_rks == rks
    @test tto.orthogonality == [1, nsites(tto)]

end


@testset "TT constructors and properties" begin
    # Test TTVector constructor
    N = 3
    vec = [randn(2, 1, 2), randn(2, 2, 2), randn(2, 2, 1)]
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    tt = TTVector{Float64, 3}(vec, dims, rks)
    @test nsites(tt) == N
    @test tt.ttv_vec == vec
    @test tt.ttv_dims == dims
    @test tt.ttv_rks == rks
    @test tt.orthogonality == [1, nsites(tt)]

    # Test TTOperator constructor
    N = 3
    vec = [randn(2, 2, 1, 2), randn(2, 2, 2, 2), randn(2, 2, 2, 1)]
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    tto = TTOperator{Float64, 3}(vec, dims, rks)
    @test nsites(tto) == N
    @test tto.tto_vec == vec
    @test tto.tto_dims == dims
    @test tto.tto_rks == rks
    @test tto.orthogonality == [1, nsites(tto)]

end

@testset "TTVector and TTOperator functions" begin
    dims1 = (2, 2)
    rks1 = [1, 2, 1]
    tt1 = rand_tt(dims1, rks1)
    dims2 = (2, 2)
    rks2 = [1, 2, 1]
    tt2 = rand_tt(dims2, rks2)
    tt_concat = concatenate(tt1, tt2)
    @test nsites(tt_concat) == nsites(tt1) + nsites(tt2)
    @test tt_concat.ttv_dims == (tt1.ttv_dims..., tt2.ttv_dims...)
    @test tt_concat.ttv_rks == vcat(tt1.ttv_rks[1:(end - 1)], tt2.ttv_rks)
    @test tt_concat.orthogonality == [tt1.orthogonality[1], nsites(tt1) + tt2.orthogonality[2]]

    # Test concatenate function for TTOperator
    dims_op1 = (2, 2)
    tto1 = rand_tto(dims_op1, 3)
    dims_op2 = (2, 2)
    tto2 = rand_tto(dims_op2, 3)
    tto_concat = concatenate(tto1, tto2)
    @test nsites(tto_concat) == nsites(tto1) + nsites(tto2)
    @test tto_concat.tto_dims == (tto1.tto_dims..., tto2.tto_dims...)
    @test tto_concat.tto_rks == vcat(tto1.tto_rks[1:(end - 1)], tto2.tto_rks)
    @test tto_concat.orthogonality == [tto1.orthogonality[1], nsites(tto1) + tto2.orthogonality[2]]

    # Test concatenate function
    dims1 = (2, 2)
    rks1 = [1, 2, 1]
    tt1 = rand_tt(dims1, rks1)
    dims2 = (2, 2)
    rks2 = [1, 2, 1]
    tt2 = rand_tt(dims2, rks2)
    tt_concat = concatenate(tt1, tt2)
    @test nsites(tt_concat) == nsites(tt1) + nsites(tt2)
    @test tt_concat.ttv_dims == (tt1.ttv_dims..., tt2.ttv_dims...)
    @test tt_concat.ttv_rks == vcat(tt1.ttv_rks[1:(end - 1)], tt2.ttv_rks)
    @test tt_concat.orthogonality == [tt1.orthogonality[1], nsites(tt1) + tt2.orthogonality[2]]
end

@testset "TT constructors and properties" begin
    # Test TTVector constructor
    N = 3
    vec = [randn(2, 1, 2), randn(2, 2, 2), randn(2, 2, 1)]
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    tt = TTVector{Float64, 3}(vec, dims, rks)
    @test nsites(tt) == N
    @test tt.ttv_vec == vec
    @test tt.ttv_dims == dims
    @test tt.ttv_rks == rks
    @test tt.orthogonality == [1, nsites(tt)]

    # Test TTOperator constructor
    N = 3
    vec = [randn(2, 2, 1, 2), randn(2, 2, 2, 2), randn(2, 2, 2, 1)]
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    tto = TTOperator{Float64, 3}(vec, dims, rks)
    @test nsites(tto) == N
    @test tto.tto_vec == vec
    @test tto.tto_dims == dims
    @test tto.tto_rks == rks
    @test tto.orthogonality == [1, nsites(tto)]

end

@testset "TTVector and TTOperator functions" begin
    dims1 = (2, 2)
    rks1 = [1, 2, 1]
    tt1 = rand_tt(dims1, rks1)
    dims2 = (2, 2)
    rks2 = [1, 2, 1]
    tt2 = rand_tt(dims2, rks2)
    tt_concat = concatenate(tt1, tt2)
    @test nsites(tt_concat) == nsites(tt1) + nsites(tt2)
    @test tt_concat.ttv_dims == (tt1.ttv_dims..., tt2.ttv_dims...)
    @test tt_concat.ttv_rks == vcat(tt1.ttv_rks[1:(end - 1)], tt2.ttv_rks)
    @test tt_concat.orthogonality == [tt1.orthogonality[1], nsites(tt1) + tt2.orthogonality[2]]

    # Test concatenate function for TTOperator
    dims_op1 = (2, 2)
    tto1 = rand_tto(dims_op1, 3)
    dims_op2 = (2, 2)
    tto2 = rand_tto(dims_op2, 3)
    tto_concat = concatenate(tto1, tto2)
    @test nsites(tto_concat) == nsites(tto1) + nsites(tto2)
    @test tto_concat.tto_dims == (tto1.tto_dims..., tto2.tto_dims...)
    @test tto_concat.tto_rks == vcat(tto1.tto_rks[1:(end - 1)], tto2.tto_rks)
    @test tto_concat.orthogonality == [tto1.orthogonality[1], nsites(tto1) + tto2.orthogonality[2]]
end


@testset "Base.eltype and Base.complex for TTVector and TTOperator" begin
    # Test eltype for TTVector
    N = 2
    vec = [randn(2, 1, 2), randn(2, 2, 1)]
    dims = (2, 2)
    rks = [1, 2, 1]
    tt = TTVector{Float64, 2}(vec, dims, rks)
    @test eltype(tt) == Float64

    # Test eltype for TTOperator
    op_vec = [randn(2, 2, 1, 2), randn(2, 2, 2, 1)]
    op_dims = (2, 2)
    op_rks = [1, 2, 1]
    tto = TTOperator{Float64, 2}(op_vec, op_dims, op_rks)
    @test eltype(tto) == Float64

    # Test Base.complex for TTVector
    tt_c = TTVector{ComplexF64, 2}([complex.(core) for core in tt.ttv_vec], tt.ttv_dims, tt.ttv_rks; orthogonality = tt.orthogonality)
    @test typeof(tt_c) == TTVector{ComplexF64, 2}
    @test nsites(tt_c) == nsites(tt)
    @test tt_c.ttv_dims == tt.ttv_dims
    @test tt_c.ttv_rks == tt.ttv_rks
    @test tt_c.orthogonality == tt.orthogonality
    @test all(eltype(core) == ComplexF64 for core in tt_c.ttv_vec)

    # Test Base.complex for TTOperator
    tto_c = TTOperator{ComplexF64, 2}([complex.(core) for core in tto.tto_vec], tto.tto_dims, tto.tto_rks; orthogonality = tto.orthogonality)
    @test typeof(tto_c) == TTOperator{ComplexF64, 2}
    @test nsites(tto_c) == nsites(tto)
    @test tto_c.tto_dims == tto.tto_dims
    @test tto_c.tto_rks == tto.tto_rks
    @test tto_c.orthogonality == tto.orthogonality
    @test all(eltype(core) == ComplexF64 for core in tto_c.tto_vec)
end


@testset "Base.eltype for TTVector" begin
    # Test with Float64
    N = 2
    vec = [randn(2, 1, 2), randn(2, 2, 1)]
    dims = (2, 2)
    rks = [1, 2, 1]
    tt_float = TTVector{Float64, 2}(vec, dims, rks)
    @test eltype(tt_float) == Float64

    # Test with Int
    vec_int = [rand(Int, 2, 1, 2), rand(Int, 2, 2, 1)]
    tt_int = TTVector{Int, 2}(vec_int, dims, rks)
    @test eltype(tt_int) == Int

    # Test with ComplexF64
    vec_c = [randn(ComplexF64, 2, 1, 2), randn(ComplexF64, 2, 2, 1)]
    tt_c = TTVector{ComplexF64, 2}(vec_c, dims, rks)
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

            current_tensor = reshape(current_tensor, r * dims[i + 1], :)
            rpre = r
        end

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

    tensor_centered = randn(2, 3, 2)
    tt_centered = ttv_decomp(tensor_centered; index = 2)
    @test tt_centered.orthogonality == [2, 2]
    @test isapprox(ttv_to_tensor(tt_centered), tensor_centered; atol = 1.0e-10)

end

@testset "tto_decomp" begin

    @testset "tto_to_tensor" begin
        N = 3
        dims = (2, 2, 2)
        rks = [1, 2, 2, 1]
        tto_vec = [randn(2, 2, 1, 2), randn(2, 2, 2, 2), randn(2, 2, 2, 1)]
        tto = TTOperator{Float64, N}(tto_vec, dims, rks)

        M = tto_to_tensor(tto)
        tto2 = tto_decomp(M)

        @test nsites(tto2) == N
        @test tto2.tto_dims == dims
        @test isapprox(tto_to_tensor(tto2), M; rtol = 1.0e-10)
    end

    @testset "reproduces dense non-symmetric matvec" begin
        # Guards the (x_1,y_1,...,x_d,y_d) interleave on a non-symmetric operator.
        dims = (2, 2)
        n = prod(dims)
        A_mat = randn(n, n)
        A_tt = tto_decomp(reshape(A_mat, dims..., dims...))

        @test isapprox(reshape(tto_to_tensor(A_tt), n, n), A_mat; rtol = 1.0e-10)

        v = randn(n)
        v_tt = ttv_decomp(reshape(v, dims...))
        Av = ttv_to_tensor(A_tt * v_tt)
        @test isapprox(vec(Av), A_mat * v; rtol = 1.0e-10)
    end

    @testset "non-uniform dimensions" begin
        dims = (2, 3)
        n = prod(dims)
        A_mat = randn(n, n)
        A_tt = tto_decomp(reshape(A_mat, dims..., dims...))

        @test A_tt.tto_dims == dims
        @test isapprox(reshape(tto_to_tensor(A_tt), n, n), A_mat; rtol = 1.0e-10)
    end

    @testset "preserves eltype" begin
        @test eltype(tto_to_tensor(tto_decomp(randn(Float32, 2, 2, 2, 2)))) == Float32
    end

end

@testset "random TT" begin

    dims = (1, 2, 2, 1)

    max_r = 2

    A = rand_tt(dims, max_r; normalize = true, orthogonal = true)

    @test maximum(A.ttv_rks) == max_r
    @test nsites(A) == length(dims)
    @test isapprox(norm(A), 1.0, atol = 1.0e-10)
    @test A.orthogonality == [1, 4]
end

@testset "Copy" begin

    dims = (1, 2, 3, 4, 5, 1)
    A = rand_tt(dims, 5; normalize = true, orthogonal = true)
    B = copy(A)
    @test A.ttv_dims == B.ttv_dims
    @test A.orthogonality == B.orthogonality
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

@testset "tt truncation behavior" begin
    @testset "tt_compress! reduces rank to max_bond and updates core shapes" begin
        N = 3
        dims = (2, 2, 2)
        rks = [1, 4, 4, 1]
        vec = Array{Array{Float64, 3}}(undef, N)
        vec[1] = randn(2, 1, 4)
        vec[2] = randn(2, 4, 4)
        vec[3] = randn(2, 4, 1)
        tt = TTVector{Float64, 3}(vec, dims, rks)

        y = tt_compress!(tt, 2)
        @test y === tt
        @test maximum(tt.ttv_rks) ≤ 2
        for i in 1:N
            @test size(tt.ttv_vec[i]) == (dims[i], tt.ttv_rks[i], tt.ttv_rks[i + 1])
        end
    end

    @testset "exact rank-1 vector rounds to rank 1" begin
        N = 2
        n1 = 2; n2 = 2
        rks = [1, 2, 1]
        u = [1.2, -0.5]
        v = [0.7, 0.3]
        p = [2.0, 3.0]
        q = [4.0, 5.0]

        core1 = zeros(Float64, n1, 1, 2) # shape (n1, r0=1, r1=2)
        core2 = zeros(Float64, n2, 2, 1) # shape (n2, r1=2, r2=1)

        for s1 in 1:n1, γ in 1:2
            core1[s1, 1, γ] = p[γ] * u[s1]
        end
        for s2 in 1:n2, γ in 1:2
            core2[s2, γ, 1] = q[γ] * v[s2]
        end

        tt = TTVector{Float64, 2}([core1, core2], (n1, n2), rks)
        expected = (p' * q) .* (u * v')

        # rank cap finds the exact rank-1 structure
        t1 = tt_compress!(copy(tt), 1)
        @test t1.ttv_rks[2] == 1
        @test ttv_to_tensor(t1) ≈ expected

        # tolerance-based rounding detects the zero singular value on its own
        t2 = tt_round!(copy(tt); trunc_tol = 1.0e-13)
        @test t2.ttv_rks[2] == 1
        @test ttv_to_tensor(t2) ≈ expected
    end
end

@testset "tt_compress! behavior" begin
    @testset "no-op for large max_bond" begin
        N = 3
        dims = (2, 2, 2)
        rks = [1, 2, 2, 1]
        vec = Array{Array{Float64, 3}}(undef, N)
        vec[1] = randn(dims[1], rks[1], rks[2])
        vec[2] = randn(dims[2], rks[2], rks[3])
        vec[3] = randn(dims[3], rks[3], rks[4])
        tt = TTVector{Float64, N}(vec, dims, rks)

        before_rks = copy(tt.ttv_rks)
        y = TensorTrainNumerics.tt_compress!(tt, 10; sweeps = 1)
        @test y === tt
        @test tt.ttv_rks == before_rks
    end

    @testset "reduces rank to max_bond and updates shapes" begin
        N = 4
        dims = (2, 2, 2, 2)
        rks = [1, 4, 4, 4, 1]
        vec = Array{Array{Float64, 3}}(undef, N)
        for i in 1:N
            vec[i] = randn(dims[i], rks[i], rks[i + 1])
        end
        tt = TTVector{Float64, N}(vec, dims, rks)

        y = TensorTrainNumerics.tt_compress!(tt, 2; sweeps = 1)
        @test y === tt
        @test maximum(tt.ttv_rks) ≤ 2

        for i in 1:N
            @test size(tt.ttv_vec[i], 1) == dims[i]
            @test size(tt.ttv_vec[i], 2) == tt.ttv_rks[i]
            @test size(tt.ttv_vec[i], 3) == tt.ttv_rks[i + 1]
        end
    end

    @testset "sweeps validation" begin
        N = 3
        dims = (2, 2, 2)
        rks = [1, 2, 2, 1]
        vec = [randn(dims[1], rks[1], rks[2]), randn(dims[2], rks[2], rks[3]), randn(dims[3], rks[3], rks[4])]
        tt = TTVector{Float64, N}(vec, dims, rks)

        @test_throws "`sweeps` must be ≥ 1; got 0" TensorTrainNumerics.tt_compress!(tt, 2; sweeps = 0)
    end

    @testset "multiple sweeps return and type" begin
        N = 3
        dims = (2, 2, 2)
        rks = [1, 3, 3, 1]
        vec = [randn(dims[1], rks[1], rks[2]), randn(dims[2], rks[2], rks[3]), randn(dims[3], rks[3], rks[4])]
        tt = TTVector{Float64, N}(vec, dims, rks)

        y = TensorTrainNumerics.tt_compress!(tt, 3; sweeps = 2, trunc_tol = 0.0)
        @test typeof(y) == typeof(tt)
        @test y === tt
    end

    @testset "verbosity 2 logs each rounding pass" begin
        N = 3
        dims = (2, 2, 2)
        rks = [1, 2, 2, 1]
        vec = [randn(dims[1], rks[1], rks[2]), randn(dims[2], rks[2], rks[3]), randn(dims[3], rks[3], rks[4])]
        tt = TTVector{Float64, N}(vec, dims, rks)

        @test_logs (:info, r"TT compress: sweep 1") TensorTrainNumerics.tt_compress!(tt, 2; verbosity = 2)
    end
end

@testset "Base.eltype for TTVector" begin
    N = 3
    vec = [randn(2, 1, 2), randn(2, 2, 2), randn(2, 2, 1)]
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    tt_float = TTVector{Float64, 3}(vec, dims, rks)
    @test eltype(tt_float) == Float64

    vec_int = [rand(Int, 2, 1, 2), rand(Int, 2, 2, 2), rand(Int, 2, 2, 1)]
    tt_int = TTVector{Int, 3}(vec_int, dims, rks)
    @test eltype(tt_int) == Int

    vec_complex = [randn(ComplexF64, 2, 1, 2), randn(ComplexF64, 2, 2, 2), randn(ComplexF64, 2, 2, 1)]
    tt_complex = TTVector{ComplexF64, 3}(vec_complex, dims, rks)
    @test eltype(tt_complex) == ComplexF64

    vec_complex32 = [randn(Complex{Float32}, 2, 1, 2), randn(Complex{Float32}, 2, 2, 2), randn(Complex{Float32}, 2, 2, 1)]
    tt_complex32 = TTVector{Complex{Float32}, 3}(vec_complex32, dims, rks)
    @test eltype(tt_complex32) == Complex{Float32}
end

@testset "derived TT values own their mutable storage" begin
    for T in (Float32, Float64, ComplexF64)
        @testset "noisy copy: $T" begin
            for ε in (0.0, 1.0e-3)
                x = orthogonalize(rand_tt(T, (2, 2, 2), [1, 2, 2, 1]); i = 2)
                before = copy(x)
                dense = ttv_to_tensor(x)
                y = rand_tt(x; ε)
                y.ttv_vec[1][1, 1, 1] += one(T)
                tt_round!(y; max_bond = 1)
                @test x.ttv_vec == before.ttv_vec
                @test x.ttv_rks == before.ttv_rks
                @test x.orthogonality == before.orthogonality
                @test norm(x) ≈ norm(dense)
            end
        end

        @testset "operator to vector: $T" begin
            A = rand_tto((2, 2, 2), 2; T)
            dense = tto_to_tensor(A)
            ranks = copy(A.tto_rks)
            flags = copy(A.orthogonality)
            x = tto_to_ttv(A)
            x.ttv_vec[1][1, 1, 1] += one(T)
            x.orthogonality[1] = 2
            @test tto_to_tensor(A) == dense
            @test A.orthogonality == flags
            tt_round!(x; max_bond = 1)
            @test A.tto_rks == ranks
            @test A.orthogonality == flags
            @test tto_to_tensor(A) == dense
        end

        @testset "vector to operator: $T" begin
            x = rand_tt(T, (4, 4, 4), [1, 2, 2, 1])
            before = copy(x)
            A = ttv_to_tto(x)
            A.tto_vec[1][1, 1, 1, 1] += one(T)
            A.orthogonality[1] = 2
            @test x.ttv_vec == before.ttv_vec
            @test x.orthogonality == before.orthogonality

            A = ttv_to_tto(x)
            dense = tto_to_tensor(A)
            tt_round!(x; max_bond = 1)
            @test A.tto_rks == before.ttv_rks
            @test A.orthogonality == before.orthogonality
            @test tto_to_tensor(A) == dense
        end
    end
end

@testset "rand_tt with TTVector noise addition" begin
    N = 3
    vec = [randn(2, 1, 2), randn(2, 2, 2), randn(2, 2, 1)]
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    tt = TTVector{Float64, 3}(vec, dims, rks)

    tt_noisy = rand_tt(tt)

    @test nsites(tt_noisy) == nsites(tt)
    @test tt_noisy.ttv_dims == tt.ttv_dims
    @test tt_noisy.ttv_rks == tt.ttv_rks
    @test tt_noisy.orthogonality == [1, N]
    @test length(tt_noisy.ttv_vec) == length(tt.ttv_vec)

    @test !all(isapprox(tt_noisy.ttv_vec[i], tt.ttv_vec[i]) for i in eachindex(tt.ttv_vec))

    ε_custom = convert(Float64, 0.1)
    tt_noisy_custom = rand_tt(tt; ε = ε_custom)

    @test nsites(tt_noisy_custom) == nsites(tt)
    @test tt_noisy_custom.ttv_dims == tt.ttv_dims
    @test tt_noisy_custom.ttv_rks == tt.ttv_rks
    @test tt_noisy_custom.orthogonality == [1, N]

    tt_copy = rand_tt(tt; ε = convert(Float64, 0.0))

    @test all(isapprox(tt_copy.ttv_vec[i], tt.ttv_vec[i]) for i in eachindex(tt.ttv_vec))

    vec_c = [randn(ComplexF64, 2, 1, 2), randn(ComplexF64, 2, 2, 2), randn(ComplexF64, 2, 2, 1)]
    tt_c = TTVector{ComplexF64, 3}(vec_c, dims, rks)

    tt_noisy_c = rand_tt(tt_c; ε = convert(ComplexF64, 1.0e-3))

    @test eltype(tt_noisy_c.ttv_vec[1]) == ComplexF64
    @test tt_noisy_c.ttv_dims == tt_c.ttv_dims
    @test tt_noisy_c.ttv_rks == tt_c.ttv_rks

    tt_noisy1 = rand_tt(tt)
    tt_noisy2 = rand_tt(tt)

    @test !all(isapprox(tt_noisy1.ttv_vec[i], tt_noisy2.ttv_vec[i]) for i in eachindex(tt.ttv_vec))
end

@testset "tto_to_ttv conversion" begin
    @testset "basic conversion preserves structure" begin
        N = 3
        dims = (2, 3, 2)
        rks = [1, 2, 3, 1]
        tto_vec = [randn(2, 2, 1, 2), randn(3, 3, 2, 3), randn(2, 2, 3, 1)]
        tto = TTOperator{Float64, 3}(tto_vec, dims, rks)

        ttv = tto_to_ttv(tto)

        @test nsites(ttv) == nsites(tto)
        @test ttv.ttv_dims == tto.tto_dims .^ 2
        @test ttv.ttv_rks == tto.tto_rks
        @test ttv.orthogonality == tto.orthogonality
        @test length(ttv.ttv_vec) == length(tto.tto_vec)
    end

    @testset "core reshaping correctness" begin
        N = 2
        dims = (2, 3)
        rks = [1, 2, 1]
        tto_vec = [randn(2, 2, 1, 2), randn(3, 3, 2, 1)]
        tto = TTOperator{Float64, 2}(tto_vec, dims, rks)

        ttv = tto_to_ttv(tto)

        for i in 1:N
            @test size(ttv.ttv_vec[i], 1) == dims[i]^2
            @test size(ttv.ttv_vec[i], 2) == rks[i]
            @test size(ttv.ttv_vec[i], 3) == rks[i + 1]
        end
    end

    @testset "data preservation through reshape" begin
        N = 2
        dims = (2, 2)
        rks = [1, 2, 1]
        tto_vec = [randn(2, 2, 1, 2), randn(2, 2, 2, 1)]
        tto = TTOperator{Float64, 2}(tto_vec, dims, rks)

        ttv = tto_to_ttv(tto)

        for i in 1:N
            tto_flat = reshape(tto.tto_vec[i], dims[i]^2, rks[i], rks[i + 1])
            @test isapprox(ttv.ttv_vec[i], tto_flat)
        end
    end

    @testset "eltype preservation" begin
        N = 2
        dims = (2, 2)
        rks = [1, 2, 1]
        tto_vec_f32 = [randn(Float32, 2, 2, 1, 2), randn(Float32, 2, 2, 2, 1)]
        tto_f32 = TTOperator{Float32, 2}(tto_vec_f32, dims, rks)

        ttv_f32 = tto_to_ttv(tto_f32)

        @test eltype(ttv_f32) == Float32
        @test all(eltype(core) == Float32 for core in ttv_f32.ttv_vec)
    end

    @testset "orthogonalization info preserved" begin
        N = 3
        dims = (2, 2, 2)
        rks = [1, 2, 2, 1]
        tto_vec = [randn(2, 2, 1, 2), randn(2, 2, 2, 2), randn(2, 2, 2, 1)]
        tto = TTOperator{Float64, 3}(tto_vec, dims, rks; orthogonality = (2, 2))

        ttv = tto_to_ttv(tto)

        @test ttv.orthogonality == tto.orthogonality
    end

    @testset "single core operator" begin
        N = 1
        dims = (3,)
        rks = [1, 1]
        tto_vec = [randn(3, 3, 1, 1)]
        tto = TTOperator{Float64, 1}(tto_vec, dims, rks)

        ttv = tto_to_ttv(tto)

        @test nsites(ttv) == 1
        @test ttv.ttv_dims == (9,)
        @test size(ttv.ttv_vec[1]) == (9, 1, 1)
    end

    @testset "large operator conversion" begin
        N = 4
        dims = (2, 2, 2, 2)
        rks = [1, 2, 3, 2, 1]
        tto_vec = [randn(2, 2, rks[i], rks[i + 1]) for i in 1:N]
        tto = TTOperator{Float64, 4}(tto_vec, dims, rks)

        ttv = tto_to_ttv(tto)

        @test nsites(ttv) == 4
        @test all(ttv.ttv_dims .== 4)
        @test ttv.ttv_rks == rks
    end
end

@testset "ttv_to_tto conversion" begin
    @testset "basic conversion preserves structure" begin
        N = 3
        dims = (4, 9, 16)
        rks = [1, 2, 3, 1]
        ttv_vec = [randn(4, 1, 2), randn(9, 2, 3), randn(16, 3, 1)]
        ttv = TTVector{Float64, 3}(ttv_vec, dims, rks)

        tto = ttv_to_tto(ttv)

        @test nsites(tto) == nsites(ttv)
        @test tto.tto_dims == (2, 3, 4)
        @test tto.tto_rks == ttv.ttv_rks
        @test tto.orthogonality == ttv.orthogonality
        @test length(tto.tto_vec) == length(ttv.ttv_vec)
    end

    @testset "core reshaping correctness" begin
        N = 2
        dims = (4, 9)
        rks = [1, 2, 1]
        ttv_vec = [randn(4, 1, 2), randn(9, 2, 1)]
        ttv = TTVector{Float64, 2}(ttv_vec, dims, rks)

        tto = ttv_to_tto(ttv)

        @test size(tto.tto_vec[1]) == (2, 2, 1, 2)
        @test size(tto.tto_vec[2]) == (3, 3, 2, 1)
    end

    @testset "data preservation through reshape" begin
        N = 2
        dims = (4, 4)
        rks = [1, 2, 1]
        ttv_vec = [randn(4, 1, 2), randn(4, 2, 1)]
        ttv = TTVector{Float64, 2}(ttv_vec, dims, rks)

        tto = ttv_to_tto(ttv)

        for i in 1:N
            ttv_flat = reshape(ttv.ttv_vec[i], isqrt(dims[i]), isqrt(dims[i]), rks[i], rks[i + 1])
            @test isapprox(tto.tto_vec[i], ttv_flat)
        end
    end

    @testset "eltype preservation" begin
        N = 2
        dims = (4, 4)
        rks = [1, 2, 1]
        ttv_vec_f32 = [randn(Float32, 4, 1, 2), randn(Float32, 4, 2, 1)]
        ttv_f32 = TTVector{Float32, 2}(ttv_vec_f32, dims, rks)

        tto_f32 = ttv_to_tto(ttv_f32)

        @test eltype(tto_f32) == Float32
        @test all(eltype(core) == Float32 for core in tto_f32.tto_vec)
    end

    @testset "orthogonalization info preserved" begin
        N = 3
        dims = (4, 4, 4)
        rks = [1, 2, 2, 1]
        ttv_vec = [randn(4, 1, 2), randn(4, 2, 2), randn(4, 2, 1)]
        ttv = TTVector{Float64, 3}(ttv_vec, dims, rks; orthogonality = (2, 2))

        tto = ttv_to_tto(ttv)

        @test tto.orthogonality == ttv.orthogonality
    end

    @testset "single core operator" begin
        N = 1
        dims = (9,)
        rks = [1, 1]
        ttv_vec = [randn(9, 1, 1)]
        ttv = TTVector{Float64, 1}(ttv_vec, dims, rks)

        tto = ttv_to_tto(ttv)

        @test nsites(tto) == 1
        @test tto.tto_dims == (3,)
        @test size(tto.tto_vec[1]) == (3, 3, 1, 1)
    end

    @testset "non-perfect square dimensions throw error" begin
        N = 2
        dims = (4, 5)
        rks = [1, 2, 1]
        ttv_vec = [randn(4, 1, 2), randn(5, 2, 1)]
        ttv = TTVector{Float64, 2}(ttv_vec, dims, rks)

        @test_throws AssertionError ttv_to_tto(ttv)
    end

    @testset "large operator conversion" begin
        N = 4
        dims = (4, 4, 4, 4)
        rks = [1, 2, 3, 2, 1]
        ttv_vec = [randn(4, rks[i], rks[i + 1]) for i in 1:N]
        ttv = TTVector{Float64, 4}(ttv_vec, dims, rks)

        tto = ttv_to_tto(ttv)

        @test nsites(tto) == 4
        @test tto.tto_dims == (2, 2, 2, 2)
        @test tto.tto_rks == rks
    end

    @testset "complex types" begin
        N = 2
        dims = (4, 9)
        rks = [1, 2, 1]
        ttv_vec = [randn(ComplexF64, 4, 1, 2), randn(ComplexF64, 9, 2, 1)]
        ttv = TTVector{ComplexF64, 2}(ttv_vec, dims, rks)

        tto = ttv_to_tto(ttv)

        @test eltype(tto) == ComplexF64
        @test tto.tto_dims == (2, 3)
        @test all(eltype(core) == ComplexF64 for core in tto.tto_vec)
    end

    @testset "round-trip conversion tto_to_ttv → ttv_to_tto" begin
        N = 2
        dims = (2, 3)
        rks = [1, 2, 1]
        tto_vec = [randn(2, 2, 1, 2), randn(3, 3, 2, 1)]
        tto_orig = TTOperator{Float64, 2}(tto_vec, dims, rks)

        ttv = tto_to_ttv(tto_orig)
        tto_back = ttv_to_tto(ttv)

        @test nsites(tto_back) == nsites(tto_orig)
        @test tto_back.tto_dims == tto_orig.tto_dims
        @test tto_back.tto_rks == tto_orig.tto_rks
        @test tto_back.orthogonality == tto_orig.orthogonality
        for i in 1:N
            @test isapprox(tto_back.tto_vec[i], tto_orig.tto_vec[i])
        end
    end
end

@testset "ttv_to_tensor roundtrip" begin
    tensor = rand(3, 4, 5)
    tt = ttv_decomp(tensor)
    @test isapprox(ttv_to_tensor(tt), tensor; atol = 1.0e-12)

    # Rank-1 trivial case
    v = rand(Float64, 4)
    core = reshape(v, 4, 1, 1)
    tt1 = TTVector{Float64, 1}([core], (4,), [1, 1])
    @test vec(ttv_to_tensor(tt1)) ≈ v
end

@testset "tto_to_tensor" begin
    # Build a known rank-1 TTOperator and verify contraction
    # tto_vec cores have layout (row_dim, col_dim, left_rank, right_rank)
    tto = rand_tto((2, 3, 4), 3)
    M = tto_to_tensor(tto)
    @test size(M) == (2, 3, 4, 2, 3, 4)

    # For a scalar (N=1) operator the result should be the 2N-dim tensor
    tto1 = rand_tto((3,), 1)
    M1 = tto_to_tensor(tto1)
    @test size(M1) == (3, 3)
end

@testset "r_and_d_to_rks" begin
    dims = (2, 3, 4)
    # Request ranks higher than the matrix rank bound
    rks_in = [1, 100, 100, 1]
    rks_out = r_and_d_to_rks(rks_in, dims)
    @test rks_out[1] == 1
    @test rks_out[end] == 1
    # rank[2] ≤ min(prod(dims[1:1]), prod(dims[2:3])) = min(2, 12) = 2
    @test rks_out[2] ≤ 2
    # rank[3] ≤ min(prod(dims[1:2]), prod(dims[3:3])) = min(6, 4) = 4
    @test rks_out[3] ≤ 4

    # Requested ranks below bounds are preserved
    rks_small = [1, 1, 1, 1]
    @test r_and_d_to_rks(rks_small, dims) == [1, 1, 1, 1]

    @test r_and_d_to_rks([5, 5, 5], (0, 2); rmax = 4) == [1, 2, 1]
    @test r_and_d_to_rks([5, 5, 5], (0, 0); rmax = 4) == [1, 4, 1]
end

@testset "increase_ranks preserves represented tensor and validates rank growth" begin
    dims = (2, 2, 2)
    tt = rand_tt(dims, [1, 1, 1, 1])
    dense = ttv_to_tensor(tt)

    enlarged = TensorTrainNumerics.increase_ranks(tt, 3; noise = 0.0)
    @test enlarged isa TTVector{Float64, 3}
    @test maximum(enlarged.ttv_rks) <= 3
    @test maximum(enlarged.ttv_rks) > maximum(tt.ttv_rks)
    @test ttv_to_tensor(enlarged) ≈ dense

    noisy = TensorTrainNumerics.increase_ranks(tt, 2; noise = 1.0e-3)
    @test noisy isa TTVector{Float64, 3}
    @test noisy.ttv_rks == [1, 2, 2, 1]
    @test any(!iszero, noisy.ttv_vec[1][:, :, 2])
    @test any(!iszero, noisy.ttv_vec[2][:, 2, 2])
    @test any(!iszero, noisy.ttv_vec[3][:, 2, :])

    @test_throws AssertionError TensorTrainNumerics.increase_ranks(tt, 1)
    @test_deprecated TensorTrainNumerics.tt_up_rks(tt, 3; ϵ_wn = 0.0)
end

@testset "concatenate rejects incompatible boundary ranks" begin
    tt1 = rand_tt((2, 2), [1, 2, 2])
    tt2 = rand_tt((2, 2), [1, 2, 1])
    @test_throws ArgumentError concatenate(tt1, tt2)

    A1 = TTOperator{Float64, 1}([randn(2, 2, 1, 2)], (2,), [1, 2])
    A2 = TTOperator{Float64, 1}([randn(2, 2, 1, 1)], (2,), [1, 1])
    @test_throws ArgumentError concatenate(A1, A2)
end

@testset "orthogonalize" begin
    dims = (2, 3, 4)
    tt = rand_tt(dims, [1, 2, 3, 1])
    original_tensor = ttv_to_tensor(tt)

    for center in 1:3
        orth = orthogonalize(tt; i = center)

        # Reconstruction is preserved
        @test isapprox(ttv_to_tensor(orth), original_tensor; atol = 1.0e-12)

        @test orth.orthogonality == [center, center]

        # Left-orthogonal cores (j < center): reshape to (r_l*d, r_r) has orthonormal columns
        for j in 1:(center - 1)
            G = orth.ttv_vec[j]          # (d, r_l, r_r)
            d, r_l, r_r = size(G)
            A = reshape(permutedims(G, (2, 1, 3)), r_l * d, r_r)
            @test isapprox(A' * A, I(r_r); atol = 1.0e-12)
        end

        # Right-orthogonal cores (j > center): reshape to (r_l, r_r*d) has orthonormal rows
        for j in (center + 1):3
            G = orth.ttv_vec[j]          # (d, r_l, r_r)
            d, r_l, r_r = size(G)
            A = reshape(permutedims(G, (2, 3, 1)), r_l, r_r * d)
            @test isapprox(A * A', I(r_l); atol = 1.0e-12)
        end
    end
end

@testset "entanglement_entropy" begin
    product_state = qtt_basis_vector(4, 1)
    @test entanglement_entropy(product_state) ≈ zeros(3)

    bell_tensor = zeros(Float64, 2, 2)
    bell_tensor[1, 1] = inv(sqrt(2))
    bell_tensor[2, 2] = inv(sqrt(2))
    bell_state = ttv_decomp(bell_tensor)
    @test entanglement_entropy(bell_state) ≈ [log(2)]
    @test entanglement_entropy(bell_state; base = 2) ≈ [1.0]

    ghz_tensor = zeros(ComplexF64, 2, 2, 2)
    ghz_tensor[1, 1, 1] = inv(sqrt(2))
    ghz_tensor[2, 2, 2] = im / sqrt(2)
    ghz_state = ttv_decomp(ghz_tensor)
    @test entanglement_entropy(ghz_state) ≈ fill(log(2), 2)
end


@testset "matricize" begin
    d = 3
    N = 2^d
    # qtt_basis_vector(d, pos) has exactly one nonzero entry
    for pos in [1, 3, 5, 8]
        tt = qtt_basis_vector(d, pos)
        v = matricize(tt, d)
        @test length(v) == N
        @test sum(abs2, v) ≈ 1.0
        @test count(!iszero, v) == 1
    end
end

@testset "Base.show for TTVector and TTOperator" begin
    tt = rand_tt((2, 3, 4), [1, 2, 3, 1])

    # Compact show: MPS{T}(N sites)
    s = sprint(show, tt)
    @test occursin("MPS", s)
    @test occursin("Float64", s)
    @test occursin("sites", s)

    # text/plain show: informational block
    s_plain = sprint(show, MIME("text/plain"), tt)
    @test occursin("MPS", s_plain)
    @test occursin("Physical dims", s_plain)
    @test occursin("Bond dims", s_plain)
    @test occursin("Orthogonality", s_plain)

    # Orthogonality strings
    orth = orthogonalize(tt; i = 2)
    s_orth = sprint(show, MIME("text/plain"), orth)
    @test occursin("center @ site 2", s_orth)

    tt_partial = TTVector{Float64, 3}(tt.ttv_vec, tt.ttv_dims, tt.ttv_rks; orthogonality = (2, 3))
    @test occursin("center within sites 2:3", sprint(show, MIME("text/plain"), tt_partial))

    tt_none = TTVector{Float64, 3}(tt.ttv_vec, tt.ttv_dims, tt.ttv_rks)
    @test occursin("none", sprint(show, MIME("text/plain"), tt_none))

    # TTOperator
    tto = rand_tto((2, 3), 2)
    s_op = sprint(show, tto)
    @test occursin("MPO", s_op)
    @test occursin("Float64", s_op)

    s_op_plain = sprint(show, MIME("text/plain"), tto)
    @test occursin("MPO", s_op_plain)
    @test occursin("Physical dims", s_op_plain)
    @test occursin("Bond dims", s_op_plain)
    @test occursin("Orthogonality", s_op_plain)

    # visualize still works (ASCII diagram, returns nothing)
    @test visualize(tt) === nothing
    @test visualize(tto) === nothing
end

@testset "ones_tt" begin
    o = TensorTrainNumerics.ones_tt(2, 3)
    @test ttv_to_tensor(o) ≈ ones(2, 2, 2)
    o2 = TensorTrainNumerics.ones_tt(Float64, (2, 3))
    @test ttv_to_tensor(o2) ≈ ones(2, 3)
end

@testset "tt_round!" begin
    d = 6
    y = qtt_sin(d) + qtt_cos(d)          # exact QTT rank 2: sin + cos is a shifted sine
    pad = 0.0 * rand_tt(y.ttv_dims, 6)
    x = y + pad                          # same vector, inflated ranks
    before = qtt_to_vector(x)
    @test maximum(x.ttv_rks) > 4
    r = tt_round!(x; trunc_tol = 1.0e-12)
    @test r === x
    @test maximum(x.ttv_rks) ≤ 2
    @test qtt_to_vector(x) ≈ before

    # max_bond cap
    Random.seed!(3)
    z = rand_tt((2, 2, 2, 2, 2, 2), 4; normalize = true)
    tt_round!(z; max_bond = 2)
    @test maximum(z.ttv_rks) ≤ 2

    # exact rounding with tol = 0 and no cap preserves the tensor
    w = rand_tt((2, 3, 2, 3), 3; normalize = true)
    before_w = ttv_to_tensor(w)
    tt_round!(w)
    @test ttv_to_tensor(w) ≈ before_w

    # non-mutating variant leaves the input untouched
    v = qtt_sin(d) + 0.0 * rand_tt(y.ttv_dims, 5)
    rks_before = copy(v.ttv_rks)
    v2 = tt_round(v; trunc_tol = 1.0e-12)
    @test maximum(v2.ttv_rks) ≤ 2
    @test v.ttv_rks == rks_before
end

@testset "tt_compress! and tt_round! mutate through the QTT wrapper" begin
    d = 6
    x = qtt_sin(d) + 0.0 * rand_tt(qtt_sin(d).ttv_dims, 5)
    q = QTTVector(x, 1, d, :serial)
    tt_compress!(q, 3)
    @test maximum(q.ttv_rks) ≤ 3
    tt_round!(q; trunc_tol = 1.0e-12)
    @test maximum(q.ttv_rks) ≤ 2
end

@testset "dense conversion layouts (non-uniform dims)" begin
    Random.seed!(5)
    dims = (2, 3, 4)
    x = rand_tt(dims, [1, 2, 3, 1])
    dense = ttv_to_tensor(x)
    for t in CartesianIndices(dense)
        v = x.ttv_vec[1][t[1], :, :] * x.ttv_vec[2][t[2], :, :] * x.ttv_vec[3][t[3], :, :]
        @test dense[t] ≈ v[1, 1]
    end

    A = rand_tto((2, 3), 2)
    T4 = tto_to_tensor(A)
    for t in CartesianIndices(T4)
        M = A.tto_vec[1][t[1], t[3], :, :] * A.tto_vec[2][t[2], t[4], :, :]
        @test T4[t] ≈ M[1, 1]
    end
end

@testset "matricize agrees with dense extraction" begin
    d = 5
    x = qtt_sin(d) + qtt_polynom([0.5, 1.0], d)
    dense = ttv_to_tensor(x)
    for core in (2, d)
        vals = matricize(x, core)
        @test length(vals) == 2^core
        ok = true
        for i in 1:(2^core)
            bits = reverse(digits(i - 1, base = 2, pad = core)) .+ 1
            idx = CartesianIndex(Tuple(vcat(bits, ones(Int, d - core))))
            ok &= isapprox(vals[i], dense[idx])
        end
        @test ok
    end
    @test matricize(x, d) ≈ qtt_to_vector(x)
end

@testset "trunc_tol is shared by tt_round! and tt_compress!" begin
    Random.seed!(2024)
    x = rand_tt((2, 2, 2, 2, 2, 2), 6; normalize = true)
    for ε in (1.0e-1, 1.0e-2)
        a = tt_round!(copy(x); trunc_tol = ε)
        b = tt_compress!(copy(x), typemax(Int); trunc_tol = ε)
        @test a.ttv_rks == b.ttv_rks
        @test norm(orthogonalize(a - x)) ≤ ε * norm(x) * (1 + 1.0e-8)
    end
    @test_throws MethodError tt_round!(copy(x); tol = 1.0e-2)
    @test_throws MethodError tt_compress!(copy(x), 4; truncerr = 1.0e-2)
    @test_throws MethodError tt_compress!(copy(x), 4; verbose = true)
end

@testset "tt_compress! verbosity" begin
    x = rand_tt((2, 2, 2, 2), 3)
    @test_logs tt_compress!(copy(x), 2)
    @test_logs (:info, "TT compress: sweep 1") (:info, "TT compress: sweep 2") tt_compress!(copy(x), 2; sweeps = 2, verbosity = 2)
end

@testset "deprecated names" begin
    for (old, new) in (
            :AbstractTTvector => AbstractTTVector, :AbstractTToperator => AbstractTTOperator,
            :TTvector => TTVector, :TToperator => TTOperator,
            :QTTvector => QTTVector, :QTToperator => QTTOperator,
        )
        @test Base.isdeprecated(TensorTrainNumerics, old)
        @test getglobal(TensorTrainNumerics, old) === new
    end
    x = rand_tt((2, 3, 2), 2)
    A = id_tto(3)
    @test nsites(x) == 3
    @test nsites(A) == 3
    @test (@test_deprecated x.N) == 3
    @test (@test_deprecated A.N) == 3
    y = @test_deprecated TTVector{Float64, 3}(3, x.ttv_vec, x.ttv_dims, x.ttv_rks, [1, 0, -1])
    @test y.ttv_vec === x.ttv_vec
    @test y.orthogonality == [2, 2]
    y = @test_deprecated TTVector(3, x.ttv_vec, x.ttv_dims, x.ttv_rks, [0, 0, 0])
    @test y isa TTVector{Float64, 3}
    B = @test_deprecated TTOperator{Float64, 3}(3, A.tto_vec, A.tto_dims, A.tto_rks, [1, 1, 1])
    @test B.tto_vec === A.tto_vec
    @test B.orthogonality == [3, 3]
    B = @test_deprecated TTOperator(3, A.tto_vec, A.tto_dims, A.tto_rks, [0, 0, 0])
    @test B isa TTOperator{Float64, 3}
    @test (@test_deprecated entanglemententropy(x)) == entanglement_entropy(x)
    z = @test_deprecated rand_tt((2, 2), 2; normalise = true)
    @test z isa TTVector{Float64, 2}
end

# The orthogonality interval of `x` if its cores have the recorded
# orthogonality, otherwise a description of the first core that does not.
function verified_orthogonality(x; atol = 1.0e-8)
    left, right = x.orthogonality
    for k in 1:nsites(x)
        core = x.ttv_vec[k]
        n, rl, rr = size(core)
        if k < left
            Q = reshape(permutedims(core, (2, 1, 3)), rl * n, rr)
            isapprox(Q' * Q, I; atol) || return "core $k is not left-orthogonal"
        elseif k > right
            Q = reshape(permutedims(core, (2, 1, 3)), rl, n * rr)
            isapprox(Q * Q', I; atol) || return "core $k is not right-orthogonal"
        end
    end
    return (left, right)
end

@testset "orthogonality interval" begin
    TTN = TensorTrainNumerics
    dims = (2, 3, 2, 3)
    x = rand_tt(dims, 3)
    @test x.orthogonality == [1, 4]

    @testset "constructor" begin
        y = TTVector{Float64, 4}(x.ttv_vec, x.ttv_dims, x.ttv_rks; orthogonality = (2, 3))
        @test y.orthogonality == [2, 3]
        shared = [2, 2]
        y = TTVector(x.ttv_vec, x.ttv_dims, x.ttv_rks; orthogonality = shared)
        @test y.orthogonality === shared
        @test_throws "1 ≤ left ≤ right ≤ 4" TTVector{Float64, 4}(x.ttv_vec, x.ttv_dims, x.ttv_rks; orthogonality = (3, 2))
        @test_throws "1 ≤ left ≤ right ≤ 4" TTVector{Float64, 4}(x.ttv_vec, x.ttv_dims, x.ttv_rks; orthogonality = (1, 5))
        @test_throws "orthogonality must be an interval (left, right)" TTVector{Float64, 4}(x.ttv_vec, x.ttv_dims, x.ttv_rks; orthogonality = [0, 0, 0, 0])
        A = rand_tto((2, 2, 2), 2)
        B = TTOperator(A.tto_vec, A.tto_dims, A.tto_rks; orthogonality = (2, 2))
        @test B.orthogonality == [2, 2]
        @test_throws "1 ≤ left ≤ right ≤ 3" TTOperator(A.tto_vec, A.tto_dims, A.tto_rks; orthogonality = (0, 2))
    end

    @testset "legacy flags" begin
        @test TTN._flags_to_orthogonality([0, 0, 0], 3) == (1, 3)
        @test TTN._flags_to_orthogonality([1, 0, -1], 3) == (2, 2)
        @test TTN._flags_to_orthogonality([1, 0, 0], 3) == (2, 3)
        @test TTN._flags_to_orthogonality([1, 1, 1], 3) == (3, 3)
        @test TTN._flags_to_orthogonality([-1, -1, -1], 3) == (1, 1)
        @test TTN._flags_to_orthogonality([1, 1, -1], 3) == (2, 2)
        # Only the leading left-orthogonal and trailing right-orthogonal runs count.
        @test TTN._flags_to_orthogonality([-1, 0, 1], 3) == (1, 3)
        @test TTN._flags_to_orthogonality([1, 0, 1, -1], 4) == (2, 3)
        @test_throws "expected 3 orthogonality flags" TTN._flags_to_orthogonality([1, 0], 3)

        y = @test_deprecated TTVector{Float64, 4}(x.ttv_vec, x.ttv_dims, x.ttv_rks, [1, 1, 0, -1])
        @test y.orthogonality == [3, 3]
        @test (@test_deprecated y.ttv_ot) == [1, 1, 0, -1]
        A = rand_tto((2, 2, 2), 2)
        B = @test_deprecated TTOperator(A.tto_vec, A.tto_dims, A.tto_rks, [1, 0, 0])
        @test B.orthogonality == [2, 3]
        @test (@test_deprecated B.tto_ot) == [1, 0, 0]
        z = @test_deprecated zeros_tt(dims, x.ttv_rks; ot = [1, 0, -1, -1])
        @test z.orthogonality == [2, 2]
        @test zeros_tt(dims, x.ttv_rks; orthogonality = (2, 4)).orthogonality == [2, 4]
    end

    @testset "updates" begin
        y = orthogonalize(x; i = 2)
        TTN._center_moved_right!(y, 2)
        @test y.orthogonality == [3, 3]
        TTN._center_moved_left!(y, 3)
        @test y.orthogonality == [2, 2]
        TTN._core_replaced!(y, 4)
        @test y.orthogonality == [2, 4]
        # A move outside the interval keeps only what still holds.
        TTN._set_orthogonality!(y, 3, 3)
        TTN._center_moved_right!(y, 1)
        @test y.orthogonality == [2, 3]
        TTN._set_orthogonality!(y, 2, 2)
        TTN._center_moved_left!(y, 4)
        @test y.orthogonality == [2, 3]
        TTN._forget_orthogonality!(y)
        @test y.orthogonality == [1, 4]
        @test TTN._orthogonality_center(y) === nothing
        @test TTN._orthogonality_center(orthogonalize(x; i = 3)) == 3
        @test_throws "1 ≤ left ≤ right ≤ 4" TTN._set_orthogonality!(y, 0, 2)
    end

    @testset "operations record what the cores satisfy" begin
        tensor = randn(dims)
        for i in 1:4
            @test verified_orthogonality(orthogonalize(x; i)) == (i, i)
            @test verified_orthogonality(ttv_decomp(tensor; index = i)) == (i, i)
            @test verified_orthogonality(2.0 * orthogonalize(x; i)) == (i, i)
            @test verified_orthogonality(reverse_qtt_bits(orthogonalize(x; i))) == (5 - i, 5 - i)
        end
        @test verified_orthogonality(tt_round(x)) == (4, 4)
        y = copy(x)
        tt_compress!(y, 2)
        @test verified_orthogonality(y) == (4, 4)
        a = orthogonalize(x; i = 2)
        b = orthogonalize(x; i = 3)
        @test verified_orthogonality(concatenate(a, b)) == (2, 7)
        @test verified_orthogonality(kron(a, b)) == (2, 7)
        @test verified_orthogonality(a + b) == (1, 4)
        @test verified_orthogonality(hadamard_ttm(a, b)) == (1, 1)

        q = QTTVector(orthogonalize(rand_tt((2, 2, 2, 2), 2); i = 3), 1, 4, :serial)
        @test q.orthogonality === TTVector(q).orthogonality
        tt_round!(q)
        @test verified_orthogonality(q) == (4, 4)
    end

    @testset "solver results" begin
        d = 4
        sdims = ntuple(_ -> 2, d)
        A = Δ(d) + id_tto(d)
        b = rand_tt(sdims, 2; normalize = true)
        x0 = rand_tt(sdims, 2; normalize = true)
        for alg in (ALS(), MALS(), DMRG(), DMRG(nsites = 1), AMEn())
            sol = linear_solve(A, b, x0, alg)
            @test verified_orthogonality(sol) isa Tuple
        end
        for alg in (ALS(), MALS(), DMRG())
            sol = eigen_solve(A, x0, alg)[2]
            @test verified_orthogonality(sol) isa Tuple
        end
        _, sol = als_gen_eigsolve(A, id_tto(d), x0)
        @test verified_orthogonality(sol) isa Tuple
        @test verified_orthogonality(tdvp(A, x0, [0.01]; show_progress = false)) isa Tuple
        @test verified_orthogonality(tdvp2(A, x0, [0.01]; show_progress = false)) isa Tuple
    end
end
