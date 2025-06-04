using Test
using Random


@testset "permute for TTvector" begin

    # Test 1: Permute a 3-dimensional TTvector (trivial ranks)
    dims = (2, 3, 4)
    rks = [1,2,2,1]
    x = rand_tt(dims,rks)
    order = [2, 1, 3]
    eps = 1e-12
    y = permute(x, order, eps)
    @test y.ttv_dims == dims[order]
    @test y.N == 3
    @test length(y.ttv_vec) == 3

    # Test 2: Identity permutation (should not change anything)
    order_id = [1, 2, 3]
    y_id = permute(x, order_id, eps)
    @test y_id.ttv_dims == dims
    @test y_id.N == 3
    @test length(y_id.ttv_vec) == 3

    # Test 3: Permute a 4-dimensional TTvector
    dims4 = (2, 2, 3, 2)
    rks4 = [1, 2, 2, 2, 1]
    x4 = rand_tt(dims4, rks4)
    order4 = [3, 1, 4, 2]
    y4 = permute(x4, order4, eps)
    @test y4.ttv_dims == dims4[order4]
    @test y4.N == 4
    @test length(y4.ttv_vec) == 4

    # Test 4: Permute twice with inverse order returns original dims
    invorder4 = invperm(order4)
    y4b = permute(y4, invorder4, eps)
    @test y4b.ttv_dims == x4.ttv_dims
    @test y4b.N == x4.N

end

@testset "TTdiag for TTvector" begin
    # Test 1: TTdiag on a simple 2D TTvector (all ranks 1, should match diag of full vector)
    dims = (3, 2)
    rks = [1, 1, 1]
    # Construct a TTvector with explicit values
    core1 = reshape([1.0, 2.0, 3.0], 3, 1, 1)
    core2 = reshape([4.0, 5.0], 2, 1, 1)
    x = TTvector{Float64,2}(2, [core1, core2], dims, rks, zeros(Int,2))
    # Full vector
    full_x = vec([core1[i,1,1]*core2[j,1,1] for i=1:3, j=1:2])
    # Diagonal TT-matrix
    Xdiag = TTdiag(x)
    # Reconstruct full matrix from TToperator
    # For all (i1,i2), (j1,j2): sum over ranks (but all ranks are 1)
    mat = zeros(Float64, 6, 6)
    for i1=1:3, i2=1:2, j1=1:3, j2=1:2
        v = Xdiag.tto_vec[1][i1, j1, 1, 1] * Xdiag.tto_vec[2][i2, j2, 1, 1]
        mat[(i1-1)*2+i2, (j1-1)*2+j2] = v
    end
    # Should be diagonal with full_x on the diagonal
    @test all(mat[i,j] == 0.0 for i=1:6, j=1:6 if i != j)

    # Test 2: TTdiag preserves dimensions and ranks
    dims3 = (2, 2, 2)
    rks3 = [1, 2, 2, 1]
    x3 = rand_tt(dims3, rks3)
    X3diag = TTdiag(x3)
    @test X3diag.tto_dims == x3.ttv_dims
    @test X3diag.tto_rks == x3.ttv_rks
    @test length(X3diag.tto_vec) == 3
    @test all(size(core,1) == size(core,2) == d for (core,d) in zip(X3diag.tto_vec, x3.ttv_dims))
    @test all(size(core,3) == r for (core,r) in zip(X3diag.tto_vec, x3.ttv_rks[1:end-1]))
    @test all(size(core,4) == r for (core,r) in zip(X3diag.tto_vec, x3.ttv_rks[2:end]))

end


@testset "hadamard for TTvector" begin
    # Test 1: Hadamard product of two TTvectors with all ranks 1 (should match elementwise product)
    dims = (2, 3)
    rks = [1, 1, 1]
    # Construct two TTvectors with explicit values
    core1a = reshape([1.0, 2.0], 2, 1, 1)
    core2a = reshape([3.0, 4.0, 5.0], 3, 1, 1)
    a = TTvector{Float64,2}(2, [core1a, core2a], dims, rks, zeros(Int,2))
    core1b = reshape([2.0, 1.0], 2, 1, 1)
    core2b = reshape([1.0, 0.5, 2.0], 3, 1, 1)
    b = TTvector{Float64,2}(2, [core1b, core2b], dims, rks, zeros(Int,2))
    eps = 1e-12
    c = hadamard(a, b, eps)
    # Full vector reconstruction
    full_a = vec([core1a[i,1,1]*core2a[j,1,1] for i=1:2, j=1:3])
    full_b = vec([core1b[i,1,1]*core2b[j,1,1] for i=1:2, j=1:3])
    full_c = vec([c.ttv_vec[1][i,1,1]*c.ttv_vec[2][j,1,1] for i=1:2, j=1:3])
    @test isapprox(full_c, full_a .* full_b; atol=1e-10)

    # Test 2: Hadamard product with random TTvectors (ranks > 1)
    dims3 = (2, 2, 2)
    rks3 = [1, 2, 2, 1]
    a3 = rand_tt(dims3, rks3)
    b3 = rand_tt(dims3, rks3)
    c3 = hadamard(a3, b3, 1e-10)
    # Check output TTvector has correct dimensions and length
    @test c3.ttv_dims == a3.ttv_dims
    @test c3.N == a3.N
    @test length(c3.ttv_vec) == a3.N

    # Test 3: Error on mismatched dimensions
    dims4 = (2, 2, 3)
    rks4 = [1, 2, 2, 1]
    a4 = rand_tt(dims4, rks4)
    @test_throws ErrorException hadamard(a3, a4, 1e-10)
    # Error on mismatched mode sizes
    dims_bad = (2, 3, 2)
    a_bad = rand_tt(dims_bad, rks3)
    @test_throws ErrorException hadamard(a3, a_bad, 1e-10)
end
