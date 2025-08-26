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

@testset "ttv_to_diag_tto for TTvector" begin
    # Test 1: ttv_to_diag_tto on a simple 2D TTvector (all ranks 1, should match diag of full vector)
    dims = (3, 2)
    rks = [1, 1, 1]
    # Construct a TTvector with explicit values
    core1 = reshape([1.0, 2.0, 3.0], 3, 1, 1)
    core2 = reshape([4.0, 5.0], 2, 1, 1)
    x = TTvector{Float64,2}(2, [core1, core2], dims, rks, zeros(Int,2))
    # Full vector
    full_x = vec([core1[i,1,1]*core2[j,1,1] for i=1:3, j=1:2])
    # Diagonal TT-matrix
    Xdiag = ttv_to_diag_tto(x)
    # Reconstruct full matrix from TToperator
    # For all (i1,i2), (j1,j2): sum over ranks (but all ranks are 1)
    mat = zeros(Float64, 6, 6)
    for i1=1:3, i2=1:2, j1=1:3, j2=1:2
        v = Xdiag.tto_vec[1][i1, j1, 1, 1] * Xdiag.tto_vec[2][i2, j2, 1, 1]
        mat[(i1-1)*2+i2, (j1-1)*2+j2] = v
    end
    # Should be diagonal with full_x on the diagonal
    @test all(mat[i,j] == 0.0 for i=1:6, j=1:6 if i != j)

    # Test 2: ttv_to_diag_tto preserves dimensions and ranks
    dims3 = (2, 2, 2)
    rks3 = [1, 2, 2, 1]
    x3 = rand_tt(dims3, rks3)
    X3diag = ttv_to_diag_tto(x3)
    @test X3diag.tto_dims == x3.ttv_dims
    @test X3diag.tto_rks == x3.ttv_rks
    @test length(X3diag.tto_vec) == 3
    @test all(size(core,1) == size(core,2) == d for (core,d) in zip(X3diag.tto_vec, x3.ttv_dims))
    @test all(size(core,3) == r for (core,r) in zip(X3diag.tto_vec, x3.ttv_rks[1:end-1]))
    @test all(size(core,4) == r for (core,r) in zip(X3diag.tto_vec, x3.ttv_rks[2:end]))

end


@testset "Hadamard product and function reconstruction tests" begin
    d = 8
    x_points = LinRange(0, 1, 2^d)

    # Define tensor train vectors
    A1 = qtt_exp(d)
    A2 = qtt_sin(d, λ = π)
    A3 = qtt_cos(d, λ = π)
    A4 = qtt_polynom([0.0, 2.0, 3.0, -8.0, -5.0], d; a = 0.0, b = 1.0)

    # Test 1: cos(π^2 * x) * sin(π^2 * x)
    expected1 = cos.(π^2 * x_points) .* sin.(π^2 * x_points)
    result1 = qtt_to_function(A2 ⊕ A3)
    @test isapprox(result1, expected1; atol=1e-12)

    # Test 2: exp(x) * sin(π^2 * x)
    expected2 = exp.(x_points) .* sin.(π^2 * x_points)
    result2 = qtt_to_function(A1 ⊕ A2)
    @test isapprox(result2, expected2; atol=1e-12)

    # Test 3: Polynomial values * sin(π^2 * x)
    values_polynom(x) = 2 * x + 3 * x^2 - 8 * x^3 - 5 * x^4
    expected3 = values_polynom.(x_points) .* sin.(π^2 * x_points)
    result3 = qtt_to_function(A4 ⊕ A2)
    @test isapprox(result3, expected3; atol=1e-12)

    # Test 4: Polynomial values * cos(π^2 * x)
    expected4 = values_polynom.(x_points) .* cos.(π^2 * x_points)
    result4 = qtt_to_function(A4 ⊕ A3)
    @test isapprox(result4, expected4; atol=1e-12)
end