using Test
using Random

@testset "ttv_to_diag_tto for TTvector" begin
    # Test 1: ttv_to_diag_tto on a simple 2D TTvector (all ranks 1, should match diag of full vector)
    dims = (3, 2)
    rks = [1, 1, 1]
    # Construct a TTvector with explicit values
    core1 = reshape([1.0, 2.0, 3.0], 3, 1, 1)
    core2 = reshape([4.0, 5.0], 2, 1, 1)
    x = TTvector{Float64, 2}(2, [core1, core2], dims, rks, zeros(Int, 2))
    # Full vector
    full_x = vec([core1[i, 1, 1] * core2[j, 1, 1] for i in 1:3, j in 1:2])
    # Diagonal TT-matrix
    Xdiag = ttv_to_diag_tto(x)
    # Reconstruct full matrix from TToperator
    # For all (i1,i2), (j1,j2): sum over ranks (but all ranks are 1)
    mat = zeros(Float64, 6, 6)
    for i1 in 1:3, i2 in 1:2, j1 in 1:3, j2 in 1:2
        v = Xdiag.tto_vec[1][i1, j1, 1, 1] * Xdiag.tto_vec[2][i2, j2, 1, 1]
        mat[(i1 - 1) * 2 + i2, (j1 - 1) * 2 + j2] = v
    end
    # Should be diagonal with full_x on the diagonal
    @test all(mat[i, j] == 0.0 for i in 1:6, j in 1:6 if i != j)

    # Test 2: ttv_to_diag_tto preserves dimensions and ranks
    dims3 = (2, 2, 2)
    rks3 = [1, 2, 2, 1]
    x3 = rand_tt(dims3, rks3)
    X3diag = ttv_to_diag_tto(x3)
    @test X3diag.tto_dims == x3.ttv_dims
    @test X3diag.tto_rks == x3.ttv_rks
    @test length(X3diag.tto_vec) == 3
    @test all(size(core, 1) == size(core, 2) == d for (core, d) in zip(X3diag.tto_vec, x3.ttv_dims))
    @test all(size(core, 3) == r for (core, r) in zip(X3diag.tto_vec, x3.ttv_rks[1:(end - 1)]))
    @test all(size(core, 4) == r for (core, r) in zip(X3diag.tto_vec, x3.ttv_rks[2:end]))

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
    @test isapprox(result1, expected1; atol = 1.0e-12)

    # Test 2: exp(x) * sin(π^2 * x)
    expected2 = exp.(x_points) .* sin.(π^2 * x_points)
    result2 = qtt_to_function(A1 ⊕ A2)
    @test isapprox(result2, expected2; atol = 1.0e-12)

    # Test 3: Polynomial values * sin(π^2 * x)
    values_polynom(x) = 2 * x + 3 * x^2 - 8 * x^3 - 5 * x^4
    expected3 = values_polynom.(x_points) .* sin.(π^2 * x_points)
    result3 = qtt_to_function(A4 ⊕ A2)
    @test isapprox(result3, expected3; atol = 1.0e-12)

    # Test 4: Polynomial values * cos(π^2 * x)
    expected4 = values_polynom.(x_points) .* cos.(π^2 * x_points)
    result4 = qtt_to_function(A4 ⊕ A3)
    @test isapprox(result4, expected4; atol = 1.0e-12)
end


@testset "Hadamard TTM algorithm vs naive" begin
    d = 8
    x_points = LinRange(0, 1, 2^d)

    A1 = qtt_exp(d)
    A2 = qtt_sin(d, λ = π)
    A3 = qtt_cos(d, λ = π)
    A4 = qtt_polynom([0.0, 2.0, 3.0, -8.0, -5.0], d; a = 0.0, b = 1.0)

    expected1 = cos.(π^2 * x_points) .* sin.(π^2 * x_points)
    @test isapprox(qtt_to_function(hadamard_ttm(A2, A3)), expected1; atol = 1.0e-10)

    expected2 = exp.(x_points) .* sin.(π^2 * x_points)
    @test isapprox(qtt_to_function(hadamard_ttm(A1, A2)), expected2; atol = 1.0e-10)

    values_polynom(x) = 2 * x + 3 * x^2 - 8 * x^3 - 5 * x^4
    expected3 = values_polynom.(x_points) .* sin.(π^2 * x_points)

    @test isapprox(qtt_to_function(hadamard_ttm(A4, A2)), expected3; atol = 1.0e-4)

    expected4 = values_polynom.(x_points) .* cos.(π^2 * x_points)
    @test isapprox(qtt_to_function(hadamard_ttm(A4, A3)), expected4; atol = 1.0e-4)

    @test euclidean_distance(hadamard_ttm(A2, A3), hadamard(A2, A3)) / norm(hadamard(A2, A3)) < 1e-5
    @test euclidean_distance(hadamard_ttm(A1, A2), hadamard(A1, A2)) / norm(hadamard(A1, A2)) < 1e-5
    @test euclidean_distance(hadamard_ttm(A4, A2), hadamard(A4, A2)) / norm(hadamard(A4, A2)) < 1e-5
    @test euclidean_distance(hadamard_ttm(A4, A3), hadamard(A4, A3)) / norm(hadamard(A4, A3)) < 1e-5

    ttm_loose = hadamard_ttm(A1, A2; tol = 1e-8)
    @test isapprox(qtt_to_function(ttm_loose), expected2; atol = 1.0e-3)
end

@testset "add!" begin
    x = rand_tt((2, 3), [1, 2, 1])
    y = rand_tt((2, 3), [1, 3, 1])
    expected_tensor = ttv_to_tensor(x + y)
    add!(x, y)
    @test isapprox(ttv_to_tensor(x), expected_tensor; atol = 1e-12)
    @test x.ttv_rks == [1, 5, 1]
    @test all(x.ttv_ot .== 0)
end

@testset "TToperator callable" begin
    dims = (2, 3)
    A = rand_tto(dims, 2)
    v = rand_tt(dims, [1, 2, 1])
    @test isapprox(ttv_to_tensor(A(v)), ttv_to_tensor(A * v); atol = 1e-12)
    @test isapprox(ttv_to_tensor(A(v, Val(:x))), ttv_to_tensor(A * v); atol = 1e-12)
end

@testset "TToperator * TToperator" begin
    dims = (2, 3)
    A = rand_tto(dims, 2)
    B = rand_tto(dims, 2)
    C = A * B
    n = prod(dims)
    A_mat = reshape(tto_to_tensor(A), n, n)
    B_mat = reshape(tto_to_tensor(B), n, n)
    C_mat = reshape(tto_to_tensor(C), n, n)
    @test isapprox(C_mat, A_mat * B_mat; atol = 1e-12)
    @test C.tto_dims == A.tto_dims
    @test C.tto_rks == A.tto_rks .* B.tto_rks
end

@testset "Array{TTvector} * Vector (linear combination)" begin
    dims = (2, 3)
    a = rand_tt(dims, [1, 2, 1])
    b = rand_tt(dims, [1, 3, 1])
    coeffs = [2.0, -1.5]
    result = [a, b] * coeffs
    expected = 2.0 * ttv_to_tensor(a) + (-1.5) * ttv_to_tensor(b)
    @test isapprox(ttv_to_tensor(result), expected; atol = 1e-12)
end

@testset "TTvector / scalar" begin
    x = rand_tt((2, 3), [1, 2, 1])
    a = 3.0
    y = x / a
    @test isapprox(ttv_to_tensor(y), ttv_to_tensor(x) / a; atol = 1e-12)
end

@testset "outer_product" begin
    dims = (2, 3)
    x = rand_tt(dims, [1, 2, 1])
    y = rand_tt(dims, [1, 3, 1])
    M = outer_product(x, y)

    @test M.tto_dims == dims
    @test M.tto_rks == x.ttv_rks .* y.ttv_rks

    # Full tensor: M[i1,i2, j1,j2] = x[i1,i2] * conj(y[j1,j2])
    Tx = ttv_to_tensor(x)
    Ty = ttv_to_tensor(y)
    TM = tto_to_tensor(M)
    for i1 in 1:dims[1], i2 in 1:dims[2], j1 in 1:dims[1], j2 in 1:dims[2]
        @test isapprox(TM[i1, i2, j1, j2], Tx[i1, i2] * conj(Ty[j1, j2]); atol = 1e-12)
    end

    # Key property: (x ⊗ y†) * z = ⟨y, z⟩ * x
    z = rand_tt(dims, [1, 2, 1])
    Mz = M * z
    @test isapprox(ttv_to_tensor(Mz), TensorTrainNumerics.dot(y, z) * ttv_to_tensor(x); atol = 1e-12)
end

@testset "kron for TTvector" begin
    a = rand_tt((2, 3), [1, 2, 1])
    b = rand_tt((4, 5), [1, 3, 1])
    c = kron(a, b)
    @test c.N == a.N + b.N
    @test c.ttv_dims == (2, 3, 4, 5)
    @test c.ttv_rks[1] == 1 && c.ttv_rks[end] == 1
    Ta = ttv_to_tensor(a)
    Tb = ttv_to_tensor(b)
    Tc = ttv_to_tensor(c)
    for i1 in 1:2, i2 in 1:3, j1 in 1:4, j2 in 1:5
        @test isapprox(Tc[i1, i2, j1, j2], Ta[i1, i2] * Tb[j1, j2]; atol = 1e-12)
    end
end

@testset "kron for TToperator" begin
    A = rand_tto((2, 3), 2)
    B = rand_tto((4, 5), 2)
    C = kron(A, B)
    @test C.N == A.N + B.N
    @test C.tto_dims == (2, 3, 4, 5)
    @test C.tto_rks[1] == 1 && C.tto_rks[end] == 1

    # Property: kron(A, B) * kron(a, b) ≈ kron(A*a, B*b)
    a = rand_tt((2, 3), [1, 2, 1])
    b = rand_tt((4, 5), [1, 3, 1])
    lhs = ttv_to_tensor(kron(A, B) * kron(a, b))
    rhs = ttv_to_tensor(kron(A * a, B * b))
    @test isapprox(lhs, rhs; atol = 1e-12)
end

@testset "Norms" begin

    d = 8
    A1 = qtt_exp(d)
    A2 = qtt_sin(d, λ = π)
    A3 = qtt_cos(d, λ = π)
    A4 = qtt_polynom([0.0, 2.0, 3.0, -8.0, -5.0], d; a = 0.0, b = 1.0)

    @test euclidean_distance(A1, A1) == 0.0
    @test euclidean_distance_normalized(A1, A1) == 0.0

    S1 = qtt_to_function(A1)
    S2 = qtt_to_function(A2)

    @test isapprox(sqrt(LinearAlgebra.dot(S1, S1) - 2 * real(LinearAlgebra.dot(S1, S2)) + LinearAlgebra.dot(S2, S2)), euclidean_distance(A1, A2), atol=1.0e-10)
    @test isapprox(sqrt(1.0 + LinearAlgebra.dot(S1, S1) / LinearAlgebra.dot(S2, S2) - 2.0 * real(LinearAlgebra.dot(S2, S1)) / LinearAlgebra.dot(S2, S2)), euclidean_distance_normalized(A1, A2), atol = 1.0e-12)

end
