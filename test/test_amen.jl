using Test
using LinearAlgebra
using Logging
using Random
using TensorTrainNumerics

const TTN = TensorTrainNumerics

amen_matrix(A) = reshape(tto_to_tensor(A), prod(A.tto_dims), :)
amen_vector(x) = vec(ttv_to_tensor(x))
amen_relres(A, x, b) = norm(amen_matrix(A) * amen_vector(x) - amen_vector(b)) / norm(amen_vector(b))

@testset "AMEn interfaces contract to inner products ($T)" for T in (Float64, ComplexF64)
    Random.seed!(1)
    dims = (2, 3, 2, 2)
    d = length(dims)
    x = rand_tt(T, dims, [1, 2, 3, 2, 1])
    b = rand_tt(T, dims, [1, 2, 2, 2, 1])
    A = rand_tto(dims, 2; T)
    xAx = dot(x, A * x)
    xb = dot(x, b)

    L3 = [ones(T, 1, 1, 1)]
    L2 = [ones(T, 1, 1)]
    for k in 1:d
        push!(L3, TTN._left_interface(L3[k], x.ttv_vec[k], A.tto_vec[k], x.ttv_vec[k]))
        push!(L2, TTN._left_interface(L2[k], x.ttv_vec[k], b.ttv_vec[k]))
    end
    R3 = Vector{Array{T, 3}}(undef, d + 1)
    R2 = Vector{Array{T, 2}}(undef, d + 1)
    R3[d + 1] = ones(T, 1, 1, 1)
    R2[d + 1] = ones(T, 1, 1)
    for k in d:-1:1
        R3[k] = TTN._right_interface(R3[k + 1], x.ttv_vec[k], A.tto_vec[k], x.ttv_vec[k])
        R2[k] = TTN._right_interface(R2[k + 1], x.ttv_vec[k], b.ttv_vec[k])
    end

    for k in 1:(d + 1)
        @test size(L3[k]) == (x.ttv_rks[k], A.tto_rks[k], x.ttv_rks[k])
        @test size(L2[k]) == (x.ttv_rks[k], b.ttv_rks[k])
        @test sum(L3[k] .* R3[k]) ≈ xAx
        @test sum(L2[k] .* R2[k]) ≈ xb
    end

    for k in 1:d
        ΦL, ΦR, Ak, xk = L3[k], R3[k + 1], A.tto_vec[k], x.ttv_vec[k]
        v = randn(T, size(xk))
        @test vec(TTN._local_matvec(ΦL, Ak, ΦR, v)) ≈ TTN._local_matrix(ΦL, Ak, ΦR) * vec(v)
        @test dot(xk, TTN._local_matvec(ΦL, Ak, ΦR, xk)) ≈ xAx
        @test dot(xk, TTN._project(L2[k], b.ttv_vec[k], R2[k + 1])) ≈ xb
    end
end

@testset "AMEn orthogonalization helpers ($T)" for T in (Float64, ComplexF64)
    Random.seed!(2)
    dims = (2, 2, 2)
    # Bond ranks 5 and 3 exceed `n * r_right` of the core to their right (4 and 2).
    cores = [randn(T, 2, 1, 5), randn(T, 2, 5, 3), randn(T, 2, 3, 1)]
    as_tt(c) = TTvector{T, 3}(c, dims, [1; [size(ck, 3) for ck in c]], zeros(Int, 3))
    before = ttv_to_tensor(as_tt(copy(cores)))

    for k in 3:-1:2
        TTN._orthogonalize_right!(cores, k)
        n, rl, rr = size(cores[k])
        M = reshape(permutedims(cores[k], (2, 1, 3)), rl, n * rr)
        @test M * M' ≈ I
        @test size(cores[k - 1], 3) == rl
    end
    @test size(cores[3], 2) == 2
    @test size(cores[2], 2) == 4
    @test ttv_to_tensor(as_tt(cores)) ≈ before

    c = randn(T, 2, 3, 4)
    q = TTN._left_orthonormal(c)
    Q = reshape(q, 6, :)
    @test size(q) == (2, 3, 4)
    @test Q' * Q ≈ I
    @test Q * (Q' * reshape(c, 6, 4)) ≈ reshape(c, 6, 4)
    @test size(TTN._left_orthonormal(randn(T, 2, 1, 4))) == (2, 1, 2)
    @test all(isfinite, TTN._left_orthonormal(zeros(T, 2, 2, 3)))
end

@testset "AMEn solves an SPD system from a rank-1 guess" begin
    Random.seed!(3)
    d = 6
    dims = ntuple(_ -> 2, d)
    A = Δ(d) + 0.5 * id_tto(d)
    b = rand_tt(dims, 3; normalize = true)
    x0 = rand_tt(dims, 1)
    x, info = linear_solve(A, b, x0, AMEn(tol = 1.0e-8, return_info = true, show_progress = false))
    ref = amen_matrix(A) \ amen_vector(b)
    @test x isa TTvector{Float64}
    @test info.converged
    @test info.sweeps ≤ 20
    @test info.residual ≤ 1.0e-6
    @test amen_relres(A, x, b) ≤ 1.0e-7
    @test norm(amen_vector(x) - ref) / norm(ref) ≤ 1.0e-6
    @test linear_solve(A, b, x0, AMEn(tol = 1.0e-8, show_progress = false)) isa TTvector
    @test amen_relres(A, linear_solve(A, b, x0; alg = AMEn(show_progress = false)), b) ≤ 1.0e-5
end

@testset "AMEn adapts the ranks to the solution" begin
    Random.seed!(4)
    d = 6
    dims = ntuple(_ -> 2, d)
    A = Δ(d) + 0.5 * id_tto(d)
    x_true = rand_tt(dims, 3; normalize = true)
    b = A * x_true
    x = linear_solve(A, b, rand_tt(dims, 1), AMEn(tol = 1.0e-8, show_progress = false))
    @test norm(amen_vector(x) - amen_vector(x_true)) ≤ 1.0e-6
    @test maximum(x.ttv_rks) ≤ 4
end

@testset "AMEn solves a non-symmetric system" begin
    Random.seed!(5)
    d = 6
    dims = ntuple(_ -> 2, d)
    A = Δ(d) + 0.3 * ∇(d) + 0.5 * id_tto(d)
    @test !issymmetric(amen_matrix(A))
    b = rand_tt(dims, 3; normalize = true)
    x, info = linear_solve(A, b, rand_tt(dims, 1), AMEn(tol = 1.0e-8, return_info = true, show_progress = false))
    @test info.converged
    @test amen_relres(A, x, b) ≤ 1.0e-7
end

@testset "AMEn element types and dimensions" begin
    Random.seed!(6)
    dims = (2, 2, 2, 2)
    B = rand_tto(dims, 2; T = ComplexF64)
    A = B' * B + 5.0 * id_tto(ComplexF64, 4)
    b = rand_tt(ComplexF64, dims, [1, 2, 2, 2, 1])
    x = linear_solve(A, b, rand_tt(ComplexF64, dims, [1, 1, 1, 1, 1]), AMEn(tol = 1.0e-8, show_progress = false))
    @test eltype(x) == ComplexF64
    @test amen_relres(A, x, b) ≤ 1.0e-7

    # Real operator and guess with a complex right-hand side.
    Ar = Δ(4) + 0.5 * id_tto(4)
    x = linear_solve(Ar, b, rand_tt(dims, 1), AMEn(tol = 1.0e-8, show_progress = false))
    @test eltype(x) == ComplexF64
    @test amen_relres(Ar, x, b) ≤ 1.0e-7

    A32 = TTN._convert_eltype(Float32, Ar)
    b32 = TTN._convert_eltype(Float32, rand_tt(dims, 2; normalize = true))
    x32 = linear_solve(A32, b32, TTN._convert_eltype(Float32, rand_tt(dims, 1)), AMEn(tol = 1.0e-4, show_progress = false))
    @test eltype(x32) == Float32
    @test amen_relres(A32, x32, b32) ≤ 1.0e-3

    dims = (2, 3, 4, 2)
    B = rand_tto(dims, 2)
    A = B' * B + 10.0 * TTN._identity_like(B)
    b = rand_tt(dims, [1, 2, 3, 2, 1])
    x = linear_solve(A, b, rand_tt(dims, [1, 1, 1, 1, 1]), AMEn(tol = 1.0e-8, show_progress = false))
    @test x.ttv_dims == dims
    @test amen_relres(A, x, b) ≤ 1.0e-7
end

@testset "AMEn rank controls and local solvers" begin
    Random.seed!(7)
    d = 6
    dims = ntuple(_ -> 2, d)
    A = Δ(d) + 0.5 * id_tto(d)
    b = rand_tt(dims, 3; normalize = true)

    x = linear_solve(A, b, rand_tt(dims, 1), AMEn(max_bond = 2, max_sweeps = 4, verbosity = 0, show_progress = false))
    @test maximum(x.ttv_rks) ≤ 2

    x0 = rand_tt(dims, 2)
    x = linear_solve(A, b, x0, AMEn(kickrank = 0, max_sweeps = 3, verbosity = 0, show_progress = false))
    @test all(x.ttv_rks .≤ x0.ttv_rks)

    x0 = rand_tt(dims, 1)
    xd = linear_solve(A, b, x0, AMEn(tol = 1.0e-8, local_solver = :direct, show_progress = false))
    xi = linear_solve(A, b, x0, AMEn(tol = 1.0e-8, local_solver = :iterative, show_progress = false))
    @test amen_relres(A, xi, b) ≤ 1.0e-7
    @test norm(amen_vector(xd) - amen_vector(xi)) / norm(amen_vector(xd)) ≤ 1.0e-6
end

@testset "AMEn reports non-convergence" begin
    Random.seed!(8)
    d = 6
    dims = ntuple(_ -> 2, d)
    A = Δ(d) + 0.5 * id_tto(d)
    b = rand_tt(dims, 3; normalize = true)
    x0 = rand_tt(dims, 1)
    @test_logs (:warn, r"did not converge") match_mode = :any linear_solve(A, b, x0, AMEn(max_sweeps = 1, tol = 1.0e-12, show_progress = false))
    @test_logs min_level = Logging.Warn linear_solve(A, b, x0, AMEn(max_sweeps = 1, tol = 1.0e-12, verbosity = 0, show_progress = false))
    x, info = linear_solve(A, b, x0, AMEn(max_sweeps = 1, tol = 1.0e-12, verbosity = 0, return_info = true, show_progress = false))
    @test !info.converged
    @test info.sweeps == 1
    @test info.residual > 1.0e-12
end

@testset "AMEn input checks" begin
    Random.seed!(9)
    dims = (2, 2, 2, 2)
    A = Δ(4) + 0.5 * id_tto(4)
    b = rand_tt(dims, 2)
    x0 = rand_tt(dims, 1)
    alg = AMEn(show_progress = false)
    @test_throws "do not match" linear_solve(A, rand_tt((2, 2, 2), 1), x0, alg)
    @test_throws "do not match" linear_solve(A, b, rand_tt((2, 2, 2), 1), alg)
    @test_throws "right-hand side is zero" linear_solve(A, zeros_tt(dims, [1, 1, 1, 1, 1]), x0, alg)
    @test_throws "at least 2 cores" linear_solve(Δ(1), rand_tt((2,), [1, 1]), rand_tt((2,), [1, 1]), alg)
end

@testset "AMEn handles unusual guesses and wrappers" begin
    Random.seed!(10)
    d = 5
    dims = ntuple(_ -> 2, d)
    A = Δ(d) + 0.5 * id_tto(d)
    b = rand_tt(dims, 2; normalize = true)

    # Ranks above what the dimensions allow.
    x = linear_solve(A, b, rand_tt(dims, [1, 4, 6, 4, 2, 1]), AMEn(tol = 1.0e-8, show_progress = false))
    @test amen_relres(A, x, b) ≤ 1.0e-7

    # A guess that already solves the system.
    x_true = rand_tt(dims, 2; normalize = true)
    x, info = linear_solve(A, A * x_true, x_true, AMEn(tol = 1.0e-8, return_info = true, show_progress = false))
    @test info.converged
    @test info.sweeps ≤ 2
    @test all(c -> all(isfinite, c), x.ttv_vec)
    @test norm(amen_vector(x) - amen_vector(x_true)) ≤ 1.0e-10

    Aq = QTToperator(A, 1, d, :serial)
    bq = QTTvector(b, 1, d, :serial)
    xq = linear_solve(Aq, bq, QTTvector(rand_tt(dims, 1), 1, d, :serial), AMEn(tol = 1.0e-8, show_progress = false))
    @test xq isa QTTvector
    @test (xq.n_dims, xq.bits_per_dim, xq.ordering) == (1, d, :serial)
    @test amen_relres(A, TTvector(xq), b) ≤ 1.0e-7
end

@testset "AMEn finds the smallest eigenpair" begin
    Random.seed!(11)
    d = 6
    dims = ntuple(_ -> 2, d)
    A = Δ(d)
    M = amen_matrix(A)
    λref = eigmin(Symmetric(M))
    E, x, r_hist = eigen_solve(A, rand_tt(dims, 1), AMEn(tol = 1.0e-8, show_progress = false))
    @test abs(E[end] - λref) ≤ 1.0e-8
    @test length(E) == length(r_hist)
    @test all(diff(E) .≤ 1.0e-8)
    @test x isa TTvector
    xv = amen_vector(x)
    @test norm(xv) ≈ 1
    @test norm(M * xv - E[end] * xv) ≤ 1.0e-5

    A = heisenberg_xyz_tto(6)
    λref = eigmin(Hermitian(amen_matrix(A)))
    E, x, _ = eigen_solve(A, rand_tt(dims, 1), AMEn(tol = 1.0e-8, show_progress = false))
    @test abs(E[end] - λref) ≤ 1.0e-7 * abs(λref)

    Ei, _, _ = eigen_solve(Δ(d), rand_tt(dims, 1), AMEn(tol = 1.0e-8, local_solver = :iterative, show_progress = false))
    @test abs(Ei[end] - eigmin(Symmetric(M))) ≤ 1.0e-7
    @test eigen_solve(Δ(d), rand_tt(dims, 1); alg = AMEn(show_progress = false))[2] isa TTvector
end

@testset "AMEn eigen_solve options and limits" begin
    Random.seed!(12)
    d = 6
    dims = ntuple(_ -> 2, d)
    A = Δ(d)
    x0 = rand_tt(dims, 1)
    @test_throws "not used by eigen_solve" eigen_solve(A, x0, AMEn(return_info = true))
    _, _, r_hist = eigen_solve(A, x0, AMEn(max_bond = 2, max_sweeps = 3, verbosity = 0, show_progress = false))
    @test maximum(r_hist) ≤ 2
    @test_logs (:warn, r"did not converge") match_mode = :any eigen_solve(A, x0, AMEn(max_sweeps = 1, tol = 1.0e-12, show_progress = false))
    @test_throws "do not match" eigen_solve(A, rand_tt((2, 2, 2), 1), AMEn(show_progress = false))
end

@testset "AMEn agrees with ALS, MALS, and DMRG" begin
    Random.seed!(13)
    # Rank 4 is the largest possible rank for four sites of dimension 2, so
    # fixed-rank ALS can represent the solution exactly.
    d = 4
    dims = ntuple(_ -> 2, d)
    x0 = rand_tt(dims, 4)
    b = rand_tt(dims, 2; normalize = true)

    A = Δ(d) + 0.5 * id_tto(d)
    ref = amen_matrix(A) \ amen_vector(b)
    for alg in (
            AMEn(tol = 1.0e-10, show_progress = false),
            ALS(max_sweeps = 8, show_progress = false),
            MALS(max_sweeps = 4, trunc_tol = 1.0e-12, show_progress = false),
            DMRG(max_sweeps = 4, trunc_tol = 1.0e-12, max_bond = 4, local_solver = :direct, show_progress = false),
        )
        x = linear_solve(A, b, x0, alg)
        @test norm(amen_vector(x) - ref) / norm(ref) ≤ 1.0e-6
    end

    # MALS and DMRG symmetrize the local systems, so only ALS is compared here.
    A = Δ(d) + 0.3 * ∇(d) + 0.5 * id_tto(d)
    ref = amen_matrix(A) \ amen_vector(b)
    for alg in (AMEn(tol = 1.0e-10, show_progress = false), ALS(max_sweeps = 8, show_progress = false))
        x = linear_solve(A, b, x0, alg)
        @test norm(amen_vector(x) - ref) / norm(ref) ≤ 1.0e-6
    end

    A = Δ(d)
    λref = eigmin(Symmetric(amen_matrix(A)))
    for alg in (
            AMEn(tol = 1.0e-10, show_progress = false),
            ALS(max_sweeps = 6, show_progress = false),
            MALS(max_sweeps = 4, trunc_tol = 1.0e-12, max_bond = 4, show_progress = false),
            DMRG(max_sweeps = 4, trunc_tol = 1.0e-12, max_bond = 4, local_solver = :direct, show_progress = false),
        )
        E = first(eigen_solve(A, x0, alg))
        @test abs(E[end] - λref) ≤ 1.0e-6
    end
end

@testset "AMEn in the implicit time steppers" begin
    Random.seed!(14)
    d = 5
    dims = ntuple(_ -> 2, d)
    A = -1.0 * Δ(d)
    M = amen_matrix(A)
    u0 = rand_tt(dims, 2; normalize = true)
    h = 0.01
    steps = fill(h, 3)

    u = implicit_euler_method(A, u0, u0, steps; alg = AMEn(tol = 1.0e-10), show_progress = false)
    ref = (I - h * M)^3 \ amen_vector(u0)
    @test norm(amen_vector(u) - ref) / norm(ref) ≤ 1.0e-6

    u = crank_nicolson_method(A, u0, u0, steps; alg = AMEn(tol = 1.0e-10), show_progress = false)
    ref = ((I - h / 2 * M) \ (I + h / 2 * M))^3 * amen_vector(u0)
    @test norm(amen_vector(u) - ref) / norm(ref) ≤ 1.0e-6

    # Stepper keyword arguments replace fields of the algorithm object.
    u = implicit_euler_method(A, u0, u0, steps; alg = AMEn(), tol = 1.0e-10, show_progress = false)
    ref = (I - h * M)^3 \ amen_vector(u0)
    @test norm(amen_vector(u) - ref) / norm(ref) ≤ 1.0e-6
end

@testset "AMEn eigen_solve converges when the smallest eigenvalue is zero" begin
    Random.seed!(15)
    d = 5
    dims = ntuple(_ -> 2, d)
    λmin = eigmin(Symmetric(amen_matrix(Δ(d))))
    H = Δ(d) - λmin * id_tto(d)
    alg = AMEn(tol = 1.0e-8, show_progress = false)
    E, x, _ = @test_logs min_level = Logging.Warn eigen_solve(H, rand_tt(dims, 1), alg)
    @test abs(E[end]) ≤ 1.0e-8
    @test length(E) < 20 * d
end

@testset "AMEn handles a zero projected right-hand side and rejects a zero guess" begin
    Random.seed!(16)
    dims = (2, 2, 2, 2)
    A = Δ(4) + 0.5 * id_tto(4)
    b = rand_tt(dims, 1)
    # The last core of the guess is orthogonal to the last core of `b`, so the
    # projected right-hand side of every other site is zero in the first sweep.
    x0 = rand_tt(dims, 1)
    v = vec(b.ttv_vec[4])
    x0.ttv_vec[4] = reshape([-v[2], v[1]], 2, 1, 1)
    @test abs(dot(x0, b)) ≤ 1.0e-12 * norm(x0) * norm(b)
    for local_solver in (:direct, :iterative)
        x = linear_solve(A, b, x0, AMEn(; tol = 1.0e-8, local_solver, show_progress = false))
        @test amen_relres(A, x, b) ≤ 1.0e-7
    end
    @test_throws "guess is zero" eigen_solve(A, zeros_tt(dims, [1, 1, 1, 1, 1]), AMEn(show_progress = false))
end

@testset "AMEn reports an accurate residual for a badly scaled operator" begin
    Random.seed!(17)
    d = 8
    dims = ntuple(_ -> 2, d)
    A = 4.0^d * Δ(d)
    b = rand_tt(dims, 2; normalize = true)
    for tol in (1.0e-4, 1.0e-8)
        x, info = linear_solve(A, b, rand_tt(dims, 1), AMEn(; tol, return_info = true, show_progress = false))
        @test info.converged
        @test abs(info.residual - amen_relres(A, x, b)) ≤ 1.0e-10
    end
end

@testset "AMEn rejects QTT inputs with different metadata" begin
    Random.seed!(18)
    d = 4
    dims = ntuple(_ -> 2, d)
    A = Δ(d) + 0.5 * id_tto(d)
    b = rand_tt(dims, 2)
    x0 = rand_tt(dims, 1)
    Aq = QTToperator(A, 2, 2, :serial)
    serial(x) = QTTvector(x, 2, 2, :serial)
    interleaved(x) = QTTvector(x, 2, 2, :interleaved)
    for alg in (AMEn(show_progress = false), AMEn(return_info = true, show_progress = false))
        @test_throws "ordering mismatch" linear_solve(Aq, interleaved(b), serial(x0), alg)
        @test_throws "ordering mismatch" linear_solve(Aq, serial(b), interleaved(x0), alg)
        @test_throws "ordering mismatch" linear_solve(A, serial(b), interleaved(x0), alg)
    end
    @test_throws "ordering mismatch" eigen_solve(Aq, interleaved(x0), AMEn(show_progress = false))
    @test_throws "n_dims mismatch" linear_solve(Aq, QTTvector(b, 1, 4, :serial), serial(x0), AMEn(show_progress = false))

    # A plain TT carries no QTT metadata, so it is accepted next to a QTT wrapper.
    x = linear_solve(Aq, b, x0, AMEn(tol = 1.0e-8, show_progress = false))
    @test x isa TTvector
    @test amen_relres(A, x, b) ≤ 1.0e-7
end

@testset "AMEn eigen_solve converges for the zero operator" begin
    Random.seed!(19)
    A = zeros_tto(2, 4, 1)
    x0 = rand_tt((2, 2, 2, 2), 2)
    E, x, _ = @test_logs min_level = Logging.Warn eigen_solve(A, x0, AMEn(show_progress = false))
    @test all(iszero, E)
    @test length(E) ≤ 2 * 4
    @test norm(amen_vector(x)) ≈ 1
end
