using Test
using LinearAlgebra
using Random
using TensorTrainNumerics

@testset "solver algorithm constructors" begin
    @test ALS(sweep_count = 3).sweep_count == 3
    @test !ALS().show_progress
    @test ALS(show_progress = true).show_progress
    @test MALS(tol = 1.0e-9, rmax = 4).tol == 1.0e-9
    @test MALS(tol = 1.0e-9, rmax = 4).rmax == 4
    @test !MALS().show_progress
    @test MALS(show_progress = true).show_progress
    @test isnothing(MALS().linsolv_tol)
    @test DMRG(N = 2, tol = 1.0e-8, rmax_schedule = [4]).N == 2
    @test DMRG(N = 2, tol = 1.0e-8, rmax_schedule = [4]).rmax_schedule == [4]
    @test DMRG().it_solver
    @test !DMRG().show_progress
    @test DMRG(show_progress = true).show_progress
    @test isnothing(DMRG().linsolv_tol)
    @test Krylov(max_bond = 3, krylov_solver = :gmres).max_bond == 3
    @test Krylov(max_bond = 3, krylov_solver = :gmres).krylov_solver == :gmres
    @test !Krylov().show_progress
    @test Krylov(show_progress = true).show_progress
    @test ALSSolver === ALS
    @test MALSSolver === MALS
    @test DMRGSolver === DMRG
    @test KrylovSolver === Krylov
end

@testset "linear_solve front door with ALS" begin
    Random.seed!(11)
    dims = (2, 2, 2)
    A = id_tto(3)
    b = rand_tt(dims, [1, 2, 2, 1])
    guess = rand_tt(dims, [1, 2, 2, 1])

    x = linear_solve(A, b, guess, ALS(sweep_count = 2))
    x_ref = als_linsolve(A, b, guess; sweep_count = 2)
    @test x isa TensorTrainNumerics.AbstractTTvector
    @test qtt_to_vector(x) ≈ qtt_to_vector(x_ref) rtol = 1.0e-12 atol = 1.0e-12
end

@testset "linear_solve front door with DMRG preserves legacy defaults" begin
    Random.seed!(17)
    d = 4
    h = 1 / d^2
    A = -h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    dims = ntuple(_ -> 2, d)
    ranks = [1; fill(2, d - 1); 1]
    b = rand_tt(dims, ranks)
    guess = rand_tt(dims, ranks)

    x_alg = linear_solve(A, b, guess, DMRG())
    x_ref = dmrg_linsolve(A, b, guess)

    @test qtt_to_vector(x_alg) ≈ qtt_to_vector(x_ref) rtol = 1.0e-10 atol = 1.0e-10
end

@testset "legacy linear wrappers delegate to linear_solve (seed=$seed)" for seed in (1, 3)
    relvec(x, y) = norm(qtt_to_vector(x) - qtt_to_vector(y)) / max(norm(qtt_to_vector(y)), eps())

    Random.seed!(seed)
    dims = (2, 2, 2)
    ranks = [1, 2, 2, 1]
    A = id_tto(3)
    b = rand_tt(dims, ranks)
    guess = rand_tt(dims, ranks)

    # A is the identity, so the relative residual is the dense distance to b.
    # Contracting the TT difference can inflate roundoff to O(sqrt(eps())).
    x_als = linear_solve(A, b, guess, ALS(sweep_count = 2))
    x_als_wrapper = als_linsolve(A, b, guess; sweep_count = 2)
    @test relvec(x_als, b) < 1.0e-8
    @test relvec(x_als_wrapper, x_als) < 1.0e-12

    x_mals = linear_solve(A, b, guess, MALS(tol = 1.0e-10, rmax = 4))
    x_mals_wrapper = mals_linsolve(A, b, guess; tol = 1.0e-10, rmax = 4)
    @test relvec(x_mals, b) < 1.0e-8
    @test relvec(x_mals_wrapper, x_mals) < 1.0e-12

    x_dmrg = linear_solve(A, b, guess, DMRG(N = 2, sweep_schedule = [2], rmax_schedule = [4]))
    x_dmrg_wrapper = dmrg_linsolve(A, b, guess; N = 2, sweep_schedule = [2], rmax_schedule = [4])
    @test relvec(x_dmrg, b) < 1.0e-8
    @test relvec(x_dmrg_wrapper, x_dmrg) < 1.0e-12
end

@testset "eigen_solve front door" begin
    Random.seed!(23)
    dims = (2, 2, 2)
    A = id_tto(3)
    guess = rand_tt(dims, [1, 2, 2, 1])

    E_als, x_als = eigen_solve(
        A, guess,
        ALS(sweep_schedule = [2], rmax_schedule = [2], noise_schedule = [0.0])
    )
    @test !isempty(E_als)
    @test x_als isa TensorTrainNumerics.AbstractTTvector
    @test abs(last(E_als) - 1.0) < 1.0e-8

    E_mals, x_mals, r_hist_mals = eigen_solve(A, guess, MALS(sweep_schedule = [2], rmax_schedule = [4]))
    @test !isempty(E_mals)
    @test !isempty(r_hist_mals)
    @test x_mals isa TensorTrainNumerics.AbstractTTvector

    E_dmrg, x_dmrg, r_hist_dmrg = eigen_solve(A, guess, DMRG(N = 2, sweep_schedule = [2], rmax_schedule = [4]))
    @test !isempty(E_dmrg)
    @test !isempty(r_hist_dmrg)
    @test x_dmrg isa TensorTrainNumerics.AbstractTTvector
end

@testset "legacy eigen wrappers delegate to eigen_solve" begin
    Random.seed!(31)
    dims = (2, 2, 2)
    ranks = [1, 2, 2, 1]
    A = id_tto(3)
    guess = rand_tt(dims, ranks)

    E_als, x_als = eigen_solve(
        A, guess,
        ALS(sweep_schedule = [2], rmax_schedule = [2], noise_schedule = [0.0])
    )
    E_als_wrapper, x_als_wrapper = als_eigsolve(
        A, guess;
        sweep_schedule = [2], rmax_schedule = [2], noise_schedule = [0.0]
    )
    @test E_als_wrapper ≈ E_als
    @test qtt_to_vector(x_als_wrapper) ≈ qtt_to_vector(x_als) rtol = 1.0e-12 atol = 1.0e-12

    E_mals, x_mals, r_hist_mals = eigen_solve(A, guess, MALS(tol = 1.0e-10, sweep_schedule = [2], rmax_schedule = [4]))
    E_mals_wrapper, x_mals_wrapper, r_hist_mals_wrapper = mals_eigsolve(A, guess; tol = 1.0e-10, sweep_schedule = [2], rmax_schedule = [4])
    @test E_mals_wrapper ≈ E_mals
    @test r_hist_mals_wrapper == r_hist_mals
    @test qtt_to_vector(x_mals_wrapper) ≈ qtt_to_vector(x_mals) rtol = 1.0e-12 atol = 1.0e-12

    E_dmrg, x_dmrg, r_hist_dmrg = eigen_solve(A, guess, DMRG(N = 2, tol = 1.0e-10, sweep_schedule = [2], rmax_schedule = [4]))
    E_dmrg_wrapper, x_dmrg_wrapper, r_hist_dmrg_wrapper = dmrg_eigsolve(A, guess; N = 2, tol = 1.0e-10, sweep_schedule = [2], rmax_schedule = [4])
    @test E_dmrg_wrapper ≈ E_dmrg
    @test r_hist_dmrg_wrapper == r_hist_dmrg
    @test qtt_to_vector(x_dmrg_wrapper) ≈ qtt_to_vector(x_dmrg) rtol = 1.0e-12 atol = 1.0e-12
end

@testset "solver options that a problem does not read are rejected" begin
    Random.seed!(41)
    dims = (2, 2, 2, 2)
    B = rand_tto(dims, 2)
    A = B' * B + id_tto(4)                    # symmetric positive definite
    b = rand_tt(dims, [1, 2, 2, 2, 1])
    x0 = rand_tt(dims, [1, 2, 2, 2, 1])

    for alg in (
            ALS(sweep_schedule = [4]), ALS(rmax_schedule = [3]), ALS(noise_schedule = [0.1]), ALS(itslv_thresh = 10),
            MALS(sweep_schedule = [4]), MALS(rmax_schedule = [3]), MALS(it_solver = true),
            MALS(linsolv_maxiter = 5), MALS(linsolv_tol = 1.0e-3), MALS(itslv_thresh = 10),
        )
        @test_throws "not used by linear_solve" linear_solve(A, b, x0, alg)
        # The implicit time steppers rebuild the algorithm, so they check first.
        @test_throws "not used by linear_solve" implicit_euler_method(A, x0, x0, [0.1]; tt_solver = alg, show_progress = false)
    end
    for alg in (ALS(sweep_count = 6), ALS(r_itsolver = 10), ALS(return_info = true), MALS(return_info = true), DMRG(return_info = true))
        @test_throws "not used by eigen_solve" eigen_solve(A, x0, alg)
    end

    # `sweep_count` is shorthand for a one-stage `sweep_schedule` in DMRG.
    @test qtt_to_vector(linear_solve(A, b, x0, DMRG(sweep_count = 5))) ≈
        qtt_to_vector(linear_solve(A, b, x0, DMRG(sweep_schedule = [5])))
    E1, _, _ = eigen_solve(A, x0, DMRG(sweep_count = 5))
    E2, _, _ = eigen_solve(A, x0, DMRG(sweep_schedule = [5]))
    @test E1 ≈ E2
    @test_throws "either `sweep_count` or `sweep_schedule`" linear_solve(A, b, x0, DMRG(sweep_count = 5, sweep_schedule = [3]))

    # `rmax` is shorthand for a constant `rmax_schedule` in MALS eigenvalue problems.
    _, _, r_hist = eigen_solve(A, x0, MALS(rmax = 2))
    @test maximum(r_hist) ≤ 2
    @test_throws "either `rmax` or `rmax_schedule`" eigen_solve(A, x0, MALS(rmax = 2, rmax_schedule = [3]))
end

@testset "Krylov reports whether the solve converged" begin
    Random.seed!(43)
    dims = (2, 2, 2, 2)
    B = rand_tto(dims, 2)
    A = B' * B + id_tto(4)
    b = rand_tt(dims, [1, 2, 2, 2, 1])
    x0 = rand_tt(dims, [1, 2, 2, 2, 1])

    # Too few iterations: KrylovKit's non-convergence warning is shown by default.
    @test_logs (:warn, r"without converging") match_mode = :any linear_solve(A, b, x0, Krylov(krylovdim = 2, maxiter = 1, rtol = 1.0e-14))
    x, info = linear_solve(A, b, x0, Krylov(krylovdim = 2, maxiter = 1, rtol = 1.0e-14, verbosity = 0, return_info = true))
    @test !info.converged
    @test info.residual > 1.0e-10

    # A converged solve reports its relative residual.
    x, info = linear_solve(A, b, x0, Krylov(krylovdim = 16, maxiter = 20, rtol = 1.0e-10, return_info = true))
    @test info.converged
    M = reshape(tto_to_tensor(A), prod(dims), :)
    bv = vec(ttv_to_tensor(b))
    @test info.residual ≈ norm(M * vec(ttv_to_tensor(x)) - bv) / norm(bv) rtol = 1.0e-3
    @test info.residual < 1.0e-9
end
