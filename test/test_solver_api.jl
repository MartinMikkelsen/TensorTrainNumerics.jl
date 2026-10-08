using Test
using LinearAlgebra
using Random
using TensorTrainNumerics

@testset "solver algorithm constructors" begin
    @test ALS(max_sweeps = 3).max_sweeps == 3
    @test ALS().show_progress
    @test !ALS(show_progress = false).show_progress
    @test MALS(trunc_tol = 1.0e-9, max_bond = 4).trunc_tol == 1.0e-9
    @test MALS(trunc_tol = 1.0e-9, max_bond = 4).max_bond == 4
    @test MALS().show_progress
    @test MALS().local_tol == 1.0e-6
    @test DMRG(nsites = 2, trunc_tol = 1.0e-8, max_bond = [4]).nsites == 2
    @test DMRG(nsites = 2, trunc_tol = 1.0e-8, max_bond = [4]).max_bond == [4]
    @test DMRG().local_solver === :iterative
    @test DMRG().show_progress
    @test DMRG().local_tol == 1.0e-6
    @test Krylov(max_bond = 3, krylov_solver = :gmres).max_bond == 3
    @test Krylov(max_bond = 3, krylov_solver = :gmres).krylov_solver == :gmres
    @test Krylov().show_progress
    @test !Krylov(show_progress = false).show_progress
    @test ALSSolver === ALS
    @test MALSSolver === MALS
    @test DMRGSolver === DMRG
    @test KrylovSolver === Krylov
    @test AMEn().tol == 1.0e-6
    @test AMEn().max_sweeps == 20
    @test AMEn().kickrank == 4
    @test isnothing(AMEn().max_bond)
    @test isnothing(AMEn().local_tol)
    @test AMEn(tol = 1.0e-9, max_bond = 12, kickrank = 2).max_bond == 12
    @test AMEn(local_tol = 1.0e-3).local_tol == 1.0e-3
    @test AMEn().show_progress
    @test AMEnSolver === AMEn
    @test AMEn() isa EigenSolverAlgorithm
    @test_throws "`tol` must be ≥ 0" AMEn(tol = -1.0)
    @test_throws "`max_sweeps` must be ≥ 1" AMEn(max_sweeps = 0)
    @test_throws "`kickrank` must be ≥ 0" AMEn(kickrank = -1)
    @test_throws "`max_bond` must be ≥ 1" AMEn(max_bond = 0)
    @test_throws "`local_solver` must be" AMEn(local_solver = :gmres)
    opts = TensorTrainNumerics._amen_options(AMEn(tol = 1.0e-4))
    @test opts.max_bond == typemax(Int)
    @test opts.local_threshold == 256
    @test opts.local_tol == 0.5e-4
    @test !haskey(opts, :return_info)
end

@testset "linear_solve front door with ALS" begin
    Random.seed!(11)
    dims = (2, 2, 2)
    A = id_tto(3)
    b = rand_tt(dims, [1, 2, 2, 1])
    guess = rand_tt(dims, [1, 2, 2, 1])

    x = linear_solve(A, b, guess, ALS(max_sweeps = 1))
    x_ref = als_linsolve(A, b, guess; max_sweeps = 1)
    @test x isa TensorTrainNumerics.AbstractTTVector
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
    x_als = linear_solve(A, b, guess, ALS(max_sweeps = 1))
    x_als_wrapper = als_linsolve(A, b, guess; max_sweeps = 1)
    @test relvec(x_als, b) < 1.0e-8
    @test relvec(x_als_wrapper, x_als) < 1.0e-12

    x_mals = linear_solve(A, b, guess, MALS(trunc_tol = 1.0e-5, max_bond = 4))
    x_mals_wrapper = mals_linsolve(A, b, guess; trunc_tol = 1.0e-5, max_bond = 4)
    @test relvec(x_mals, b) < 1.0e-8
    @test relvec(x_mals_wrapper, x_mals) < 1.0e-12

    x_dmrg = linear_solve(A, b, guess, DMRG(nsites = 2, max_sweeps = 1, max_bond = 4))
    x_dmrg_wrapper = dmrg_linsolve(A, b, guess; nsites = 2, max_sweeps = 1, max_bond = 4)
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
        ALS(max_sweeps = 1, max_bond = 2, noise = 0.0)
    )
    @test !isempty(E_als)
    @test x_als isa TensorTrainNumerics.AbstractTTVector
    @test abs(last(E_als) - 1.0) < 1.0e-8

    E_mals, x_mals, r_hist_mals = eigen_solve(A, guess, MALS(max_sweeps = 1, max_bond = 4))
    @test !isempty(E_mals)
    @test !isempty(r_hist_mals)
    @test x_mals isa TensorTrainNumerics.AbstractTTVector

    E_dmrg, x_dmrg, r_hist_dmrg = eigen_solve(A, guess, DMRG(nsites = 2, max_sweeps = 1, max_bond = 4))
    @test !isempty(E_dmrg)
    @test !isempty(r_hist_dmrg)
    @test x_dmrg isa TensorTrainNumerics.AbstractTTVector
end

@testset "legacy eigen wrappers delegate to eigen_solve" begin
    Random.seed!(31)
    dims = (2, 2, 2)
    ranks = [1, 2, 2, 1]
    A = id_tto(3)
    guess = rand_tt(dims, ranks)

    E_als, x_als = eigen_solve(
        A, guess,
        ALS(max_sweeps = 1, max_bond = 2, noise = 0.0)
    )
    E_als_wrapper, x_als_wrapper = als_eigsolve(
        A, guess;
        max_sweeps = 1, max_bond = 2, noise = 0.0
    )
    @test E_als_wrapper ≈ E_als
    @test qtt_to_vector(x_als_wrapper) ≈ qtt_to_vector(x_als) rtol = 1.0e-12 atol = 1.0e-12

    E_mals, x_mals, r_hist_mals = eigen_solve(A, guess, MALS(trunc_tol = 1.0e-5, max_sweeps = 1, max_bond = 4))
    E_mals_wrapper, x_mals_wrapper, r_hist_mals_wrapper = mals_eigsolve(A, guess; trunc_tol = 1.0e-5, max_sweeps = 1, max_bond = 4)
    @test E_mals_wrapper ≈ E_mals
    @test r_hist_mals_wrapper == r_hist_mals
    @test qtt_to_vector(x_mals_wrapper) ≈ qtt_to_vector(x_mals) rtol = 1.0e-12 atol = 1.0e-12

    E_dmrg, x_dmrg, r_hist_dmrg = eigen_solve(A, guess, DMRG(nsites = 2, trunc_tol = 1.0e-10, max_sweeps = 1, max_bond = 4))
    E_dmrg_wrapper, x_dmrg_wrapper, r_hist_dmrg_wrapper = dmrg_eigsolve(A, guess; nsites = 2, trunc_tol = 1.0e-10, max_sweeps = 1, max_bond = 4)
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
            ALS(max_bond = 3), ALS(noise = 0.1), ALS(max_sweeps = [1, 1]),
            MALS(local_solver = :iterative), MALS(local_maxiter = 5), MALS(local_tol = 1.0e-3), MALS(local_threshold = 10),
            MALS(max_bond = [2, 3]),
        )
        @test_throws ArgumentError linear_solve(A, b, x0, alg)
        @test_throws ArgumentError implicit_euler_method(A, x0, x0, [0.1]; alg, show_progress = false)
    end
    for alg in (ALS(return_info = true), MALS(return_info = true), DMRG(return_info = true))
        @test_throws "not used by eigen_solve" eigen_solve(A, x0, alg)
    end

    _, _, r_hist = eigen_solve(A, x0, MALS(max_bond = 2))
    @test maximum(r_hist) ≤ 2
end

@testset "Krylov reports whether the solve converged" begin
    Random.seed!(43)
    dims = (2, 2, 2, 2)
    B = rand_tto(dims, 2)
    # The shift keeps A well conditioned, so GMRES reaches `rtol` before its Krylov
    # dimension equals the system size and the residual stays far above rounding error.
    A = B' * B + 1000 * id_tto(4)
    b = rand_tt(dims, [1, 2, 2, 2, 1])
    x0 = rand_tt(dims, [1, 2, 2, 2, 1])

    # Too few iterations: KrylovKit's non-convergence warning is shown by default.
    @test_logs (:warn, r"without converging") match_mode = :any linear_solve(A, b, x0, Krylov(krylovdim = 2, maxiter = 1, rtol = 1.0e-14))
    x, info = linear_solve(A, b, x0, Krylov(krylovdim = 2, maxiter = 1, rtol = 1.0e-14, verbosity = 0, return_info = true))
    @test !info.converged
    @test info.residual > 1.0e-10

    # A converged solve reports its relative residual.
    x, info = linear_solve(A, b, x0, Krylov(krylovdim = 16, maxiter = 20, rtol = 1.0e-6, return_info = true))
    @test info.converged
    M = reshape(tto_to_tensor(A), prod(dims), :)
    bv = vec(ttv_to_tensor(b))
    @test info.residual ≈ norm(M * vec(ttv_to_tensor(x)) - bv) / norm(bv) rtol = 1.0e-3
    @test info.residual ≤ 1.0e-6
end

@testset "shared solver option helpers" begin
    TTN = TensorTrainNumerics

    s = [1.0, 0.1, 0.01, 0.001]
    ε = 0.05 / norm(s)                                   # δ = 0.05 for d = 2
    @test TTN._trunc_rank(s, 0.0, 2, typemax(Int)) == 4
    @test TTN._trunc_rank(s, ε, 2, typemax(Int)) == 2     # drops 0.01 and 0.001
    @test TTN._trunc_rank(s, ε, 101, typemax(Int)) == 3   # δ = 0.005 drops only 0.001
    @test TTN._trunc_rank(s, ε, 2, 1) == 1
    @test TTN._trunc_rank(s, 10.0, 2, typemax(Int)) == 1  # never below rank 1

    M = randn(6, 5)
    U, S, Vt = TTN._truncated_svd(M, 0.0, 2, 3)
    @test size(U, 2) == size(S, 1) == size(Vt, 1) == 3
    F = svd(M)
    @test norm(M - U * S * Vt) ≈ norm(F.S[4:end])

    st = TTN._stages(; max_sweeps = [2, 3], max_bond = 8)
    @test st.max_sweeps == [2, 3]
    @test st.max_bond == [8, 8]
    @test TTN._stages(; max_sweeps = 4, max_bond = 5) == (; max_sweeps = [4], max_bond = [5])
    @test_throws "`max_sweeps` has length 2 and `max_bond` has length 3" TTN._stages(; max_sweeps = [1, 1], max_bond = [2, 3, 4])
    @test_throws "`max_sweeps` entries must be ≥ 1" TTN._stages(; max_sweeps = [0], max_bond = 4)
    @test_throws "per-stage vectors must not be empty" TTN._stages(; max_sweeps = Int[], max_bond = 4)

    @test TTN._as_stage(1:3, Int) == [1, 2, 3]
    @test TTN._as_stage(2, Float64) === 2.0

    @test TTN._use_iterative(:auto, 300, 256)
    @test !TTN._use_iterative(:auto, 256, 256)
    @test TTN._use_iterative(:iterative, 1, 256)
    @test !TTN._use_iterative(:direct, 10^6, 256)
    @test TTN._check_local_solver(:direct) === :direct
    @test_throws "`local_solver` must be :auto, :direct, or :iterative; got :gmres" TTN._check_local_solver(:gmres)
end

# Text that the solver progress bars print while `f` runs, redrawing on every update.
function progress_output(f)
    TTN = TensorTrainNumerics
    old = TTN.PROGRESS_DT[]
    TTN.PROGRESS_DT[] = 0.0
    try
        return mktemp() do path, io
            redirect_stderr(f, io)
            flush(io)
            read(path, String)
        end
    finally
        TTN.PROGRESS_DT[] = old
    end
end

@testset "progress bars show sweep information" begin
    TTN = TensorTrainNumerics
    Random.seed!(81)
    d = 5
    dims = ntuple(_ -> 2, d)
    A = Δ(d) + 3.0 * id_tto(d)
    b = A * rand_tt(dims, 2; normalize = true)
    x0 = rand_tt(dims, 2; normalize = true)

    out = progress_output(() -> linear_solve(A, b, x0, ALS(; max_sweeps = 3)))
    @test occursin(r"sweep: 3/3", out) && occursin("largest rank:", out)
    out = progress_output(() -> eigen_solve(A, x0, ALS(; max_sweeps = 3)))
    @test occursin(r"sweep: 3/3", out) && occursin("eigenvalue:", out)
    out = progress_output(() -> eigen_solve(A, x0, MALS(; max_sweeps = 3)))
    @test occursin("eigenvalue:", out) && occursin("truncation error:", out)
    out = progress_output(() -> linear_solve(A, b, x0, DMRG(; max_sweeps = [1, 1, 1], max_bond = [2, 3, 4])))
    @test occursin(r"sweep: 3/3", out) && occursin("largest rank:", out) && occursin("truncation error:", out)

    out = progress_output(() -> linear_solve(A, b, rand_tt(dims, 1), AMEn(; tol = 1.0e-10)))
    @test occursin("sweep:", out) && occursin("largest rank:", out) && occursin("residual:", out)
    out = progress_output(() -> eigen_solve(A, rand_tt(dims, 1), AMEn(; tol = 1.0e-10)))
    @test occursin("eigenvalue:", out) && occursin("residual:", out)

    out = progress_output(() -> tdvp2(A, x0, fill(0.01, 3); max_bond = 4))
    @test occursin(r"step: 3/3", out) && occursin("time:", out) && occursin("truncation error:", out)
    out = progress_output(() -> implicit_euler_method(-1.0 * A, x0, x0, fill(0.01, 3); alg = ALS()))
    @test occursin(r"step: 3/3", out) && occursin("largest rank:", out)

    u0 = function_to_qtt(x -> sin(π * x), d)
    out = progress_output(() -> non_linear_solve((4.0^d / 2) * Δ(d), u0 / norm(u0), PenaltyALS(; penalty_schedule = [1.0e2], max_sweeps = 3, tol = 0.0); g = 1.0))
    @test occursin("penalty:", out)
    out = progress_output(() -> non_linear_solve(k -> (4.0^k / 2) * Δ(k), u0 / norm(u0), MGR(; inner = PenaltyALS(; penalty_schedule = [1.0e2], max_sweeps = 2)); g_builder = k -> 1.0, target_sites = d + 2))
    @test occursin(r"level: 3/3", out)

    f(X) = vec(1 ./ (1 .+ sum(X, dims = 2)))
    domain = [collect(range(0.0, 1.0, length = 8)) for _ in 1:3]
    out = progress_output(() -> TTN.tt_cross(f, domain, TTN.MaxVol(; max_sweeps = 3, max_bond = 2, verbosity = 0)))
    @test occursin("validation error:", out)
    # ProgressMeter never draws on the first update and draws the final state only
    # after an earlier draw, so every run above has at least three iterations.
end
