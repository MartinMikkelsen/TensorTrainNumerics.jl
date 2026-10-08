using Test
using Random
using LinearAlgebra
using Logging
using TensorTrainNumerics

Random.seed!(1234)


# Relative residual ‖Ax - b‖ / ‖b‖
function dmrg_rel_residual(A, x, b)
    r = A * x - b
    nb = norm(b)
    return nb > 0 ? norm(r) / nb : norm(r)
end

# Build a small SPD operator: Δ(d) + shift * I  (positive-definite for shift > 0)
dmrg_spd_op(d, shift = 3.0) = Δ(d) + shift * id_tto(d)


@testset "dmrg_linsolve: return type and structure" begin
    d = 4
    A = dmrg_spd_op(d)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])

    x = dmrg_linsolve(A, b, x0; nsites = 2, max_sweeps = 1, max_bond = 4)

    @test x isa TTVector{Float64}
    @test nsites(x) == d
    @test x.dims == b.dims
    @test all(isfinite, x.ranks)
end

@testset "dmrg_linsolve nsites = 2: residual decreases" begin
    d = 4
    A = dmrg_spd_op(d, 10.0)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])

    x = dmrg_linsolve(A, b, x0; nsites = 2, max_sweeps = 3, max_bond = 8)

    @test dmrg_rel_residual(A, x, b) < 0.5
end

@testset "dmrg_linsolve: two-stage sweep schedule" begin
    d = 4
    A = dmrg_spd_op(d, 5.0)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])

    x = dmrg_linsolve(A, b, x0; nsites = 2, max_sweeps = [1, 2], max_bond = [2, 8])

    @test x isa TTVector{Float64}
    @test x.dims == b.dims
end

@testset "dmrg_linsolve: identity operator → residual near zero" begin
    d = 4
    A = id_tto(d)
    b = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])

    x = dmrg_linsolve(A, b, x0; nsites = 2, max_sweeps = 3, max_bond = 4)

    @test dmrg_rel_residual(A, x, b) < 0.05
end

@testset "dmrg_linsolve nsites = 1: full local solve returns residual info" begin
    d = 3
    A = dmrg_spd_op(d, 5.0)
    b = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1])

    x, info = dmrg_linsolve(
        A, b, x0;
        nsites = 1,
        max_sweeps = 1,
        max_bond = 2,
        local_solver = :direct,
        return_info = true,
    )

    @test x isa TTVector{Float64}
    @test haskey(info, :residual)
    @test isfinite(info.residual)
end


@testset "dmrg_eigsolve: return type and structure" begin
    d = 4
    A = dmrg_spd_op(d)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalize = true)

    E, x_opt, r_hist = dmrg_eigsolve(A, x0; nsites = 2, max_sweeps = 1, max_bond = 4)

    @test E isa Vector{Float64}
    @test x_opt isa TTVector{Float64}
    @test r_hist isa Vector{<:Integer}
    @test length(E) == length(r_hist)
    @test nsites(x_opt) == d
    @test x_opt.dims == ntuple(_ -> 2, d)
end

@testset "dmrg_eigsolve nsites = 2: eigenvalue positive for SPD operator" begin
    d = 4
    shift = 3.0
    A = dmrg_spd_op(d, shift)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalize = true)

    E, x_opt, _ = dmrg_eigsolve(A, x0; nsites = 2, max_sweeps = 3, max_bond = 4)

    λ = E[end]
    @test λ > 0.0
    # Rayleigh quotient should be close to λ
    rq = real(TensorTrainNumerics.dot(x_opt, A * x_opt)) / real(TensorTrainNumerics.dot(x_opt, x_opt))
    @test isapprox(rq, λ; rtol = 0.1)
end

@testset "dmrg_eigsolve: sweep schedule with rank growth" begin
    d = 4
    A = dmrg_spd_op(d, 2.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1]; normalize = true)

    E, x_opt, r_hist = dmrg_eigsolve(A, x0; nsites = 2, max_sweeps = [1, 2], max_bond = [2, 4])

    @test length(E) ≥ 2
    @test x_opt isa TTVector{Float64}
    @test maximum(x_opt.ranks) ≤ 4
end

@testset "dmrg_eigsolve: eigenvalues are real and finite" begin
    d = 4
    A = dmrg_spd_op(d, 1.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalize = true)

    E, _, _ = dmrg_eigsolve(A, x0; nsites = 2, max_sweeps = 1, max_bond = 4)

    @test all(isreal, E)
    @test all(isfinite, E)
end

@testset "dmrg_eigsolve: iterative local eigensolver path" begin
    d = 3
    A = dmrg_spd_op(d, 2.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 1]; normalize = true)

    E, x_opt, r_hist = dmrg_eigsolve(
        A, x0;
        nsites = 2,
        max_sweeps = 1,
        max_bond = 2,
        local_solver = :iterative,
        local_maxiter = 20,
    )

    @test all(isfinite, E)
    @test x_opt isa TTVector{Float64}
    @test length(r_hist) == length(E)
end

@testset "dmrg_eigsolve nsites = 1: finalizes single-site core" begin
    d = 3
    A = dmrg_spd_op(d, 2.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1]; normalize = true)

    E, x_opt, r_hist = dmrg_eigsolve(
        A, x0;
        nsites = 1,
        max_sweeps = 1,
        max_bond = 2,
        local_solver = :direct,
    )

    @test all(isfinite, E)
    @test x_opt isa TTVector{Float64}
    @test x_opt.orthogonality == [1, 1]
    @test length(r_hist) == length(E)
end

@testset "dmrg solvers are quiet by default" begin
    d = 5
    A = id_tto(d) + 0.1 * Δ(d)
    b = qtt_sin(d)
    x0 = rand_tt(b.dims, 2)
    @test_logs min_level = Logging.Info begin
        dmrg_linsolve(A, b, x0; max_sweeps = 1, max_bond = 4)
    end
    @test_logs min_level = Logging.Info begin
        dmrg_eigsolve(A, x0; max_sweeps = 1, max_bond = 4)
    end
end

@testset "DMRG preserves nonsymmetric and complex projected operators" begin
    Random.seed!(9102)
    # Three sites exercise nontrivial left/right environments; two sites expose
    # the complete local problem directly. Dense data supply independent oracles.
    for d in (2, 3), T in (Float64, ComplexF64)
        dims = ntuple(_ -> 2, d)
        A_dense = 4I + 0.2 * randn(T, 2^d, 2^d)
        b_dense = randn(T, 2^d)
        A = tto_decomp(reshape(A_dense, dims..., dims...))
        b = ttv_decomp(reshape(b_dense, dims))
        x0 = ttv_decomp(randn(T, dims))
        expected = A_dense \ b_dense

        for options in (NamedTuple(), (; local_solver = :direct), (; local_solver = :auto, local_threshold = 1))
            alg = DMRG(; local_tol = 1.0e-12, options...)
            x = linear_solve(A, b, x0, alg)
            values = vec(ttv_to_tensor(x))
            @test norm(A_dense * values - b_dense) / norm(b_dense) < 1.0e-10
            @test values ≈ expected rtol = 1.0e-10 atol = 1.0e-12
        end
    end
end

@testset "DMRG preserves complex Hermitian eigenvectors and linear solves" begin
    Random.seed!(9103)
    for d in (2, 3)
        dims = ntuple(_ -> 2, d)
        M = randn(ComplexF64, 2^d, 2^d)
        A_dense = 8I + M + M'
        A = tto_decomp(reshape(A_dense, dims..., dims...))
        x0 = ttv_decomp(randn(ComplexF64, dims))
        b_dense = randn(ComplexF64, 2^d)
        b = ttv_decomp(reshape(b_dense, dims))
        λ_exact = first(eigvals(Hermitian(A_dense)))

        for options in (NamedTuple(), (; local_solver = :direct), (; local_solver = :auto, local_threshold = 1))
            alg = DMRG(; local_tol = 1.0e-12, options...)
            E, x, _ = eigen_solve(A, x0, alg)
            values = vec(ttv_to_tensor(x))
            @test E[end] ≈ λ_exact atol = 1.0e-10
            @test norm(A_dense * values - E[end] * values) / norm(values) < 1.0e-10

            solution = linear_solve(A, b, x0, alg)
            @test vec(ttv_to_tensor(solution)) ≈ A_dense \ b_dense rtol = 1.0e-10 atol = 1.0e-12
        end
    end
end

@testset "DMRG keyword names" begin
    TTN = TensorTrainNumerics
    Random.seed!(61)
    d = 6
    dims = ntuple(_ -> 2, d)
    A = Δ(d) + 3.0 * id_tto(d)
    b = A * rand_tt(dims, 2; normalize = true)
    x0 = rand_tt(dims, 2; normalize = true)

    x = linear_solve(A, b, x0, DMRG(; max_sweeps = 2, max_bond = 8, show_progress = false))
    @test norm(A * x - b) / norm(b) < 1.0e-4

    E, x, r_hist = eigen_solve(A, x0, DMRG(; max_sweeps = 2, show_progress = false))
    @test length(E) == 2 * 2 * (d - 2) + 1
    E, x, r_hist = eigen_solve(A, x0, DMRG(; max_bond = [2, 4], max_sweeps = [1, 1], show_progress = false))
    @test all(≤(2), r_hist[1:(2 * (d - 2))])
    @test maximum(x.ranks) ≤ 4

    @test_logs eigen_solve(A, x0, DMRG(; show_progress = false))
    @test_logs (:info, "DMRG core move") match_mode = :any eigen_solve(A, x0, DMRG(; verbosity = 3, show_progress = false))
    logs, _ = Test.collect_test_logs() do
        eigen_solve(A, x0, DMRG(; max_sweeps = 3, verbosity = 2, show_progress = false))
    end
    @test count(l -> l.message == "DMRG eigen solve", logs) == 3

    @test_throws MethodError DMRG(; N = 2)
    @test_throws MethodError DMRG(; sweep_count = 2)
    @test_throws MethodError DMRG(; rmax_schedule = [4])
    @test_throws "`return_info` is not used by eigen_solve with DMRG" eigen_solve(A, x0, DMRG(; return_info = true))

    # The core move keeps the rank that `_trunc_rank` predicts for the block's spectrum.
    xm = orthogonalize(rand_tt((2, 2, 2, 2), 4))       # ranks [1, 2, 4, 2, 1]
    s = [1.0, 0.1, 0.01, 0.001]
    Q1 = Matrix(qr(randn(4, 4)).Q)
    Q2 = Matrix(qr(randn(4, 4)).Q)
    V = reshape(Q1 * Diagonal(s) * Q2', 2, 4, 2)       # (r₂, n₂·n₃, r₄)
    ε = 0.05 / norm(s)
    TTN.right_core_move!(xm, V, zeros(8, 8, 8), 2, ε, typemax(Int))
    @test xm.ranks[3] == TTN._trunc_rank(s, ε, 4, typemax(Int)) == 2
end
