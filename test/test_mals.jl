using Test
using Random
using LinearAlgebra
using Logging

Random.seed!(5678)


function mals_rel_residual(A, x, b)
    r = A * x - b
    nb = norm(b)
    return nb > 0 ? norm(r) / nb : norm(r)
end

mals_spd_op(d, shift = 3.0) = Δ(d) + shift * id_tto(d)


@testset "mals_linsolve: return type and structure" begin
    d = 4
    A = mals_spd_op(d)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])

    x = mals_linsolve(A, b, x0)

    @test x isa TTVector{Float64}
    @test nsites(x) == d
    @test x.dims == b.dims
    @test all(isfinite, x.ranks)
end

@testset "mals_linsolve: residual decreases for well-conditioned system" begin
    d = 4
    A = mals_spd_op(d, 10.0)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])

    x = mals_linsolve(A, b, x0; trunc_tol = 1.0e-5, max_bond = 8)

    @test mals_rel_residual(A, x, b) < 0.5
end

@testset "mals_linsolve: identity operator gives x ≈ b" begin
    d = 4
    A = id_tto(d)
    b = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])

    x = mals_linsolve(A, b, x0; trunc_tol = 1.0e-6, max_bond = 4)

    @test mals_rel_residual(A, x, b) < 0.05
end

@testset "mals_linsolve: rank adaptation respects max_bond" begin
    d = 4
    A = mals_spd_op(d, 5.0)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])
    max_bond = 4

    x = mals_linsolve(A, b, x0; trunc_tol = 1.0e-5, max_bond)

    @test maximum(x.ranks) ≤ max_bond
end

@testset "mals_linsolve: looser trunc_tol gives smaller or equal ranks" begin
    d = 4
    A = mals_spd_op(d, 3.0)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])

    x_loose = mals_linsolve(A, b, x0; trunc_tol = 0.1, max_bond = 8)
    x_tight = mals_linsolve(A, b, x0; trunc_tol = 0.0, max_bond = 8)

    @test maximum(x_loose.ranks) ≤ maximum(x_tight.ranks) + 2
end


@testset "mals_eigsolve: return type and structure" begin
    d = 4
    A = mals_spd_op(d)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalize = true)

    E, x_opt, r_hist = mals_eigsolve(A, x0; max_sweeps = 1, max_bond = 4)

    @test E isa Vector{Float64}
    @test x_opt isa TTVector{Float64}
    @test r_hist isa Vector{<:Integer}
    @test length(E) == length(r_hist)
    @test nsites(x_opt) == d
    @test x_opt.dims == ntuple(_ -> 2, d)
end

@testset "mals_eigsolve: eigenvalue positive for SPD operator" begin
    d = 4
    shift = 3.0
    A = mals_spd_op(d, shift)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalize = true)

    E, x_opt, _ = mals_eigsolve(A, x0; max_sweeps = 3, max_bond = 4)

    λ = E[end]
    @test λ > 0.0
    rq = real(TensorTrainNumerics.dot(x_opt, A * x_opt)) / real(TensorTrainNumerics.dot(x_opt, x_opt))
    @test isapprox(rq, λ; rtol = 0.1)
end

@testset "mals_eigsolve: eigenvalue non-increasing over sweeps" begin
    d = 4
    A = mals_spd_op(d, 2.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalize = true)

    E, _, _ = mals_eigsolve(A, x0; max_sweeps = 3, max_bond = 4)

    @test E[end] ≤ E[1] + 1.0e-8
end

@testset "mals_eigsolve: multi-stage sweep schedule with rank growth" begin
    d = 4
    A = mals_spd_op(d, 2.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1]; normalize = true)

    E, x_opt, r_hist = mals_eigsolve(A, x0; max_sweeps = [1, 2], max_bond = [2, 4])

    @test length(E) ≥ 2
    @test x_opt isa TTVector{Float64}
    @test maximum(x_opt.ranks) ≤ 4
end

@testset "mals_eigsolve: rank history is non-empty and positive" begin
    d = 4
    A = mals_spd_op(d, 1.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalize = true)

    E, x_opt, r_hist = mals_eigsolve(A, x0; max_sweeps = 1, max_bond = 4)

    @test all(r > 0 for r in r_hist)
    @test all(isfinite, E)
    @test all(isreal, E)
end

@testset "mals_eigsolve: iterative solver path" begin
    d = 4
    A = mals_spd_op(d, 2.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalize = true)

    E, x_opt, _ = mals_eigsolve(
        A, x0;
        max_sweeps = 1, max_bond = 4,
        local_solver = :auto, local_threshold = 1
    )

    @test x_opt isa TTVector{Float64}
    @test isfinite(E[end])
end

@testset "MALS preserves real and complex nonsymmetric linear systems" begin
    Random.seed!(9202)
    for d in (2, 3), T in (Float64, ComplexF64)
        dims = ntuple(_ -> 2, d)
        A_dense = 4I + 0.2 * randn(T, 2^d, 2^d)
        b_dense = randn(T, 2^d)
        A = tto_decomp(reshape(A_dense, dims..., dims...))
        b = tt_decomp(reshape(b_dense, dims))
        x0 = tt_decomp(randn(T, dims))

        x = linear_solve(A, b, x0, MALS())
        values = vec(tt_to_tensor(x))
        @test norm(A_dense * values - b_dense) / norm(b_dense) < 1.0e-10
        @test values ≈ A_dense \ b_dense rtol = 1.0e-10 atol = 1.0e-12
    end
end

@testset "MALS dense eigen solve retains Hermitian semantics" begin
    Random.seed!(9203)
    dims = (2, 2, 2)
    M = randn(ComplexF64, 8, 8)
    A_dense = 8I + M + M'
    A = tto_decomp(reshape(A_dense, dims..., dims...))
    x0 = tt_decomp(randn(ComplexF64, dims))

    E, x, _ = eigen_solve(A, x0, MALS())
    values = vec(tt_to_tensor(x))
    @test E[end] ≈ first(eigvals(Hermitian(A_dense))) atol = 1.0e-10
    @test norm(A_dense * values - E[end] * values) / norm(values) < 1.0e-10
end

@testset "MALS keyword names" begin
    TTN = TensorTrainNumerics
    Random.seed!(51)
    d = 6
    dims = ntuple(_ -> 2, d)
    A = Δ(d) + 3.0 * id_tto(d)
    b = A * rand_tt(dims, 2; normalize = true)
    x0 = rand_tt(dims, 2; normalize = true)

    logs, x = Test.collect_test_logs() do
        linear_solve(A, b, x0, MALS(; max_sweeps = 2, max_bond = 8, verbosity = 2, show_progress = false))
    end
    @test count(l -> l.level == Logging.Info, logs) == 2
    @test norm(A * x - b) / norm(b) < 1.0e-4

    @test_throws "`local_tol` is not used by linear_solve with MALS" linear_solve(A, b, x0, MALS(; local_tol = 1.0e-3))
    @test_throws MethodError MALS(; rmax = 4)
    @test_throws MethodError MALS(; tol = 1.0e-8)
    @test_throws MethodError MALS(; sweep_schedule = [2])

    E, x, r_hist = eigen_solve(A, x0, MALS(; max_sweeps = 2, show_progress = false))
    @test length(E) == 2 * 2 * (d - 1)
    E, x, r_hist = eigen_solve(A, x0, MALS(; max_bond = [2, 4], max_sweeps = [1, 1], show_progress = false))
    @test all(≤(2), r_hist[1:(2 * (d - 1))])
    @test maximum(r_hist) ≤ 4

    # The core move keeps the rank that `_trunc_rank` predicts for the block's spectrum.
    xm = orthogonalize(rand_tt((2, 2, 2, 2), 4))       # ranks [1, 2, 4, 2, 1]
    s = [1.0, 0.1, 0.01, 0.001]
    Q1 = Matrix(qr(randn(4, 4)).Q)
    Q2 = Matrix(qr(randn(4, 4)).Q)
    V = reshape(Q1 * Diagonal(s) * Q2', 2, 2, 2, 2)    # (n₂, r₂, n₃, r₄)
    ε = 0.05 / norm(s)                                 # δ = 0.05/√3 for d = 4
    TTN.right_core_move_mals(xm, 2, V, ε, typemax(Int))
    @test xm.ranks[3] == TTN._trunc_rank(s, ε, 4, typemax(Int)) == 2
end
