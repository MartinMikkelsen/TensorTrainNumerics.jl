using Test
using Random
using LinearAlgebra
using Logging
using TensorTrainNumerics

import TensorTrainNumerics: Ksolve, update_H!

Random.seed!(9999)


als_rel_residual(A, x, b) = norm(A * x - b) / max(norm(b), eps())
als_spd_op(d, shift = 3.0) = Δ(d) + shift * id_tto(d)


@testset "update_H!" begin
    n, r1, r2, r3 = 2, 1, 1, 1
    x_vec = randn(Float64, n, r1, r2)
    A_vec = randn(Float64, n, n, r3, r1)
    Hi = randn(Float64, r3, r2, r2)
    Him = zeros(Float64, r1, r2, r2)
    update_H!(x_vec, A_vec, Hi, Him)
    @test size(Him) == (r1, r2, r2)
    @test eltype(Him) <: Number
end


@testset "als_linsolve: return type and structure" begin
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    A = rand_tto(dims, 3)
    b = rand_tt(dims, rks)
    x0 = rand_tt(dims, rks)

    x = als_linsolve(A, b, x0)

    @test x isa TTvector{Float64, 3}
    @test x.N == 3
    @test x.ttv_dims == dims
end

@testset "als_linsolve: residual decreases for well-conditioned system" begin
    d = 4
    A = als_spd_op(d, 10.0)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])

    x = als_linsolve(A, b, x0; max_sweeps = 2)

    @test als_rel_residual(A, x, b) < 0.5
end

@testset "als_linsolve: identity operator gives x ≈ b" begin
    d = 4
    A = id_tto(d)
    b = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])

    x = als_linsolve(A, b, x0; max_sweeps = 2)

    @test als_rel_residual(A, x, b) < 0.05
end

@testset "als_linsolve: max_sweeps = 1" begin
    d = 3
    A = als_spd_op(d, 5.0)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 1])

    x = als_linsolve(A, b, x0; max_sweeps = 1)

    @test x isa TTvector{Float64}
    @test x.ttv_dims == b.ttv_dims
end

@testset "Ksolve iterative local solve agrees with the dense solve" begin
    Gi = zeros(Float64, 2, 1, 2, 1, 1)
    Gi[1, 1, 1, 1, 1] = 2.0
    Gi[2, 1, 2, 1, 1] = 3.0
    G_bi = reshape([4.0, 9.0], 2, 1, 1)
    Hi = ones(Float64, 1, 1, 1)
    H_bi = ones(Float64, 1, 1)

    dense = Ksolve(Gi, G_bi, Hi, H_bi)
    iterative = Ksolve(
        Gi, G_bi, Hi, H_bi;
        local_solver = :auto,
        local_threshold = 1,
        local_maxiter = 20,
        local_tol = 1.0e-12,
    )

    @test vec(dense) ≈ [2.0, 3.0]
    @test iterative ≈ dense atol = 1.0e-10

    disabled = Ksolve(
        Gi, G_bi, Hi, H_bi;
        local_solver = :direct,
        local_threshold = 0,
        local_tol = Inf,
    )
    at_threshold = Ksolve(
        Gi, G_bi, Hi, H_bi;
        local_solver = :auto,
        local_threshold = 2,
        local_tol = Inf,
    )
    forced_iterative = Ksolve(
        Gi, G_bi, Hi, H_bi;
        local_solver = :auto,
        local_threshold = 1,
        local_tol = Inf,
    )

    @test disabled == dense
    @test at_threshold == dense
    @test iszero(forced_iterative)
end

@testset "als_linsolve forwards iterative options through a complete sweep" begin
    d = 3
    A = id_tto(d)
    dims = ntuple(_ -> 2, d)
    ranks = [1, 2, 2, 1]
    b = rand_tt(dims, ranks)
    x0 = rand_tt(dims, ranks)

    dense = als_linsolve(A, b, x0; max_sweeps = 1)
    iterative = als_linsolve(
        A, b, x0;
        max_sweeps = 1,
        local_solver = :auto,
        local_threshold = 1,
        local_maxiter = 50,
        local_tol = 1.0e-12,
    )

    dense_values = vec(ttv_to_tensor(dense))
    iterative_values = vec(ttv_to_tensor(iterative))
    @test norm(iterative_values - dense_values) / norm(dense_values) < 1.0e-10
end


@testset "als_eigsolve: return type and structure" begin
    d = 4
    A = als_spd_op(d)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    E, x_opt = als_eigsolve(A, x0; max_sweeps = 1, max_bond = 2, noise = 0.0)

    @test E isa Vector{Float64}
    @test x_opt isa TTvector{Float64}
    @test x_opt.N == d
    @test x_opt.ttv_dims == ntuple(_ -> 2, d)
    @test length(E) ≥ 1
    @test all(isfinite, E)
end

@testset "als_eigsolve: eigenvalue positive for SPD operator" begin
    d = 4
    shift = 3.0
    A = als_spd_op(d, shift)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    E, x_opt = als_eigsolve(A, x0; max_sweeps = 3, max_bond = 2)

    λ = E[end]
    @test λ > 0.0
    rq = real(TensorTrainNumerics.dot(x_opt, A * x_opt)) / real(TensorTrainNumerics.dot(x_opt, x_opt))
    @test isapprox(rq, λ; rtol = 0.1)
end

@testset "als_eigsolve: eigenvalue non-increasing over sweeps" begin
    d = 4
    A = als_spd_op(d, 2.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    E, _ = als_eigsolve(A, x0; max_sweeps = 3, max_bond = 2)

    @test E[end] ≤ E[1] + 1.0e-8
end

@testset "als_eigsolve: multi-stage schedule with rank growth and noise" begin
    d = 4
    A = als_spd_op(d, 2.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1]; normalise = true)

    E, x_opt = als_eigsolve(
        A, x0;
        max_sweeps = [1, 2],
        max_bond = [1, 2],
        noise = [0.0, 1.0e-3]
    )

    @test x_opt isa TTvector{Float64}
    @test maximum(x_opt.ttv_rks) ≤ 2
    @test all(isfinite, E)
end

@testset "als_eigsolve: iterative solver path" begin
    d = 4
    A = als_spd_op(d, 2.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    E, x_opt = als_eigsolve(
        A, x0;
        max_sweeps = 1, max_bond = 2,
        local_solver = :auto, local_threshold = 1
    )

    @test x_opt isa TTvector{Float64}
    @test isfinite(E[end])
end


@testset "als_gen_eigsolv: return type and structure (S = I)" begin
    d = 4
    A = als_spd_op(d, 3.0)
    S = id_tto(d)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    result = als_gen_eigsolv(A, S, x0; sweep_schedule = [2], rmax_schedule = [2])

    @test result !== nothing
    E, x_opt = result
    @test E isa AbstractVector
    @test x_opt isa TTvector{Float64}
    @test x_opt.N == d
end

@testset "als_gen_eigsolv: Ax = λx with S=I matches als_eigsolve" begin
    d = 4
    shift = 2.0
    A = als_spd_op(d, shift)
    S = id_tto(d)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    result = als_gen_eigsolv(A, S, x0; sweep_schedule = [2], rmax_schedule = [2])
    @test result !== nothing
    E_gen, _ = result

    E_std, _ = als_eigsolve(A, x0; max_sweeps = 1, max_bond = 2)

    @test isapprox(E_gen[end], E_std[end]; rtol = 0.05)
end

@testset "als_gen_eigsolv: multi-stage schedule grows ranks" begin
    d = 3
    A = als_spd_op(d, 2.0)
    S = id_tto(d)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1]; normalise = true)

    result = als_gen_eigsolv(A, S, x0; sweep_schedule = [1, 2], rmax_schedule = [1, 2])

    @test result !== nothing
    E, x_opt = result
    @test x_opt isa TTvector{Float64}
    @test maximum(x_opt.ttv_rks) ≤ 2
    @test all(isfinite, E)
end

@testset "als_gen_eigsolv: iterative local eigensolver path" begin
    d = 3
    A = als_spd_op(d, 2.0)
    S = id_tto(d)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 1]; normalise = true)

    result = als_gen_eigsolv(
        A, S, x0;
        sweep_schedule = [2],
        rmax_schedule = [2],
        it_solver = true,
        itslv_thresh = 1,
    )

    @test result !== nothing
    E, x_opt = result
    @test x_opt isa TTvector{Float64}
    @test isfinite(E[end])
end

@testset "als_linsolve return_info for one and two sweeps" begin
    Random.seed!(1)
    d = 5
    A = id_tto(d) + 0.1 * Δ(d)
    b = qtt_sin(d)
    x0 = rand_tt(b.ttv_dims, 4)
    x1, info1 = als_linsolve(A, b, x0; max_sweeps = 1, return_info = true)
    @test info1.residual ≥ 0
    x2, info2 = als_linsolve(A, b, x0; max_sweeps = 2, return_info = true)
    @test info2.residual < 1.0e-6
end

@testset "ALS keyword names" begin
    Random.seed!(41)
    d = 6
    dims = ntuple(_ -> 2, d)
    A = Δ(d) + 3.0 * id_tto(d)
    b = A * rand_tt(dims, 2; normalise = true)          # exact solution has rank 2
    x0 = rand_tt(dims, 2; normalise = true)

    logs, x = Test.collect_test_logs() do
        linear_solve(A, b, x0, ALS(; max_sweeps = 3, verbosity = 2, show_progress = false))
    end
    @test count(l -> l.level == Logging.Info, logs) == 3
    @test norm(A * x - b) / norm(b) < 1.0e-4

    @test_throws "`max_bond` is not used by linear_solve with ALS" linear_solve(A, b, x0, ALS(; max_bond = 4))
    @test_throws "ALS linear_solve runs a single stage" linear_solve(A, b, x0, ALS(; max_sweeps = [1, 1]))
    @test_throws MethodError ALS(; sweep_count = 2)
    @test_throws MethodError ALS(; it_solver = true)
    @test_throws "`local_solver` must be" ALS(; local_solver = :gmres)

    E, x = eigen_solve(A, x0, ALS(; max_sweeps = 3, show_progress = false))
    @test length(E) == 3 * 2 * (d - 1)

    # Stage 1 applies max_bond from the start; a scalar max_bond with two stages
    # raises the ranks once.
    g1 = rand_tt(dims, 1; normalise = true)
    E, x = eigen_solve(A, g1, ALS(; max_bond = 4, max_sweeps = [1, 1], noise = 1.0e-3, show_progress = false))
    @test maximum(x.ttv_rks) == 4
    @test length(E) == 2 * 2 * (d - 1)
    Ad = reshape(tto_to_tensor(A), 2^d, 2^d)
    @test E[end] ≈ eigmin(Symmetric(Ad)) rtol = 1.0e-6

    @test_throws "ALS cannot lower ranks" eigen_solve(A, x0, ALS(; max_bond = 1))
    @test_throws "`max_sweeps` has length 2 and `max_bond` has length 3" eigen_solve(A, x0, ALS(; max_sweeps = [1, 1], max_bond = [3, 4, 5]))
end
