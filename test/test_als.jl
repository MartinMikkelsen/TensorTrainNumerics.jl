using Test
using Random
using LinearAlgebra

import TensorTrainNumerics: update_H!

Random.seed!(9999)

# ── helpers ───────────────────────────────────────────────────────────────────

als_rel_residual(A, x, b) = norm(A * x - b) / max(norm(b), eps())
als_spd_op(d, shift = 3.0) = Δ(d) + shift * id_tto(d)

# ── internal: update_H! ───────────────────────────────────────────────────────

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

# ── als_linsolve ──────────────────────────────────────────────────────────────

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

    x = als_linsolve(A, b, x0; sweep_count = 4)

    @test als_rel_residual(A, x, b) < 0.5
end

@testset "als_linsolve: identity operator gives x ≈ b" begin
    d = 4
    A = id_tto(d)
    b = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])

    x = als_linsolve(A, b, x0; sweep_count = 4)

    @test als_rel_residual(A, x, b) < 0.05
end

@testset "als_linsolve: sweep_count=1 (single forward half-sweep)" begin
    d = 3
    A = als_spd_op(d, 5.0)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 1])

    x = als_linsolve(A, b, x0; sweep_count = 1)

    @test x isa TTvector{Float64}
    @test x.ttv_dims == b.ttv_dims
end

# ── als_eigsolve ──────────────────────────────────────────────────────────────

@testset "als_eigsolve: return type and structure" begin
    d = 4
    A = als_spd_op(d)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    E, x_opt = als_eigsolve(A, x0; sweep_schedule = [2], rmax_schedule = [2], noise_schedule = [0.0])

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

    E, x_opt = als_eigsolve(A, x0; sweep_schedule = [4], rmax_schedule = [2])

    λ = E[end]
    @test λ > 0.0
    rq = real(TensorTrainNumerics.dot(x_opt, A * x_opt)) / real(TensorTrainNumerics.dot(x_opt, x_opt))
    @test isapprox(rq, λ; rtol = 0.1)
end

@testset "als_eigsolve: eigenvalue non-increasing over sweeps" begin
    d = 4
    A = als_spd_op(d, 2.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    E, _ = als_eigsolve(A, x0; sweep_schedule = [4], rmax_schedule = [2])

    @test E[end] ≤ E[1] + 1.0e-8
end

@testset "als_eigsolve: multi-stage schedule with rank growth and noise" begin
    d = 4
    A = als_spd_op(d, 2.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1]; normalise = true)

    E, x_opt = als_eigsolve(
        A, x0;
        sweep_schedule = [2, 4],
        rmax_schedule = [1, 2],
        noise_schedule = [0.0, 1.0e-3]
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
        sweep_schedule = [2], rmax_schedule = [2],
        it_solver = true, itslv_thresh = 1
    )

    @test x_opt isa TTvector{Float64}
    @test isfinite(E[end])
end

# ── als_gen_eigsolv ───────────────────────────────────────────────────────────

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

    E_std, _ = als_eigsolve(A, x0; sweep_schedule = [2], rmax_schedule = [2])

    @test isapprox(E_gen[end], E_std[end]; rtol = 0.05)
end
