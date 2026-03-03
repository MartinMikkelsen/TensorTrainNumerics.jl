using Test
using Random
using LinearAlgebra

Random.seed!(5678)

# ── helpers ──────────────────────────────────────────────────────────────────

function mals_rel_residual(A, x, b)
    r = A * x - b
    nb = norm(b)
    return nb > 0 ? norm(r) / nb : norm(r)
end

mals_spd_op(d, shift = 3.0) = Δ(d) + shift * id_tto(d)

# ── mals_linsolve ─────────────────────────────────────────────────────────────

@testset "mals_linsolve: return type and structure" begin
    d = 4
    A = mals_spd_op(d)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])

    x = mals_linsolve(A, b, x0)

    @test x isa TTvector{Float64}
    @test x.N == d
    @test x.ttv_dims == b.ttv_dims
    @test all(isfinite, x.ttv_rks)
end

@testset "mals_linsolve: residual decreases for well-conditioned system" begin
    d = 4
    A = mals_spd_op(d, 10.0)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])

    x = mals_linsolve(A, b, x0; tol = 1.0e-10, rmax = 8)

    @test mals_rel_residual(A, x, b) < 0.5
end

@testset "mals_linsolve: identity operator gives x ≈ b" begin
    d = 4
    A = id_tto(d)
    b = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])

    x = mals_linsolve(A, b, x0; tol = 1.0e-12, rmax = 4)

    @test mals_rel_residual(A, x, b) < 0.05
end

@testset "mals_linsolve: rank adaptation respects rmax" begin
    d = 4
    A = mals_spd_op(d, 5.0)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])
    rmax = 4

    x = mals_linsolve(A, b, x0; tol = 1.0e-10, rmax = rmax)

    @test maximum(x.ttv_rks) ≤ rmax
end

@testset "mals_linsolve: tighter tol gives smaller or equal ranks" begin
    d = 4
    A = mals_spd_op(d, 3.0)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])

    x_loose = mals_linsolve(A, b, x0; tol = 1.0e-2, rmax = 8)
    x_tight = mals_linsolve(A, b, x0; tol = 0.0, rmax = 8)

    @test maximum(x_loose.ttv_rks) ≤ maximum(x_tight.ttv_rks) + 2
end

# ── mals_eigsolve ─────────────────────────────────────────────────────────────

@testset "mals_eigsolve: return type and structure" begin
    d = 4
    A = mals_spd_op(d)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    E, x_opt, r_hist = mals_eigsolve(A, x0; sweep_schedule = [2], rmax_schedule = [4])

    @test E isa Vector{Float64}
    @test x_opt isa TTvector{Float64}
    @test r_hist isa Vector{<:Integer}
    @test length(E) == length(r_hist)
    @test x_opt.N == d
    @test x_opt.ttv_dims == ntuple(_ -> 2, d)
end

@testset "mals_eigsolve: eigenvalue positive for SPD operator" begin
    d = 4
    shift = 3.0
    A = mals_spd_op(d, shift)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    E, x_opt, _ = mals_eigsolve(A, x0; sweep_schedule = [4], rmax_schedule = [4])

    λ = E[end]
    @test λ > 0.0
    rq = real(TensorTrainNumerics.dot(x_opt, A * x_opt)) / real(TensorTrainNumerics.dot(x_opt, x_opt))
    @test isapprox(rq, λ; rtol = 0.1)
end

@testset "mals_eigsolve: eigenvalue non-increasing over sweeps" begin
    d = 4
    A = mals_spd_op(d, 2.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    E, _, _ = mals_eigsolve(A, x0; sweep_schedule = [4], rmax_schedule = [4])

    @test E[end] ≤ E[1] + 1.0e-8
end

@testset "mals_eigsolve: multi-stage sweep schedule with rank growth" begin
    d = 4
    A = mals_spd_op(d, 2.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1]; normalise = true)

    E, x_opt, r_hist = mals_eigsolve(A, x0; sweep_schedule = [2, 4], rmax_schedule = [2, 4])

    @test length(E) ≥ 2
    @test x_opt isa TTvector{Float64}
    @test maximum(x_opt.ttv_rks) ≤ 4
end

@testset "mals_eigsolve: rank history is non-empty and positive" begin
    d = 4
    A = mals_spd_op(d, 1.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    E, x_opt, r_hist = mals_eigsolve(A, x0; sweep_schedule = [2], rmax_schedule = [4])

    @test all(r > 0 for r in r_hist)
    @test all(isfinite, E)
    @test all(isreal, E)
end

@testset "mals_eigsolve: iterative solver path" begin
    d = 4
    A = mals_spd_op(d, 2.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    E, x_opt, _ = mals_eigsolve(
        A, x0;
        sweep_schedule = [2], rmax_schedule = [4],
        it_solver = true, itslv_thresh = 1
    )

    @test x_opt isa TTvector{Float64}
    @test isfinite(E[end])
end
