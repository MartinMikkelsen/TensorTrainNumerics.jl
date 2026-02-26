using Test
using Random
using LinearAlgebra

Random.seed!(1234)

# ── helpers ──────────────────────────────────────────────────────────────────

# Relative residual ‖Ax - b‖ / ‖b‖
function dmrg_rel_residual(A, x, b)
    r = A * x - b
    nb = norm(b)
    return nb > 0 ? norm(r) / nb : norm(r)
end

# Build a small SPD operator: Δ(d) + shift * I  (positive-definite for shift > 0)
dmrg_spd_op(d, shift = 3.0) = Δ(d) + shift * id_tto(d)

# ── dmrg_linsolve ─────────────────────────────────────────────────────────────

@testset "dmrg_linsolve: return type and structure" begin
    d = 4
    A = dmrg_spd_op(d)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])

    x = dmrg_linsolve(A, b, x0; N = 2, sweep_schedule = [2], rmax_schedule = [4])

    @test x isa TTvector{Float64}
    @test x.N == d
    @test x.ttv_dims == b.ttv_dims
    @test all(isfinite, x.ttv_rks)
end

@testset "dmrg_linsolve N=2: residual decreases" begin
    d = 4
    A = dmrg_spd_op(d, 10.0)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])

    x = dmrg_linsolve(A, b, x0; N = 2, sweep_schedule = [4], rmax_schedule = [8])

    @test dmrg_rel_residual(A, x, b) < 0.5
end

@testset "dmrg_linsolve: two-stage sweep schedule" begin
    d = 4
    A = dmrg_spd_op(d, 5.0)
    b = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])

    x = dmrg_linsolve(A, b, x0; N = 2, sweep_schedule = [2, 4], rmax_schedule = [2, 8])

    @test x isa TTvector{Float64}
    @test x.ttv_dims == b.ttv_dims
end

@testset "dmrg_linsolve: identity operator → residual near zero" begin
    d = 4
    A = id_tto(d)
    b = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1])

    x = dmrg_linsolve(A, b, x0; N = 2, sweep_schedule = [4], rmax_schedule = [4])

    @test dmrg_rel_residual(A, x, b) < 0.05
end

# ── dmrg_eigsolve ─────────────────────────────────────────────────────────────

@testset "dmrg_eigsolve: return type and structure" begin
    d = 4
    A = dmrg_spd_op(d)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    E, x_opt, r_hist = dmrg_eigsolve(A, x0; N = 2, sweep_schedule = [2], rmax_schedule = [4])

    @test E isa Vector{Float64}
    @test x_opt isa TTvector{Float64}
    @test r_hist isa Vector{<:Integer}
    @test length(E) == length(r_hist)
    @test x_opt.N == d
    @test x_opt.ttv_dims == ntuple(_ -> 2, d)
end

@testset "dmrg_eigsolve N=2: eigenvalue positive for SPD operator" begin
    d = 4
    shift = 3.0
    A = dmrg_spd_op(d, shift)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    E, x_opt, _ = dmrg_eigsolve(A, x0; N = 2, sweep_schedule = [4], rmax_schedule = [4])

    λ = E[end]
    @test λ > 0.0
    # Rayleigh quotient should be close to λ
    rq = real(TensorTrainNumerics.dot(x_opt, A * x_opt)) / real(TensorTrainNumerics.dot(x_opt, x_opt))
    @test isapprox(rq, λ; rtol = 0.1)
end

@testset "dmrg_eigsolve: sweep schedule with rank growth" begin
    d = 4
    A = dmrg_spd_op(d, 2.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 1, 1, 1, 1]; normalise = true)

    E, x_opt, r_hist = dmrg_eigsolve(A, x0; N = 2, sweep_schedule = [2, 4], rmax_schedule = [2, 4])

    @test length(E) ≥ 2
    @test x_opt isa TTvector{Float64}
    @test maximum(x_opt.ttv_rks) ≤ 4
end

@testset "dmrg_eigsolve: eigenvalues are real and finite" begin
    d = 4
    A = dmrg_spd_op(d, 1.0)
    x0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    E, _, _ = dmrg_eigsolve(A, x0; N = 2, sweep_schedule = [2], rmax_schedule = [4])

    @test all(isreal, E)
    @test all(isfinite, E)
end
