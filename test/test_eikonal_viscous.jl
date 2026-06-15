using Test
using LinearAlgebra
using Random
using TensorTrainNumerics
using InterpolativeQTT

@testset "viscous Eikonal QTT solver" begin

@testset "eikonal_viscous_solve matches dense same-scheme reference" begin
    d = 5
    N = 2^d
    h = 1.0 / (N + 1)
    sfun = (x, y) -> 1.0 + 0.5 * exp(-((x - 0.6)^2 + (y - 0.6)^2) / 0.02)
    ε_schedule = [0.2, 0.1, 0.05]

    Random.seed!(11)
    T, info = eikonal_viscous_solve(d;
        slowness = sfun,
        ε_schedule = ε_schedule,
        max_scf = 40,
        scf_tol = 1.0e-11,
        max_bond = 32,
        als_sweeps = 6,
    )

    # Dense reference: identical Picard fixed-point iteration with the same
    # discrete operators and exact coefficients.
    L_d = qtto_to_matrix(info.Lap)
    Dx_d = qtto_to_matrix(info.Dx)
    Dy_d = qtto_to_matrix(info.Dy)
    s2_d = real.(qtt_to_function(info.s2))

    T_d = zeros(length(s2_d))
    for ε in ε_schedule
        for _ in 1:200
            T_old = T_d
            A = ε * L_d + Diagonal(Dx_d * T_d) * Dx_d + Diagonal(Dy_d * T_d) * Dy_d
            T_d = A \ s2_d
            norm(T_d - T_old) / max(norm(T_d), eps(Float64)) < 1.0e-12 && break
        end
    end

    T_vals = real.(qtt_to_function(T))
    @test norm(T_vals - T_d) / norm(T_d) < 5.0e-3
    @test maximum(T.ttv_rks) <= 32
    @test length(info.eikonal_residual) == length(ε_schedule)
    @test info.eikonal_residual[end] < info.eikonal_residual[1]
end

@testset "constant slowness gives distance-like solution" begin
    d = 5
    Random.seed!(11)
    T, info = eikonal_viscous_solve(d;
        slowness = (x, y) -> 1.0,
        ε_schedule = [0.2, 0.1, 0.05, 0.03],
        max_scf = 40,
        scf_tol = 1.0e-11,
        max_bond = 32,
        als_sweeps = 6,
    )

    N = 2^d
    h = 1.0 / (N + 1)
    dist = [min(i * h, 1 - i * h, j * h, 1 - j * h) for i in 1:N for j in 1:N]
    T_vals = real.(qtt_to_function(T))
    # viscosity smooths the ridge: agreement is O(ε), not tight
    @test norm(T_vals - dist) / norm(dist) < 0.2
    @test maximum(T_vals) < 0.55
end

end
