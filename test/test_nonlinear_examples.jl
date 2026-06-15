using Test
using InterpolativeQTT

include(joinpath(@__DIR__, "..", "examples", "nonlinear_benchmark_utils.jl"))

@testset "nonlinear example diagnostics" begin
    @testset "KdV soliton benchmark reports error and ranks" begin
        bench = kdv_soliton_benchmark(;
            d = 5,
            T_end = 0.1,
            Nt = 5,
            method = :als,
            max_scf = 5,
            scf_tol = 1.0e-6,
            max_bond = 20,
            verbose_steps = false
        )

        @test bench.metrics.relative_error < 2.0e-2
        @test bench.metrics.max_rank <= 20
        @test bench.metrics.snapshot_count == 6
        @test bench.method_label == "InterpolativeQTT-SCF-ALS"
        @test !haskey(pairs(bench), :coefficient_projection)
    end

    @testset "Allen-Cahn benchmark reports bounded phase separation" begin
        bench = allen_cahn_benchmark(;
            d = 5,
            T_end = 0.2,
            Nt = 5,
            max_scf = 5,
            scf_tol = 1.0e-7,
            max_bond = 12,
            verbose_steps = false
        )

        @test bench.metrics.max_abs_u <= 1.05
        @test bench.metrics.max_rank <= 12
        @test bench.metrics.snapshot_count == 6
        @test bench.method_label == "InterpolativeQTT-SCF-MALS"
        @test !haskey(pairs(bench), :coefficient_projection)
    end

    @testset "2D GPE benchmark compares ALS and MALS" begin
        bench = gpe_2d_benchmark(;
            L = 4,
            κ = 50.0,
            g_vals = [0.0, 100.0],
            random_rank = 4,
            linear_sweeps = 8,
            nonlinear_sweeps = 5,
            mals_rmax = 8,
            mals_tol = 1.0e-8,
            projection_degree = 12,
            projection_tolerance = 1.0e-10,
            seed = 42
        )

        @test bench.metrics.max_mu_gap < 1.0e-4
        @test bench.metrics.max_rank_als <= 8
        @test bench.metrics.max_rank_mals <= 8
        @test bench.metrics.linear_mu > 0
        @test length(bench.g_vals) == 2
        @test bench.method_label == "InterpolativeQTT-SCF-ALS / InterpolativeQTT-SCF-MALS"
        @test !haskey(pairs(bench), :coefficient_projection)
    end

    @testset "1D KdV analytical solver comparison reports relative errors" begin
        bench = kdv_1d_solver_comparison_benchmark(;
            d = 5,
            T_end = 0.05,
            Nt = 3,
            methods = (:als_direct, :als_krylov, :mals_direct),
            max_scf = 3,
            scf_tol = 1.0e-6,
            max_bond = 20,
            projection_degree = 8,
            projection_tolerance = 1.0e-9,
        )

        @test bench.equation == "KdV"
        @test bench.analytical_reference == :soliton
        @test length(bench.results) == 3
        @test Set(result.method for result in bench.results) == Set([:als_direct, :als_krylov, :mals_direct])
        for result in bench.results
            @test occursin("InterpolativeQTT", result.method_label)
            @test isfinite(result.metrics.relative_error)
            @test result.metrics.relative_error < 5.0e-2
            @test result.metrics.max_rank <= 20
            @test isfinite(result.metrics.runtime_seconds)
            @test isfinite(result.metrics.final_pde_residual_norm)
            @test isfinite(result.metrics.final_pde_relative_residual)
            @test result.metrics.final_pde_relative_residual < 1.0
        end
    end

    @testset "1D GPE analytical solver comparison includes local Krylov variants" begin
        bench = gpe_1d_solver_comparison_benchmark(;
            L = 5,
            κ = 50.0,
            g_vals = [0.0, 25.0],
            methods = (:als_direct, :als_krylov, :mals_direct, :mals_krylov),
            random_rank = 4,
            linear_sweeps = 6,
            nonlinear_sweeps = 3,
            mals_rmax = 8,
            projection_degree = 8,
            projection_tolerance = 1.0e-9,
            seed = 42,
        )

        @test bench.equation == "1D Gross-Pitaevskii"
        @test bench.analytical_reference == :linear_mu
        @test length(bench.results) == 4
        @test Set(result.method for result in bench.results) == Set([:als_direct, :als_krylov, :mals_direct, :mals_krylov])
        for result in bench.results
            @test occursin("InterpolativeQTT", result.method_label)
            @test isfinite(result.metrics.linear_mu_relative_error)
            @test result.metrics.linear_mu_relative_error < 0.25
            @test isfinite(result.metrics.final_mu)
            @test result.metrics.max_rank <= 8
            @test isfinite(result.metrics.runtime_seconds)
            @test isfinite(result.metrics.final_nonlinear_residual_norm)
            @test isfinite(result.metrics.final_nonlinear_relative_residual)
            @test result.metrics.final_nonlinear_relative_residual < 1.0
            @test isfinite(result.metrics.final_mu_step_change)
        end
    end

    @testset "Allen-Cahn benchmark supports adaptive projection" begin
        bench = allen_cahn_benchmark(;
            d = 5,
            T_end = 0.1,
            Nt = 3,
            max_scf = 3,
            scf_tol = 1.0e-6,
            max_bond = 16,
            projection_degree = 8,
            projection_tolerance = 1.0e-9,
            projection_mode = :adaptive,
            projection_adaptive_tolerance = 1.0e-9,
            verbose_steps = false
        )

        @test bench.metrics.max_abs_u <= 1.1
        @test bench.metrics.max_rank <= 16
    end

    @testset "1D GPE comparison supports adaptive projection" begin
        bench = gpe_1d_solver_comparison_benchmark(;
            L = 5,
            κ = 50.0,
            g_vals = [0.0, 25.0],
            methods = (:als_direct,),
            random_rank = 4,
            linear_sweeps = 6,
            nonlinear_sweeps = 3,
            mals_rmax = 8,
            projection_degree = 8,
            projection_tolerance = 1.0e-9,
            projection_mode = :adaptive,
            projection_adaptive_tolerance = 1.0e-9,
            seed = 42,
        )

        @test length(bench.results) == 1
        @test isfinite(bench.results[1].metrics.linear_mu_relative_error)
        @test bench.results[1].metrics.linear_mu_relative_error < 0.25
    end

    @testset "2D Allen-Cahn dense comparison benchmark" begin
        bench = allen_cahn_2d_dense_benchmark(;
            d = 5,
            ε = 0.15,
            T_end = 0.2,
            Nt = 2,
            u0_fun = c -> 0.9 * cos(pi * c[1]) * cos(pi * c[2]),
            max_scf = 12,
            scf_tol = 1.0e-11,
            max_bond = 32,
            projection_degree = 10,
            projection_tolerance = 1.0e-10,
        )

        @test bench.equation == "Allen-Cahn 2D"
        @test occursin("InterpolativeQTT", bench.method_label)
        @test length(bench.metrics.stepwise_relative_error) == 2
        @test all(isfinite, bench.metrics.stepwise_relative_error)
        @test bench.metrics.final_relative_error < 5.0e-3
        @test bench.metrics.max_rank <= 32
        @test isfinite(bench.metrics.qtt_runtime_seconds)
        @test isfinite(bench.metrics.dense_runtime_seconds)
    end

    @testset "2D GPE dense comparison benchmark" begin
        bench = gpe_2d_dense_benchmark(;
            L = 4,
            κ = 50.0,
            g_vals = [0.0, 25.0],
            random_rank = 4,
            linear_sweeps = 10,
            nonlinear_sweeps = 12,
            mals_rmax = 12,
            projection_degree = 8,
            projection_tolerance = 1.0e-9,
            seed = 7,
        )

        @test bench.equation == "2D Gross-Pitaevskii"
        @test occursin("InterpolativeQTT", bench.method_label)
        @test length(bench.μ_dense) == 2
        @test bench.metrics.max_mu_relative_error_als < 2.0e-2
        @test bench.metrics.max_mu_relative_error_mals < 2.0e-2
        @test bench.metrics.max_density_relative_error_mals < 5.0e-2
        @test isfinite(bench.metrics.dense_runtime_seconds)
    end

    @testset "1D KdV dense comparison benchmark" begin
        bench = kdv_1d_dense_benchmark(;
            d = 5,
            T_end = 0.05,
            Nt = 3,
            method = :als,
            max_scf = 5,
            scf_tol = 1.0e-8,
            max_bond = 20,
            projection_degree = 8,
            projection_tolerance = 1.0e-9,
        )

        @test bench.equation == "KdV"
        @test occursin("InterpolativeQTT", bench.method_label)
        @test length(bench.metrics.stepwise_relative_error) == 3
        @test all(isfinite, bench.metrics.stepwise_relative_error)
        @test bench.metrics.final_dense_relative_error < 2.0e-2
        @test isfinite(bench.metrics.qtt_vs_analytical_error)
        @test isfinite(bench.metrics.dense_vs_analytical_error)

        bench_adaptive = kdv_1d_dense_benchmark(;
            d = 5,
            T_end = 0.05,
            Nt = 3,
            method = :als,
            max_scf = 5,
            scf_tol = 1.0e-8,
            max_bond = 20,
            projection_degree = 8,
            projection_tolerance = 1.0e-9,
            projection_mode = :adaptive,
            projection_adaptive_tolerance = 1.0e-9,
        )

        @test bench_adaptive.metrics.final_dense_relative_error < 5.0e-3
        @test bench_adaptive.metrics.final_dense_relative_error <
            bench.metrics.final_dense_relative_error
    end

    @testset "viscous Eikonal benchmark with adaptive slowness field" begin
        bench = eikonal_2d_benchmark(;
            d = 5,
            lens_strength = 2.0,
            ε_schedule = [0.2, 0.1, 0.05],
            max_scf = 30,
            scf_tol = 1.0e-11,
            max_bond = 32,
            als_sweeps = 6,
            projection_degree = 8,
            residual_field = false,
            seed = 11,
        )

        @test bench.equation == "Viscous Eikonal 2D"
        @test occursin("InterpolativeQTT", bench.method_label)
        @test bench.metrics.slowness_build_relative_error < 1.0e-6
        @test bench.metrics.qtt_vs_dense_relative_error < 5.0e-3
        @test bench.metrics.eikonal_residuals[end] < bench.metrics.eikonal_residuals[1]
        @test isfinite(bench.metrics.fast_sweeping_gap)
        @test bench.metrics.max_rank <= 32
    end

    @testset "paper benchmarks are InterpolativeQTT-only" begin
        ac = allen_cahn_benchmark(;
            d = 5,
            T_end = 0.1,
            Nt = 3,
            max_scf = 3,
            scf_tol = 1.0e-6,
            max_bond = 16,
            projection_degree = 8,
            projection_tolerance = 1.0e-9,
            verbose_steps = false
        )

        @test occursin("InterpolativeQTT", ac.method_label)
        @test !haskey(pairs(ac), :coefficient_projection)
        @test ac.metrics.max_abs_u <= 1.1
        @test ac.metrics.max_rank <= 16

        kdv = kdv_soliton_benchmark(;
            d = 5,
            T_end = 0.05,
            Nt = 3,
            method = :cn_mals,
            max_scf = 3,
            scf_tol = 1.0e-6,
            max_bond = 24,
            projection_degree = 8,
            projection_tolerance = 1.0e-9,
            verbose_steps = false
        )

        @test occursin("InterpolativeQTT", kdv.method_label)
        @test !haskey(pairs(kdv), :coefficient_projection)
        @test isfinite(kdv.metrics.relative_error)
        @test kdv.metrics.relative_error < 5.0e-2
        @test kdv.metrics.max_rank <= 24

        gpe = gpe_2d_benchmark(;
            L = 4,
            κ = 50.0,
            g_vals = [0.0, 25.0],
            random_rank = 4,
            linear_sweeps = 6,
            nonlinear_sweeps = 3,
            mals_rmax = 8,
            mals_tol = 1.0e-8,
            projection_degree = 8,
            projection_tolerance = 1.0e-9,
            seed = 42
        )

        @test occursin("InterpolativeQTT", gpe.method_label)
        @test !haskey(pairs(gpe), :coefficient_projection)
        @test isfinite(gpe.metrics.max_mu_gap)
        @test gpe.metrics.max_rank_als <= 8
        @test gpe.metrics.max_rank_mals <= 8
    end
end
