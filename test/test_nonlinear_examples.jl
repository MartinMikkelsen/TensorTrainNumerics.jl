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
