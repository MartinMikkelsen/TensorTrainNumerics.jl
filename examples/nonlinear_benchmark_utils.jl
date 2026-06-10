using LinearAlgebra
using Random
using TensorTrainNumerics
using InterpolativeQTT

import TensorTrainNumerics: dot

const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "output")

function ensure_example_output_dir(path::AbstractString = DEFAULT_OUTPUT_DIR)
    mkpath(path)
    return path
end

function _qtt_snapshots_to_matrix(sol)
    return reduce(hcat, real.(qtt_to_function(u)) for u in sol)
end

function _max_rank(sol)
    return maximum(maximum(u.ttv_rks) for u in sol)
end

_interpolative_label(label::AbstractString) = "InterpolativeQTT-$label"

const LOCAL_SOLVER_VARIANTS = (:als_direct, :als_krylov, :mals_direct, :mals_krylov)

function _solver_variant_label(method::Symbol)
    method == :als_direct && return _interpolative_label("SCF-ALS")
    method == :als_krylov && return _interpolative_label("SCF-ALS-Krylov")
    method == :mals_direct && return _interpolative_label("SCF-MALS")
    method == :mals_krylov && return _interpolative_label("SCF-MALS-Krylov")
    throw(ArgumentError("unknown solver comparison method $(repr(method)); use one of $(LOCAL_SOLVER_VARIANTS)"))
end

_solver_site_count(method::Symbol) = method in (:als_direct, :als_krylov) ? 1 :
    method in (:mals_direct, :mals_krylov) ? 2 :
    throw(ArgumentError("unknown solver comparison method $(repr(method)); use one of $(LOCAL_SOLVER_VARIANTS)"))

_solver_local_backend(method::Symbol) = method in (:als_krylov, :mals_krylov) ? :krylov :
    method in (:als_direct, :mals_direct) ? :dense :
    throw(ArgumentError("unknown solver comparison method $(repr(method)); use one of $(LOCAL_SOLVER_VARIANTS)"))

_is_krylov_variant(method::Symbol) = _solver_local_backend(method) == :krylov

function _relative_residual_metrics(residual, terms...)
    residual_norm = norm(residual)
    residual_scale = sum(norm, terms)
    return (
        residual_norm = Float64(residual_norm),
        residual_scale = Float64(residual_scale),
        relative_residual = Float64(residual_norm / max(residual_scale, eps(Float64))),
    )
end

function kdv_soliton_benchmark(;
        d::Int = 8,
        L::Real = 25.0,
        T_end::Real = 1.0,
        Nt::Int = 250,
        c::Real = 1.0,
        x0::Real = 9.0,
        method::Symbol = :cn_mals,
        max_scf::Int = 25,
        scf_tol::Real = 1.0e-8,
        max_bond::Int = 50,
        it_solver::Bool = false,
        itslv_thresh::Int = 256,
        linsolv_maxiter::Int = 200,
        linsolv_tol::Float64 = 1.0e-8,
        projection_degree::Int = 8,
        projection_tolerance::Real = 1.0e-10,
        projection_q::Int = 1,
        projection_mode::Symbol = :singlescale,
        projection_adaptive_tolerance::Real = 1.0e-8,
        verbose::Bool = false,
        verbose_steps::Bool = false
    )
    N = 2^d
    dx = L / (N - 1)
    dt = T_end / Nt

    D_x = (1 / (2dx)) * ∇_c_P(d)
    D_xxx = (1 / (2dx^3)) * ∇3_P(d)

    u0 = function_to_qtt(x -> (c / 2) * sech(sqrt(c) / 2 * (x - x0))^2, d; b = L)
    x = collect((0:N-1) ./ (N - 1) .* L)
    t = collect((0:Nt) .* dt)

    sol, method_label = if method == :als
        kdv_als(u0, D_x, D_xxx, dt, Nt;
            max_scf = max_scf,
            scf_tol = scf_tol,
            max_bond = max_bond,
            it_solver = it_solver,
            r_itsolver = itslv_thresh,
            linsolv_maxiter = linsolv_maxiter,
            linsolv_tol = linsolv_tol,
            projection_degree = projection_degree,
            projection_tolerance = projection_tolerance,
            projection_q = projection_q,
            projection_mode = projection_mode,
            projection_adaptive_tolerance = projection_adaptive_tolerance,
            projection_a = 0.0,
            projection_b = Float64(L),
            verbose = verbose,
            verbose_steps = verbose_steps), "SCF-ALS"
    elseif method == :mals
        kdv_mals(u0, D_x, D_xxx, dt, Nt;
            max_scf = max_scf,
            scf_tol = scf_tol,
            max_bond = max_bond,
            projection_degree = projection_degree,
            projection_tolerance = projection_tolerance,
            projection_q = projection_q,
            projection_mode = projection_mode,
            projection_adaptive_tolerance = projection_adaptive_tolerance,
            projection_a = 0.0,
            projection_b = Float64(L),
            verbose = verbose,
            verbose_steps = verbose_steps), "SCF-MALS"
    elseif method == :cn_mals
        kdv_cn_mals(u0, D_x, D_xxx, dt, Nt;
            max_scf = max_scf,
            scf_tol = scf_tol,
            max_bond = max_bond,
            projection_degree = projection_degree,
            projection_tolerance = projection_tolerance,
            projection_q = projection_q,
            projection_mode = projection_mode,
            projection_adaptive_tolerance = projection_adaptive_tolerance,
            projection_a = 0.0,
            projection_b = Float64(L),
            verbose = verbose,
            verbose_steps = verbose_steps), "CN-SCF-MALS"
    else
        throw(ArgumentError("unknown KdV method $(repr(method)); use :als, :mals, or :cn_mals"))
    end
    method_label = _interpolative_label(method_label)

    u_exact = function_to_qtt(x -> (c / 2) * sech(sqrt(c) / 2 * (x - x0 - c * T_end))^2, d; b = L)
    U = _qtt_snapshots_to_matrix(sol)
    exact_values = real.(qtt_to_function(u_exact))
    relative_error = norm(sol[end] - u_exact) / norm(u_exact)

    return (
        equation = "KdV",
        method = method,
        method_label = method_label,
        projection_degree = projection_degree,
        projection_tolerance = Float64(projection_tolerance),
        d = d,
        N = N,
        L = Float64(L),
        T_end = Float64(T_end),
        Nt = Nt,
        c = Float64(c),
        x0 = Float64(x0),
        x = x,
        t = t,
        solution = sol,
        U = U,
        exact = u_exact,
        exact_values = exact_values,
        metrics = (
            relative_error = Float64(relative_error),
            max_rank = _max_rank(sol),
            final_rank = maximum(sol[end].ttv_rks),
            snapshot_count = length(sol),
            dx = Float64(dx),
            dt = Float64(dt),
        ),
    )
end

function _kdv_comparison_base_method(method::Symbol)
    method == :als_direct && return :als
    method == :als_krylov && return :als
    method == :mals_direct && return :mals
    method == :mals_krylov &&
        throw(ArgumentError("KdV MALS Krylov is not available yet because the KdV two-site step uses a nonsymmetric local linear solve"))
    throw(ArgumentError("unknown KdV comparison method $(repr(method))"))
end

function _kdv_final_pde_residual_metrics(bench)
    length(bench.solution) >= 2 ||
        throw(ArgumentError("KdV residual needs at least one time step"))

    d = bench.d
    dx = bench.metrics.dx
    dt = bench.metrics.dt
    D_x = (1 / (2dx)) * ∇_c_P(d)
    D_xxx = (1 / (2dx^3)) * ∇3_P(d)

    u = bench.solution[end]
    u_prev = bench.solution[end - 1]
    u_vals = real.(qtt_to_function(u))
    u_prev_vals = real.(qtt_to_function(u_prev))
    ux_vals = real.(qtt_to_function(D_x * u))
    uxxx_vals = real.(qtt_to_function(D_xxx * u))

    time_term = (u_vals .- u_prev_vals) ./ dt
    nonlinear_term = 6.0 .* u_vals .* ux_vals
    dispersion_term = uxxx_vals
    residual = time_term .+ nonlinear_term .+ dispersion_term
    metrics = _relative_residual_metrics(residual, time_term, nonlinear_term, dispersion_term)
    return (
        final_pde_residual_norm = metrics.residual_norm,
        final_pde_residual_scale = metrics.residual_scale,
        final_pde_relative_residual = metrics.relative_residual,
    )
end

function kdv_1d_solver_comparison_benchmark(;
        methods = (:als_direct, :als_krylov, :mals_direct),
        kwargs...
    )
    results = map(collect(methods)) do method
        base_method = _kdv_comparison_base_method(method)
        elapsed = @elapsed bench = kdv_soliton_benchmark(;
            method = base_method,
            it_solver = _is_krylov_variant(method),
            itslv_thresh = 0,
            kwargs...
        )
        residual_metrics = _kdv_final_pde_residual_metrics(bench)
        (
            method = method,
            method_label = _solver_variant_label(method),
            site_count = _solver_site_count(method),
            local_solver = _solver_local_backend(method),
            benchmark = bench,
            metrics = (
                relative_error = bench.metrics.relative_error,
                runtime_seconds = elapsed,
                max_rank = bench.metrics.max_rank,
                final_rank = bench.metrics.final_rank,
                snapshot_count = bench.metrics.snapshot_count,
                final_pde_residual_norm = residual_metrics.final_pde_residual_norm,
                final_pde_residual_scale = residual_metrics.final_pde_residual_scale,
                final_pde_relative_residual = residual_metrics.final_pde_relative_residual,
            ),
        )
    end
    return (
        equation = "KdV",
        analytical_reference = :soliton,
        methods = collect(methods),
        results = results,
        metrics = (
            best_relative_error = minimum(result.metrics.relative_error for result in results),
            max_relative_error = maximum(result.metrics.relative_error for result in results),
            total_runtime_seconds = sum(result.metrics.runtime_seconds for result in results),
        ),
    )
end

function _allen_cahn_energy_values(U::AbstractMatrix, dx::Real, ε::Real)
    energies = Float64[]
    for u in eachcol(U)
        grad = diff(u) ./ dx
        bulk = 0.25 .* (u .^ 2 .- 1) .^ 2
        push!(energies, dx * (0.5 * ε^2 * sum(abs2, grad) + sum(bulk)))
    end
    return energies
end

function allen_cahn_benchmark(;
        d::Int = 8,
        L::Real = 1.0,
        T_end::Real = 5.0,
        Nt::Int = 100,
        ε::Real = 0.05,
        max_scf::Int = 5,
        scf_tol::Real = 1.0e-8,
        max_bond::Int = 20,
        projection_degree::Int = 8,
        projection_tolerance::Real = 1.0e-10,
        projection_q::Int = 1,
        projection_mode::Symbol = :singlescale,
        projection_adaptive_tolerance::Real = 1.0e-8,
        verbose::Bool = false,
        verbose_steps::Bool = false
    )
    N = 2^d
    dx = L / (N - 1)
    dt = T_end / Nt

    Dxx = (1 / dx^2) * Δ_NN(d)
    u0 = function_to_qtt(x -> sin(2π * x), d)
    x = collect((0:N-1) ./ (N - 1) .* L)
    t = collect((0:Nt) .* dt)

    elapsed = @elapsed sol = allen_cahn_mals(u0, Dxx, ε, dt, Nt;
        max_scf = max_scf,
        scf_tol = scf_tol,
        max_bond = max_bond,
        projection_degree = projection_degree,
        projection_tolerance = projection_tolerance,
        projection_q = projection_q,
        projection_a = 0.0,
        projection_b = Float64(L),
        verbose = verbose,
        verbose_steps = verbose_steps
    )
    U = _qtt_snapshots_to_matrix(sol)
    final_values = U[:, end]
    energy_history = _allen_cahn_energy_values(U, dx, ε)

    return (
        equation = "Allen-Cahn",
        method = :mals,
        method_label = _interpolative_label("SCF-MALS"),
        projection_degree = projection_degree,
        projection_tolerance = Float64(projection_tolerance),
        d = d,
        N = N,
        L = Float64(L),
        T_end = Float64(T_end),
        Nt = Nt,
        ε = Float64(ε),
        x = x,
        t = t,
        solution = sol,
        U = U,
        energy_history = energy_history,
        metrics = (
            runtime_seconds = elapsed,
            min_u = minimum(final_values),
            max_u = maximum(final_values),
            max_abs_u = maximum(abs.(final_values)),
            max_rank = _max_rank(sol),
            final_rank = maximum(sol[end].ttv_rks),
            snapshot_count = length(sol),
            initial_energy = first(energy_history),
            final_energy = last(energy_history),
            dx = Float64(dx),
            dt = Float64(dt),
        ),
    )
end

function _gpe_1d_nonlinear_residual_metrics(ψ, H_lin, g::Real, μ::Real)
    ψ_vals = real.(qtt_to_function(ψ))
    Hψ_vals = real.(qtt_to_function(H_lin * ψ))
    nonlinear_term = Float64(g) .* abs2.(ψ_vals) .* ψ_vals
    eigen_term = Float64(μ) .* ψ_vals
    residual = Hψ_vals .+ nonlinear_term .- eigen_term
    metrics = _relative_residual_metrics(residual, Hψ_vals, nonlinear_term, eigen_term)
    return (
        final_nonlinear_residual_norm = metrics.residual_norm,
        final_nonlinear_residual_scale = metrics.residual_scale,
        final_nonlinear_relative_residual = metrics.relative_residual,
    )
end

function gpe_1d_solver_comparison_benchmark(;
        L::Int = 8,
        κ::Real = 200.0,
        g_vals = [0.0, 50.0, 200.0],
        methods = LOCAL_SOLVER_VARIANTS,
        random_rank::Int = 4,
        linear_sweeps::Int = 10,
        nonlinear_sweeps::Int = 12,
        mals_rmax::Int = 8,
        mals_tol::Real = 1.0e-10,
        projection_degree::Int = 8,
        projection_tolerance::Real = 1.0e-10,
        projection_q::Int = 1,
        projection_maxbonddim::Int = mals_rmax,
        seed::Int = 42
    )
    N = 2^L
    h = 1.0 / (N - 1)
    x = collect(LinRange(0.0, 1.0, N))

    H_kin = (1.0 / (2h^2)) * Δ(L)
    V_trap = function_to_qtt(t -> κ * (t - 0.5)^2, L)
    H_lin = H_kin + ttv_to_diag_tto(V_trap)

    Random.seed!(seed)
    ψ0 = rand_tt(fill(2, L), random_rank; normalise = true)
    _, ψ_lin = als_eigsolve(H_lin, ψ0; sweep_schedule = [linear_sweeps])
    μ_lin = real(dot(ψ_lin, H_lin * ψ_lin))
    expected_linear_mu = sqrt(Float64(κ) / 2)

    results = map(collect(methods)) do method
        local_solver = _solver_local_backend(method)
        krylov = local_solver == :krylov
        ψ = ψ_lin
        μ_values = zeros(Float64, length(g_vals))
        μ_values[1] = μ_lin
        μ_step_changes = fill(NaN, length(g_vals))
        rank_values = [maximum(ψ.ttv_rks)]
        elapsed = @elapsed begin
            for k in eachindex(g_vals)[2:end]
                g = g_vals[k]
                if method in (:als_direct, :als_krylov)
                    μ_hist, ψ = nonlinear_als_eigsolve(H_lin, g, ψ;
                        sweep_count = nonlinear_sweeps,
                        it_solver = krylov,
                        itslv_thresh = 0,
                        linsolv_maxiter = 200,
                        linsolv_tol = 1.0e-8,
                        projection_degree = projection_degree,
                        projection_tolerance = projection_tolerance,
                        projection_maxbonddim = projection_maxbonddim,
                        projection_q = projection_q,
                        verbose = false,
                    )
                    μ_values[k] = last(μ_hist)
                    μ_step_changes[k] = length(μ_hist) >= 2 ?
                        abs(μ_hist[end] - μ_hist[end - 1]) / max(abs(μ_hist[end]), eps(Float64)) :
                        NaN
                elseif method in (:mals_direct, :mals_krylov)
                    μ_hist, ψ, _ = nonlinear_mals_eigsolve(H_lin, g, ψ;
                        tol = Float64(mals_tol),
                        sweep_schedule = [nonlinear_sweeps + 1],
                        rmax_schedule = [mals_rmax],
                        it_solver = krylov,
                        itslv_thresh = 0,
                        linsolv_maxiter = 200,
                        linsolv_tol = 1.0e-8,
                        projection_degree = projection_degree,
                        projection_tolerance = projection_tolerance,
                        projection_maxbonddim = projection_maxbonddim,
                        projection_q = projection_q,
                        verbose = false,
                    )
                    μ_values[k] = last(μ_hist)
                    μ_step_changes[k] = length(μ_hist) >= 2 ?
                        abs(μ_hist[end] - μ_hist[end - 1]) / max(abs(μ_hist[end]), eps(Float64)) :
                        NaN
                else
                    throw(ArgumentError("unknown GPE comparison method $(repr(method))"))
                end
                push!(rank_values, maximum(ψ.ttv_rks))
            end
        end
        density = abs2.(real.(qtt_to_function(ψ)))
        final_g = Float64(last(g_vals))
        final_mu = last(μ_values)
        residual_metrics = _gpe_1d_nonlinear_residual_metrics(ψ, H_lin, final_g, final_mu)
        (
            method = method,
            method_label = _solver_variant_label(method),
            site_count = _solver_site_count(method),
            local_solver = local_solver,
            μ_values = μ_values,
            final_state = ψ,
            density = density,
            metrics = (
                linear_mu = μ_lin,
                expected_linear_mu = expected_linear_mu,
                linear_mu_relative_error = abs(μ_lin - expected_linear_mu) / max(abs(expected_linear_mu), eps(Float64)),
                final_mu = final_mu,
                runtime_seconds = elapsed,
                max_rank = maximum(rank_values),
                final_rank = maximum(ψ.ttv_rks),
                max_norm_error = abs(norm(ψ) - 1),
                final_nonlinear_residual_norm = residual_metrics.final_nonlinear_residual_norm,
                final_nonlinear_residual_scale = residual_metrics.final_nonlinear_residual_scale,
                final_nonlinear_relative_residual = residual_metrics.final_nonlinear_relative_residual,
                final_mu_step_change = μ_step_changes[end],
            ),
        )
    end

    return (
        equation = "1D Gross-Pitaevskii",
        analytical_reference = :linear_mu,
        methods = collect(methods),
        L = L,
        N = N,
        h = h,
        κ = Float64(κ),
        g_vals = collect(Float64, g_vals),
        x = x,
        H_lin = H_lin,
        results = results,
        metrics = (
            expected_linear_mu = expected_linear_mu,
            max_linear_mu_relative_error = maximum(result.metrics.linear_mu_relative_error for result in results),
            total_runtime_seconds = sum(result.metrics.runtime_seconds for result in results),
            max_rank = maximum(result.metrics.max_rank for result in results),
        ),
    )
end

function gpe_density_grid(ψ, N::Int)
    v = real.(qtt_to_function(ψ))
    return permutedims(reshape(abs2.(v), N, N))
end

function gpe_2d_benchmark(;
        L::Int = 6,
        κ::Real = 200.0,
        g_vals = [0.0, 500.0, 2000.0, 10000.0],
        random_rank::Int = 8,
        linear_sweeps::Int = 20,
        nonlinear_sweeps::Int = 30,
        mals_rmax::Int = 12,
        mals_tol::Real = 1.0e-10,
        projection_degree::Int = 12,
        projection_tolerance::Real = 1.0e-10,
        projection_q::Int = 1,
        projection_maxbonddim::Int = mals_rmax,
        seed::Int = 42
    )
    N = 2^L
    h = 1.0 / (N - 1)
    x = collect(LinRange(0.0, 1.0, N))
    y = collect(LinRange(0.0, 1.0, N))

    I_L = id_tto(L)
    H_kin = (1.0 / (2h^2)) * (Δ(L) ⊗ I_L + I_L ⊗ Δ(L))
    o_L = ones_tt(2, L)
    V_x = function_to_qtt(t -> κ * (t - 0.5)^2, L)
    V_y = function_to_qtt(t -> κ * (t - 0.5)^2, L)
    H_lin = H_kin + ttv_to_diag_tto(V_x ⊗ o_L + o_L ⊗ V_y)

    Random.seed!(seed)
    ψ0 = rand_tt(fill(2, 2L), random_rank; normalise = true)
    _, ψ_lin = als_eigsolve(H_lin, ψ0; sweep_schedule = [linear_sweeps])
    μ_lin = real(dot(ψ_lin, H_lin * ψ_lin))

    ψ_list_als = Vector{Any}(undef, length(g_vals))
    ψ_list_mals = Vector{Any}(undef, length(g_vals))
    μ_list_als = Vector{Float64}(undef, length(g_vals))
    μ_list_mals = Vector{Float64}(undef, length(g_vals))
    ψ_list_als[1] = ψ_lin
    ψ_list_mals[1] = ψ_lin
    μ_list_als[1] = μ_lin
    μ_list_mals[1] = μ_lin
    elapsed_als  = zeros(Float64, length(g_vals))
    elapsed_mals = zeros(Float64, length(g_vals))

    for k in eachindex(g_vals)[2:end]
        g = g_vals[k]
        elapsed_als[k] = @elapsed begin
        μ_hist_als, ψg_als = nonlinear_als_eigsolve(H_lin, g, ψ_list_als[k - 1];
            sweep_count = nonlinear_sweeps,
            projection_degree = projection_degree,
            projection_tolerance = projection_tolerance,
            projection_maxbonddim = projection_maxbonddim,
                projection_q = projection_q,
                projection_ndims = 2,
                projection_ordering = :serial,
                verbose = false
            )
        end
        elapsed_mals[k] = @elapsed begin
            μ_hist_mals, ψg_mals, _ = nonlinear_mals_eigsolve(H_lin, g, ψ_list_mals[k - 1];
            tol = Float64(mals_tol),
            sweep_schedule = [nonlinear_sweeps + 1],
            rmax_schedule = [mals_rmax],
            projection_degree = projection_degree,
            projection_tolerance = projection_tolerance,
            projection_maxbonddim = projection_maxbonddim,
                projection_q = projection_q,
                projection_ndims = 2,
                projection_ordering = :serial,
                verbose = false
            )
        end
        ψ_list_als[k] = ψg_als
        ψ_list_mals[k] = ψg_mals
        μ_list_als[k] = last(μ_hist_als)
        μ_list_mals[k] = last(μ_hist_mals)
    end

    μ_gaps = abs.(μ_list_als .- μ_list_mals)
    density_grids = [gpe_density_grid(ψ, N) for ψ in ψ_list_mals]
    norm_als = [norm(ψ) for ψ in ψ_list_als]
    norm_mals = [norm(ψ) for ψ in ψ_list_mals]

    return (
        equation = "2D Gross-Pitaevskii",
        method = :scf_als_mals,
        method_label = "InterpolativeQTT-SCF-ALS / InterpolativeQTT-SCF-MALS",
        projection_degree = projection_degree,
        projection_tolerance = Float64(projection_tolerance),
        L = L,
        N = N,
        h = h,
        κ = Float64(κ),
        g_vals = collect(Float64, g_vals),
        x = x,
        y = y,
        H_lin = H_lin,
        ψ_list_als = ψ_list_als,
        ψ_list_mals = ψ_list_mals,
        μ_list_als = μ_list_als,
        μ_list_mals = μ_list_mals,
        μ_gaps = μ_gaps,
        density_grids = density_grids,
        elapsed_als  = elapsed_als,
        elapsed_mals = elapsed_mals,
        metrics = (
            linear_mu = μ_lin,
            max_mu_gap = maximum(μ_gaps),
            max_rank_als = maximum(maximum(ψ.ttv_rks) for ψ in ψ_list_als),
            max_rank_mals = maximum(maximum(ψ.ttv_rks) for ψ in ψ_list_mals),
            max_norm_error_als = maximum(abs.(norm_als .- 1)),
            max_norm_error_mals = maximum(abs.(norm_mals .- 1)),
            expected_linear_mu = sqrt(2 * κ),
            total_elapsed_als  = sum(elapsed_als),
            total_elapsed_mals = sum(elapsed_mals),
        ),
    )
end

function _allen_cahn_2d_energy(grid::AbstractMatrix, dx::Real, ε::Real)
    ux = diff(grid; dims = 2) ./ dx
    uy = diff(grid; dims = 1) ./ dx
    grad2 = sum(abs2, ux) + sum(abs2, uy)
    bulk  = sum(((grid .^ 2 .- 1) .^ 2) ./ 4)
    return dx^2 * (0.5 * ε^2 * grad2 + bulk)
end

function _interface_radius(grid, x::AbstractVector)
    N   = size(grid, 1)
    mid = div(N, 2) + 1
    slice = grid[mid, :]
    for k in 1:(N - 1)
        if slice[k] > 0 && slice[k + 1] <= 0
            dx = x[2] - x[1]
            xc = x[k] + (0 - slice[k]) / (slice[k + 1] - slice[k]) * dx
            return abs(xc - 0.5)
        end
    end
    return 0.0
end

"""
    allen_cahn_2d_benchmark(; d, R0, ε, T_end, Nt, ...)

2D Allen-Cahn benchmark on [0,1]² with a circular phase-field initial condition.

    ∂_t u = ε²·Δu + u - u³,   u₀(x,y) = tanh((r - R₀) / (ε√2))

where r = √((x-½)² + (y-½)²).  The interface radius shrinks asymptotically as
R(t) = √(R₀² - 2ε²t).  The benchmark compares the measured QTT zero-crossing
radius at time T to this formula and tracks max bond dimension over time.

"""
function allen_cahn_2d_benchmark(;
        d::Int                         = 7,
        R0::Real                       = 0.3,
        ε::Real                        = 0.05,
        T_end::Real                    = 0.5,
        Nt::Int                        = 50,
        max_scf::Int                   = 5,
        scf_tol::Real                  = 1.0e-8,
        max_bond::Int                  = 30,
        projection_degree::Int         = 8,
        projection_tolerance::Real     = 1.0e-10,
        projection_q::Int              = 1,
        verbose::Bool                  = false,
        verbose_steps::Bool            = false
    )
    N  = 2^d
    dx = 1.0 / (N - 1)
    dt = T_end / Nt

    I_d  = id_tto(d)
    D_xx = (1.0 / dx^2) * (Δ_NN(d) ⊗ I_d + I_d ⊗ Δ_NN(d))

    R0f = Float64(R0)
    εf  = Float64(ε)
    u0_q = function_to_qttv(
        coords -> tanh((sqrt((coords[1] - 0.5)^2 + (coords[2] - 0.5)^2) - R0f) / (εf * sqrt(2.0))),
        2, d; ordering = :serial
    )
    u0 = tt_compress!(TTvector(u0_q), max_bond)
    x  = collect(LinRange(0.0, 1.0, N))
    t  = collect((0:Nt) .* dt)

    elapsed = @elapsed sol = allen_cahn_2d_mals(u0, D_xx, d, ε, dt, Nt;
        max_scf                = max_scf,
        scf_tol                = scf_tol,
        max_bond               = max_bond,
        verbose                = verbose,
        verbose_steps          = verbose_steps,
        projection_degree      = projection_degree,
        projection_tolerance   = Float64(projection_tolerance),
        projection_q           = projection_q,
        projection_maxbonddim  = max_bond,
    )

    vals_final = real.(qtt_to_function(sol[end]))
    grid_final = permutedims(reshape(vals_final, N, N))

    R_asym    = sqrt(max(R0^2 - 2 * ε^2 * T_end, 0.0))
    R_measured = _interface_radius(grid_final, x)

    initial_grid = permutedims(reshape(real.(qtt_to_function(u0)), N, N))
    energy_initial = _allen_cahn_2d_energy(initial_grid, dx, ε)
    energy_final   = _allen_cahn_2d_energy(grid_final,   dx, ε)

    return (
        equation              = "Allen-Cahn 2D",
        method                = :mals,
        method_label          = _interpolative_label("SCF-MALS"),
        projection_degree     = projection_degree,
        projection_tolerance  = Float64(projection_tolerance),
        d                     = d,
        N                     = N,
        R0                    = R0f,
        ε                     = εf,
        T_end                 = Float64(T_end),
        Nt                    = Nt,
        x                     = x,
        t                     = t,
        solution              = sol,
        grid                  = grid_final,
        metrics               = (
            runtime_seconds = elapsed,
            R_asymptotic    = Float64(R_asym),
            R_measured      = Float64(R_measured),
            R_error         = Float64(abs(R_measured - R_asym)),
            max_rank        = _max_rank(sol),
            final_rank      = maximum(sol[end].ttv_rks),
            snapshot_count  = length(sol),
            energy_initial  = Float64(energy_initial),
            energy_final    = Float64(energy_final),
            energy_decrease = Float64(energy_initial - energy_final),
            dx              = Float64(dx),
            dt              = Float64(dt),
        ),
    )
end

# ─── Dense-reference comparison benchmarks ───────────────────────────────────
#
# Each benchmark runs the QTT solver and an identically discretized dense
# reference (same grid, same time scheme, same Picard linearisation, but with
# exact pointwise nonlinear coefficients), so the reported gap isolates the
# QTT-specific error: local solves, rank truncation, and the InterpolativeQTT
# coefficient projection.

function _dense_allen_cahn_2d_integrate(v0, A_lin, dt, Nt; max_picard::Int, picard_tol::Real)
    snapshots = [copy(v0)]
    v = copy(v0)
    for _ in 1:Nt
        v_prev = v
        for _ in 1:max_picard
            v_old = v
            v = (A_lin + Diagonal(v .^ 2)) \ (v_prev ./ dt)
            norm(v - v_old) / max(norm(v), eps(Float64)) < picard_tol && break
        end
        push!(snapshots, copy(v))
    end
    return snapshots
end

"""
    allen_cahn_2d_dense_benchmark(; d, ε, T_end, Nt, u0_fun, ...)

Run the 2D Allen-Cahn InterpolativeQTT SCF-MALS integrator next to a dense
implicit-Euler + Picard reference with exact u² coefficients, on the same
serial QTT grid and operators. `u0_fun` receives a 2-vector of coordinates;
the default is the circular tanh interface used by `allen_cahn_2d_benchmark`.
"""
function allen_cahn_2d_dense_benchmark(;
        d::Int                       = 5,
        R0::Real                     = 0.3,
        ε::Real                      = 0.15,
        T_end::Real                  = 0.2,
        Nt::Int                      = 4,
        u0_fun                       = nothing,
        max_scf::Int                 = 12,
        scf_tol::Real                = 1.0e-10,
        max_bond::Int                = 32,
        projection_degree::Int       = 10,
        projection_tolerance::Real   = 1.0e-10,
        projection_q::Int            = 1,
        dense_picard_iters::Int      = 30,
        dense_picard_tol::Real       = 1.0e-13,
        verbose::Bool                = false,
        verbose_steps::Bool          = false
    )
    N  = 2^d
    dx = 1.0 / (N - 1)
    dt = T_end / Nt

    I_d  = id_tto(d)
    D_xx = (1.0 / dx^2) * (Δ_NN(d) ⊗ I_d + I_d ⊗ Δ_NN(d))

    R0f = Float64(R0)
    εf  = Float64(ε)
    ic  = u0_fun === nothing ?
        (c -> tanh((sqrt((c[1] - 0.5)^2 + (c[2] - 0.5)^2) - R0f) / (εf * sqrt(2.0)))) :
        u0_fun
    u0 = tt_compress!(TTvector(function_to_qttv(ic, 2, d; ordering = :serial)), max_bond)
    x  = collect(LinRange(0.0, 1.0, N))
    t  = collect((0:Nt) .* dt)

    qtt_runtime = @elapsed sol = allen_cahn_2d_mals(u0, D_xx, d, ε, dt, Nt;
        max_scf               = max_scf,
        scf_tol               = scf_tol,
        max_bond              = max_bond,
        verbose               = verbose,
        verbose_steps         = verbose_steps,
        projection_degree     = projection_degree,
        projection_tolerance  = Float64(projection_tolerance),
        projection_q          = projection_q,
        projection_maxbonddim = max_bond,
    )

    A_lin = (1.0 / dt - 1.0) * I + εf^2 * qtto_to_matrix(D_xx)
    v0 = real.(qtt_to_function(u0))
    dense_runtime = @elapsed dense_snapshots = _dense_allen_cahn_2d_integrate(
        v0, A_lin, dt, Nt;
        max_picard = dense_picard_iters,
        picard_tol = dense_picard_tol,
    )

    stepwise = [
        norm(real.(qtt_to_function(sol[k + 1])) - dense_snapshots[k + 1]) /
            max(norm(dense_snapshots[k + 1]), eps(Float64))
            for k in 1:Nt
    ]

    grid_qtt   = permutedims(reshape(real.(qtt_to_function(sol[end])), N, N))
    grid_dense = permutedims(reshape(dense_snapshots[end], N, N))

    return (
        equation             = "Allen-Cahn 2D",
        method               = :mals,
        method_label         = _interpolative_label("SCF-MALS"),
        reference            = :dense,
        projection_degree    = projection_degree,
        projection_tolerance = Float64(projection_tolerance),
        d                    = d,
        N                    = N,
        ε                    = εf,
        T_end                = Float64(T_end),
        Nt                   = Nt,
        x                    = x,
        t                    = t,
        solution             = sol,
        dense_snapshots      = dense_snapshots,
        grid_qtt             = grid_qtt,
        grid_dense           = grid_dense,
        metrics              = (
            stepwise_relative_error = stepwise,
            final_relative_error    = stepwise[end],
            max_rank                = _max_rank(sol),
            final_rank              = maximum(sol[end].ttv_rks),
            qtt_runtime_seconds     = qtt_runtime,
            dense_runtime_seconds   = dense_runtime,
            dx                      = Float64(dx),
            dt                      = Float64(dt),
        ),
    )
end

function _dense_gpe_scf_ground_state(H_d, g::Real, ψ_start; max_iter::Int, tol::Real)
    ψ = normalize(copy(ψ_start))
    for _ in 1:max_iter
        F = eigen(Hermitian(Matrix(H_d) + g * Diagonal(abs2.(ψ))), 1:1)
        ψ_new = normalize(F.vectors[:, 1])
        (ψ' * ψ_new) < 0 && (ψ_new = -ψ_new)
        converged = norm(ψ_new - ψ) < tol
        ψ = ψ_new
        converged && break
    end
    ρ = abs2.(ψ)
    μ = real(ψ' * (H_d * ψ)) + g * sum(ρ .^ 2)
    return μ, ψ
end

"""
    gpe_2d_dense_benchmark(; L, κ, g_vals, ...)

Run the 2D GPE SCF-ALS and SCF-MALS ground-state solvers (with multivariate
InterpolativeQTT density projection) next to a dense SCF reference with exact
densities, warm-starting both through the same `g_vals` continuation.
"""
function gpe_2d_dense_benchmark(;
        L::Int                       = 4,
        κ::Real                      = 50.0,
        g_vals                       = [0.0, 25.0],
        random_rank::Int             = 4,
        linear_sweeps::Int           = 10,
        nonlinear_sweeps::Int        = 12,
        mals_rmax::Int               = 12,
        mals_tol::Real               = 1.0e-10,
        projection_degree::Int       = 8,
        projection_tolerance::Real   = 1.0e-9,
        projection_q::Int            = 1,
        projection_maxbonddim::Int   = mals_rmax,
        dense_scf_iters::Int         = 400,
        dense_scf_tol::Real          = 1.0e-12,
        seed::Int                    = 42
    )
    N = 2^L
    h = 1.0 / (N - 1)
    x = collect(LinRange(0.0, 1.0, N))

    I_L   = id_tto(L)
    H_kin = (1.0 / (2h^2)) * (Δ(L) ⊗ I_L + I_L ⊗ Δ(L))
    o_L   = ones_tt(2, L)
    V_x   = function_to_qtt(s -> κ * (s - 0.5)^2, L)
    H_lin = H_kin + ttv_to_diag_tto(V_x ⊗ o_L + o_L ⊗ V_x)

    Random.seed!(seed)
    ψ0 = rand_tt(fill(2, 2L), random_rank; normalise = true)
    _, ψ_lin = als_eigsolve(H_lin, ψ0; sweep_schedule = [linear_sweeps])
    μ_lin = real(dot(ψ_lin, H_lin * ψ_lin))

    nk = length(g_vals)
    μ_dense = zeros(Float64, nk)
    ψ_dense = Vector{Vector{Float64}}(undef, nk)
    H_d = Hermitian(qtto_to_matrix(H_lin))
    ψ_d = normalize(real.(qtt_to_function(ψ_lin)))
    dense_runtime = @elapsed for k in 1:nk
        μ_dense[k], ψ_d = _dense_gpe_scf_ground_state(H_d, Float64(g_vals[k]), ψ_d;
            max_iter = dense_scf_iters, tol = dense_scf_tol)
        ψ_dense[k] = ψ_d
    end

    μ_als   = zeros(Float64, nk)
    μ_mals  = zeros(Float64, nk)
    ψ_list_als  = Vector{Any}(undef, nk)
    ψ_list_mals = Vector{Any}(undef, nk)
    μ_als[1]  = μ_lin
    μ_mals[1] = μ_lin
    ψ_list_als[1]  = ψ_lin
    ψ_list_mals[1] = ψ_lin

    qtt_runtime = @elapsed for k in 2:nk
        g = Float64(g_vals[k])
        μ_hist_als, ψg_als = nonlinear_als_eigsolve(H_lin, g, ψ_list_als[k - 1];
            sweep_count           = nonlinear_sweeps,
            projection_degree     = projection_degree,
            projection_tolerance  = projection_tolerance,
            projection_maxbonddim = projection_maxbonddim,
            projection_q          = projection_q,
            projection_ndims      = 2,
            projection_ordering   = :serial,
            verbose               = false,
        )
        μ_hist_mals, ψg_mals, _ = nonlinear_mals_eigsolve(H_lin, g, ψ_list_mals[k - 1];
            tol                   = Float64(mals_tol),
            sweep_schedule        = [nonlinear_sweeps + 1],
            rmax_schedule         = [mals_rmax],
            projection_degree     = projection_degree,
            projection_tolerance  = projection_tolerance,
            projection_maxbonddim = projection_maxbonddim,
            projection_q          = projection_q,
            projection_ndims      = 2,
            projection_ordering   = :serial,
            verbose               = false,
        )
        ψ_list_als[k]  = ψg_als
        ψ_list_mals[k] = ψg_mals
        μ_als[k]  = last(μ_hist_als)
        μ_mals[k] = last(μ_hist_mals)
    end

    nonlinear_idx = 2:nk
    μ_err_als  = [abs(μ_als[k] - μ_dense[k]) / max(abs(μ_dense[k]), eps(Float64)) for k in nonlinear_idx]
    μ_err_mals = [abs(μ_mals[k] - μ_dense[k]) / max(abs(μ_dense[k]), eps(Float64)) for k in nonlinear_idx]

    ρ_err(ψ_tt, ψ_ref) = begin
        ρ_tt  = abs2.(real.(qtt_to_function(ψ_tt)))
        ρ_ref = abs2.(ψ_ref)
        norm(ρ_tt - ρ_ref) / max(norm(ρ_ref), eps(Float64))
    end
    ρ_err_als  = [ρ_err(ψ_list_als[k], ψ_dense[k]) for k in nonlinear_idx]
    ρ_err_mals = [ρ_err(ψ_list_mals[k], ψ_dense[k]) for k in nonlinear_idx]

    return (
        equation             = "2D Gross-Pitaevskii",
        method               = :scf_als_mals,
        method_label         = "InterpolativeQTT-SCF-ALS / InterpolativeQTT-SCF-MALS",
        reference            = :dense,
        projection_degree    = projection_degree,
        projection_tolerance = Float64(projection_tolerance),
        L                    = L,
        N                    = N,
        κ                    = Float64(κ),
        g_vals               = collect(Float64, g_vals),
        x                    = x,
        μ_als                = μ_als,
        μ_mals               = μ_mals,
        μ_dense              = μ_dense,
        ψ_list_als           = ψ_list_als,
        ψ_list_mals          = ψ_list_mals,
        ψ_dense              = ψ_dense,
        metrics              = (
            linear_mu                          = μ_lin,
            mu_relative_errors_als             = μ_err_als,
            mu_relative_errors_mals            = μ_err_mals,
            max_mu_relative_error_als          = maximum(μ_err_als),
            max_mu_relative_error_mals         = maximum(μ_err_mals),
            density_relative_errors_als        = ρ_err_als,
            density_relative_errors_mals       = ρ_err_mals,
            max_density_relative_error_als     = maximum(ρ_err_als),
            max_density_relative_error_mals    = maximum(ρ_err_mals),
            max_rank_als                       = maximum(maximum(ψ.ttv_rks) for ψ in ψ_list_als),
            max_rank_mals                      = maximum(maximum(ψ.ttv_rks) for ψ in ψ_list_mals),
            qtt_runtime_seconds                = qtt_runtime,
            dense_runtime_seconds              = dense_runtime,
        ),
    )
end

function _dense_kdv_step_implicit_euler(v_prev, D_x, D_xxx, dt; max_scf::Int, scf_tol::Real)
    v = copy(v_prev)
    for _ in 1:max_scf
        v_old = v
        A = (1.0 / dt) * I + 6.0 * Diagonal(v) * D_x + D_xxx
        v = A \ (v_prev ./ dt)
        norm(v - v_old) / max(norm(v), eps(Float64)) < scf_tol && break
    end
    return v
end

function _dense_kdv_step_crank_nicolson(v_prev, D_x, D_xxx, dt; max_scf::Int, scf_tol::Real)
    v = copy(v_prev)
    for _ in 1:max_scf
        v_old = v
        A   = (1.0 / dt) * I + 3.0 * Diagonal(v) * D_x + 0.5 * D_xxx
        rhs = v_prev ./ dt .- 3.0 .* (v .* (D_x * v_prev)) .- 0.5 .* (D_xxx * v_prev)
        v = A \ rhs
        norm(v - v_old) / max(norm(v), eps(Float64)) < scf_tol && break
    end
    return v
end

"""
    kdv_1d_dense_benchmark(; d, L, T_end, Nt, c, x0, method, ...)

Run `kdv_soliton_benchmark` next to a dense reference using the matching time
scheme (implicit Euler for `:als`/`:mals`, Crank-Nicolson for `:cn_mals`) and
the same Picard linearisation with exact advection coefficients. Reports the
stepwise QTT-vs-dense gap as well as both solutions' analytical soliton error.
"""
function kdv_1d_dense_benchmark(;
        d::Int                     = 8,
        L::Real                    = 25.0,
        T_end::Real                = 1.0,
        Nt::Int                    = 250,
        c::Real                    = 1.0,
        x0::Real                   = 9.0,
        method::Symbol             = :cn_mals,
        max_scf::Int               = 25,
        scf_tol::Real              = 1.0e-8,
        max_bond::Int              = 50,
        projection_degree::Int     = 8,
        projection_tolerance::Real = 1.0e-10,
        projection_q::Int          = 1,
        projection_mode::Symbol    = :singlescale,
        projection_adaptive_tolerance::Real = 1.0e-8,
        verbose::Bool              = false,
        verbose_steps::Bool        = false
    )
    qtt_runtime = @elapsed bench = kdv_soliton_benchmark(;
        d = d, L = L, T_end = T_end, Nt = Nt, c = c, x0 = x0,
        method = method,
        max_scf = max_scf,
        scf_tol = scf_tol,
        max_bond = max_bond,
        projection_degree = projection_degree,
        projection_tolerance = projection_tolerance,
        projection_q = projection_q,
        projection_mode = projection_mode,
        projection_adaptive_tolerance = projection_adaptive_tolerance,
        verbose = verbose,
        verbose_steps = verbose_steps,
    )

    dx = bench.metrics.dx
    dt = bench.metrics.dt
    D_x_m   = qtto_to_matrix((1 / (2dx)) * ∇_c_P(d))
    D_xxx_m = qtto_to_matrix((1 / (2dx^3)) * ∇3_P(d))
    dense_step = method == :cn_mals ?
        _dense_kdv_step_crank_nicolson : _dense_kdv_step_implicit_euler

    v = real.(qtt_to_function(bench.solution[1]))
    dense_snapshots = [copy(v)]
    dense_runtime = @elapsed for _ in 1:Nt
        v = dense_step(v, D_x_m, D_xxx_m, dt; max_scf = max_scf, scf_tol = scf_tol)
        push!(dense_snapshots, copy(v))
    end

    stepwise = [
        norm(real.(qtt_to_function(bench.solution[k + 1])) - dense_snapshots[k + 1]) /
            max(norm(dense_snapshots[k + 1]), eps(Float64))
            for k in 1:Nt
    ]
    dense_vs_analytical = norm(dense_snapshots[end] - bench.exact_values) /
        max(norm(bench.exact_values), eps(Float64))

    return (
        equation             = "KdV",
        method               = method,
        method_label         = bench.method_label,
        reference            = :dense,
        projection_degree    = projection_degree,
        projection_tolerance = Float64(projection_tolerance),
        d                    = d,
        N                    = bench.N,
        L                    = Float64(L),
        T_end                = Float64(T_end),
        Nt                   = Nt,
        x                    = bench.x,
        t                    = bench.t,
        solution             = bench.solution,
        U                    = bench.U,
        dense_snapshots      = dense_snapshots,
        exact_values         = bench.exact_values,
        metrics              = (
            stepwise_relative_error    = stepwise,
            final_dense_relative_error = stepwise[end],
            qtt_vs_analytical_error    = bench.metrics.relative_error,
            dense_vs_analytical_error  = dense_vs_analytical,
            max_rank                   = bench.metrics.max_rank,
            qtt_runtime_seconds        = qtt_runtime,
            dense_runtime_seconds      = dense_runtime,
            dx                         = Float64(dx),
            dt                         = Float64(dt),
        ),
    )
end
