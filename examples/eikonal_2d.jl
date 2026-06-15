# Nonlinear Eikonal equation |∇T| = s(x,y) solved in QTT format.
#
# A slow lens bends the first-arrival fronts. The solver continues the viscous
# regularisation -ε·ΔT + |∇T|² = s² down in ε with Picard-frozen advection and
# fixed-rank ALS solves. The adaptive (multiscale) InterpolativeQTT machinery
# appears twice where it is genuinely needed:
#   1. the steep lens slowness field s² is built with mode = :adaptive;
#   2. the pointwise Eikonal residual |∇T| − s is reconstructed as a QTT field
#      through one multivariate adaptive project_nonlinearity call over the
#      three kinked fields (∂x T, ∂y T, s).
# References: a dense Godunov fast-sweeping solution of the true ε = 0 problem
# (shows the O(ε) viscosity bias); for small grids (d ≤ 7) a dense same-scheme
# Picard solve additionally isolates the QTT discretisation error.

using CairoMakie

include("nonlinear_benchmark_utils.jl")

d = 8      # 64 × 64 interior grid

bench = eikonal_2d_benchmark(;
    d = d,
    lens_center = (0.62, 0.62),
    lens_width = 0.012,
    lens_strength = 3.0,
    ε_schedule = [0.2, 0.1, 0.05],
    max_scf = 30,
    scf_tol = 1.0e-11,
    max_bond = 100,
    als_sweeps = 6,
    projection_degree = 10,
    residual_field = true,
    verbose = true,
)

@info "Viscous Eikonal 2D" method=bench.method_label slowness_build_err=round(bench.metrics.slowness_build_relative_error, sigdigits = 3) qtt_vs_dense=round(bench.metrics.qtt_vs_dense_relative_error, sigdigits = 3) fast_sweeping_gap=round(bench.metrics.fast_sweeping_gap, sigdigits = 3) eikonal_residuals=round.(bench.metrics.eikonal_residuals, sigdigits = 3) max_rank=bench.metrics.max_rank qtt_runtime=round(bench.metrics.qtt_runtime_seconds, digits = 1) residual_field_runtime=round(bench.metrics.residual_field_runtime_seconds, digits = 1)

set_theme!(Theme(fontsize = 13))
fig = Figure(size = (1280, 820))

x = bench.x
N = bench.N
T_grid = bench.T_grid

ax_s = Axis(fig[1, 1]; title = "Slowness s(x,y) (adaptive QTT build)", xlabel = "x", ylabel = "y", aspect = DataAspect())
hm_s = heatmap!(ax_s, x, x, bench.slowness; colormap = :thermal)
Colorbar(fig[2, 1], hm_s; label = "s", vertical = false)

_T_title = bench.T_fast_sweeping !== nothing ? "QTT solution T (contours: fast sweeping)" : "QTT solution T"
ax_T = Axis(fig[1, 2]; title = _T_title, xlabel = "x", ylabel = "y", aspect = DataAspect())
hm_T = heatmap!(ax_T, x, x, T_grid; colormap = :viridis)
bench.T_fast_sweeping !== nothing && contour!(ax_T, x, x, bench.T_fast_sweeping; color = :white, linewidth = 1.2, levels = 12)
Colorbar(fig[2, 2], hm_T; label = "T", vertical = false)

if bench.T_dense !== nothing
    err_dense = abs.(T_grid .- collect(reshape(bench.T_dense, N, N)))
    ax_e = Axis(fig[1, 3]; title = "|T_qtt − T_dense| (same scheme)", xlabel = "x", ylabel = "y", aspect = DataAspect())
    hm_e = heatmap!(ax_e, x, x, err_dense; colormap = :viridis)
    Colorbar(fig[2, 3], hm_e; label = "|ΔT|", vertical = false)
else
    _colors = [:steelblue, :firebrick, :seagreen, :darkorange]
    ax_e = Axis(fig[1, 3]; title = "Picard convergence per ε level",
        xlabel = "iteration", ylabel = "relative change", yscale = log10)
    for (i, hist) in enumerate(bench.info.picard_history)
        scatterlines!(ax_e, 1:length(hist), hist;
            color = _colors[mod1(i, 4)], linewidth = 1.5,
            label = "ε = $(bench.ε_schedule[i])")
    end
    axislegend(ax_e; position = :rt)
end

if bench.residual_field !== nothing
    res_grid = collect(reshape(real.(qtt_to_function(TTvector(bench.residual_field))), N, N))
    ax_r = Axis(fig[3, 1]; title = "Eikonal residual |∇T| − s (adaptive projection)", xlabel = "x", ylabel = "y", aspect = DataAspect())
    hm_r = heatmap!(ax_r, x, x, res_grid; colormap = :balance, colorrange = (-0.5, 0.5))
    Colorbar(fig[4, 1], hm_r; label = "|∇T| − s", vertical = false)
end

ax_c = Axis(fig[3, 2];
    title = "ε-continuation",
    xlabel = "ε",
    ylabel = "mean | |∇T| − s |",
    xscale = log10,
    yscale = log10)
scatterlines!(ax_c, bench.ε_schedule, bench.metrics.eikonal_residuals; color = :firebrick, linewidth = 2)

ax_rk = Axis(fig[3, 3]; title = "Effective rank per ε level (trunc 1e-10)", xlabel = "continuation level", ylabel = "effective max rank")
scatterlines!(ax_rk, 1:length(bench.metrics.rank_history), bench.metrics.rank_history; color = :steelblue, linewidth = 2)

out = joinpath(ensure_example_output_dir(), "eikonal_2d.png")
save(out, fig)
@info "Figure saved" path=out
display(fig)
