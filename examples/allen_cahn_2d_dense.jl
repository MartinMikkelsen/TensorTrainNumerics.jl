# 2D Allen-Cahn: InterpolativeQTT SCF-MALS vs dense reference.
#
# Both sides use the identical implicit-Euler + Picard discretization; the dense
# side evaluates u² exactly, so the reported gap isolates the QTT-specific error
# (MALS local solves, rank truncation, and the InterpolativeQTT coefficient
# projection with its O(f''·4^-d) inversion floor).

using CairoMakie

include("nonlinear_benchmark_utils.jl")

d   = 6       # bits per spatial dimension, N = 64 grid points per axis
R0  = 0.3
ε   = 0.06
T   = 0.25
Nt  = 10
rmax = 30

bench = allen_cahn_2d_dense_benchmark(;
    d = d,
    R0 = R0,
    ε = ε,
    T_end = T,
    Nt = Nt,
    max_bond = rmax,
    projection_degree = 12,
    projection_tolerance = 1.0e-10,
    dense_picard_iters = 12,
    dense_picard_tol = 1.0e-12,
    verbose_steps = true,
)

@info "Allen-Cahn 2D dense comparison" method=bench.method_label final_relative_error=round(bench.metrics.final_relative_error, sigdigits = 3) max_rank=bench.metrics.max_rank qtt_runtime=round(bench.metrics.qtt_runtime_seconds, digits = 2) dense_runtime=round(bench.metrics.dense_runtime_seconds, digits = 2)

set_theme!(Theme(fontsize = 13))
fig = Figure(size = (1200, 800))

x = bench.x
err_grid = abs.(bench.grid_qtt - bench.grid_dense)
ranks = [maximum(u.ttv_rks) for u in bench.solution]

ax_q = Axis(fig[1, 1]; title = "QTT u(x,y,T)", xlabel = "x", ylabel = "y", aspect = DataAspect())
hm_q = heatmap!(ax_q, x, x, bench.grid_qtt; colormap = :RdBu, colorrange = (-1, 1))
Colorbar(fig[2, 1], hm_q; label = "u", vertical = false)

ax_d = Axis(fig[1, 2]; title = "Dense u(x,y,T)", xlabel = "x", ylabel = "y", aspect = DataAspect())
hm_d = heatmap!(ax_d, x, x, bench.grid_dense; colormap = :RdBu, colorrange = (-1, 1))
Colorbar(fig[2, 2], hm_d; label = "u", vertical = false)

ax_e = Axis(fig[1, 3]; title = "|QTT − dense|", xlabel = "x", ylabel = "y", aspect = DataAspect())
hm_e = heatmap!(ax_e, x, x, err_grid; colormap = :viridis)
Colorbar(fig[2, 3], hm_e; label = "|Δu|", vertical = false)

ax_err = Axis(fig[3, 1:2];
    title = "Relative error vs dense reference",
    xlabel = "time step",
    ylabel = "‖u_qtt − u_dense‖ / ‖u_dense‖",
    yscale = log10)
lines!(ax_err, 1:Nt, bench.metrics.stepwise_relative_error;
    color = :firebrick, linewidth = 2, label = bench.method_label)
scatter!(ax_err, 1:Nt, bench.metrics.stepwise_relative_error; color = :firebrick, markersize = 7)
axislegend(ax_err; position = :rb)

ax_r = Axis(fig[3, 3]; title = "Max bond dimension", xlabel = "time step", ylabel = "max rank")
lines!(ax_r, 0:Nt, ranks; color = :steelblue, linewidth = 2)
scatter!(ax_r, 0:Nt, ranks; color = :steelblue, markersize = 6)

out = joinpath(ensure_example_output_dir(), "allen_cahn_2d_dense.png")
save(out, fig)
@info "Figure saved" path=out
display(fig)
