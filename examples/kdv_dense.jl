# KdV soliton: InterpolativeQTT QTT integrator vs dense reference.
#
# The dense reference uses the matching time scheme (Crank-Nicolson here) and
# Picard linearisation with exact advection coefficients, so the QTT-vs-dense
# gap isolates the QTT-side error.
#
# The dominant QTT error for this problem is the I_q rebuild floor: the
# single-scale coefficient projection rebuilds u with degree-p Chebyshev pieces
# at the coarsest scale, and a localized soliton on [0, 25] needs high degree.
# The degree sweep below makes that floor visible. `projection_mode = :adaptive`
# refines sub-intervals until the local interpolation error is resolved, which
# removes the floor entirely: at degree 8 it reaches the time-scheme error.

using CairoMakie

include("nonlinear_benchmark_utils.jl")

d     = 8        # N = 128 grid points
Ldom  = 25.0
T     = 1.0
Nt    = 200
c     = 1.0
x0    = 9.0
degrees = [8, 10, 12, 14, 16]

benches = map(degrees) do degree
    kdv_1d_dense_benchmark(;
        d = d,
        L = Ldom,
        T_end = T,
        Nt = Nt,
        c = c,
        x0 = x0,
        method = :cn_mals,
        max_scf = 25,
        scf_tol = 1.0e-9,
        max_bond = 40,
        projection_degree = degree,
        projection_tolerance = 1.0e-10,
    )
end
degree_errors = [b.metrics.final_dense_relative_error for b in benches]

bench = kdv_1d_dense_benchmark(;
    d = d,
    L = Ldom,
    T_end = T,
    Nt = Nt,
    c = c,
    x0 = x0,
    method = :cn_mals,
    max_scf = 25,
    scf_tol = 1.0e-9,
    max_bond = 40,
    projection_degree = 8,
    projection_tolerance = 1.0e-10,
    projection_mode = :adaptive,
    projection_adaptive_tolerance = 1.0e-9,
)
adaptive_error = bench.metrics.final_dense_relative_error

@info "KdV dense comparison" method=bench.method_label singlescale_degrees=degrees singlescale_qtt_vs_dense=round.(degree_errors, sigdigits = 3) adaptive_deg8_qtt_vs_dense=round(adaptive_error, sigdigits = 3) qtt_vs_analytical_adaptive=round(bench.metrics.qtt_vs_analytical_error, sigdigits = 3) dense_vs_analytical=round(bench.metrics.dense_vs_analytical_error, sigdigits = 3) max_rank=bench.metrics.max_rank

set_theme!(Theme(fontsize = 13))
fig = Figure(size = (1200, 800))

x = bench.x
t = bench.t
U_dense = reduce(hcat, bench.dense_snapshots)
Δ_spacetime = abs.(bench.U - U_dense)

ax_u = Axis(fig[1, 1:2];
    title = "Final profiles at T = $(T)  (adaptive projection, degree 8)",
    xlabel = "x",
    ylabel = "u(x, T)")
lines!(ax_u, x, bench.exact_values; color = :black, linewidth = 2, label = "analytical soliton")
lines!(ax_u, x, bench.dense_snapshots[end]; color = :forestgreen, linewidth = 2, label = "dense (same scheme)")
lines!(ax_u, x, real.(qtt_to_function(bench.solution[end])); color = :firebrick, linewidth = 2, linestyle = :dash, label = bench.method_label)
axislegend(ax_u; position = :rt)

ax_deg = Axis(fig[1, 3];
    title = "I_q rebuild floor",
    xlabel = "projection degree",
    ylabel = "final QTT-vs-dense error",
    yscale = log10)
scatterlines!(ax_deg, degrees, degree_errors; color = :firebrick, linewidth = 2, label = "single-scale")
hlines!(ax_deg, [adaptive_error]; color = :steelblue, linestyle = :dot, linewidth = 2, label = "adaptive, degree 8")
hlines!(ax_deg, [bench.metrics.dense_vs_analytical_error];
    color = :forestgreen, linestyle = :dash, label = "scheme error vs analytical")
axislegend(ax_deg; position = :rt)

ax_e = Axis(fig[2, 1];
    title = "QTT vs dense over time (adaptive, degree 8)",
    xlabel = "t",
    ylabel = "relative error",
    yscale = log10)
lines!(ax_e, t[2:end], bench.metrics.stepwise_relative_error; color = :firebrick, linewidth = 2)

ax_h = Axis(fig[2, 2]; title = "|U_qtt − U_dense|(x, t)", xlabel = "t", ylabel = "x")
hm = heatmap!(ax_h, t, x, permutedims(Δ_spacetime); colormap = :viridis)
Colorbar(fig[2, 3], hm; label = "|Δu|")

out = joinpath(ensure_example_output_dir(), "kdv_dense.png")
save(out, fig)
@info "Figure saved" path=out
display(fig)
