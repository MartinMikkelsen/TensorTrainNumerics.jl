# 2D Gross-Pitaevskii ground states: InterpolativeQTT SCF-ALS / SCF-MALS vs a
# dense SCF reference with exact densities, continued through increasing g.
#
# The QTT solvers project |ψ|² through the multivariate InterpolativeQTT path
# (invert to Chebyshev tables, square pointwise, rebuild); the dense reference
# diagonalizes the exact frozen-density Hamiltonian, so μ and density gaps
# measure the QTT-specific error directly.

# Note on projection_mode: these 2D solves keep the default :singlescale
# projection. The fields here are globally smooth, so single-scale Chebyshev
# rebuild is already accurate; :adaptive would refine to the inversion-table
# cells wherever the field has curvature (4^level interval counts in 2D),
# measured at ~50x slower per projection at d = 6 for no accuracy gain.

using CairoMakie

include("nonlinear_benchmark_utils.jl")

L      = 5        # bits per dimension, N = 32 grid points per axis
κ      = 100.0
g_vals = [0.0, 50.0, 200.0, 500.0]

bench = gpe_2d_dense_benchmark(;
    L = L,
    κ = κ,
    g_vals = g_vals,
    random_rank = 6,
    linear_sweeps = 14,
    nonlinear_sweeps = 18,
    mals_rmax = 14,
    projection_degree = 10,
    projection_tolerance = 1.0e-10,
    dense_scf_iters = 300,
    dense_scf_tol = 1.0e-11,
    seed = 42,
)

@info "GPE 2D dense comparison" method=bench.method_label max_mu_err_als=round(bench.metrics.max_mu_relative_error_als, sigdigits = 3) max_mu_err_mals=round(bench.metrics.max_mu_relative_error_mals, sigdigits = 3) max_density_err_mals=round(bench.metrics.max_density_relative_error_mals, sigdigits = 3) qtt_runtime=round(bench.metrics.qtt_runtime_seconds, digits = 2) dense_runtime=round(bench.metrics.dense_runtime_seconds, digits = 2)

set_theme!(Theme(fontsize = 13))
fig = Figure(size = (1200, 800))

N = bench.N
x = bench.x
g_nl = bench.g_vals[2:end]

ax_mu = Axis(fig[1, 1];
    title = "Chemical potential vs g",
    xlabel = "g",
    ylabel = "μ")
lines!(ax_mu, bench.g_vals, bench.μ_dense; color = :black, linewidth = 2, label = "dense SCF")
scatter!(ax_mu, bench.g_vals, bench.μ_als; color = :firebrick, marker = :circle, markersize = 11, label = "QTT SCF-ALS")
scatter!(ax_mu, bench.g_vals, bench.μ_mals; color = :steelblue, marker = :utriangle, markersize = 11, label = "QTT SCF-MALS")
axislegend(ax_mu; position = :lt)

ax_err = Axis(fig[1, 2];
    title = "Relative μ error vs dense",
    xlabel = "g",
    ylabel = "|μ_qtt − μ_dense| / |μ_dense|",
    yscale = log10)
scatterlines!(ax_err, g_nl, bench.metrics.mu_relative_errors_als; color = :firebrick, label = "SCF-ALS")
scatterlines!(ax_err, g_nl, bench.metrics.mu_relative_errors_mals; color = :steelblue, label = "SCF-MALS")
axislegend(ax_err; position = :rb)

ρ_qtt = permutedims(reshape(abs2.(real.(qtt_to_function(bench.ψ_list_mals[end]))), N, N))
ρ_dense = permutedims(reshape(abs2.(bench.ψ_dense[end]), N, N))

ax_rq = Axis(fig[2, 1]; title = "QTT MALS density, g = $(bench.g_vals[end])", xlabel = "x", ylabel = "y", aspect = DataAspect())
hm_rq = heatmap!(ax_rq, x, x, ρ_qtt; colormap = :inferno)
Colorbar(fig[3, 1], hm_rq; label = "|ψ|²", vertical = false)

ax_rd = Axis(fig[2, 2]; title = "Dense density, g = $(bench.g_vals[end])", xlabel = "x", ylabel = "y", aspect = DataAspect())
hm_rd = heatmap!(ax_rd, x, x, ρ_dense; colormap = :inferno)
Colorbar(fig[3, 2], hm_rd; label = "|ψ|²", vertical = false)

ax_sl = Axis(fig[2, 3];
    title = "Density slice y = 1/2",
    xlabel = "x",
    ylabel = "|ψ(x, 1/2)|²")
mid = N ÷ 2
lines!(ax_sl, x, ρ_dense[mid, :]; color = :black, linewidth = 2, label = "dense")
lines!(ax_sl, x, ρ_qtt[mid, :]; color = :steelblue, linewidth = 2, linestyle = :dash, label = "QTT MALS")
axislegend(ax_sl; position = :rt)

out = joinpath(ensure_example_output_dir(), "gpe_2d_dense.png")
save(out, fig)
@info "Figure saved" path=out
display(fig)
