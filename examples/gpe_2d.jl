# 2D Gross-Pitaevskii equation in QTT format.
#
# (-1/(2h^2) Delta + V(x,y) + g|psi|^2) psi = mu psi
# on [0,1]^2 with a separable harmonic trap.

using CairoMakie

include("nonlinear_benchmark_utils.jl")

bench = gpe_2d_benchmark(;
    L = 6,
    κ = 200.0,
    g_vals = [0.0, 500.0, 2000.0, 10000.0],
    random_rank = 8,
    linear_sweeps = 20,
    nonlinear_sweeps = 30,
    mals_rmax = 12,
    mals_tol = 1.0e-10,
    projection_degree = 12,
    projection_tolerance = 1.0e-10,
    seed = 42
)

@info "GPE 2D benchmark" method=bench.method_label expected_linear_mu=round(bench.metrics.expected_linear_mu, digits = 4) linear_mu=round(bench.metrics.linear_mu, digits = 6) max_mu_gap=round(bench.metrics.max_mu_gap, sigdigits = 4) max_rank_als=bench.metrics.max_rank_als max_rank_mals=bench.metrics.max_rank_mals max_norm_error_als=round(bench.metrics.max_norm_error_als, sigdigits = 3) max_norm_error_mals=round(bench.metrics.max_norm_error_mals, sigdigits = 3)

set_theme!(Theme(fontsize = 14))
fig = Figure(size = (1080, 760))

for k in eachindex(bench.g_vals)
    g = bench.g_vals[k]
    ax = Axis(fig[1, k],
        xlabel = "x",
        ylabel = k == 1 ? "y" : "",
        title = "g = $(Int(g)), mu = $(round(bench.μ_list_mals[k], digits = 2))",
        aspect = DataAspect())
    k > 1 && hideydecorations!(ax; grid = false)
    hm = heatmap!(ax, bench.x, bench.y, bench.density_grids[k]; colormap = :inferno)
    Colorbar(fig[2, k], hm;
        vertical = false,
        label = k == 2 ? "|psi(x,y)|^2" : "",
        ticklabelsize = 11)
end

ax3 = Axis(fig[3, 1:3],
    xlabel = "nonlinear coupling g",
    ylabel = "chemical potential mu",
    title = "2D GPE continuation: SCF-ALS and SCF-MALS agreement",
    xticks = Int.(bench.g_vals))
lines!(ax3, bench.g_vals, bench.μ_list_als; color = :black, linewidth = 2, label = "SCF-ALS")
scatter!(ax3, bench.g_vals, bench.μ_list_als; color = :black, markersize = 10)
lines!(ax3, bench.g_vals, bench.μ_list_mals;
    color = :firebrick,
    linewidth = 2,
    linestyle = :dash,
    label = "SCF-MALS")
scatter!(ax3, bench.g_vals, bench.μ_list_mals; color = :firebrick, markersize = 10)
for k in eachindex(bench.g_vals)
    text!(ax3, bench.g_vals[k], bench.μ_list_mals[k] + 0.3;
        text = "$(round(bench.μ_list_mals[k], digits = 1))",
        fontsize = 11,
        align = (:center, :bottom),
        color = :black)
end
axislegend(ax3; position = :lt, framevisible = false)

out = joinpath(ensure_example_output_dir(), "gpe_2d.png")
save(out, fig)
@info "Figure saved" path=out

for k in eachindex(bench.g_vals)
    @info "GPE continuation point" g=bench.g_vals[k] mu_als=round(bench.μ_list_als[k], digits = 4) mu_mals=round(bench.μ_list_mals[k], digits = 4) mu_gap=round(bench.μ_gaps[k], sigdigits = 3) rank_als=maximum(bench.ψ_list_als[k].ttv_rks) rank_mals=maximum(bench.ψ_list_mals[k].ttv_rks)
end

display(fig)
