using CairoMakie

include("nonlinear_benchmark_utils.jl")

bench = kdv_soliton_benchmark(;
    d = 8,
    L = 25.0,
    T_end = 1.0,
    Nt = 250,
    c = 1.0,
    method = :mals,
    max_scf = 25,
    scf_tol = 1.0e-8,
    max_bond = 50,
    projection_degree = 15,
    projection_tolerance = 1.0e-10,
    verbose_steps = true
)

@info "KdV benchmark" method=bench.method_label max_rank=bench.metrics.max_rank relative_l2_error=round(bench.metrics.relative_error, sigdigits = 3)

let
    fig = Figure(size = (920, 560))

    ax1 = Axis(fig[1, 1],
        xlabel = "x",
        ylabel = "t",
        title = "KdV soliton ($(bench.method_label), c = $(bench.c), T = $(bench.T_end), N = $(bench.N))")
    hm = heatmap!(ax1, bench.x, bench.t, bench.U; colormap = :viridis)
    Colorbar(fig[1, 2], hm; label = "u")

    ax2 = Axis(fig[2, 1:2],
        xlabel = "x",
        ylabel = "u(x,t)",
        title = "Final-time soliton error = $(round(bench.metrics.relative_error, sigdigits = 3))")
    lines!(ax2, bench.x, bench.exact_values;
        label = "exact t = $(bench.T_end)",
        linewidth = 2,
        color = :black,
        linestyle = :dash)
    lines!(ax2, bench.x, bench.U[:, end];
        label = "$(bench.method_label) t = $(bench.T_end)",
        linewidth = 2,
        color = :blue)
    lines!(ax2, bench.x, bench.U[:, 1];
        label = "t = 0",
        linewidth = 2,
        color = :gray,
        linestyle = :dot)
    axislegend(ax2; position = :rt)

    out = joinpath(ensure_example_output_dir(), "KdV.png")
    save(out, fig)
    @info "Figure saved" path=out
    display(fig)
end
