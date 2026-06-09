using CairoMakie

include("nonlinear_benchmark_utils.jl")

bench = allen_cahn_benchmark(;
    d = 8,
    L = 1.0,
    T_end = 5.0,
    Nt = 100,
    ε = 0.05,
    max_scf = 5,
    scf_tol = 1.0e-8,
    max_bond = 20,
    projection_degree = 25,
    projection_tolerance = 1.0e-10,
    verbose_steps = true
)

@info "Allen-Cahn benchmark" method=bench.method_label runtime_seconds=round(bench.metrics.runtime_seconds, digits = 2) max_rank=bench.metrics.max_rank min_u=round(bench.metrics.min_u, digits = 4) max_u=round(bench.metrics.max_u, digits = 4) initial_energy=round(bench.metrics.initial_energy, sigdigits = 5) final_energy=round(bench.metrics.final_energy, sigdigits = 5)

let
    fig = Figure(size = (920, 760))

    ax1 = Axis(fig[1, 1],
        xlabel = "x",
        ylabel = "t",
        title = "Allen-Cahn ($(bench.method_label), eps = $(bench.ε), T = $(bench.T_end), N = $(bench.N))")
    hm = heatmap!(ax1, bench.x, bench.t, bench.U; colormap = :RdBu, colorrange = (-1, 1))
    Colorbar(fig[1, 2], hm; label = "u")

    ax2 = Axis(fig[2, 1],
        xlabel = "x",
        ylabel = "u(x,t)",
        title = "Selected snapshots")
    times = [0, div(bench.Nt, 4), div(bench.Nt, 2), bench.Nt]
    colors = [:gray, :steelblue, :orange, :red]
    for (k, color) in zip(times, colors)
        lines!(ax2, bench.x, bench.U[:, k + 1];
            label = "t = $(round(bench.t[k + 1], digits = 2))",
            linewidth = 2,
            color = color)
    end
    axislegend(ax2; position = :rb)

    ax3 = Axis(fig[3, 1:2],
        xlabel = "t",
        ylabel = "free energy",
        title = "Discrete energy diagnostic")
    lines!(ax3, bench.t, bench.energy_history; linewidth = 2, color = :black)
    scatter!(ax3, bench.t[1:10:end], bench.energy_history[1:10:end]; markersize = 6, color = :black)

    out = joinpath(ensure_example_output_dir(), "Allen_Cahn.png")
    save(out, fig)
    @info "Figure saved" path=out
    display(fig)
end
