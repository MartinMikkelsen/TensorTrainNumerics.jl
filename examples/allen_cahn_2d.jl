# 2D Allen-Cahn equation with a circular interface initial condition.
#
# Solves:   ∂_t u = ε²·Δu + u - u³   on [0,1]²
#
# Initial condition: u₀(x,y) = tanh((r - R₀) / (ε√2)).
# The nonlinear coefficient u² is rebuilt with InterpolativeQTT each SCF step.

using CairoMakie

include("nonlinear_benchmark_utils.jl")

d    = 7      # bits per spatial dimension, N = 128 grid points per axis
R0   = 0.3
ε    = 0.05
T    = 0.5
Nt   = 50
rmax = 30

bench = allen_cahn_2d_benchmark(;
    d = d,
    R0 = R0,
    ε = ε,
    T_end = T,
    Nt = Nt,
    max_bond = rmax,
    projection_degree = 12,
    projection_tolerance = 1.0e-10,
    verbose_steps = true,
)

@info "Allen-Cahn 2D benchmark" method=bench.method_label R_asymptotic=round(bench.metrics.R_asymptotic, digits = 4) R_measured=round(bench.metrics.R_measured, digits = 4) R_error=round(bench.metrics.R_error, sigdigits = 3) max_rank=bench.metrics.max_rank energy_initial=round(bench.metrics.energy_initial, sigdigits = 5) energy_final=round(bench.metrics.energy_final, sigdigits = 5)

set_theme!(Theme(fontsize = 13))
fig = Figure(size = (1080, 760))

x = bench.x
initial_grid = permutedims(reshape(real.(qtt_to_function(bench.solution[1])), bench.N, bench.N))
ranks = [maximum(u.ttv_rks) for u in bench.solution]

ax_u0 = Axis(fig[1, 1]; title = "u₀(x,y)", xlabel = "x", ylabel = "y", aspect = DataAspect())
hm0 = heatmap!(ax_u0, x, x, initial_grid; colormap = :RdBu, colorrange = (-1, 1))
Colorbar(fig[2, 1], hm0; label = "u", vertical = false)

ax_uT = Axis(fig[1, 2]; title = "u(x,y,T)", xlabel = "x", ylabel = "y", aspect = DataAspect())
hmT = heatmap!(ax_uT, x, x, bench.grid; colormap = :RdBu, colorrange = (-1, 1))
Colorbar(fig[2, 2], hmT; label = "u", vertical = false)

ax_r = Axis(fig[3, 1:2];
    title = "Max bond dimension vs time step",
    xlabel = "time step",
    ylabel = "max rank")
lines!(ax_r, 0:Nt, ranks; color = :firebrick, linewidth = 2, label = bench.method_label)
scatter!(ax_r, 0:Nt, ranks; color = :firebrick, markersize = 6)
axislegend(ax_r; position = :lt)

out = joinpath(ensure_example_output_dir(), "allen_cahn_2d.png")
save(out, fig)
@info "Figure saved" path=out
display(fig)
