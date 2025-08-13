using TensorTrainNumerics
using CairoMakie
using KrylovKit

d = 8
N = 2^d
h = 1 / (N-1)
A = h^2 * toeplitz_to_qtto(-2, 1.0, 1.0, d)
xes = collect(range(0.0, 1.0, 2^d))

u₀ = qtt_sin(d, λ = π)
init = rand_tt(u₀.ttv_dims, u₀.ttv_rks)

dt = 1e-2
nsteps = 1000
steps = fill(dt, nsteps)


solution_tdvp1 = tdvp_method(A, u₀, steps; imaginary_time = true, sweeps_per_step=2, verbose = true)
solution_tdvp2 = tdvp2_method(A, u₀, steps; imaginary_time = true, sweeps_per_step=2, verbose = true )

let
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "Comparison of Time-Stepping Methods")
    lines!(ax, xes, qtt_to_function(solution_tdvp), label = "tdvp", linestyle = :dash, linewidth = 3)
    lines!(ax, xes, qtt_to_function(solution_tdvp2), label = "tdvp2", linestyle = :dash, linewidth = 3)
    axislegend(ax)
    display(fig)
end