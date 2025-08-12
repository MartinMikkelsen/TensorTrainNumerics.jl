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
steps = collect(range(0.0, 10.0, 1000))

solution_explicit, error_explicit = euler_method(A, u₀, steps; return_error = true)

solution_implicit, rel_implicit = implicit_euler_method(A, u₀, init, steps; return_error = true)

solution_crank, rel_crank = crank_nicholson_method(A, u₀, init, steps; return_error = true, tt_solver = "mals")

solution_krylov, rel_krylov = expintegrator(A, last(steps), u₀)

let
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "Comparison of Time-Stepping Methods")
    lines!(ax, xes, qtt_to_function(solution_explicit), label = "Explicit Euler", linestyle = :solid, linewidth = 3)
    lines!(ax, xes, qtt_to_function(solution_implicit), label = "Implicit Euler", linestyle = :dot, linewidth = 3)
    lines!(ax, xes, qtt_to_function(solution_crank), label = "Crank-Nicolson", linestyle = :dash, linewidth = 3)
    axislegend(ax)
    display(fig)
end

let
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "Comparison of Time-Stepping Methods")
    lines!(ax, xes, qtt_to_function(solution_krylov), label = "Krylov", linestyle = :solid, linewidth = 3)
    axislegend(ax)
    display(fig)
end
