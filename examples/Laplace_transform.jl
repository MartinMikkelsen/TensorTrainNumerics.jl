using CairoMakie
using TensorTrainNumerics
using TensorOperations
using LinearAlgebra

Smax = 10.0
Tmax = 10.0
ω = 5.0
f = x -> sin(ω * x)
num_cores = 10
N = 75

qtt = lagrange_rank_revealing(f, num_cores, N)

qtt_values = matricize(qtt, num_cores)

x_points = LinRange(1.0e-6, 1, 2^num_cores)
original_values = f.(x_points)

L = laplace_qtto(num_cores, N, 10.0)

A = (L * qtt)
approx_values = matricize(A, num_cores)

s_points = LinRange(1.0e-5, Smax, 2^num_cores)
analytical_laplace(s) = ω / (s^2 + ω^2)

let
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Laplace transform Approximation", xlabel = "s", ylabel = "f(x)")
    lines!(ax, s_points, analytical_laplace.(s_points), label = "Original Function")
    lines!(ax, s_points, approx_values, label = "QTT", linestyle = :dash, color = :red)
    axislegend(ax)
    fig
end
