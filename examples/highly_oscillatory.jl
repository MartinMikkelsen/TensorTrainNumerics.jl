using CairoMakie
using TensorTrainNumerics
using LinearAlgebra
using InterpolativeQTT

f = x -> cos(1 / (x^3 + 0.01)) + sin(π * x)
num_cores = 10
degree = 150

# Chebyshev interpolation on [0, 1]; the multiscale variant refines towards x = 0,
# where the oscillation frequency of f is highest.
qtt = to_ttvector(interpolatesinglescale(f, 0.0, 1.0, num_cores, degree))
qtt_multiscale = to_ttvector(interpolatemultiscale(f, 0.0, 1.0, num_cores, degree, [0.0]))

qtt_values = matricize(qtt, num_cores)
qtt_values_multiscale = matricize(qtt_multiscale, num_cores)

# QTT grid points x_k = k / 2^num_cores, k = 0, …, 2^num_cores - 1
x_points = (0:(2^num_cores - 1)) ./ 2^num_cores
original_values = f.(x_points)

let
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Function Approximation", xlabel = "x", ylabel = "f(x)")

    lines!(ax, x_points, original_values, label = "Original Function")
    lines!(ax, x_points, qtt_values_multiscale, label = "QTT, multiscale", linestyle = :dash, color = :green)
    lines!(ax, x_points, qtt_values, label = "QTT, single-scale", linestyle = :dash, color = :red)

    axislegend(ax)
    fig
end

A = copy(qtt)
Q = tt_compress!(A, 10; truncerr = 1.0e-8, sweeps = 10, verbose = true)

let
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Function Approximation", xlabel = "x", ylabel = "f(x)")

    lines!(ax, x_points, qtt_to_function(qtt), label = "Original Function")
    lines!(ax, x_points, qtt_to_function(Q), label = "QTT, compressed", linestyle = :dash, color = :red)

    axislegend(ax)
    fig
end
