using CairoMakie
using TensorTrainNumerics

f = x -> cos(1 / (x^3 + 0.01)) + sin(π * x)
num_cores = 10
N = 150

qtt = interpolating_qtt(f, num_cores, N)
qtt_rank_revealing = lagrange_rank_revealing(f, num_cores, N)

qtt_values = matricize(qtt, num_cores)
qtt_values_rank_revealing = matricize(qtt_rank_revealing, num_cores)

x_points = LinRange(0, 1, 2^num_cores)
original_values = f.(x_points)

let
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Function Approximation", xlabel = "x", ylabel = "f(x)")

    lines!(ax, x_points, original_values, label = "Original Function")
    lines!(ax, x_points, qtt_values_rank_revealing, label = "QTT, rank rev.", linestyle = :dash, color = :green)
    lines!(ax, x_points, qtt_values, label = "QTT", linestyle = :dash, color = :red)

    axislegend(ax)
    fig
end
