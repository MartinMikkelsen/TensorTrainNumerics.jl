using TensorTrainNumerics
using CairoMakie

A1 = qtt_exp(8)
A2 = qtt_sin(8,λ=π)
qtt_values_exponential = qtt_to_function(A1)
qtt_values_sin  = qtt_to_function(A2)
values_exp(x) = exp(x)
values_sin(x) = sin(x*π^2)
x_points = LinRange(0, 1, 2^8)
original_values_exponential = values_exp.(x_points)
original_values_sin = values_sin.(x_points)

fig = Figure()
ax1 = Axis(fig[1, 2], title="Exp Approximation", xlabel="x", ylabel="f(x)")
ax2 = Axis(fig[1, 1], title="Sin Approximation", xlabel="x", ylabel="f(x)")

lines!(ax1, x_points, original_values_exponential, label="Exponential function")
lines!(ax1, x_points, qtt_values_exponential, label="QTT exponential function", linestyle=:dash, color=:green)

lines!(ax2, x_points, original_values_sin, label="Sine function")
lines!(ax2, x_points, qtt_values_sin, label="QTT sine function", linestyle=:dash, color=:red)

axislegend(ax1)
axislegend(ax2)
fig