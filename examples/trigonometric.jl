using TensorTrainNumerics
using CairoMakie

d = 8

A1 = qtt_exp(d)
A2 = qtt_sin(d, λ = π)
A3 = qtt_cos(d, λ = π)
A4 = qtt_polynom([0.0, 2.0, 3.0, -8.0, -5.0], d; a = 0.0, b = 1.0)

qtt_values_exponential = qtt_to_function(A1)
qtt_values_sin = qtt_to_function(A2)
qtt_values_cos = qtt_to_function(A3)
qtt_values_polynom = qtt_to_function(A4)


values_exp(x) = exp(x)
values_sin(x) = sin(x * π^2)
values_cos(x) = cos(x * π^2)
values_polynom(x) = 2 * x + 3 * x^2 - 8 * x^3 - 5 * x^4

x_points = LinRange(0, 1, 2^8)
original_values_exponential = values_exp.(x_points)
original_values_sin = values_sin.(x_points)
original_values_cos = values_cos.(x_points)
original_values_polynom = values_polynom.(x_points)

let
    fig = Figure()
    ax1 = Axis(fig[2, 2], title = "Exp Approximation", xlabel = "x", ylabel = "f(x)")
    ax2 = Axis(fig[1, 1], title = "Sin Approximation", xlabel = "x", ylabel = "f(x)")
    ax3 = Axis(fig[1, 2], title = "Cos Approximation", xlabel = "x", ylabel = "f(x)")
    ax4 = Axis(fig[2, 1], title = "Polynomial Approximation", xlabel = "x", ylabel = "f(x)")


    lines!(ax1, x_points, original_values_exponential, label = "Exponential function")
    lines!(ax1, x_points, qtt_values_exponential, label = "QTT exponential function", linestyle = :dash, color = :green)

    lines!(ax2, x_points, original_values_sin, label = "Sine function")
    lines!(ax2, x_points, qtt_values_sin, label = "QTT sine function", linestyle = :dash, color = :red)

    lines!(ax3, x_points, original_values_cos, label = "Sine function")
    lines!(ax3, x_points, qtt_values_cos, label = "QTT sine function", linestyle = :dash, color = :red)

    lines!(ax4, x_points, original_values_polynom, label = "Sine function")
    lines!(ax4, x_points, qtt_values_polynom, label = "QTT sine function", linestyle = :dash, color = :red)

    fig
end


let
    fig = Figure()
    ax1 = Axis(fig[2, 2], xlabel = "x", ylabel = "f(x)")
    ax2 = Axis(fig[1, 1], xlabel = "x", ylabel = "f(x)")
    ax3 = Axis(fig[1, 2], xlabel = "x", ylabel = "f(x)")
    ax4 = Axis(fig[2, 1], xlabel = "x", ylabel = "f(x)")

    lines!(ax1, x_points, cos.(π^2 * x_points) .* sin.(π^2 * x_points))
    lines!(ax1, x_points, qtt_to_function(A2 ⊕ A3), linestyle = :dash, color = :green)

    lines!(ax2, x_points, exp.(x_points) .* sin.(π^2 * x_points))
    lines!(ax2, x_points, qtt_to_function(A1 ⊕ A2), linestyle = :dash, color = :green)

    lines!(ax3, x_points, values_polynom.(x_points) .* sin.(π^2 * x_points))
    lines!(ax3, x_points, qtt_to_function(A4 ⊕ A2), linestyle = :dash, color = :green)

    lines!(ax4, x_points, values_polynom.(x_points) .* cos.(π^2 * x_points))
    lines!(ax4, x_points, qtt_to_function(A4 ⊕ A3), linestyle = :dash, color = :green)

    fig
end
