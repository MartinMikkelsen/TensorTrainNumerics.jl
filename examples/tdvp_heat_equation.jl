using TensorTrainNumerics
using CairoMakie

cores = 8

x_points = collect(range(0.0, 1.0, length = 2^cores))

# Finite difference coefficients for the Laplacian
h = 1 / (2^cores)
p = 1.0
s = 0.0
v = 0.0
α = h^2 * v - 2 * p
β = p + h * s / 2
γ = p - h * s / 2

Δ = toeplitz_to_qtto(α, β, γ, cores)

# Solve the heat equation \partial_t u = Δ u with initial state cos(π x)
A = Δ
u0 = qtt_cos(cores)

final_time = 0.1
u_final = tdvp_solve(A, u0, final_time; dt = 0.01)

solution = qtt_to_function(u_final)
analytic = exp(-π^2 * final_time) .* cos.(π .* x_points)

fig = Figure()
ax = Axis(fig[1, 1], title = "Heat Equation", xlabel = "x", ylabel = "u(x,t)")
lines!(ax, x_points, analytic, label = "Analytic")
lines!(ax, x_points, solution, label = "TT Solution", linestyle = :dash, color = :red)
axislegend(ax)
fig
