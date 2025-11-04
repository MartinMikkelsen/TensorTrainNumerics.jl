using TensorTrainNumerics
using OptimKit
using CairoMakie
import TensorTrainNumerics: dot

d = 6
L = 1.0
T = 1.0
Nt = 1000
ν = 0.01

N = 2^d
dx = L / N
dt = T / Nt

Dx = 1 / dx * ∇(d)
Dxx = 1 / dx^2 * Δ_DN(d)

u₀ = qtt_sin(d, λ = π / 2)
x0 = (u₀)
v = u₀
max_bond = 20
"""
    residual: (u - v)/dt + 1/2*∂x(u²) + ν uₓₓ
"""
function burgers_residual(u)
    t1 = (u - v) * (1 / dt)

    nl = 0.5 * tt_compress!(Dx * ((u ⊕ u)), max_bond)

    lin = Dxx * u

    R = t1 + nl + ν * lin
    return tt_compress!(R, max_bond)
end

"""
    gradient of J = 1/2 ∫∫ |R|² dx dt
"""
function burgers_cost_grad(u)
    R = burgers_residual(u)
    J = 0.5 * dx * dt * dot(R, R)

    g = (1 / dt) * R
    g += ν * (Dxx * R)

    Dxu = Dx * u
    g += Dxu ⊕ R
    g += tt_compress!(Dx * (u ⊕ R), max_bond)

    g = (dx * dt) * g
    return J, tt_compress!(g, max_bond)
end

solver = GradientDescent(verbosity = 2, gradtol = 1.0e-6)

v = x0
for _ in 1:150
    x, _, _, _, _ = optimize(burgers_cost_grad, v, solver)
    v = tt_compress!(x, max_bond)
end

let
    vals_init = qtt_to_function(x0)
    vals_final = qtt_to_function(v)
    xes = (1:N) ./ N

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "Variational Burgers")
    lines!(ax, xes, vals_init, label = "initial", linewidth = 2)
    lines!(ax, xes, vals_final, label = "final", linewidth = 2)
    axislegend(ax)
    display(fig)
end
