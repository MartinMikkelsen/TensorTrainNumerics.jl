using TensorTrainNumerics
using OptimKit
using CairoMakie
import TensorTrainNumerics: dot, orthogonalize

orth2(x) = orthogonalize(orthogonalize(x; i = 1); i = x.N)

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
x0 = orth2(u₀)
v = u₀
v_ref = Ref(u₀)

"""
    residual: (u - v)/dt + 1/2*∂x(u²) + ν uₓₓ
"""
function burgers_residual(u)
    t1 = (u - v_ref[]) * (1 / dt)
    nl = 0.5 * orth2(Dx * (u ⊕ u))
    lin = Dxx * u
    R = t1 + nl + ν * lin
    return orth2(R)
end

"""
    gradient of J = 1/2 ∫∫ |R|² dx dt
"""
function burgers_cost_grad(u)
    R = burgers_residual(u)
    J = 0.5 * dx * dt * dot(R, R)

    g = (1 / dt) * R
    g += ν * (Dxx * R)

    Dxu = orth2(Dx * u)
    g += orth2(Dxu ⊕ R)
    g += orth2(Dx * orth2(u ⊕ R))

    g = (dx * dt) * g
    return J, orth2(g)
end

solver = GradientDescent(verbosity = 2, gradtol = 1.0e-6)

v_ref[] = x0
for _ in 1:150
    x, _, _, _, _ = optimize(burgers_cost_grad, v_ref[], solver)
    v_ref[] = orth2(x)
end

vals_init = qtt_to_function(x0)
vals_final = qtt_to_function(v_ref[])
xes = (1:N) ./ N

let
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "Variational Burgers")
    lines!(ax, xes, vals_init, label = "initial", linewidth = 2)
    lines!(ax, xes, vals_final, label = "final", linewidth = 2)
    axislegend(ax)
    fig
end
