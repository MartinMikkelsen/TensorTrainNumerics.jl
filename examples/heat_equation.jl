using TensorTrainNumerics
using CairoMakie

d = 10
h = 1 / (2^d)
p = 1.0
s = 0.0
v = 0.0
α = h^2 * v - 2 * p
β = p + h * s / 2
γ = p - h * s / 2

Δ = toeplitz_to_qtto(α, β, γ, d)
kappa = 0.1
A = kappa * Δ

u0 = qtt_sin(d, λ = π)

dt = 1.0e-2
nsteps = 1000
steps = fill(dt, nsteps)

Lx = Δ
Ly = Δ
I = id_tto(d)
A2 = kappa * ((I ⊗ Ly) + (Lx ⊗ I))
u0 = qtt_sin(d, λ = 1 / π) ⊗ qtt_cos(d, λ = 1 / π)
sol = tdvp2(A2, u0, steps, imaginary_time = true, truncerr = 1.0e-3)

solution = reshape(qtt_to_function(sol), 2^d, 2^d)

let
    x = range(0, 1, length = 2^d)
    y = range(0, 1, length = 2^d)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "TDVP Heatmap")
    hm = heatmap!(ax, x, y, solution)
    Colorbar(fig[1, 2], hm, label = "u(x, y)")
    display(fig)
end
