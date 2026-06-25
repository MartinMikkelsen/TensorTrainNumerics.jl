using CairoMakie
using LinearAlgebra
using TensorTrainNumerics

ω₀ = 1.5
C = 2.0
ν = 1.7

x₀ = 5.0
v₀ = 0.0
u₀_dense = [x₀, v₀, 0.0, 1.0]

A_dense = [
    0.0 1.0 0.0 0.0
    -(ω₀^2) 0.0 C 0.0
    0.0 0.0 0.0 ν
    0.0 0.0 -ν 0.0
]

function rk4_step(A, u, Δt)
    k₁ = A * u
    k₂ = A * (u + (Δt / 2) * k₁)
    k₃ = A * (u + (Δt / 2) * k₂)
    k₄ = A * (u + Δt * k₃)
    return u + (Δt / 6) * (k₁ + 2k₂ + 2k₃ + k₄)
end

function solve_forced_oscillator(A_dense, A, u₀_dense, Δt, T, max_bond)
    t = collect(0.0:Δt:T)
    u_dense = copy(u₀_dense)
    ψ = ttv_decomp(reshape(u₀_dense, 2, 2))
    x_dense = [u_dense[1]]
    x_tt = [u₀_dense[1]]

    for _ in 2:length(t)
        u_dense = rk4_step(A_dense, u_dense, Δt)

        ψ = rk4_method(A, ψ, [Δt], max_bond; normalize = false)
        push!(x_dense, u_dense[1])
        push!(x_tt, vec(ttv_to_tensor(ψ))[1])
    end

    return t, x_dense, x_tt
end

T = 50.0
Δt = 0.15
max_bond = 15

A = tto_decomp(reshape(A_dense, 2, 2, 2, 2))
t, x_dense, x_tt = solve_forced_oscillator(A_dense, A, u₀_dense, Δt, T, max_bond)

@info rel_error = norm(x_tt - x_dense) / norm(x_dense)

fig = Figure(size = (900, 520))
ax = Axis(fig[1, 1], xlabel = "t", ylabel = "x(t)", title = "Forced harmonic oscillator")
lines!(ax, t, x_dense, label = "Dense RK4", linewidth = 3)
scatter!(ax, t, x_tt, label = "TT RK4", markersize = 5, color = :tomato)
axislegend(ax)

display(fig)
