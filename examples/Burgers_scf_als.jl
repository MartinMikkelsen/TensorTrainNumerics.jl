using TensorTrainNumerics
using CairoMakie

d    = 6
L    = 1.0
T_end = 0.15
Nt   = 150
ν    = 0.01

N   = 2^d
dx  = L / N
dt  = T_end / Nt

Dx  = (1/dx) * ∇(d)
Dxx = (1/dx^2) * Δ_DN(d)

u₀  = qtt_sin(d, λ = π/2)
xes = (1:N) ./ N

t_scf = @elapsed v_scf = burgers_scf_als(u₀, Dx, Dxx, ν, dt, Nt;
    max_scf = 5, scf_tol = 1e-8, max_bond = 20, verbose_steps = true)
println("SCF-ALS: $(round(t_scf, digits=2))s,  max_rank = $(maximum(v_scf.ttv_rks))")

let
    fig = Figure(size = (800, 420))
    ax  = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x, T)",
               title = "Burgers equation  (ν = $ν,  T = $T_end,  N = $N)")
    lines!(ax, xes, qtt_to_function(u₀),   label = "initial", linewidth = 2, color = :gray, linestyle = :dash)
    lines!(ax, xes, qtt_to_function(v_scf), label = "SCF-ALS", linewidth = 2, color = :blue)
    axislegend(ax)
    display(fig)
    save("burgers_scf_als.pdf", fig)
end
