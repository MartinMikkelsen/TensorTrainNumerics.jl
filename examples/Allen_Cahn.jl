using TensorTrainNumerics
using CairoMakie

d     = 6
L     = 1.0
T_end = 5.0
Nt    = 100
ε     = 0.05

N   = 2^d
dx  = L / (N - 1)
dt  = T_end / Nt

Dxx = (1/dx^2) * Δ_NN(d)

# Initial condition: sin(2πx) ∈ [-1, 1] — will phase-separate into ±1 plateaus
u₀  = function_to_qtt(x -> sin(2π * x), d)
xes = (0:N-1) ./ (N-1)
tes = (0:Nt) .* dt

t_mals = @elapsed sol = allen_cahn_mals(u₀, Dxx, ε, dt, Nt;
    max_scf = 5, scf_tol = 1e-8, max_bond = 20, verbose_steps = true)
println("MALS: $(round(t_mals, digits=2))s,  max_rank = $(maximum(sol[end].ttv_rks))")

# Build space × time solution matrix: columns are snapshots, rows are grid points
U = reduce(hcat, qtt_to_function(u) for u in sol)   # N × (Nt+1)

let
    fig = Figure(size = (800, 500))

    # Space-time heatmap
    ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = "t",
               title = "Allen-Cahn  (ε = $ε,  T = $T_end,  N = $N)")
    hm  = heatmap!(ax1, xes, tes, U, colormap = :RdBu, colorrange = (-1, 1))
    Colorbar(fig[1, 2], hm, label = "u")

    # Selected snapshots
    ax2 = Axis(fig[2, 1:2], xlabel = "x", ylabel = "u(x, t)")
    times = [0, div(Nt, 4), div(Nt, 2), Nt]
    colors = [:gray, :steelblue, :orange, :red]
    for (k, c) in zip(times, colors)
        lines!(ax2, xes, U[:, k+1], label = "t = $(round(k*dt, digits=2))",
               linewidth = 2, color = c)
    end
    axislegend(ax2)

    display(fig)
end
