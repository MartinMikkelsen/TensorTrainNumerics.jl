using TensorTrainNumerics
using CairoMakie

d = 6          # 2^6 = 64 interior points per dimension
N = 2^d
h = 1.0 / (N + 1)
xes = h .* (1:N)
κ = 0.1        # diffusion coefficient

# 2D heat equation: u_t = κ∇²u on (0,1)², zero Dirichlet BCs.
# Initial condition: u0(x,y) = sin(πx)sin(πy) — lowest Dirichlet eigenfunction.
# Exact solution:    u(x,y,T) = sin(πx)sin(πy) exp(−2κπ²T).

# Discrete −∇² scaled correctly: (1/h²) toeplitz(−2,1,1) ≈ d²/dx²
Δ1d = toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
A_raw = (κ / h^2) * (Δ1d ⊗ id_tto(d) + id_tto(d) ⊗ Δ1d)
A = QTToperator(A_raw, 2, d, :serial)

u0_raw = qtt_sin(d; a = h, b = 1 - h) ⊗ qtt_sin(d; a = h, b = 1 - h)
u0 = QTTvector(u0_raw, 2, d, :serial)

T = 1.0
dt = 0.001      # (κ/h²)*dt ≈ 422*0.001 = 0.42, small enough for the local Krylov steps.
steps = fill(dt, round(Int, T / dt))

sol_tdvp = tdvp(A, u0, steps; imaginary_time = true, normalize = false)
sol_tdvp2 = tdvp2(
    A, u0, steps; imaginary_time = true, normalize = false,
    max_bond = 12, truncerr = 1.0e-12
)

u_tdvp = qttv_to_array(sol_tdvp)
u_tdvp2 = qttv_to_array(sol_tdvp2)
u_exact = [sin(π * xi) * sin(π * yi) * exp(-2κ * π^2 * T) for xi in xes, yi in xes]
err_tdvp = abs.(u_tdvp .- u_exact)
err_tdvp2 = abs.(u_tdvp2 .- u_exact)

relerr_tdvp = norm(vec(u_tdvp .- u_exact)) / norm(vec(u_exact))
relerr_tdvp2 = norm(vec(u_tdvp2 .- u_exact)) / norm(vec(u_exact))

println("TDVP  relative error: ", relerr_tdvp)
println("TDVP2 relative error: ", relerr_tdvp2)

let
    fig = Figure(size = (1200, 720))
    cmap = :viridis
    urange = extrema(u_exact)
    erange = (0.0, max(maximum(err_tdvp), maximum(err_tdvp2)))

    ax1 = Axis(fig[1, 1], title = "TDVP one-site (T=$T)", xlabel = "x", ylabel = "y")
    ax2 = Axis(fig[1, 2], title = "TDVP two-site (T=$T)", xlabel = "x", ylabel = "y")
    ax3 = Axis(fig[1, 3], title = "Exact", xlabel = "x", ylabel = "y")
    ax4 = Axis(fig[2, 1], title = "|TDVP - exact|", xlabel = "x", ylabel = "y")
    ax5 = Axis(fig[2, 2], title = "|TDVP2 - exact|", xlabel = "x", ylabel = "y")

    hm1 = heatmap!(ax1, xes, xes, u_tdvp; colormap = cmap, colorrange = urange)
    hm2 = heatmap!(ax2, xes, xes, u_tdvp2; colormap = cmap, colorrange = urange)
    hm3 = heatmap!(ax3, xes, xes, u_exact; colormap = cmap, colorrange = urange)
    hm4 = heatmap!(ax4, xes, xes, err_tdvp; colormap = :magma, colorrange = erange)
    hm5 = heatmap!(ax5, xes, xes, err_tdvp2; colormap = :magma, colorrange = erange)

    Colorbar(fig[1, 4], hm1, label = "u(x, y)")
    Colorbar(fig[2, 4], hm4, label = "|error|")
    display(fig)
end
