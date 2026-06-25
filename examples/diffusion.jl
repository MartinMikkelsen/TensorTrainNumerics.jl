using TensorTrainNumerics
using CairoMakie

d = 8          # 2^8 = 256 interior points per dimension
N = 2^d
h = 1.0 / (N + 1)
xes = h .* (1:N)

# 2D Poisson problem: −∇²u = f on (0,1)², zero Dirichlet BCs.
# Exact solution: u(x,y) = sin(πx) sin(πy),  f = 2π² sin(πx) sin(πy).

# Discrete −∇² = −(1/h²)(Δ1d ⊗ I + I ⊗ Δ1d),  Δ1d = tridiag(−2,1,1)
Δ1d = toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
A_raw = -(1 / h^2) * (Δ1d ⊗ id_tto(d) + id_tto(d) ⊗ Δ1d)
A = QTToperator(A_raw, 2, d, :serial)

# RHS: 2π² sin(πxᵢ) sin(πyⱼ) at interior points xᵢ = i·h
b_raw = 2π^2 * qtt_sin(d; a = h, b = 1 - h) ⊗ qtt_sin(d; a = h, b = 1 - h)
b = QTTvector(b_raw, 2, d, :serial)

x0 = QTTvector(rand_tt(b_raw.ttv_dims, b_raw.ttv_rks), 2, d, :serial)
x_sol = dmrg_linsolve(A, b, x0; sweep_count = 20, tol = 1.0e-10)

sol = qttv_to_array(x_sol)
u_exact = [sin(π * xi) * sin(π * yi) for xi in xes, yi in xes]

let
    fig = Figure(size = (1200, 380))
    cmap = :roma
    ax1 = Axis(fig[1, 1], title = "DMRG solution", xlabel = "x", ylabel = "y")
    ax2 = Axis(fig[1, 2], title = "Exact solution", xlabel = "x", ylabel = "y")
    ax3 = Axis(fig[1, 3], title = "|error|", xlabel = "x", ylabel = "y")
    hm1 = heatmap!(ax1, xes, xes, sol; colormap = cmap)
    hm2 = heatmap!(ax2, xes, xes, u_exact; colormap = cmap)
    hm3 = heatmap!(ax3, xes, xes, abs.(sol .- u_exact); colormap = :viridis)
    Colorbar(fig[1, 4], hm1, label = "u(x, y)")
    display(fig)
end
