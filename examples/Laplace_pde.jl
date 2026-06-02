using TensorTrainNumerics
using CairoMakie

d = 8          # 2^d interior grid points per dimension
N = 2^d
h = 1.0 / (N + 1)          # uniform interior spacing on [0,1]
xes = h .* (1:N)            # interior grid: x_i = i/(N+1)

# Discrete 2D Laplacian on N×N interior grid (zero Dirichlet BCs at x=0,1).
# The Toeplitz stencil is correct for interior-only points because the missing
# ghost-point values (u=0 at boundary) drop out of the stencil equations.
Δ1d   = toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
A_raw = (1/h^2) * (Δ1d ⊗ id_tto(d) + id_tto(d) ⊗ Δ1d)
A     = QTToperator(A_raw, 2, d, :serial)

# BCs: u(x, 0) = sin(πx),  all other boundaries zero.
# The bottom BC contributes −sin(πx_i)/h² to the first y-row (y=h) of the RHS.
# qtt_sin with a=h, b=1−h evaluates sin(πx) at interior points x_i = i·h.
b_raw = -(1/h^2) * qtt_sin(d; a = h, b = 1 - h) ⊗ qtt_basis_vector(d, 1)
b     = QTTvector(b_raw, 2, d, :serial)

# Random initial guess
x0 = QTTvector(rand_tt(b_raw.ttv_dims, b_raw.ttv_rks), 2, d, :serial)

# Solve with MALS (single sweep) and DMRG (50 sweeps)
x_mals = mals_linsolve(A, b, x0)
x_dmrg = dmrg_linsolve(A, b, x0; sweep_count = 50, tol = 1e-12)

# Solutions on the N×N interior grid
sol_mals = qttv_to_array(x_mals)
sol_dmrg = qttv_to_array(x_dmrg)

# Exact solution: u(x,y) = sin(πx) sinh(π(1−y)) / sinh(π)
u_exact = [sin(π * xi) * sinh(π * (1 - yi)) / sinh(π) for xi in xes, yi in xes]

let
    fig  = Figure(size = (1200, 380))
    cmap = :roma
    ax1 = Axis(fig[1, 1], title = "MALS (1 sweep)",   xlabel = "x", ylabel = "y")
    ax2 = Axis(fig[1, 2], title = "DMRG (50 sweeps)", xlabel = "x", ylabel = "y")
    ax3 = Axis(fig[1, 3], title = "Exact solution",   xlabel = "x", ylabel = "y")
    ax4 = Axis(fig[1, 4], title = "|DMRG − exact|",   xlabel = "x", ylabel = "y")
    hm1 = heatmap!(ax1, xes, xes, sol_mals;                  colormap = cmap)
    hm2 = heatmap!(ax2, xes, xes, sol_dmrg;                  colormap = cmap)
    hm3 = heatmap!(ax3, xes, xes, u_exact;                   colormap = cmap)
    hm4 = heatmap!(ax4, xes, xes, abs.(sol_dmrg .- u_exact); colormap = :viridis)
    Colorbar(fig[1, 5], hm1, label = "u(x, y)")
    display(fig)
end
