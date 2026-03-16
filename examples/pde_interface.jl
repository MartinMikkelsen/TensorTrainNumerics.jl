using TensorTrainNumerics
using LinearAlgebra

# ─── Example 1: 1D Poisson equation ──────────────────────────────────────────
#
# Solve  -u''(x) = f(x)  on [0, 1]  with u(0) = 0, u'(1) = 0  (DN BCs)
#
# Exact solution: u(x) = sin(πx/2),  f(x) = (π/2)² sin(πx/2)

levels = 7   # 128 grid points
g = QTTGrid(1, levels; bc = (:dn,))

A = laplacian(g)                                          # (1/h²) Δ_DN
b = source(g, x -> (π / 2)^2 * sin(π * x[1] / 2))       # RHS in QTT format

sol = solve(EllipticPDE(A, b, g), MALSLinsolve(tol = 1e-8, rmax = 20))

u     = to_array(sol)
u_ex  = sin.(π / 2 .* nodes(g, 1))
println("1D Poisson — relative L2 error: ", norm(u - u_ex) / norm(u_ex))

# ─── Example 2: 2D Poisson equation ──────────────────────────────────────────
#
# Solve  -(u_xx + u_yy) = f(x, y)  on [0,1]²  with DN BCs on both axes
#
# Exact solution: u(x,y) = sin(πx/2) sin(πy/2),  f = (π²/2) u

levels2 = 5   # 32 × 32 grid
g2 = QTTGrid(2, levels2; bc = (:dn, :dn))

A2 = laplacian(g2)
b2 = source(g2, x -> (π^2 / 2) * sin(π * x[1] / 2) * sin(π * x[2] / 2))

sol2  = solve(EllipticPDE(A2, b2, g2), MALSLinsolve(tol = 1e-8, rmax = 20))
u2    = to_array(sol2)                       # shape (32, 32)
u2_ex = sin.(π / 2 .* nodes(g2, 1)) * sin.(π / 2 .* nodes(g2, 2))'
println("2D Poisson — relative L2 error: ", norm(u2 - u2_ex) / norm(u2_ex))

D      = 10
lev    = 5
κ      = 0.1
T      = 0.5
dt     = 0.1

gD = QTTGrid(D, lev; bc = ntuple(_ -> :dn, D))
LD = diffusion_operator(gD; κ = κ)

# Rank-1 separable initial condition: u₀(x) = ∏_d sin(πx_d/2)
u0D = source(gD, x -> prod(sin.(π / 2 .* x)))

n_full = big(2)^(D * lev)
println("Grid: $D dimensions × 2^$lev points/dim  →  $(D * lev) QTT sites",
        " ($(n_full) grid points if stored densely)")

norm_u0 = norm(u0D)

t_start = time()
solD = solve(
    ParabolicPDE(LD, u0D, gD; tspan = (0.0, T), dt = dt),
    TDVP2Solver(truncerr = 1e-4, rmax = 8, imaginary_time = true, normalize = true),
)
t_elapsed = round(time() - t_start; digits = 1)

# With normalize=true the state stays unit-norm; we check max bond rank
# to confirm QTT complexity stays controlled despite 10^15 grid points.
println("Elapsed: $(t_elapsed)s")
println("$(D)D heat (t=$T) — max TT bond rank: ", maximum(solD.u.ttv_rks),
        "  (low rank confirms QTT structure is maintained over 10^15-point grid)")
