# 1D Nonlinear Schrödinger / Gross-Pitaevskii ground state
#
# Equation: ( -1/(2h²) Δ + V(x) + g|ψ|² ) ψ = μ ψ
# Domain:   x ∈ [0, 1] with Dirichlet BCs
# Trap:     V(x) = κ(x - 0.5)²  (harmonic, centered at 0.5)
# Norm:     ‖ψ‖² = 1  (discrete l² norm; physical integral norm = h·‖ψ‖²)
#
# Solvers:
#   nonlinear_als_eigsolve  — SCF-ALS   (single-site, fixed-rank)
#   nonlinear_mals_eigsolve — SCF-MALS  (two-site, rank-adaptive)
# At each outer sweep the density |ψ|² is frozen, giving a linearized
# eigenvalue problem solved by ALS or MALS micro-steps.

using TensorTrainNumerics

# ── Grid ──────────────────────────────────────────────────────────────────
L = 8             # 2^L = 256 grid points
N = 2^L
h = 1.0 / (N - 1)

# ── Physical parameters ────────────────────────────────────────────────────
κ = 200.0         # trap curvature
g = 500.0          # nonlinear coupling (repulsive)

# ── Linear Hamiltonian ─────────────────────────────────────────────────────
#  H_kin encodes -1/2 ∂²  via the QTT finite-difference Laplacian Δ(L)
#  (Dirichlet-Dirichlet, tridiagonal 2/-1 stencil, scaled by 1/(2h²))
H_kin  = (1.0 / (2h^2)) * Δ(L)
V_trap = function_to_qtt(x -> κ * (x - 0.5)^2, L)
H_lin  = H_kin + ttv_to_diag_tto(V_trap)

println("H_lin TTO ranks: ", H_lin.tto_rks)

function run_nls_mals(H, g, ψ0; n_sweeps = 12, tol = 1.0e-10, rmax = 8, verbose = false)
    return nonlinear_mals_eigsolve(
        H, g, ψ0;
        tol = tol,
        sweep_schedule = [n_sweeps + 1],
        rmax_schedule = [rmax],
        verbose = verbose
    )
end

# ── Initial guess: random rank-4 TT ──────────────────────────────────────
r = 4
tt0 = rand_tt(fill(2, L), r; normalise = true)

# ── Linear benchmark (g = 0, standard ALS) ───────────────────────────────
println("\n=== Linear ALS  (g = 0) ===")
E_lin, ψ_lin = als_eigsolve(H_lin, tt0; sweep_schedule = [10])
μ_lin = E_lin[end]
println("  μ_linear = $μ_lin")

# ── Nonlinear SCF-ALS ─────────────────────────────────────────────────────
println("\n=== Nonlinear SCF-ALS  (g = $g) ===")
μ_hist_als, ψ_als = nonlinear_als_eigsolve(H_lin, g, ψ_lin; sweep_count = 12, verbose = true)
μ_nl_als = μ_hist_als[end]

# ── Nonlinear SCF-MALS ────────────────────────────────────────────────────
println("\n=== Nonlinear SCF-MALS  (g = $g) ===")
μ_hist_mals, ψ_mals, r_hist_mals = run_nls_mals(H_lin, g, ψ_lin; n_sweeps = 12, rmax = 8, verbose = true)
μ_nl_mals = μ_hist_mals[end]

# ── Diagnostics ──────────────────────────────────────────────────────────
println("\n=== Diagnostics ===")
println("  μ_linear    = $(round(μ_lin, digits=6))")
println("  μ_ALS        = $(round(μ_nl_als,  digits=6))")
println("  μ_MALS       = $(round(μ_nl_mals, digits=6))")
println("  Δμ_ALS  = μ_ALS  - μ_lin = $(round(μ_nl_als  - μ_lin, digits=6))")
println("  Δμ_MALS = μ_MALS - μ_lin = $(round(μ_nl_mals - μ_lin, digits=6))")
println("  |μ_ALS - μ_MALS| = $(round(abs(μ_nl_als - μ_nl_mals), digits=10))")

for (name, ψ, r_hist) in [("SCF-ALS", ψ_als, Int[]), ("SCF-MALS", ψ_mals, r_hist_mals)]
    ψ_vec = qtt_to_function(ψ)
    println("\n  [$name]")
    println("    ‖ψ‖²            = $(round(sum(abs2, ψ_vec), digits=8))  (should be 1)")
    println("    ‖ψ‖² · h        = $(round(sum(abs2, ψ_vec) * h, digits=6))  (physical integral norm)")
    println("    max |ψ|         = $(round(maximum(abs, ψ_vec), digits=6))")
    println("    E_NLS           = $(round(nls_energy(ψ, H_lin, g), digits=6))")
    println("    TT ranks of ψ   = $(ψ.ttv_rks)")
    isempty(r_hist) || println("    max local rank history = $(maximum(r_hist))")
end

# ── Convergence history ───────────────────────────────────────────────────
println("\n=== μ at last 5 sweeps (should be converged) ===")
for (name, hist) in [("SCF-ALS", μ_hist_als), ("SCF-MALS", μ_hist_mals)]
    println("  [$name]")
    for μi in hist[max(1, end - 4):end]
        println("    μ = $(round(μi, digits=8))")
    end
end

# ── Coupling scan: μ vs g ─────────────────────────────────────────────────
println("\n=== μ vs g (using linear ground state as warm start) ===")
for gi in [0.0, 10.0, 50.0, 100.0, 200.0]
    if gi == 0.0
        μ_als_i = μ_lin
        μ_mals_i = μ_lin
    else
        μh_als, _ = nonlinear_als_eigsolve(H_lin, gi, ψ_lin;
                                           sweep_count = 10, verbose = false)
        μ_als_i = μh_als[end]
        μh_mals, _, _ = run_nls_mals(H_lin, gi, ψ_lin; n_sweeps = 10, rmax = 8, verbose = false)
        μ_mals_i = μh_mals[end]
    end
    @info "NLS coupling scan" g=gi mu_als=round(μ_als_i, digits = 6) mu_mals=round(μ_mals_i, digits = 6) mu_gap=round(abs(μ_als_i - μ_mals_i), sigdigits = 3)
end
