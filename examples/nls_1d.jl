# 1D Nonlinear Schrödinger / Gross-Pitaevskii ground state
#
# Equation: ( -1/(2h²) Δ + V(x) + g|ψ|² ) ψ = μ ψ
# Domain:   x ∈ [0, 1] with Dirichlet BCs
# Trap:     V(x) = κ(x - 0.5)²  (harmonic, centered at 0.5)
# Norm:     ‖ψ‖² = 1  (discrete l² norm; physical integral norm = h·‖ψ‖²)
#
# Solver: nonlinear_als_eigsolve (SCF-ALS)
# At each outer sweep the density |ψ|² is frozen, giving a linear
# eigenvalue problem solved site-by-site with standard ALS micro-steps.

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
μ_hist, ψ = nonlinear_als_eigsolve(H_lin, g, ψ_lin; sweep_count = 12, verbose = true)
μ_nl = μ_hist[end]

# ── Diagnostics ──────────────────────────────────────────────────────────
println("\n=== Diagnostics ===")
println("  μ_linear    = $(round(μ_lin, digits=6))")
println("  μ_nonlinear = $(round(μ_nl,  digits=6))")
println("  Δμ = μ_nl - μ_lin = $(round(μ_nl - μ_lin, digits=6))  " *
        "(positive for repulsive g > 0)")

ψ_vec = qtt_to_function(ψ)
println("  ‖ψ‖²            = $(round(sum(abs2, ψ_vec), digits=8))  (should be 1)")
println("  ‖ψ‖² · h        = $(round(sum(abs2, ψ_vec) * h, digits=6))  (physical integral norm)")
println("  max |ψ|         = $(round(maximum(abs, ψ_vec), digits=6))")
println("  E_NLS           = $(round(nls_energy(ψ, H_lin, g), digits=6))")
println("  TT ranks of ψ   = $(ψ.ttv_rks)")

# ── Convergence history ───────────────────────────────────────────────────
println("\n=== μ at last 5 sweeps (should be converged) ===")
for μi in μ_hist[end-4:end]
    println("  μ = $(round(μi, digits=8))")
end

# ── Coupling scan: μ vs g ─────────────────────────────────────────────────
println("\n=== μ vs g (using linear ground state as warm start) ===")
for gi in [0.0, 10.0, 50.0, 100.0, 200.0]
    if gi == 0.0
        μi = μ_lin
    else
        μh, _ = nonlinear_als_eigsolve(H_lin, gi, ψ_lin;
                                        sweep_count = 10, verbose = false)
        μi = μh[end]
    end
    println("  g = $(lpad(gi, 6))  →  μ = $(round(μi, digits=6))")
end
