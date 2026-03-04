# examples/manifolds.jl
#
# Ground-state search using Riemannian optimisation on the TT manifold.
#
# Requires:
#   pkg> add ManifoldsBase Manopt
#
# Two approaches are compared:
#   1. Riemannian gradient descent via Manopt.jl  (uses TTManifold geometry)
#   2. Imaginary-time TDVP                        (exact geodesic integrator)

using TensorTrainNumerics, ManifoldsBase, Manopt

# ── 1.  Problem setup ──────────────────────────────────────────────────────────
#
# QTT Laplacian −∇² on [0,1] with Neumann boundary conditions.
# N = 4 QTT levels  →  2^4 = 16 grid points; each site has physical dim d = 2.
# The ground state of −∇² is the constant function (eigenvalue 0).

N  = 4
H  = -Δ_NN(N)                                 # TToperator:  −∇² ≥ 0
M  = tt_manifold(ntuple(_ -> 2, N), 4)         # manifold with max bond 4

ψ₀ = rand_tt(ntuple(_ -> 2, N), [1, 2, 2, 2, 1])
ψ₀ = ψ₀ / norm(ψ₀)                            # normalise initial state

# ── 2.  Cost function and Riemannian gradient ──────────────────────────────────
#
# Cost: Rayleigh quotient  f(ψ) = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
#
# Euclidean gradient of f:
#   ∇_E f(ψ) = 2/⟨ψ|ψ⟩ · ( H|ψ⟩ − f(ψ)|ψ⟩ )
#
# Riemannian gradient  =  project onto T_ψ M via the TDVP projector.

cost(_, ψ) = real(dot(ψ, H * ψ)) / real(dot(ψ, ψ))

function riemannian_grad(M, ψ)
    n2 = real(dot(ψ, ψ))
    λ  = real(dot(ψ, H * ψ)) / n2          # current Rayleigh quotient
    g  = (2 / n2) * (H * ψ - λ * ψ)        # Euclidean gradient
    return project(M, ψ, g)                  # project onto T_ψ M
end

# ── 3.  Riemannian gradient descent (Manopt.jl) ────────────────────────────────
#
# Manopt calls:
#   retract!(M, q, p, X, t)  →  p + t·X, then SVD-compress to max_rank
#   inner(M, p, X, X)        →  real(dot(X, X))
#   project!(M, Y, p, grad)  →  TDVP tangent-space projector

result = gradient_descent(
    M, cost, riemannian_grad, ψ₀;
    stopping_criterion = StopAfterIteration(200) | StopWhenGradientNormLess(1e-8),
    stepsize           = ArmijoLinesearch(M),
)

println("Manopt  λ_min ≈ ", cost(M, result))

# ── 4.  Imaginary-time TDVP (exact geodesic integrator) ───────────────────────
#
# Evolving  |ψ(τ)⟩ ∝ exp(−τ H)|ψ₀⟩  converges to the ground state as τ → ∞.
# TDVP is the exact geodesic on the TT manifold: it projects the Schrödinger
# equation onto T_ψ M at every infinitesimal step.

steps     = fill(0.05, 100)       # 100 imaginary-time steps of δτ = 0.05
ψ_tdvp    = tdvp(H, ψ₀, steps; imaginary_time = true, verbose = false)
ψ_tdvp    = ψ_tdvp / norm(ψ_tdvp)

println("TDVP    λ_min ≈ ", cost(M, ψ_tdvp))

# ── 5.  Relation between the two approaches ────────────────────────────────────
#
# Both methods minimise f on M:
#
#   Manopt GD:   ψ_{k+1} = retract(M, ψ_k, −α · ∇ᴿf(ψ_k))
#                          = SVD-compress( ψ_k − α · grad )
#
#   TDVP:        ψ(τ+δτ) = exp_M(ψ(τ), −δτ · P_ψ(H|ψ⟩))
#                          (exact geodesic via Krylov exponential)
#
# For gradient computation and sensitivity analysis, use ManifoldDiff.jl,
# which wraps Manopt/ManifoldsBase with AD backends.
