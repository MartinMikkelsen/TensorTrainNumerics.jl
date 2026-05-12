# NLS ground state: SCF-ALS vs imaginary-time ALS — cold-start benchmark
#
# Equation: ( -1/(2h²) Δ + V(x) + g|ψ|² ) ψ = μ ψ
# Starting point: random rank-4 TT (no warm start from linear solver).
#
# Both methods use the same SCF outer loop (density frozen per sweep) and
# the same ALS site-sweeping structure.  The only difference is the local step:
#
#   SCF-ALS:  eigen(K_eff, 1:1)       — direct eigensolver (dτ = ∞ limit)
#   IT-ALS:   exp(-dτ·K_eff)·v       — imaginary-time propagation per site
#
# Fair measurement: both report the global NLS chemical potential
#   μ[k] = ⟨ψ_k|H_lin|ψ_k⟩ + g ⟨|ψ_k|², |ψ_k|²⟩
# after each full forward+backward sweep.
# SCF-ALS is run one sweep at a time so the same formula can be applied.

using TensorTrainNumerics
using Printf
import TensorTrainNumerics: dot

# ── Problem setup ─────────────────────────────────────────────────────────
L  = 8
N  = 2^L
h  = 1.0 / (N - 1)
κ  = 200.0
g  = 500.0
r  = 4

H_kin  = (1.0 / (2h^2)) * Δ(L)
V_trap = function_to_qtt(x -> κ * (x - 0.5)^2, L)
H_lin  = H_kin + ttv_to_diag_tto(V_trap)

function μ_nls(ψ)
    ρ = hadamard(conj(ψ), ψ)
    return real(dot(ψ, H_lin * ψ)) + g * real(dot(ρ, ρ))
end

# ── Shared random initial guess ───────────────────────────────────────────
rng_seed = 42
import Random; Random.seed!(rng_seed)
tt_rand = rand_tt(fill(2, L), r; normalise = true)
println("Random TT ranks: $(tt_rand.ttv_rks)")

# ── Convergence reference: well-converged SCF-ALS from random start ───────
println("\nComputing reference (SCF-ALS, 50 sweeps from random start)...")
μ_ref_hist, ψ_ref = nonlinear_als_eigsolve(H_lin, g, tt_rand;
                                             sweep_count = 50, verbose = false)
μ_ref = μ_nls(ψ_ref)
println("  Reference μ = $(round(μ_ref, digits=8))")

# ── JIT warm-up (exclude compilation from timing) ─────────────────────────
# Use rank-2 L=8 TT so Julia compiles TTvector{Float64,8} specialisations.
let tt_w = rand_tt(fill(2, L), 2; normalise = true)
    nonlinear_als_eigsolve(H_lin, g, tt_w; sweep_count = 1, verbose = false)
    nonlinear_tdvp_imagtime(H_lin, g, tt_w; dτ = 1.0, n_steps = 1, verbose = false)
end

# ── Helper: run SCF-ALS one sweep at a time, record global μ per sweep ────
function scf_als_sweep_by_sweep(H, g, tt0, n_sweeps)
    ψ = (1 / norm(tt0)) * orthogonalize(tt0)
    μ_hist = Float64[]
    t = @elapsed for _ in 1:n_sweeps
        _, ψ = nonlinear_als_eigsolve(H, g, ψ; sweep_count = 1, verbose = false)
        push!(μ_hist, μ_nls(ψ))
    end
    return μ_hist, ψ, t
end

# ── SCF-ALS: 30 sweeps ────────────────────────────────────────────────────
println("\n=== SCF-ALS (30 sweeps, cold start) ===")
n_als = 30
μ_als, ψ_als, t_als = scf_als_sweep_by_sweep(H_lin, g, tt_rand, n_als)
println("  μ final = $(round(μ_als[end], digits=8))")
println("  Time: $(round(t_als, digits=2)) s  ($(round(t_als/n_als*1000, digits=1)) ms/sweep)")

# ── IT-ALS: span from gradient-descent (small dτ) to direct-projection (large dτ) ──
# dτ × local_gap ≪ 1  →  gradient-descent per site (slow convergence)
# dτ × local_gap ≫ 1  →  direct projection to K_eff ground state (= SCF-ALS)
n_it   = 200
dτ_vals = [0.01, 0.5, 10.0]
μ_it   = Dict{Float64, Vector{Float64}}()
ψ_it   = Dict{Float64, Any}()
t_it   = Dict{Float64, Float64}()

for dτ in dτ_vals
    println("\n=== IT-ALS  dτ = $dτ  ($n_it sweeps, cold start) ===")
    t_it[dτ] = @elapsed begin
        hist, ψf = nonlinear_tdvp_imagtime(H_lin, g, tt_rand;
                                            dτ = dτ, n_steps = n_it, verbose = false)
        μ_it[dτ] = hist
        ψ_it[dτ] = ψf
    end
    println("  μ final = $(round(μ_it[dτ][end], digits=8))")
    println("  Time: $(round(t_it[dτ], digits=2)) s  ($(round(t_it[dτ]/n_it*1000, digits=1)) ms/sweep)")
end

# ── Convergence table ─────────────────────────────────────────────────────
println("\n=== Convergence: |μ(k) - μ_ref|  (μ_ref = $(round(μ_ref, digits=6))) ===")

it_hdrs = join([@sprintf(" | %-14s", "IT(dτ=$(dτ))") for dτ in dτ_vals])
hdr = @sprintf("  %4s | %-14s", "sweep", "SCF-ALS") * it_hdrs
println(hdr)
println("  " * "-"^(length(hdr) - 2))

iters = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200]
for k in iters
    als_col = k <= n_als ? @sprintf("  %.2e      ", abs(μ_als[k] - μ_ref)) : "  ---           "
    it_cols = join([k <= n_it ? @sprintf(" |  %.2e      ", abs(μ_it[dτ][k] - μ_ref)) : " |  ---           " for dτ in dτ_vals])
    println(@sprintf("  %4d |", k) * als_col * it_cols)
end

# ── Steps to tolerance ────────────────────────────────────────────────────
for tol in [1e-2, 1e-4, 1e-6]
    println("\n=== Sweeps to |μ - μ_ref| < $(tol) ===")
    k = findfirst(k -> abs(μ_als[k] - μ_ref) < tol, 1:n_als)
    k === nothing || println("  SCF-ALS:       $k sweeps  ($(round(t_als*k/n_als, digits=3)) s)")
    for dτ in dτ_vals
        ki = findfirst(k -> abs(μ_it[dτ][k] - μ_ref) < tol, 1:n_it)
        if ki !== nothing
            println("  IT-ALS dτ=$dτ:  $ki sweeps  ($(round(t_it[dτ]*ki/n_it, digits=3)) s)")
        else
            println("  IT-ALS dτ=$dτ:  not converged in $n_it sweeps")
        end
    end
end

# ── State agreement ───────────────────────────────────────────────────────
println("\n=== State agreement with reference ===")
println("  |⟨ψ_ALS,  ψ_ref⟩|² = $(round(abs(dot(ψ_als,  ψ_ref))^2, digits=6))")
for dτ in dτ_vals
    println("  |⟨ψ_IT($(dτ)), ψ_ref⟩|² = $(round(abs(dot(ψ_it[dτ], ψ_ref))^2, digits=6))")
end

# ── Per-sweep timing (after JIT) ──────────────────────────────────────────
println("\n=== Per-sweep timing (post-JIT) ===")
println("  SCF-ALS:          $(round(t_als/n_als*1000, digits=1)) ms/sweep")
for dτ in dτ_vals
    println("  IT-ALS dτ=$dτ:  $(round(t_it[dτ]/n_it*1000, digits=1)) ms/sweep")
end
