using TensorTrainNumerics
using CairoMakie
using LinearAlgebra
using Logging
using Random

# Local-solver comparison for the nonlinear penalty sweep (arXiv:1802.07259, Fig. 5).
# Each site update can be driven by three inner solvers:
#   :newton — one damped-Newton step per site (ν_update = 1). Needs the interaction
#             Hessian B = Φᵀdiag(u²)Φ (cost O(χ⁶)/site), converges quadratically ⇒ few sweeps.
#   :cg     — ν_local line-searched nonlinear-CG (Polak–Ribière+) steps per site. No B
#             (cost O(χ⁵)/site), converges linearly ⇒ more sweeps.
#   :sd     — ν_local steepest-descent steps per site. Cheapest per step, slowest to converge.
# The paper's point: Newton wins on sweep count; the trade-off is per-sweep cost. All three
# descend toward the same ground state. We reproduce that on the 1D Gross–Pitaevskii box, g=100.
#
# NOTE on the seed: the fixed-grid sweep holds the TT rank fixed at the seed's, so the seed
# must carry enough bond dimension to represent the interacting ground state (χ≈6 here, Fig 8).
# A rank-2 sin seed would strand all three solvers at the same rank-starved wrong minimum —
# the paper's "without MGR" failure. We seed sin(πx) enriched to rank 8 with a tiny random pad.

Random.seed!(1)
L = 6
g = 100.0
g_eff = g * 2.0^L            # discrete interaction coefficient
A = (4.0^L / 2) * Δ(L)       # (1/(2h²))·tridiag(−1,2,−1), h = 2⁻ᴸ

seed = function_to_qtt(x -> sin(π * x), L)
pad = rand_tt(ntuple(_ -> 2, L), 8; normalise = true)
u0 = orthogonalize(seed + (1.0e-3 * norm(seed)) * pad)
u0 = (1 / norm(u0)) * u0

# Independent dense reference (implicit imaginary-time flow; the flow decreases the GPE
# functional uᵀAu + (g_eff/2)Σu⁴, so acceptance uses that, while E is the full-g readout).
function dense_gpe(L::Int; g_eff::Real, tol::Real = 1.0e-13, maxiter::Int = 100_000)
    N = 2^L
    A = SymTridiagonal(fill(4.0^L, N), fill(-(4.0^L) / 2, N - 1))
    f = [sin(π * m / (N + 1)) for m in 1:N]
    f ./= norm(f)
    E(f) = dot(f, A * f) + g_eff * sum(x -> x^4, f)
    Ê(f) = dot(f, A * f) + (g_eff / 2) * sum(x -> x^4, f)
    Êprev = Ê(f)
    τ = 1.0
    for _ in 1:maxiter
        H = SymTridiagonal(A.dv .+ g_eff .* f .^ 2, A.ev)
        fn = (I + τ * H) \ f
        fn ./= norm(fn)
        Ên = Ê(fn)
        if Ên ≤ Êprev + 1.0e-13 * abs(Êprev)
            converged = abs(Êprev - Ên) ≤ tol * abs(Ên)
            f, Êprev = fn, Ên
            converged && break
        else
            τ /= 2
            τ < 1.0e-12 && break
        end
    end
    return f, E(f)
end

_, E_ref = dense_gpe(L; g_eff = g_eff)
println("dense reference energy E = $E_ref\n")

# Energy after exactly k sweeps, per solver. The sweep driver is deterministic given the
# seed, so re-running with max_sweeps = k reproduces the state after k sweeps (tol = 0 keeps
# a stage from stopping early). A single large η isolates the local-solver rate; production
# runs use η-continuation (see the GPE example). Below ~10 sweeps the norm constraint isn't
# yet enforced, which the solver warns about — expected here, so we silence it for the scan
# (the reported energy is normalization-invariant regardless).
solvers = (:newton, :cg, :sd)
labels = Dict(:newton => "Newton (ν=1)", :cg => "nonlinear CG (ν=4)", :sd => "steepest descent (ν=4)")
Ksweeps = 30
curves = Dict{Symbol, Vector{Float64}}()
with_logger(NullLogger()) do
    for s in solvers
        Es = Float64[]
        for k in 1:Ksweeps
            alg = PenaltyALS(; local_solver = s, η_schedule = [1.0e6], tol = 0.0, max_sweeps = k, return_info = true)
            _, info = non_linear_solve(A, u0, alg; g = g_eff)
            push!(Es, info.energy)
        end
        curves[s] = Es
    end
end
for s in solvers
    hit = findfirst(E -> abs(E - E_ref) / abs(E_ref) < 1.0e-5, curves[s])
    println("$(labels[s]):  E→$(round(curves[s][end]; digits = 6)),  sweeps to ΔE/E<1e-5 = $(hit === nothing ? ">$Ksweeps" : hit)")
end

# Wall-clock for a full η-continuation solve (each solver already warmed by the loop above).
println("\nfull η-continuation solve (tol 1e-8):")
for s in solvers
    alg = PenaltyALS(; local_solver = s, tol = 1.0e-8, max_sweeps = 100, return_info = true)
    t = @elapsed ((_, info) = non_linear_solve(A, u0, alg; g = g_eff))
    println("  $(labels[s]):  $(info.sweeps) sweeps, $(round(1000t; digits = 1)) ms, ΔE/E = $(round(abs(info.energy - E_ref) / abs(E_ref); sigdigits = 3))")
end

# Convergence plot: |E_k − E_ref| vs sweep number (log scale).
fig = Figure(size = (760, 460))
ax = Axis(fig[1, 1]; xlabel = "penalty sweep", ylabel = "|E − E_ref|", yscale = log10,
    title = "Local-solver convergence, GPE box g = $(Int(g)) (L = $L, single η = 1e6)")
for s in solvers
    err = max.(abs.(curves[s] .- E_ref), 1.0e-16)
    scatterlines!(ax, 1:Ksweeps, err; label = labels[s])
end
axislegend(ax; position = :rt)
display(fig)
