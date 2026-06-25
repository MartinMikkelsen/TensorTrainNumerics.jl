using TensorTrainNumerics
using CairoMakie
import LinearAlgebra as LA
using Random

# Quantum ground state of a double-well potential by imaginary-time propagation,
# in QTT — solved two ways and cross-checked against dense diagonalisation.
#
# Hamiltonian  H = -½ ∂²/∂x² + V(x),  V(x) = λ(x²-a²)²  (wells at ±a).
# Imaginary time:  ∂ψ/∂τ = -H ψ  → renormalised relaxation projects onto the
# ground state,  E₀ = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩  (variational, converges from above).
#
#   • Crank–Nicolson + ALS:  normalize=true is the standard normalised
#     imaginary-time propagation. ALS is *fixed-rank*, so the initial state is
#     rank-enriched (increase_ranks) up front to give it room.
#   • TDVP2:  two-site, *rank-adaptive* — grows the bond dimension itself, so a
#     plain low-rank initial state suffices. Same convention as the steppers
#     (evolves exp(+Aτ)), so pass the same A = -H with imaginary_time=true.
#
# (Potential built with function_to_qtt; for large d build it with
# InterpolativeQTT.jl and convert via to_ttvector.)

# --- parameters --------------------------------------------------------------
λ = 0.2; xa = 2.0                         # double well  V = λ(x²-a²)²
d = 8; N = 2^d; a, b = -5.0, 5.0
h = (b - a) / (N - 1); xes = collect(range(a, b, N))

# --- Hamiltonian  H = -½ ∂xx + V ---------------------------------------------
∂xx = -(1 / h^2) * Δ(d)                                       # = d²/dx²
Vfun(x) = λ * (x^2 - xa^2)^2
Vop = ttv_to_diag_tto(function_to_qtt(t -> Vfun(a + (b - a) * t), d))
H = -0.5 * ∂xx + Vop
A = (-1.0) * H                                                # both methods evolve ∂ψ/∂τ = A ψ = -H ψ

# --- dense reference: ground state of the same discrete H --------------------
F = LA.eigen(LA.Symmetric(qtto_to_matrix(H)), 1:1)
E0_dense = F.values[1]
ψ0_dense = F.vectors[:, 1]; ψ0_dense ./= sqrt(sum(abs2, ψ0_dense) * h)

nrm(v) = sqrt(real(dot(v, v)))
Energy(ψ) = real(dot(ψ, H * ψ)) / real(dot(ψ, ψ))
gauss() = (g = function_to_qtt(t -> exp(-0.5 * (a + (b - a) * t)^2), d); (1 / nrm(g)) * g)

τstep = 0.02; record_dt = 0.5; T = 8.0
blk = round(Int, record_dt / τstep); nblk = round(Int, T / record_dt)
times = collect(0.0:record_dt:T)

# --- method 1: Crank–Nicolson + ALS (fixed rank → rank-enrich the IC) --------
Random.seed!(42)                                              # reproducible enrichment noise
ψ_cn = TensorTrainNumerics.increase_ranks(gauss(), 12; noise = 1.0e-3); ψ_cn = (1 / nrm(ψ_cn)) * ψ_cn
E_cn = Float64[Energy(ψ_cn)]
for _ in 1:nblk
    global ψ_cn = crank_nicholson_method(A, ψ_cn, ψ_cn, fill(τstep, blk); normalize = true, tt_solver = "als")
    push!(E_cn, Energy(ψ_cn))
end

# --- method 2: TDVP2 (two-site, rank-adaptive → plain low-rank IC) -----------
ψ_td = gauss()
E_td = Float64[Energy(ψ_td)]
for _ in 1:nblk
    global ψ_td = tdvp2(
        A, ψ_td, fill(τstep, blk); imaginary_time = true, normalize = true,
        max_bond = 24, truncerr = 1.0e-12
    )
    push!(E_td, Energy(ψ_td))
end

ψ0 = qtt_to_function(ψ_cn); ψ0 ./= sqrt(sum(abs2, ψ0) * h)   # ground-state wavefunction on the grid
@info "ground state" E0_dense = E0_dense E0_CN = E_cn[end] E0_TDVP2 = E_td[end] overlap_CN = abs(sum(ψ0 .* ψ0_dense) * h) rank_CN = maximum(ψ_cn.ttv_rks) rank_TDVP2 = maximum(ψ_td.ttv_rks)

# --- Figure 1: potential, ground-state energy and density --------------------
let
    sc = 3.0
    fig = Figure(size = (760, 480))
    ax = Axis(
        fig[1, 1], xlabel = "x", ylabel = "energy  /  V(x)",
        title = "Double-well Schrödinger ground state  (λ=$λ, a=$xa)"
    )
    lines!(ax, xes, Vfun.(xes), color = :black, linewidth = 2, label = "V(x)")
    hlines!(ax, [E_cn[end]], color = :red, linestyle = :dash, label = "E₀ = $(round(E_cn[end]; digits = 4))")
    band!(ax, xes, fill(E_cn[end], N), E_cn[end] .+ sc .* ψ0 .^ 2, color = (:dodgerblue, 0.35))
    lines!(ax, xes, E_cn[end] .+ sc .* ψ0 .^ 2, color = :dodgerblue, linewidth = 2, label = "E₀ + $(sc)·|ψ₀|²")
    xlims!(ax, -4.5, 4.5); ylims!(ax, -0.2, 4.5)
    axislegend(ax; position = :ct)
    display(fig)
end

# --- Figure 2: convergence of both methods to the dense ground state --------
let
    fig = Figure(size = (760, 480))
    ax = Axis(
        fig[1, 1], xlabel = "imaginary time τ", ylabel = "E(τ) − E₀(dense)",
        yscale = log10, title = "Imaginary-time relaxation: CN+ALS vs TDVP2"
    )
    lines!(ax, times, abs.(E_cn .- E0_dense) .+ 1.0e-16, linewidth = 2.5, label = "Crank–Nicolson + ALS")
    lines!(ax, times, abs.(E_td .- E0_dense) .+ 1.0e-16, linewidth = 2.5, label = "TDVP2 (rank-adaptive)", linestyle = :dash)
    axislegend(ax; position = :rt)
    display(fig)
end
