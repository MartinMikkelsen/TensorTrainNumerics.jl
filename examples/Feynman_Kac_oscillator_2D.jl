using TensorTrainNumerics
using CairoMakie
import LinearAlgebra as LA
using Random

# Feynman–Kac for the COUPLED 2D quantum harmonic oscillator (imaginary time), QTT.
#
# Brownian L = ½∇² and a coupled harmonic potential V = ½ zᵀK z (z=(x,y)):
#   ∂u/∂τ = ½∇²u − ½ zᵀK z · u = −H u ,   H = −½∇² + ½ zᵀK z ,   K = [a c; c b].
# By Feynman–Kac, u(x,y,τ) = E[ exp(−½∫₀^τ X_sᵀK X_s ds) g(X_τ) | X₀ ].
#
# Diagonalising K = R diag(λ₁,λ₂) Rᵀ gives normal modes with frequencies
# Ω_i = √λ_i. From an ISOTROPIC Gaussian payoff the matrix Riccati decouples in
# K's eigenbasis, so everything is exact:
#   • energy ⟨u|H|u⟩/⟨u|u⟩ → E₀ = ½(Ω₁+Ω₂)  (and = Σ_i[β_i/4 + λ_i/4β_i], β_i the
#     1D Riccati for frequency Ω_i),
#   • the ground state is a *correlated* Gaussian with covariance ½(√K)⁻¹.
# The coupling makes the ground state non-separable (rank > 1 across x|y), so the
# product IC is rank-enriched. Marched with Crank–Nicholson + ALS, normalize=false.

# --- model: coupled harmonic potential  V = ½(a x² + b y² + 2c xy) -----------
a_, b_, c_ = 1.0, 2.0, 0.8                 # K = [a c; c b] (positive definite)
α = 2.0                                     # isotropic initial Gaussian width
d = 7; N = 2^d; lo, hi = -5.0, 5.0
h = (hi - lo) / (N - 1); xes = collect(range(lo, hi, N))

# --- operator  A = ½∇² − V  (= −H), serial 2D layout (sites 1:d = x, d+1:2d = y)
∂xx = -(1 / h^2) * Δ(d)                                       # = d²/dx²
idd = id_tto(d)
X2 = ttv_to_diag_tto(qtt_polynom([0.0, 0.0, 1.0], d; a = lo, b = hi))   # diag(x²)
X1 = ttv_to_diag_tto(qtt_polynom([0.0, 1.0], d; a = lo, b = hi))        # diag(x)
A = 0.5 * (∂xx ⊗ idd + idd ⊗ ∂xx) -
    (0.5a_ * (X2 ⊗ idd) + 0.5b_ * (idd ⊗ X2) + c_ * (X1 ⊗ X1))
H = (-1.0) * A                                               # H_HO, for the energy

# --- analytic: normal modes, ground-state energy and covariance --------------
K = [a_ c_; c_ b_]; ev = LA.eigen(LA.Symmetric(K)); λ = ev.values; Ω = sqrt.(λ)
E0 = 0.5 * sum(Ω)
covGS = 0.5 * ev.vectors * LA.diagm(1 ./ Ω) * ev.vectors'    # ½(√K)⁻¹  (ground-state cov)
ρ∞ = covGS[1, 2] / sqrt(covGS[1, 1] * covGS[2, 2])
βi(λi, τ) = (ωi = sqrt(λi); ωi * (α + ωi * tanh(ωi * τ)) / (ωi + α * tanh(ωi * τ)))
E_riccati(τ) = sum(βi(λ[i], τ) / 4 + λ[i] / (4 * βi(λ[i], τ)) for i in 1:2)   # → E₀

# finite-time analytic density covariance/correlation from the SAME normal-mode
# Riccati: B(τ)=R diag(β_i(τ)) Rᵀ is the wavefunction precision, so the |u|²
# covariance is Σ(τ)=½B(τ)⁻¹ (→ ½(√K)⁻¹ as β_i→Ω_i). Gives ρ(τ) at every τ.
Σ_riccati(τ) = 0.5 * LA.inv(ev.vectors * LA.diagm([βi(λ[1], τ), βi(λ[2], τ)]) * ev.vectors')
ρ_riccati(τ) = (Σ = Σ_riccati(τ); Σ[1, 2] / sqrt(Σ[1, 1] * Σ[2, 2]))

toarr(v) = qttv_to_array(QTTvector(v, 2, d, :serial))
energy(u) = real(dot(u, H * u)) / real(dot(u, u))

# --- isotropic Gaussian payoff, rank-enriched so ALS can develop correlation -
gx = function_to_qtt(t -> exp(-0.5 * α * (lo + (hi - lo) * t)^2), d)
Random.seed!(42)                                              # reproducible enrichment noise
u₀ = TensorTrainNumerics.increase_ranks(gx ⊗ gx, 16; noise = 1.0e-2)

# --- Crank–Nicholson march in τ, recording density, energy, correlation ------
τstep = 0.02; record_dt = 0.2; T = 6.0      # record_dt must be a multiple of τstep
blk = round(Int, record_dt / τstep); nblk = round(Int, T / record_dt)
times = collect(0.0:record_dt:T)
dens = Vector{Matrix{Float64}}(); E_num = Float64[]; ρ_num = Float64[]

function record!(u)
    P = abs2.(toarr(u)); P ./= sum(P) * h^2           # quantum density |u|²
    push!(dens, P)
    push!(E_num, energy(u))
    mx = sum(xes .* vec(sum(P, dims = 2))) * h^2
    my = sum(xes .* vec(sum(P, dims = 1))) * h^2
    vx = sum((xes .- mx) .^ 2 .* vec(sum(P, dims = 2))) * h^2
    vy = sum((xes .- my) .^ 2 .* vec(sum(P, dims = 1))) * h^2
    cov = sum((xes .- mx) .* P .* (xes .- my)') * h^2
    return push!(ρ_num, cov / sqrt(vx * vy))
end

u = u₀; record!(u)
for _ in 1:nblk
    global u = crank_nicholson_method(A, u, u, fill(τstep, blk); normalize = false, tt_solver = "als")
    record!(u)
end

@info "FK 2D coupled HO" E_final = E_num[end] E0 = E0 ρ_final = ρ_num[end] ρ_analytic = ρ∞ rank = maximum(u.ttv_rks)

# 1σ ellipse of a covariance matrix Σ centred at the origin
function cov_ellipse(Σ; n = 120)
    vals, vecs = LA.eigen(LA.Symmetric(Σ))
    pts = [vecs * (sqrt.(vals) .* [cos(t), sin(t)]) for t in range(0, 2π, n)]
    return getindex.(pts, 1), getindex.(pts, 2)
end

# --- Figure 1: density relaxing from isotropic to the correlated ground state -
let
    snap = [0.0, 0.2, 0.6, 1.2, 6.0]
    cmax = maximum(dens[end])
    ex, ey = cov_ellipse(covGS)
    fig = Figure(size = (1200, 300))
    for (k, τ) in enumerate(snap)
        ax = Axis(fig[1, k], aspect = 1, xlabel = "x", ylabel = k == 1 ? "y" : "", title = "τ = $τ")
        heatmap!(ax, xes, xes, dens[round(Int, τ / record_dt) + 1], colormap = :viridis, colorrange = (0, cmax))
        lines!(ax, ex, ey, color = :red, linewidth = 1.5)      # analytic ground-state 1σ ellipse
        xlims!(ax, -3, 3); ylims!(ax, -3, 3)
    end
    Colorbar(fig[1, length(snap) + 1], limits = (0, cmax), colormap = :viridis, label = "|u|²")
    display(fig)
end

# --- Figure 2: energy → E₀ and correlation developing -----------------------
let
    fig = Figure(size = (1000, 420))
    ax1 = Axis(
        fig[1, 1], xlabel = "τ", ylabel = "energy ⟨u|H|u⟩/⟨u|u⟩",
        title = "Relaxation to the coupled ground state"
    )
    lines!(ax1, times, E_num, linewidth = 2.5, label = "numerical")
    lines!(ax1, times, E_riccati.(times), color = :black, linestyle = :dash, label = "normal-mode Riccati")
    hlines!(ax1, [E0], color = :gray, linestyle = :dot, label = "E₀ = ½(Ω₁+Ω₂)")
    axislegend(ax1; position = :rt)

    ax2 = Axis(
        fig[1, 2], xlabel = "τ", ylabel = "correlation ρ(τ)",
        title = "Correlation developing from the coupling"
    )
    lines!(ax2, times, ρ_num, linewidth = 2.5, label = "numerical")
    lines!(ax2, times, ρ_riccati.(times), color = :black, linestyle = :dash, label = "normal-mode Riccati")
    hlines!(ax2, [ρ∞], color = :gray, linestyle = :dot, label = "ρ∞ = ρ of ½(√K)⁻¹")
    axislegend(ax2; position = :rt)
    display(fig)
end
