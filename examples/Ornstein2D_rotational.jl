using TensorTrainNumerics
using CairoMakie
using LinearAlgebra

# Non-symmetric (rotational) 2D Ornstein–Uhlenbeck — a stability study.
#
# Drift matrix  Θ = [θ ω; -ω θ] = θI + ωJ,  J = [0 1; -1 0].  Eigenvalues θ±iω:
# the drift ROTATES at frequency ω while decaying at rate θ, so the mean spirals
# into μ.  Unlike the symmetric-coupling case the stationary is *isotropic*,
#   P∞ = N(μ, (D/θ)·I),
# because the antisymmetric part of Θ drops out of the Lyapunov equation.  What
# ω≠0 adds is a divergence-free *probability current* circulating around μ — the
# signature of a non-equilibrium (non-reversible) steady state.
#
# Why this is a stability test: the central-difference ∂ is EXACTLY antisymmetric
# (∂ᵀ=-∂, since shift and id-∇ are exact transposes), so the rotational coupling
# ω(∂⊗My - Mx⊗∂) is exactly antisymmetric.  Hence:
#   • it contributes nothing to (A+Aᵀ)/2, so the numerical abscissa
#     ν = λmax((A+Aᵀ)/2) — the bound on transient growth ‖e^{tA}‖ ≤ e^{tν} — is
#     INDEPENDENT of ω;
#   • it only shifts eigenvalues along the imaginary axis, so max Re λ stays ≤ 0.
# The operator becomes more non-normal as ω grows, but CN+ALS stays stable.
# Below we verify both the spectrum (small d, dense) and full evolutions.

# --- model parameters --------------------------------------------------------
θ = 1.0
μx, μy = 2.0, -2.0
σ = 1.0
D = σ^2 / 2
var∞ = D / θ                      # isotropic stationary variance

a, b = -6.0, 6.0

# rotational generator A(ω) at d bits per axis (serial layout, sites 1:d = x)
function generator(d, ω)
    N = 2^d
    h = (b - a) / (N - 1)
    ∂   = (1 / (2h)) * (shift(d) - (id_tto(d) - ∇(d)))   # central first derivative (∂ᵀ = -∂)
    ∂²  = -(1 / h^2) * Δ(d)
    idd = id_tto(d)
    Mx  = ttv_to_diag_tto(qtt_polynom([-μx, 1.0], d; a = a, b = b))
    My  = ttv_to_diag_tto(qtt_polynom([-μy, 1.0], d; a = a, b = b))
    A = θ * ((∂ * Mx) ⊗ idd + idd ⊗ (∂ * My)) +    # diagonal drift
        ω * (∂ ⊗ My - Mx ⊗ ∂) +                    # antisymmetric rotational coupling
        D * (∂² ⊗ idd + idd ⊗ ∂²)                  # isotropic diffusion
    return A, h
end

# ============================================================================
# Part 1 — spectrum sweep (small d, dense eigenvalues)
# ============================================================================
d_spec = 5
ωs = collect(0.0:0.2:2.0)
abscissa  = Float64[]   # max Re λ(A)        (spectral abscissa)
numabsc   = Float64[]   # λmax((A+Aᵀ)/2)     (numerical abscissa / log-norm)
nonnormal = Float64[]   # ‖A - Aᵀ‖           (departure from normality)
for ω in ωs
    A, _ = generator(d_spec, ω)
    M = qtto_to_matrix(A)
    push!(abscissa, maximum(real, eigvals(M)))
    push!(numabsc, maximum(real, eigvals(Symmetric((M + M') / 2))))
    push!(nonnormal, norm(M - M'))
end

# ============================================================================
# Part 2 — full CN+ALS evolutions across ω (QTT)
# ============================================================================
d = 7
N = 2^d
h = (b - a) / (N - 1)
xes = collect(range(a, b, N))
toarr(v) = qttv_to_array(QTTvector(v, 2, d, :serial))
mass(P) = sum(P) * h^2
P∞ = [exp(-((xi - μx)^2 + (yj - μy)^2) / (2var∞)) / (2π * var∞) for xi in xes, yj in xes]

# product Gaussian IC at the origin, rank-enriched so ALS can follow the rotating
# (transiently correlated) transient even though the endpoints are low-rank
gx = function_to_qtt(t -> exp(-(a + (b - a) * t)^2 / 2), d)
gy = function_to_qtt(t -> exp(-(a + (b - a) * t)^2 / 2), d)
ic() = (u = TensorTrainNumerics.tt_up_rks(gx ⊗ gy, 14; ϵ_wn = 1e-2); (1 / mass(toarr(u))) * u)

τ         = 0.02
record_dt = 0.8
T         = 8.0
blk       = round(Int, record_dt / τ)
nblk      = round(Int, T / record_dt)
times     = collect(0.0:record_dt:T)

ωs_evo = [0.0, 1.0, 2.0]
curves = Dict{Float64, Vector{Float64}}()   # L¹(t) per ω
Pfinal = Dict{Float64, Matrix{Float64}}()    # final density per ω

for ω in ωs_evo
    A, _ = generator(d, ω)
    ψ = ic()
    err = [sum(abs.((P = toarr(ψ); P ./= mass(P); P) .- P∞)) * h^2]
    for _ in 1:nblk
        ψ = crank_nicholson_method(A, ψ, ψ, fill(τ, blk); normalize = false, tt_solver = "als")
        P = toarr(ψ); P ./= mass(P)
        push!(err, sum(abs.(P .- P∞)) * h^2)
    end
    curves[ω] = err
    P = toarr(ψ); P ./= mass(P)
    Pfinal[ω] = P
    mx = sum(xes .* vec(sum(P, dims = 2))) * h^2
    vx = sum((xes .- mx) .^ 2 .* vec(sum(P, dims = 2))) * h^2
    cov = sum((xes .- mx) .* P .* (xes .- sum(xes .* vec(sum(P, dims = 1))) * h^2)') * h^2
    @info "ω = $ω" mass = mass(toarr(ψ)) mean_x = mx var = vx target_var = var∞ cov = cov L1 = err[end]
end

# steady probability current  J = -Θ(x-μ)P - D∇P  for the ω = 2 run (circulation)
let ω = 2.0, P = Pfinal[2.0]
    dx = zero(P); dy = zero(P)
    @views dx[2:(end - 1), :] .= (P[3:end, :] .- P[1:(end - 2), :]) ./ (2h)
    @views dy[:, 2:(end - 1)] .= (P[:, 3:end] .- P[:, 1:(end - 2)]) ./ (2h)
    step = N ÷ 13
    qx = Float64[]; qy = Float64[]; ju = Float64[]; jv = Float64[]
    for i in 1:step:N, j in 1:step:N
        x = xes[i]; y = xes[j]
        ax = -(θ * (x - μx) + ω * (y - μy))
        ay = -(-ω * (x - μx) + θ * (y - μy))
        push!(qx, x); push!(qy, y)
        push!(ju, ax * P[i, j] - D * dx[i, j])
        push!(jv, ay * P[i, j] - D * dy[i, j])
    end
    global current = (qx, qy, ju, jv)
end

# ============================================================================
# Figure 1 — stability of the spectrum vs ω
# ============================================================================
let
    fig = Figure(size = (1000, 420))
    ax1 = Axis(fig[1, 1], xlabel = "ω", ylabel = "abscissa",
        title = "Spectral & numerical abscissa  (dense, d=$d_spec)")
    hlines!(ax1, [0.0], color = :gray, linestyle = :dash)
    scatterlines!(ax1, ωs, abscissa, label = "max Re λ(A)  (spectral)")
    scatterlines!(ax1, ωs, numabsc, label = "λmax((A+Aᵀ)/2)  (numerical)")
    axislegend(ax1; position = :rc)
    ax2 = Axis(fig[1, 2], xlabel = "ω", ylabel = "‖A − Aᵀ‖",
        title = "Non-normality grows with ω")
    scatterlines!(ax2, ωs, nonnormal)
    display(fig)
end

# ============================================================================
# Figure 2 — evolution robustness (left) and the steady current (right)
# ============================================================================
let
    fig = Figure(size = (1050, 430))
    ax1 = Axis(fig[1, 1], xlabel = "t", ylabel = "L¹ error", yscale = log10,
        title = "CN+ALS convergence is ω-robust")
    for ω in ωs_evo
        lines!(ax1, times, curves[ω], linewidth = 2.5, label = "ω = $ω")
    end
    axislegend(ax1; position = :rt)

    ax2 = Axis(fig[1, 2], aspect = 1, xlabel = "x", ylabel = "y",
        title = "Steady probability current  (ω = 2)")
    heatmap!(ax2, xes, xes, Pfinal[2.0], colormap = :viridis)
    qx, qy, ju, jv = current
    arrows!(ax2, qx, qy, ju, jv; lengthscale = 3.0, arrowsize = 7, color = :white)
    xlims!(ax2, -1, 5); ylims!(ax2, -5, 1)
    display(fig)
end
