using TensorTrainNumerics
using CairoMakie
using Random

# Vasicek model: zero-coupon bond pricing as a Feynman–Kac (discounted backward)
# equation, in QTT.
#
# Short rate follows OU:  dr = θ(μ-r) dt + σ dW.  The zero-coupon bond price
#   P(r,τ) = E[ exp(-∫₀^τ r_s ds) | r₀ = r ]      (τ = time to maturity, payoff 1)
# solves the discounted backward equation
#   ∂P/∂τ = L P - r·P ,    P(r,0) = 1,
# where L = -θ(r-μ)∂_r + D ∂_rr is the OU generator (D = σ²/2). Compared with the
# Kolmogorov backward example the only new ingredient is the *potential* term
# -r·P, i.e. a diagonal multiply by r:
#   L_FK = -θ (M ∂_r) + D ∂_rr - X ,   M = diag(r-μ),  X = diag(r).
#
# Vasicek has an affine closed form  P(r,τ) = exp(A(τ) - B(τ) r)  — exact validation.
# Solved with Crank–Nicholson + ALS (L_FK non-symmetric); the constant terminal
# payoff is rank-enriched so ALS can develop the exp(-B r) profile.

# --- model parameters (short rate in absolute units, e.g. 0.05 = 5%) ---------
θ = 0.5; μ = 0.05; σ = 0.03; D = σ^2 / 2

d = 8; N = 2^d; a, b = -0.1, 0.25          # rate grid (allows mildly negative rates)
h = (b - a) / (N - 1); rs = collect(range(a, b, N))

# --- discounted backward generator  L_FK = -θ M ∂_r + D ∂_rr - X -------------
∂r = (1 / (2h)) * (shift(d) - (id_tto(d) - ∇(d)))      # central first derivative
∂rr = -(1 / h^2) * Δ(d)                                 # second derivative
M = ttv_to_diag_tto(qtt_polynom([-μ, 1.0], d; a = a, b = b))   # diag(r-μ)  (drift)
X = ttv_to_diag_tto(qtt_polynom([0.0, 1.0], d; a = a, b = b))  # diag(r)    (discount)
L_FK = -θ * (M * ∂r) + D * ∂rr - X

# --- terminal payoff P(r,0)=1, rank-enriched so ALS can grow the bond profile -
Random.seed!(42)                                                  # reproducible enrichment noise
u₀ = TensorTrainNumerics.increase_ranks(function_to_qtt(t -> 1.0, d), 6; noise = 1.0e-3)

# --- Vasicek closed form  P(r,τ) = exp(A(τ) - B(τ) r) ------------------------
B(τ) = (1 - exp(-θ * τ)) / θ
A(τ) = (B(τ) - τ) * (θ^2 * μ - σ^2 / 2) / θ^2 - σ^2 * B(τ)^2 / (4θ)
Panal(r, τ) = exp(A(τ) - B(τ) * r)

# --- Crank–Nicholson march in τ (= maturity), recording the price curves ------
τstep = 0.05; record_dt = 0.5; T = 10.0
blk = round(Int, record_dt / τstep); nblk = round(Int, T / record_dt)
times = collect(0.0:record_dt:T)
interior = findall(r -> -0.02 <= r <= 0.18, rs)

Pcurves = Vector{Vector{Float64}}(); relerr = Float64[]
function record!(ψ)
    P = qtt_to_function(ψ)
    push!(Pcurves, P)
    τ = (length(Pcurves) - 1) * record_dt
    return push!(
        relerr, τ == 0 ? 0.0 :
            maximum(abs.(P[interior] .- [Panal(r, τ) for r in rs[interior]]) ./ [Panal(r, τ) for r in rs[interior]])
    )
end

ψ = u₀; record!(ψ)
for _ in 1:nblk
    global ψ = crank_nicholson_method(L_FK, ψ, ψ, fill(τstep, blk); normalize = false, tt_solver = "als")
    record!(ψ)
end

@info "Vasicek bond" rel_err_max = maximum(relerr) P_μ_10y = Pcurves[end][argmin(abs.(rs .- μ))] P_μ_10y_analytic = Panal(μ, T)

# --- Figure 1: bond price P(r,τ) vs short rate, at several maturities --------
let
    snap = [1.0, 2.0, 5.0, 10.0]
    fig = Figure(size = (760, 480))
    ax = Axis(
        fig[1, 1], xlabel = "short rate r", ylabel = "bond price  P(r, τ)",
        title = "Vasicek zero-coupon bond  (θ=$θ, μ=$μ, σ=$σ)"
    )
    for τ in snap
        lines!(ax, rs, Pcurves[round(Int, τ / record_dt) + 1], linewidth = 2, label = "τ = $(τ)y")
    end
    for τ in snap
        lines!(ax, rs, [Panal(r, τ) for r in rs], color = :black, linestyle = :dash, linewidth = 1)
    end
    lines!(ax, Float64[], Float64[], color = :black, linestyle = :dash, label = "analytic")
    xlims!(ax, -0.02, 0.18); axislegend(ax; position = :rt)
    display(fig)
end

# --- Figure 2: term structure (yields) and accuracy --------------------------
let
    fig = Figure(size = (1000, 420))
    ax1 = Axis(
        fig[1, 1], xlabel = "maturity τ (years)", ylabel = "yield  y = -ln P / τ",
        title = "Term structure from a few initial short rates"
    )
    for r0 in (0.0, 0.05, 0.1)
        i = argmin(abs.(rs .- r0))
        ynum = [-log(Pcurves[k][i]) / times[k] for k in 2:length(times)]
        yana = [-log(Panal(rs[i], times[k])) / times[k] for k in 2:length(times)]
        lines!(ax1, times[2:end], ynum, linewidth = 2.5, label = "r₀ = $(round(rs[i]; digits = 3))")
        lines!(ax1, times[2:end], yana, color = :black, linestyle = :dash, linewidth = 1)
    end
    axislegend(ax1; position = :rc)

    ax2 = Axis(
        fig[1, 2], xlabel = "maturity τ (years)", ylabel = "max interior rel. error",
        yscale = log10, title = "Accuracy vs Vasicek closed form"
    )
    lines!(ax2, times[2:end], relerr[2:end], linewidth = 2.5)
    display(fig)
end
