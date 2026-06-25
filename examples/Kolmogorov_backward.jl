using TensorTrainNumerics
using CairoMakie

# Kolmogorov backward equation for the 1D Ornstein–Uhlenbeck process, in QTT.
#
# Backward generator (= transpose of the Fokker–Planck operator A_FP):
#   L u = -θ(x-μ) ∂ₓu + D ∂ₓₓu ,    D = σ²/2 ,    L = (A_FP)ᵀ
# Evolve forward in τ = T - t:   ∂u/∂τ = L u,   u(·,0) = g   (terminal payoff).
#
# By Feynman–Kac, u(x,τ) = E[g(X_τ) | X₀ = x]; since OU is Gaussian,
#   X_τ | X₀=x  ~  N( m = μ+(x-μ)e^{-θτ},  s² = (D/θ)(1-e^{-2θτ}) ),
# and for a Gaussian-bump payoff g(x)=exp(-(x-x₀)²/2w²) the expectation is the
# Gaussian convolution  u(x,τ) = (w/√(w²+s²)) exp(-(m-x₀)²/2(w²+s²))  — exact.
#
# Unlike the density problem, u is an *observable*: no mass conservation, but a
# maximum principle (u stays within [min g, max g] = [0,1]) and u relaxes to the
# stationary expectation E_stat[g] as τ → ∞. Same machinery as the FP example —
# the backward operator is just the transpose — solved with Crank–Nicholson + ALS.
#
# Boundary note: the discrete L inherits a Dirichlet-type truncation (u→0 at the
# edges), while the true observable rises to the constant E_stat[g] there. The
# interior stays accurate; the boundary layer does not. We validate/plot the
# interior; a Neumann truncation would fix the far field.

# --- parameters --------------------------------------------------------------
θ = 1.0; μ = 2.0; σ = 1.2; D = σ^2 / 2
x₀ = 2.0; w = 0.6                          # Gaussian-bump terminal payoff

d = 10; N = 2^d; a, b = -8.0, 10.0
h = (b - a) / (N - 1); xes = collect(range(a, b, N))

# --- backward generator  L = -θ M ∂ₓ + D ∂ₓₓ   (= (A_FP)ᵀ) -------------------
∂x = (1 / (2h)) * (shift(d) - (id_tto(d) - ∇(d)))   # central first derivative
∂xx = -(1 / h^2) * Δ(d)                              # second derivative
M = ttv_to_diag_tto(qtt_polynom([-μ, 1.0], d; a = a, b = b))   # diag(x-μ)
L = -θ * (M * ∂x) + D * ∂xx

# --- terminal payoff g and its QTT ------------------------------------------
g(x) = exp(-(x - x₀)^2 / (2w^2))
u₀ = function_to_qtt(t -> g(a + (b - a) * t), d)     # function_to_qtt samples on [0,1]

# --- analytic conditional expectation  u(x,τ) = E[g(X_τ) | X₀=x] -------------
m(x, τ) = μ + (x - μ) * exp(-θ * τ)
s2(τ) = (D / θ) * (1 - exp(-2θ * τ))
uA(x, τ) = (w / sqrt(w^2 + s2(τ))) * exp(-(m(x, τ) - x₀)^2 / (2 * (w^2 + s2(τ))))
E_stat = w / sqrt(w^2 + D / θ)                     # τ→∞ limit (constant)

# --- Crank–Nicholson march in τ, recording snapshots + interior diagnostics --
τstep = 0.02; record_dt = 0.25; T = 3.0
blk = round(Int, record_dt / τstep); nblk = round(Int, T / record_dt)
times = collect(0.0:record_dt:T)
interior = findall(x -> -3.0 <= x <= 7.0, xes)

sols = Vector{Vector{Float64}}(); errL∞ = Float64[]; umin = Float64[]; umax = Float64[]

function record!(ψ)
    v = qtt_to_function(ψ)
    push!(sols, v)
    τ = (length(sols) - 1) * record_dt
    push!(errL∞, maximum(abs.(v[interior] .- [uA(x, τ) for x in xes[interior]])))
    push!(umin, minimum(v[interior]))
    return push!(umax, maximum(v[interior]))
end

ψ = u₀; record!(ψ)
for _ in 1:nblk
    global ψ = crank_nicholson_method(L, ψ, ψ, fill(τstep, blk); normalize = false, tt_solver = "als")
    record!(ψ)
end

@info "KBE final" interior_L∞ = errL∞[end] u_range = (umin[end], umax[end]) E_stat = E_stat

# --- Figure 1: backward solution relaxing from payoff toward the constant -----
let
    snap = [0.0, 0.5, 1.0, 2.0, 3.0]
    fig = Figure(size = (760, 480))
    ax = Axis(
        fig[1, 1], xlabel = "x", ylabel = "u(x, τ) = E[g(X_τ) | X₀ = x]",
        title = "OU Kolmogorov backward equation  (θ=$θ, μ=$μ, σ=$σ)"
    )
    for τ in snap
        lines!(ax, xes, sols[round(Int, τ / record_dt) + 1], linewidth = 2, label = "τ = $τ")
    end
    for τ in snap
        lines!(ax, xes, [uA(x, τ) for x in xes], color = :black, linestyle = :dash, linewidth = 1)
    end
    hlines!(ax, [E_stat], color = :gray, linestyle = :dot, label = "E_stat[g]")
    lines!(ax, Float64[], Float64[], color = :black, linestyle = :dash, label = "analytic")
    xlims!(ax, -3, 7); axislegend(ax; position = :rt)
    display(fig)
end

# --- Figure 2: interior accuracy and the maximum principle ------------------
let
    fig = Figure(size = (1000, 420))
    ax1 = Axis(
        fig[1, 1], xlabel = "τ", ylabel = "interior L∞ error", yscale = log10,
        title = "Accuracy vs closed-form  E[g(X_τ) | x]"
    )
    lines!(ax1, times[2:end], errL∞[2:end], linewidth = 2.5)

    ax2 = Axis(
        fig[1, 2], xlabel = "τ", ylabel = "u (interior range)",
        title = "Maximum principle: u relaxes to E_stat[g]"
    )
    band!(ax2, times, umin, umax, color = (:dodgerblue, 0.25))
    lines!(ax2, times, umin, linewidth = 2, label = "min u")
    lines!(ax2, times, umax, linewidth = 2, label = "max u")
    hlines!(ax2, [E_stat], color = :black, linestyle = :dash, label = "E_stat[g]")
    axislegend(ax2; position = :rc)
    display(fig)
end
