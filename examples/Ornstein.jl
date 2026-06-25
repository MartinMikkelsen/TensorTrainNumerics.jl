using TensorTrainNumerics
using CairoMakie

# Ornstein–Uhlenbeck process as a Fokker–Planck equation, solved in QTT format.
#
#   ∂P/∂t = θ ∂/∂x[(x-μ)P] + (σ²/2) ∂²P/∂x²
#
# Writing M = diag(x-μ), the generator is the low-rank QTT operator
#
#   A = θ ∇ M + (σ²/2) Δ ,
#
# and the density relaxes to the stationary Gaussian  P∞ = N(μ, σ²/2θ).
# We march with Crank–Nicholson,  (I - τ/2 A) Pⁿ⁺¹ = (I + τ/2 A) Pⁿ,  solved in
# TT format with ALS.  ALS performs a direct (non-symmetric) local solve, which
# is required here because the advection operator ∇M is non-symmetric — the
# MALS/DMRG local solves symmetrise the system and diverge for this problem.
# CN+ALS is the right tool here because it is *implicit* — unconditionally stable
# for any stiffness. TDVP steps via a local matrix exponential, which is NOT
# unconditionally stable: at this fine grid (D/h² ≈ 4e4) tdvp2 needs dt ≈ 2e-4 vs
# CN's 0.02 to stay bounded (~100× more steps, hence impractical), and single-site
# tdvp is unstable at any dt on this non-normal advection operator. Not a bug —
# the inherent limit of explicit-exponential vs implicit time-stepping.

# --- model parameters --------------------------------------------------------
θ = 1.0           # mean-reversion rate
μ = 2.0           # long-term mean
σ = 1.0           # volatility
D = σ^2 / 2       # diffusion coefficient

# --- grid: 2^d points on [a, b] ----------------------------------------------
d = 12
N = 2^d
a, b = -6.0, 8.0
h = (b - a) / (N - 1)
xes = collect(range(a, b, N))

# --- QTT operators -----------------------------------------------------------
# ∂/∂x  : central difference (P_{i+1}-P_{i-1})/2h, assembled from the package
#         shifts —  shift(d) = forward shift,  id_tto(d) - ∇(d) = backward shift.
∇₁ = (1 / (2h)) * (shift(d) - (id_tto(d) - ∇(d)))
# ∂²/∂x²: Δ(d) is the positive-definite stencil tridiag(-1,2,-1) ≈ -h² ∂²/∂x².
Δ₁ = -(1 / h^2) * Δ(d)
# multiplication operator  M = diag(x - μ)
M = ttv_to_diag_tto(qtt_polynom([-μ, 1.0], d; a = a, b = b))

A = θ * (∇₁ * M) + D * Δ₁

# --- initial condition: unit-mass Gaussian centred at 0 ----------------------
mass(v) = sum(v) * h
u₀ = function_to_qtt(t -> exp(-(a + (b - a) * t)^2 / 2), d)   # samples on [0,1]
u₀ = (1 / mass(qtt_to_function(u₀))) * u₀

# --- analytic stationary distribution  N(μ, σ²/2θ) ---------------------------
var∞ = D / θ
P∞ = @. exp(-(xes - μ)^2 / (2var∞)) / sqrt(2π * var∞)

# --- Crank–Nicholson march, recording snapshots and the error to P∞ ----------
τ = 0.001
record_dt = 0.4
T = 8.0
block = round(Int, record_dt / τ)
n_blocks = round(Int, T / record_dt)

times = collect(0.0:record_dt:T)
density = Vector{Vector{Float64}}()
errL1 = Float64[]
errL2 = Float64[]

function record!(P)
    v = qtt_to_function(P)
    v ./= mass(v)
    push!(density, v)
    push!(errL1, sum(abs.(v .- P∞)) * h)
    return push!(errL2, sqrt(sum(abs2, v .- P∞) * h))
end

P = u₀
record!(P)
for _ in 1:n_blocks
    global P = crank_nicholson_method(A, P, P, fill(τ, block); normalize = false, tt_solver = "als")
    record!(P)
end

# moments of the final state vs the analytic targets
let v = density[end]
    m = sum(xes .* v) * h
    s = sum((xes .- m) .^ 2 .* v) * h
    @info "final state" mean = m target_mean = μ variance = s target_var = var∞ L1 = errL1[end]
end

# --- Figure 1: relaxation of the density toward the stationary distribution ---
let
    snap = [0.0, 0.4, 0.8, 1.6, 3.2, 8.0]
    fig = Figure(size = (760, 480))
    ax = Axis(
        fig[1, 1], xlabel = "x", ylabel = "P(x, t)",
        title = "Ornstein–Uhlenbeck relaxation  (θ=$θ, μ=$μ, σ=$σ)"
    )
    for t in snap
        lines!(
            ax, xes, density[round(Int, t / record_dt) + 1],
            linewidth = 2, label = "t = $t"
        )
    end
    lines!(
        ax, xes, P∞, color = :black, linestyle = :dash, linewidth = 2.5,
        label = "stationary  N(μ, σ²/2θ)"
    )
    xlims!(ax, -4, 6)
    axislegend(ax; position = :rt)
    display(fig)
end

# --- Figure 2: convergence to the stationary distribution --------------------
let
    fig = Figure(size = (760, 480))
    ax = Axis(
        fig[1, 1], xlabel = "t", ylabel = "‖P(·, t) − P∞‖", yscale = log10,
        title = "Convergence to the stationary distribution"
    )
    lines!(ax, times, errL1, linewidth = 2.5, label = "L¹ error")
    lines!(ax, times, errL2, linewidth = 2.5, label = "L² error")
    axislegend(ax; position = :rt)
    display(fig)
end
