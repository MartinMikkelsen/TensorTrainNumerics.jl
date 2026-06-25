using TensorTrainNumerics
using CairoMakie

# Feynman–Kac for the quantum harmonic oscillator (imaginary time), in QTT.
#
# Feynman–Kac with a potential:  ∂u/∂τ = L u − V(x)·u,  u(·,0)=g, represents
#   u(x,τ) = E[ exp(−∫₀^τ V(X_s) ds) · g(X_τ) | X₀ = x ].
# With Brownian diffusion L = ½∂ₓₓ and harmonic V = ½ω²x², this is the
# imaginary-time quantum harmonic oscillator:
#   ∂u/∂τ = ½∂ₓₓu − ½ω²x²·u = −H_HO u ,   H_HO = −½∂ₓₓ + ½ω²x² .
#
# A Gaussian payoff g(x)=exp(−½αx²) stays Gaussian: its width β(τ) solves the
# Riccati ODE  β′ = ω² − β²  (β → ω), and the Rayleigh energy ⟨u|H|u⟩/⟨u|u⟩
# relaxes to the exactly-known ground state  E₀ = ½ω.  Marched with
# Crank–Nicholson + ALS (normalize=false — u is the genuine decaying Feynman–Kac
# solution, not a renormalised state), and validated against the closed-form
# Mehler–Gaussian at every τ.  The solution stays low-rank, so no IC enrichment.

# --- parameters --------------------------------------------------------------
ω = 1.0; α = 2.5                      # frequency; initial Gaussian width (≠ ω ⇒ relaxes)
d = 8; N = 2^d; a, b = -6.0, 6.0
h = (b - a) / (N - 1); xes = collect(range(a, b, N))

# --- operators:  A = ½∂ₓₓ − ½ω²x²  (= −H_HO) --------------------------------
∂xx = -(1 / h^2) * Δ(d)                                                # = d²/dx²
V = ttv_to_diag_tto(qtt_polynom([0.0, 0.0, 0.5 * ω^2], d; a = a, b = b))   # ½ω²x²
A = 0.5 * ∂xx - V                                                   # ∂u/∂τ = A u
H = -0.5 * ∂xx + V                                                  # H_HO (= −A), for the energy

# --- Gaussian initial payoff  g(x) = exp(−½αx²) -----------------------------
u₀ = function_to_qtt(t -> exp(-0.5 * α * (a + (b - a) * t)^2), d)

# --- exact closed form: Mehler kernel applied to the Gaussian ---------------
function uA(x, τ)                                                     # u(x,τ) = Amp·exp(−½β x²)
    s = sinh(ω * τ); c = cosh(ω * τ); p = ω * c / (2s) + α / 2
    β = ω * c / s - ω^2 / (2 * p * s^2)
    return sqrt(ω / (2 * s * p)) * exp(-0.5 * β * x^2)
end
βR(τ) = ω * (α + ω * tanh(ω * τ)) / (ω + α * tanh(ω * τ))             # Riccati width β(τ)
E_riccati(τ) = βR(τ) / 4 + ω^2 / (4 * βR(τ))                          # energy of that Gaussian → ½ω
energy(u) = real(dot(u, H * u)) / real(dot(u, u))

# --- Crank–Nicholson march in τ, recording snapshots + diagnostics ----------
τstep = 0.02; record_dt = 0.2; T = 3.0    # record_dt must be an exact multiple of τstep
blk = round(Int, record_dt / τstep); nblk = round(Int, T / record_dt)
times = collect(0.0:record_dt:T)
sols = Vector{Vector{Float64}}(); E_num = Float64[]; errL2 = Float64[]

function record!(u)
    v = qtt_to_function(u); push!(sols, v)
    τ = (length(sols) - 1) * record_dt
    push!(E_num, energy(u))
    return push!(errL2, τ == 0 ? 0.0 : sqrt(sum(abs2, v .- [uA(x, τ) for x in xes]) * h))
end

u = u₀; record!(u)
for _ in 1:nblk
    global u = crank_nicholson_method(A, u, u, fill(τstep, blk); normalize = false, tt_solver = "als")
    record!(u)
end

@info "FK harmonic oscillator" E_final = E_num[end] E0_exact = 0.5ω L2_max = maximum(errL2) rank = maximum(u.ttv_rks)

# --- Figure 1: u(x,τ) relaxing toward the ground-state shape ----------------
let
    snap = [0.0, 0.2, 0.4, 1.0, 3.0]
    fig = Figure(size = (760, 480))
    ax = Axis(
        fig[1, 1], xlabel = "x", ylabel = "u(x, τ)",
        title = "Feynman–Kac, quantum harmonic oscillator  (ω=$ω)"
    )
    for τ in snap
        lines!(ax, xes, sols[round(Int, τ / record_dt) + 1], linewidth = 2, label = "τ = $τ")
    end
    for τ in snap
        τ == 0 && continue
        lines!(ax, xes, [uA(x, τ) for x in xes], color = :black, linestyle = :dash, linewidth = 1)
    end
    lines!(ax, Float64[], Float64[], color = :black, linestyle = :dash, label = "analytic (Mehler)")
    xlims!(ax, -4, 4); axislegend(ax; position = :rt)
    display(fig)
end

# --- Figure 2: energy → E₀ = ½ω, and accuracy vs the closed form ------------
let
    fig = Figure(size = (1000, 420))
    ax1 = Axis(
        fig[1, 1], xlabel = "τ", ylabel = "energy  ⟨u|H|u⟩/⟨u|u⟩",
        title = "Variational relaxation to the ground state"
    )
    lines!(ax1, times, E_num, linewidth = 2.5, label = "numerical")
    lines!(ax1, times, E_riccati.(times), color = :black, linestyle = :dash, label = "Riccati β(τ)")
    hlines!(ax1, [0.5ω], color = :gray, linestyle = :dot, label = "E₀ = ½ω")
    axislegend(ax1; position = :rt)
    ax2 = Axis(
        fig[1, 2], xlabel = "τ", ylabel = "L² error vs analytic", yscale = log10,
        title = "Accuracy vs closed-form Mehler–Gaussian"
    )
    lines!(ax2, times[2:end], errL2[2:end], linewidth = 2.5)
    display(fig)
end
