using TensorTrainNumerics
using LinearAlgebra
using Zygote
using CairoMakie

# Parameter sensitivities of a trapped Bose–Einstein condensate by automatic
# differentiation through QTT operations.
#
# The ground state of the 1D Gross–Pitaevskii equation in a harmonic trap,
#     [-½∂²ₓ + ½ω²(x − x₀)² + g|ψ|²] ψ = μψ,   ψ = 0 on the box walls,   ∫|ψ|² = 1,
# minimizes the energy functional
#     F[u; ω, g] = ⟨u|A(ω)|u⟩/⟨u|u⟩ + (g_d/2) Σ uᵢ⁴/⟨u|u⟩²,   A(ω) = K + (ω²/2) V,
# on a QTT grid with 2^L points, where K is the discrete kinetic operator, V the
# diagonal trap potential, and g_d = g·2^L the discrete interaction coefficient.
#
# The minimum E*(ω, g) = F[u*; ω, g] depends on the parameters both directly and
# through the minimizer u*. Because u* makes F stationary, the second dependence
# drops out (Hellmann–Feynman theorem):
#     dE*/dω = ∂F/∂ω (u*),   dE*/dg = ∂F/∂g (u*).
# One reverse-mode pass through F, which is built from TT operator sums, scalar
# multiples, operator application, Hadamard products, and inner products, gives
# both derivatives without re-solving for the ground state.

L0, L = 3, 10                  # coarsest and finest QTT levels
χ = 6                          # maximum TT rank of the ground state

kinetic(d) = (4.0^d / 2) * Δ(d)            # −½∂²ₓ with Dirichlet walls, h = 2⁻ᵈ

function trap(d)                            # diag((x − x₀)²) on xⱼ = j·h, j = 1, …, 2ᵈ
    h = 2.0^-d
    x0 = (1 + h) / 2                        # center of the box [0, 1 + h]
    return ttv_to_diag_tto(qtt_polynom([x0^2, -2x0, 1.0], d; a = h, b = 1.0))
end

function energy(u, K, V, ω, g)
    s = dot(u, u)
    w = hadamard(u, u)
    return dot(u, (K + (ω^2 / 2) * V) * u) / s + (g * 2.0^u.N / 2) * dot(w, w) / s^2
end

# Ground state by multigrid renormalization: solve on 2^L0 points, then prolong
# and re-solve level by level up to 2^L points.
function ground_state(ω, g)
    seed = function_to_qtt(x -> sin(π * x), L0)
    alg = MGR(; inner = PenaltyALS(; tol = 1.0e-10), max_rank = χ)
    return non_linear_solve(
        d -> kinetic(d) + (ω^2 / 2) * trap(d), seed / norm(seed), alg;
        g_builder = d -> g * 2.0^d, target_sites = L
    )
end

K, V = kinetic(L), trap(L)
Estar(ω, g) = energy(ground_state(ω, g), K, V, ω, g)

# --- 1. Ground state and a stationarity check -------------------------------------
ω, g = 50.0, 100.0
u = ground_state(ω, g)
E = energy(u, K, V, ω, g)
println("E*(ω = $ω, g = $g) = $E   (TT ranks $(u.ttv_rks))")

# Gradient of F with respect to the TT cores, holding ranks and dimensions fixed.
core_grad(v) = only(Zygote.gradient(cs -> energy(TTvector(L, cs, v.ttv_dims, v.ttv_rks, v.ttv_ot), K, V, ω, g), v.ttv_vec))
gradnorm(v) = sqrt(sum(sum(abs2, c) for c in core_grad(v)))
u_trial = orthogonalize(function_to_qtt(x -> sin(π * x), L))
println("‖∇_cores F‖ at sin(πx): $(gradnorm(u_trial / norm(u_trial)))   at the ground state: $(gradnorm(u))")

# --- 2. Sensitivities from one reverse-mode pass ----------------------------------
dEdω, dEdg = Zygote.gradient((ω, g) -> energy(u, K, V, ω, g), ω, g)

δ = 1.0e-3                     # central differences of re-solved ground-state energies
fd_ω = (Estar(ω + δ, g) - Estar(ω - δ, g)) / 2δ
fd_g = (Estar(ω, g + δ) - Estar(ω, g - δ)) / 2δ
println("dE*/dω: AD = $dEdω, finite difference = $fd_ω, relative difference = $(abs(dEdω - fd_ω) / abs(fd_ω))")
println("dE*/dg: AD = $dEdg, finite difference = $fd_g, relative difference = $(abs(dEdg - fd_g) / abs(fd_g))")

# --- 3. Calibrating the interaction strength to a target energy --------------------
# Newton's method on E*(ω, g) = E_target, with dE*/dg from AD at every iterate.
E_target = 200.0
g_k = g
newton = Tuple{Float64, Float64}[]
for it in 1:8
    global g_k
    u_k = ground_state(ω, g_k)
    E_k = energy(u_k, K, V, ω, g_k)
    push!(newton, (g_k, E_k))
    println("Newton $it: g = $g_k, E* − E_target = $(E_k - E_target)")
    abs(E_k - E_target) < 1.0e-9 && break
    slope = only(Zygote.gradient(γ -> energy(u_k, K, V, ω, γ), g_k))
    g_k -= (E_k - E_target) / slope
end
g_fit = g_k

# --- 4. Sensitivities across interaction strengths --------------------------------
gs = 0.0:25.0:300.0
states = [ground_state(ω, γ) for γ in gs]
sens = [Zygote.gradient((ω′, γ′) -> energy(v, K, V, ω′, γ′), ω, γ) for (v, γ) in zip(states, gs)]
# Without interaction the state is close to the harmonic-oscillator ground state
# (the walls are 3.5 oscillator lengths from the trap center), for which
# E* = ω/2 and first-order perturbation theory gives dE*/dg = ½∫|ψ₀|⁴ = ½√(ω/2π).
println("g = 0: dE*/dω = $(sens[1][1]) (oscillator: 0.5),  dE*/dg = $(sens[1][2]) (oscillator: $(sqrt(ω / 2π) / 2))")

gs_fd = 0.0:100.0:300.0
fd_sens = [
    ((Estar(ω + δ, γ) - Estar(ω - δ, γ)) / 2δ, (Estar(ω, γ + δ) - Estar(ω, γ - δ)) / 2δ)
    for γ in gs_fd
]

# --- 5. Figure ---------------------------------------------------------------------
h = 2.0^-L
x = h .* (1:(2^L))
density(v) = (w = qtt_to_vector(v); w .^ 2 ./ (sum(abs2, w) * h))   # ∫|ψ|² = 1

fig = Figure(size = (1400, 420))
ax1 = Axis(fig[1, 1]; xlabel = "x", ylabel = "|ψ(x)|²", title = "Ground-state density (ω = $ω)")
for (γ, label) in ((0.0, "g = 0"), (g, "g = $g"), (g_fit, "g = $(round(g_fit; digits = 2)) (fitted)"))
    lines!(ax1, x, density(ground_state(ω, γ)); label)
end
axislegend(ax1)

ax2 = Axis(fig[1, 2]; xlabel = "g", ylabel = "derivative", title = "Sensitivities: AD (lines), re-solved differences (markers)")
lines!(ax2, gs, first.(sens); label = "dE*/dω")
lines!(ax2, gs, last.(sens); label = "dE*/dg")
scatter!(ax2, gs_fd, first.(fd_sens); color = :black)
scatter!(ax2, gs_fd, last.(fd_sens); color = :black)
axislegend(ax2; position = :rc)

ax3 = Axis(
    fig[1, 3]; xlabel = "Newton iteration", ylabel = "|E* − E_target|", yscale = log10,
    title = "Calibrating g to E_target = $E_target"
)
scatterlines!(ax3, 1:length(newton), [abs(Ek - E_target) for (_, Ek) in newton])
display(fig)
