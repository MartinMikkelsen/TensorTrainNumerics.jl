using TensorTrainNumerics
using CairoMakie

# Ground state of the 1D Gross–Pitaevskii (nonlinear Schrödinger) equation in a box,
#     [-½∂²ₓ + g|ψ|²] ψ = μψ,   ψ(0) = ψ(1) = 0,   ∫|ψ|² = 1,   g = 100,
# by multigrid renormalization (arXiv:1802.07259): solve on a 2³ grid, then prolong
# and re-solve up to a 2¹² grid. Published (Table 1): E = 122.09942.

g = 100.0
L0, L = 3, 12
χ = 8

A_builder(d) = (4.0^d / 2) * Δ(d)          # (1/(2h²))·tridiag(−1,2,−1), h = 2⁻ᵈ
g_builder(d) = g * 2.0^d                    # discrete interaction coefficient

seed = function_to_qtt(x -> sin(π * x), L0)
u0 = (1 / norm(seed)) * seed

alg = MGR(;
    inner = PenaltyALS(; local_solver = :newton),
    max_rank = χ,
    return_info = true,
    show_progress = true,
)
u, info = non_linear_solve(A_builder, u0, alg; g_builder = g_builder, target_sites = L)

println("E(L=$L, χ=$χ) = $(info.energy)   [Table 1: 122.09942]")
println("per-level energies: ", round.(info.level_energies; digits = 5))

# linear (g = 0) reference on the same grid
u_lin, info_lin = non_linear_solve(A_builder, u0, alg; g_builder = d -> 0.0, target_sites = L)

# continuum-normalized wavefunctions: f = √N · u for a discretely normalized u
x = range(0, 1; length = 2^L)
f_gpe = sqrt(2.0^L) .* qtt_to_function(u)
f_lin = sqrt(2.0^L) .* qtt_to_function(u_lin)

fig = Figure(size = (800, 450))
ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "ψ(x)",
    title = "Gross–Pitaevskii ground state in a box (QTT multigrid, L = $L)")
lines!(ax, x, abs.(f_gpe); label = "g = 100 (E = $(round(info.energy; digits = 4)))")
lines!(ax, x, abs.(f_lin); linestyle = :dash, label = "g = 0 (box, √2·sin πx)")
axislegend(ax)
display(fig)
