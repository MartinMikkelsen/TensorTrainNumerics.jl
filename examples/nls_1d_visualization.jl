# 1D NLS / GPE ground-state visualization
#
# Equation: ( -1/(2h²) Δ + V(x) + g|ψ|² ) ψ = μ ψ
# Domain:   x ∈ [0, 1],  V(x) = κ(x − 0.5)²
# Norm:     ‖ψ‖² = 1  (discrete l²)
#
# Intrinsic TT rank: solved at rank-16, then SVD-compressed to ε = 1e-10
# so the rank plot shows the minimal bond dimension actually needed.
#
# Figure layout
#   (a) Top-left  — normalized density ρ(x)/ρ_max for g = 0 … 1000
#   (b) Top-right — compressed TT rank profile for each g
#   (c) Bottom    — chemical potential μ and interaction energy Δμ vs g

using TensorTrainNumerics
using CairoMakie
using Printf
import TensorTrainNumerics: dot
import Random

# ── Grid and operators ─────────────────────────────────────────────────────
L = 8;  N = 2^L;  h = 1.0 / (N - 1)
x = LinRange(0.0, 1.0, N)
κ = 200.0

H_kin  = (1.0 / (2h^2)) * Δ(L)
V_trap = function_to_qtt(x -> κ * (x - 0.5)^2, L)
H_lin  = H_kin + ttv_to_diag_tto(V_trap)
V_vec  = qtt_to_function(V_trap)

# ── Linear ground state at rank 16 ────────────────────────────────────────
# Using r=16 so the solver has freedom; after solving we compress to find
# the minimal rank actually needed.
Random.seed!(42)
r_solve = 16
ψ0 = rand_tt(fill(2, L), r_solve; normalise = true)
_, ψ_lin = als_eigsolve(H_lin, ψ0; sweep_schedule = [15])
μ_lin = real(dot(ψ_lin, H_lin * ψ_lin))

# ── NLS ground states  (warm-start chain in g, all at rank r_solve) ───────
g_vals = [0.0, 50.0, 200.0, 500.0, 1000.0]
ψ_list = Vector{Any}(undef, length(g_vals))
μ_list = Vector{Float64}(undef, length(g_vals))

ψ_list[1] = ψ_lin;  μ_list[1] = μ_lin

for k in eachindex(g_vals)[2:end]
    g = g_vals[k]
    hist, ψg = nonlinear_als_eigsolve(H_lin, g, ψ_list[k - 1];
                                       sweep_count = 25, verbose = false)
    ψ_list[k] = ψg
    μ_list[k]  = real(hist[end])
end

# ── SVD-compress each solution to ε = 1e-10 to reveal intrinsic rank ──────
compress_tol = 1e-10
ψ_comp = [tt_compress!(deepcopy(ψ), r_solve; truncerr = compress_tol)
          for ψ in ψ_list]

# ── Dense density and normalized density ──────────────────────────────────
ρ_list = [abs2.(real.(qtt_to_function(ψ))) for ψ in ψ_comp]
# Normalize to peak so the shape change (Thomas-Fermi broadening) is visible
ρ_norm = [ρ ./ maximum(ρ) for ρ in ρ_list]

# ── Figure ─────────────────────────────────────────────────────────────────
set_theme!(Theme(fontsize = 14))
fig = Figure(size = (1000, 680))

colors    = [:royalblue, :seagreen, :darkorange, :crimson, :mediumpurple]
g_labels  = ["g = $(Int(g))" for g in g_vals]

# ── (a) Normalized density ─────────────────────────────────────────────────
ax1 = Axis(fig[1, 1];
    xlabel = "x",
    ylabel = "ρ(x) / ρ_max",
    title  = "Ground-state density  (normalized to peak)")

# Trap: scaled to 50 % of peak for reference
lines!(ax1, x, 0.5 .* V_vec ./ maximum(V_vec);
       color = (:gray, 0.55), linestyle = :dash, linewidth = 1.5,
       label = "V(x)  (scaled)")
for (k, g) in enumerate(g_vals)
    lines!(ax1, x, ρ_norm[k];
           color = colors[k], linewidth = 2, label = g_labels[k])
end
axislegend(ax1; position = :rt, labelsize = 12, framevisible = false)

# ── (b) Compressed TT rank profile ────────────────────────────────────────
ax2 = Axis(fig[1, 2];
    xlabel = "Bond index  k",
    ylabel = "Rank  rₖ  (SVD-compressed, ε = 1e-10)",
    title  = "Intrinsic TT rank",
    xticks = 1:(L + 1))

for (k, g) in enumerate(g_vals)
    rks = ψ_comp[k].ttv_rks
    scatterlines!(ax2, 1:(L + 1), rks;
                  color = colors[k], linewidth = 2, markersize = 8,
                  label = g_labels[k])
end
axislegend(ax2; position = :ct, labelsize = 12, framevisible = false)

# ── (c) μ vs g and Δμ ─────────────────────────────────────────────────────
ax3 = Axis(fig[2, 1:2];
    xlabel = "Nonlinear coupling  g",
    ylabel = "Chemical potential  μ",
    title  = "Chemical potential and interaction energy vs g",
    xticks = Int.(g_vals))

lines!(ax3, g_vals, μ_list;
       color = :black, linewidth = 2, label = "μ(g)")
scatter!(ax3, g_vals, μ_list;
         color = :black, markersize = 10)

ΔE = μ_list .- μ_lin
lines!(ax3, g_vals, ΔE;
       color = :firebrick, linewidth = 2, linestyle = :dash,
       label = "Δμ = μ(g) − μ₀")
scatter!(ax3, g_vals, ΔE;
         color = :firebrick, markersize = 10)

for (k, g) in enumerate(g_vals)
    text!(ax3, g, μ_list[k] + 0.4;
          text = "$(round(μ_list[k], digits = 1))",
          fontsize = 11, align = (:center, :bottom), color = :black)
end
axislegend(ax3; position = :lt, labelsize = 12, framevisible = false)

# ── Save ───────────────────────────────────────────────────────────────────
save("nls_1d_ground_state.pdf", fig)
display(fig)
println("Saved nls_1d_ground_state.pdf")
println()
println("g        μ(g)      Δμ = μ-μ₀   max intrinsic rank  (ε=$compress_tol)")
println("-"^60)
for (k, g) in enumerate(g_vals)
    rmax = maximum(ψ_comp[k].ttv_rks)
    @printf("%-6.0f  %10.4f  %10.4f   %4d\n",
            g, μ_list[k], μ_list[k] - μ_lin, rmax)
end
