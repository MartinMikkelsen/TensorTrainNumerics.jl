# 1D NLS / GPE ground-state visualization
#
# Equation: ( -1/(2h²) Δ + V(x) + g|ψ|² ) ψ = μ ψ
# Domain:   x ∈ [0, 1],  V(x) = κ(x − 0.5)²
# Norm:     ‖ψ‖² = 1  (discrete l²)
#
# Intrinsic TT rank: solved with both SCF-ALS and SCF-MALS at rank cap 16,
# then SVD-compressed to ε = 1e-10 so the rank plot shows the minimal bond
# dimension actually needed.
#
# Figure layout
#   (a) Top-left  — normalized density ρ(x)/ρ_max for g = 0 … 1000
#                    solid = SCF-ALS, dashed = SCF-MALS
#   (b) Top-right — compressed TT rank profile for each g
#                    solid = SCF-ALS, dashed = SCF-MALS
#   (c) Bottom    — chemical potential μ vs g for both solvers

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

function run_nls_mals(H, g, ψ0; n_sweeps = 25, tol = 1.0e-10, rmax = 16, verbose = false)
    return nonlinear_mals_eigsolve(
        H, g, ψ0;
        tol = tol,
        sweep_schedule = [n_sweeps + 1],
        rmax_schedule = [rmax],
        verbose = verbose
    )
end

# ── Linear ground state at rank 16 ────────────────────────────────────────
# Using r=16 so the solver has freedom; after solving we compress to find
# the minimal rank actually needed.
Random.seed!(42)
r_solve = 16
ψ0 = rand_tt(fill(2, L), r_solve; normalise = true)
_, ψ_lin = als_eigsolve(H_lin, ψ0; sweep_schedule = [15])
μ_lin = real(dot(ψ_lin, H_lin * ψ_lin))

# ── NLS ground states  (warm-start chain in g, all at rank cap r_solve) ────
g_vals = [0.0, 50.0, 200.0, 500.0, 1000.0]
ψ_list_als = Vector{Any}(undef, length(g_vals))
ψ_list_mals = Vector{Any}(undef, length(g_vals))
μ_list_als = Vector{Float64}(undef, length(g_vals))
μ_list_mals = Vector{Float64}(undef, length(g_vals))

ψ_list_als[1] = ψ_lin
ψ_list_mals[1] = ψ_lin
μ_list_als[1] = μ_lin
μ_list_mals[1] = μ_lin

for k in eachindex(g_vals)[2:end]
    g = g_vals[k]
    hist_als, ψg_als = nonlinear_als_eigsolve(H_lin, g, ψ_list_als[k - 1];
                                              sweep_count = 25, verbose = false)
    hist_mals, ψg_mals, _ = run_nls_mals(H_lin, g, ψ_list_mals[k - 1];
                                         n_sweeps = 25, rmax = r_solve, verbose = false)
    ψ_list_als[k] = ψg_als
    ψ_list_mals[k] = ψg_mals
    μ_list_als[k] = hist_als[end]
    μ_list_mals[k] = hist_mals[end]
end

# ── SVD-compress each solution to ε = 1e-10 to reveal intrinsic rank ──────
compress_tol = 1e-10
ψ_comp_als = [tt_compress!(deepcopy(ψ), r_solve; truncerr = compress_tol)
              for ψ in ψ_list_als]
ψ_comp_mals = [tt_compress!(deepcopy(ψ), r_solve; truncerr = compress_tol)
               for ψ in ψ_list_mals]

# ── Dense density and normalized density ──────────────────────────────────
ρ_list_als = [abs2.(real.(qtt_to_function(ψ))) for ψ in ψ_comp_als]
ρ_list_mals = [abs2.(real.(qtt_to_function(ψ))) for ψ in ψ_comp_mals]
# Normalize to peak so the shape change (Thomas-Fermi broadening) is visible
ρ_norm_als = [ρ ./ maximum(ρ) for ρ in ρ_list_als]
ρ_norm_mals = [ρ ./ maximum(ρ) for ρ in ρ_list_mals]

# ── Figure ─────────────────────────────────────────────────────────────────
set_theme!(Theme(fontsize = 14))
fig = Figure(size = (1000, 680))

colors    = [:royalblue, :seagreen, :darkorange, :crimson, :mediumpurple]
g_labels  = ["g = $(Int(g))" for g in g_vals]

# ── (a) Normalized density ─────────────────────────────────────────────────
ax1 = Axis(fig[1, 1];
    xlabel = "x",
    ylabel = "ρ(x) / ρ_max",
    title  = "Ground-state density  (solid = ALS, dashed = MALS)")

# Trap: scaled to 50 % of peak for reference
lines!(ax1, x, 0.5 .* V_vec ./ maximum(V_vec);
       color = (:gray, 0.55), linestyle = :dash, linewidth = 1.5,
       label = "V(x)  (scaled)")
for (k, g) in enumerate(g_vals)
    lines!(ax1, x, ρ_norm_als[k];
           color = colors[k], linewidth = 2, label = g_labels[k])
    lines!(ax1, x, ρ_norm_mals[k];
           color = colors[k], linewidth = 2, linestyle = :dash)
end
axislegend(ax1; position = :rt, labelsize = 12, framevisible = false)

# ── (b) Compressed TT rank profile ────────────────────────────────────────
ax2 = Axis(fig[1, 2];
    xlabel = "Bond index  k",
    ylabel = "Rank  rₖ  (SVD-compressed, ε = 1e-10)",
    title  = "Intrinsic TT rank  (solid = ALS, dashed = MALS)",
    xticks = 1:(L + 1))

for (k, g) in enumerate(g_vals)
    rks_als = ψ_comp_als[k].ttv_rks
    rks_mals = ψ_comp_mals[k].ttv_rks
    scatterlines!(ax2, 1:(L + 1), rks_als;
                  color = colors[k], linewidth = 2, markersize = 8,
                  label = g_labels[k])
    scatterlines!(ax2, 1:(L + 1), rks_mals;
                  color = colors[k], linewidth = 2, markersize = 8,
                  linestyle = :dash)
end
axislegend(ax2; position = :ct, labelsize = 12, framevisible = false)

# ── (c) μ vs g ────────────────────────────────────────────────────────────
ax3 = Axis(fig[2, 1:2];
    xlabel = "Nonlinear coupling  g",
    ylabel = "Chemical potential  μ",
    title  = "Chemical potential vs g",
    xticks = Int.(g_vals))

lines!(ax3, g_vals, μ_list_als;
       color = :black, linewidth = 2, label = "SCF-ALS")
scatter!(ax3, g_vals, μ_list_als;
         color = :black, markersize = 10)
lines!(ax3, g_vals, μ_list_mals;
       color = :firebrick, linewidth = 2, linestyle = :dash,
       label = "SCF-MALS")
scatter!(ax3, g_vals, μ_list_mals;
         color = :firebrick, markersize = 10)

for (k, g) in enumerate(g_vals)
    text!(ax3, g, μ_list_mals[k] + 0.4;
          text = "$(round(μ_list_mals[k], digits = 1))",
          fontsize = 11, align = (:center, :bottom), color = :black)
end
axislegend(ax3; position = :lt, labelsize = 12, framevisible = false)

println("g        μ_ALS     μ_MALS    |Δ|        rmax_ALS  rmax_MALS  (ε=$compress_tol)")
println("-"^78)
for (k, g) in enumerate(g_vals)
    rmax_als = maximum(ψ_comp_als[k].ttv_rks)
    rmax_mals = maximum(ψ_comp_mals[k].ttv_rks)
    @printf("%-6.0f  %10.4f  %10.4f  %8.2e   %4d      %4d\n",
            g, μ_list_als[k], μ_list_mals[k], abs(μ_list_als[k] - μ_list_mals[k]),
            rmax_als, rmax_mals)
end
