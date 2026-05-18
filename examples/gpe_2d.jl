# 2D Gross-Pitaevskii equation — QTT ground state
#
# (-1/(2h²) ∇² + V(x,y) + g|ψ|²) ψ = μ ψ
# Domain:   [0,1]²,  V(x,y) = κ((x-0.5)² + (y-0.5)²)
# Grid:     N×N = (2^L)×(2^L) finite-difference points (Dirichlet BCs)
# TT format: 2L-mode QTT, block ordering — first L modes encode x, last L encode y
#
# The 2D Laplacian as a 2L-mode TToperator:
#   Δ₂D = Δ_x ⊗ I_y + I_x ⊗ Δ_y   (Kronecker sum, rank ≤ 4 everywhere)
#
# Both nonlinear_als_eigsolve (SCF-ALS) and nonlinear_mals_eigsolve (SCF-MALS)
# run unchanged on 2L-mode tensors.

using TensorTrainNumerics
using CairoMakie
using Printf
import TensorTrainNumerics: dot
import Random

# ── Grid ───────────────────────────────────────────────────────────────────
L = 6;  N = 2^L;  h = 1.0 / (N - 1)
x = LinRange(0.0, 1.0, N)
y = LinRange(0.0, 1.0, N)
κ = 200.0   # same as 1D example — tight trap, minimal boundary effects

# ── 2D Hamiltonian ─────────────────────────────────────────────────────────
I_L   = id_tto(L)
H_kin = (1.0 / (2h^2)) * (Δ(L) ⊗ I_L + I_L ⊗ Δ(L))

# Separable trap: V(x,y) = V_x(x) + V_y(y)
o_L  = ones_tt(2, L)
V_x  = function_to_qtt(t -> κ * (t - 0.5)^2, L)
V_y  = function_to_qtt(t -> κ * (t - 0.5)^2, L)
V_2D = V_x ⊗ o_L + o_L ⊗ V_y
H_lin = H_kin + ttv_to_diag_tto(V_2D)

println("H_lin TTO ranks: ", H_lin.tto_rks)
println("Expected linear μ ≈ √(2κ) = $(round(sqrt(2κ), digits=4))  (2D HO ground state, continuum)")

function run_nls_mals(H, g, ψ0; n_sweeps = 30, tol = 1.0e-10, rmax = 12, verbose = false)
    return nonlinear_mals_eigsolve(
        H, g, ψ0;
        tol = tol,
        sweep_schedule = [n_sweeps + 1],
        rmax_schedule = [rmax],
        verbose = verbose
    )
end

# ── Helper: 2L-mode QTT → N×N density array ────────────────────────────────
# qtt_to_function ordering: v[ix*N + iy + 1] = ψ(x_ix, y_iy)
# permutedims(reshape(⋅, N, N)) maps this to ρ[ix+1, iy+1] for heatmap!(ax, x, y, ρ)
function to_density(ψ, N)
    v = real.(qtt_to_function(ψ))
    return permutedims(reshape(abs2.(v), N, N))
end

# ── Linear ground state (g = 0) ────────────────────────────────────────────
Random.seed!(42)
r = 8
ψ0 = rand_tt(fill(2, 2L), r; normalise = true)
_, ψ_lin = als_eigsolve(H_lin, ψ0; sweep_schedule = [20])
μ_lin = real(dot(ψ_lin, H_lin * ψ_lin))
println("Linear μ = $(round(μ_lin, digits=6))")

# ── NLS ground states (warm-start chain along g) ───────────────────────────
g_vals = [0.0, 500.0, 2000.0, 10000.0]   # larger g needed in 2D (density ~1/N² vs 1/N in 1D)
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
                                              sweep_count = 30, verbose = false)
    hist_mals, ψg_mals, _ = run_nls_mals(H_lin, g, ψ_list_mals[k - 1];
                                         n_sweeps = 30, rmax = 12, verbose = false)
    ψ_list_als[k] = ψg_als
    ψ_list_mals[k] = ψg_mals
    μ_list_als[k] = hist_als[end]
    μ_list_mals[k] = hist_mals[end]
    println("g = $g:  μ_ALS = $(round(μ_list_als[k], digits=6))" *
            "   μ_MALS = $(round(μ_list_mals[k], digits=6))")
end

# ── Figure ─────────────────────────────────────────────────────────────────
set_theme!(Theme(fontsize = 14))
fig = Figure(size = (1050, 720))

ρ_grids = [to_density(ψ, N) for ψ in ψ_list_mals]

# Row 1: heatmaps; row 2: colorbars
for k in eachindex(g_vals)
    g = g_vals[k]
    ax = Axis(fig[1, k];
              xlabel = "x",
              ylabel = k == 1 ? "y" : "",
              title  = "g = $(Int(g)),  μ_MALS = $(round(μ_list_mals[k], digits=2))",
              aspect = DataAspect())
    k > 1 && hideydecorations!(ax; grid = false)
    hm = heatmap!(ax, x, y, ρ_grids[k]; colormap = :inferno)
    Colorbar(fig[2, k], hm; vertical = false,
             label = k == 2 ? "|ψ(x,y)|²" : "", ticklabelsize = 11)
end

# Row 3: μ vs g
ax3 = Axis(fig[3, 1:3];
    xlabel = "Nonlinear coupling  g",
    ylabel = "Chemical potential  μ",
    title  = "2D GPE  (heatmaps from SCF-MALS, μ comparison below)",
    xticks = Int.(g_vals))

lines!(ax3, g_vals, μ_list_als; color = :black, linewidth = 2, label = "SCF-ALS")
scatter!(ax3, g_vals, μ_list_als; color = :black, markersize = 10)
lines!(ax3, g_vals, μ_list_mals;
       color = :firebrick, linewidth = 2, linestyle = :dash,
       label = "SCF-MALS")
scatter!(ax3, g_vals, μ_list_mals; color = :firebrick, markersize = 10)
for k in eachindex(g_vals)
    text!(ax3, g_vals[k], μ_list_mals[k] + 0.3;
          text = "$(round(μ_list_mals[k], digits=1))",
          fontsize = 11, align = (:center, :bottom), color = :black)
end
axislegend(ax3; position = :lt, framevisible = false)


@printf("%-6s  %10s  %10s  %10s  %10s  %8s  %8s\n",
        "g", "μ_ALS", "μ_MALS", "Δμ_ALS", "|Δ|", "r_ALS", "r_MALS")
println("-"^76)
for k in eachindex(g_vals)
    @printf("%-6.0f  %10.4f  %10.4f  %10.4f  %10.2e  %8d  %8d\n",
            g_vals[k], μ_list_als[k], μ_list_mals[k], μ_list_als[k] - μ_lin,
            abs(μ_list_als[k] - μ_list_mals[k]),
            maximum(ψ_list_als[k].ttv_rks), maximum(ψ_list_mals[k].ttv_rks))
end
