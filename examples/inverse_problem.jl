# Linear inverse problem: Born inversion for 2D time-harmonic acoustic
# scattering, entirely in QTT format.
#
# Following the SwitchNet setup, the Helmholtz operator is written
# perturbatively,
#
#     L u = (-∇² - k0² - η(x)) u = f,        k(x)² = k0² + η(x),
#
# and we recover the wave-number perturbation η from scattered-field data.
# With u = u_inc + u_s and the (first) Born approximation η·u ≈ η·u_inc,
#
#     L0 u_s ≈ η ⊙ u_inc,            L0 = -∇² - k0²,
#
# i.e. the data is d_s = G0 (η ⊙ u_inc) with G0 = L0⁻¹ the background
# Green's function. Recovering η is then *linear*: applying L0 to the data
# and solving the Tikhonov-regularised normal equations
#
#     (diag(u_inc)² + λ₁·(-∇²) + λ₀·I) η = u_inc ⊙ (L0 d_s).
#
# Everything stays in QTT: the synthetic data comes from a *full*
# (non-linearised) scattering ALS solve — no inverse crime — and the
# inversion is one symmetric ALS solve. The remaining error is the Born
# linearisation bias plus the regularisation bias near nodal lines of u_inc.
# The dense grid is only touched to draw the figure.

using CairoMakie
using LinearAlgebra
using Random
using TensorTrainNumerics
using InterpolativeQTT

# ---------------------------------------------------------------- parameters
d = 7                     # bits per dimension → 128 × 128 interior grid
k0sq = 145.0              # background k0²; sits in the spectral gap of -∇²
                          # on (0,1)² (13π² ≈ 128.3 < 145 < 17π² ≈ 167.8)
max_bond = 50             # solution bond dimension (fixed-rank ALS)
coeff_bond = 25           # bond cap for compressed coefficient fields
als_sweeps = 10           # ALS sweeps per solve
λ_smooth = 1.0e-5         # Tikhonov gradient penalty λ₁·(-∇²)
λ_ridge = 1.0e-5          # ridge λ₀·I
build_degree = 10         # Chebyshev degree for the interpolative field build
build_tolerance = 1.0e-8
build_adaptive_tol = 1.0e-8
seed = 11

# true perturbation: a fast and a slow inclusion (≈ 4 % of k0², Born regime)
η_fun = (x, y) ->
    6.0 * exp(-((x - 0.35)^2 + (y - 0.62)^2) / 0.008) -
    4.0 * exp(-((x - 0.68)^2 + (y - 0.44)^2) / 0.012)

# acoustic point-like source near the bottom boundary
src_fun = (x, y) -> exp(-((x - 0.5)^2 + (y - 0.1)^2) / 0.002)

# ------------------------------------------------------------------ operators
N = 2^d
h = 1.0 / (N + 1)
x = [i * h for i in 1:N]

I_d = id_tto(d)
L1 = (1.0 / h^2) * Δ(d)              # positive discrete -∂²
Lap = L1 ⊗ I_d + I_d ⊗ L1            # -∇² (Dirichlet)
Id2 = id_tto(2 * d)
L0 = Lap + (-k0sq) * Id2             # background Helmholtz -∇² - k0²

# fields built with Lindsey's interpolative construction (adaptive/multiscale):
# dyadic grid of [a, b] = [h, 1] is exactly the interior lattice (i+1)·h
to_tt(f) = TTvector(interpolative_qttv((xx, yy) -> Float64(f(xx, yy)), 2, d;
    ordering = :serial,
    degree = build_degree,
    tolerance = build_tolerance,
    maxbonddim = coeff_bond,
    a = h, b = 1.0,
    mode = :adaptive,
    adaptive_tolerance = build_adaptive_tol,
))
cmp(v, r) = tt_compress!(orthogonalize(v), r)
grid(v) = permutedims(reshape(real.(qtt_to_function(v)), N, N))  # [i,j]=(x_i,y_j)

als_solve(A, b, x0) = als_linsolve(A, b, x0; sweep_count = als_sweeps)

η_true = to_tt(η_fun)
f_src = to_tt(src_fun)

# ------------------------------------------------- forward (synthetic data)
Random.seed!(seed)
println("incident field u_inc = G0 f ...")
u_inc = als_solve(L0, f_src, rand_tt(fill(2, 2 * d), max_bond; normalise = true))
res_inc = norm(L0 * u_inc - f_src) / norm(f_src)

# normalise the incident field to RMS 1 (fixes the data scale, all in TT)
scale = 2.0^d / norm(u_inc)
u_inc = scale * u_inc
f_src = scale * f_src

# Full (non-Born) scattering, formulated directly for the scattered field:
# (L0 - diag η) u_s = η ⊙ u_inc, exact since L0 u_inc = f. Solving for u_s
# instead of u_tot keeps the ALS residual at the data scale (no u_tot - u_inc
# cancellation, which L0 would amplify by 1/h²).
println("full scattering solve (L0 - diag η) u_s = η ⊙ u_inc ...")
A_true = L0 - ttv_to_diag_tto(η_true)
rhs_s = cmp(hadamard(η_true, u_inc), max_bond)
d_s = als_solve(A_true, rhs_s, rand_tt(fill(2, 2 * d), max_bond; normalise = true))
res_tot = norm(A_true * d_s - rhs_s) / norm(rhs_s)
u_tot = cmp(u_inc + d_s, max_bond)                # total field (for plotting)

# -------------------------------------------------------- Born inversion
println("Born inversion (one symmetric ALS solve) ...")
t_inv = @elapsed begin
    rhs_field = cmp(L0 * d_s, max_bond)           # L0 d_s ≈ η ⊙ u_inc
    b_rhs = cmp(hadamard(u_inc, rhs_field), max_bond)
    u2 = cmp(hadamard(u_inc, u_inc), coeff_bond)
    M = ttv_to_diag_tto(u2) + λ_smooth * Lap + λ_ridge * Id2
    η_rec = als_solve(M, b_rhs, rand_tt(fill(2, 2 * d), max_bond; normalise = true))
end

# ------------------------------------------------- metrics (all in TT format)
relerr = norm(η_rec - η_true) / norm(η_true)
born_relerr = norm(rhs_field - rhs_s) / norm(rhs_s)   # ‖L0 d_s − η⊙u_inc‖/‖η⊙u_inc‖

@info "Born inversion (QTT-ALS)" d N k0sq relerr = round(relerr, sigdigits = 3) born_model_error =
    round(born_relerr, sigdigits = 3) forward_residuals =
    round.((res_inc, res_tot), sigdigits = 3) ranks_u_ds_η = (maximum(u_inc.ttv_rks),
    maximum(d_s.ttv_rks), maximum(η_rec.ttv_rks)) inversion_seconds = round(t_inv, digits = 1)

# ------------------------------------------------------------------- figure
set_theme!(Theme(fontsize = 13))
fig = Figure(size = (1280, 820))

u_grid = grid(u_tot)
ds_grid = grid(d_s)
η_t = grid(η_true)
η_r = grid(η_rec)
η_lim = maximum(abs.(η_t))

ax1 = Axis(fig[1, 1]; title = "Total field u (full scattering solve)", xlabel = "x", ylabel = "y", aspect = DataAspect())
hm1 = heatmap!(ax1, x, x, u_grid; colormap = :balance,
    colorrange = (-maximum(abs.(u_grid)), maximum(abs.(u_grid))))
Colorbar(fig[2, 1], hm1; label = "u", vertical = false)

ax2 = Axis(fig[1, 2]; title = "Scattered-field data d_s = u - u_inc", xlabel = "x", ylabel = "y", aspect = DataAspect())
hm2 = heatmap!(ax2, x, x, ds_grid; colormap = :balance,
    colorrange = (-maximum(abs.(ds_grid)), maximum(abs.(ds_grid))))
Colorbar(fig[2, 2], hm2; label = "d_s", vertical = false)

ax3 = Axis(fig[1, 3]; title = "True perturbation η", xlabel = "x", ylabel = "y", aspect = DataAspect())
hm3 = heatmap!(ax3, x, x, η_t; colormap = :vik, colorrange = (-η_lim, η_lim))
Colorbar(fig[2, 3], hm3; label = "η", vertical = false)

ax4 = Axis(fig[3, 1]; title = "Recovered η (Born + Tikhonov, QTT-ALS)", xlabel = "x", ylabel = "y", aspect = DataAspect())
hm4 = heatmap!(ax4, x, x, η_r; colormap = :vik, colorrange = (-η_lim, η_lim))
Colorbar(fig[4, 1], hm4; label = "η", vertical = false)

ax5 = Axis(fig[3, 2]; title = "|η_rec − η_true|", xlabel = "x", ylabel = "y", aspect = DataAspect())
hm5 = heatmap!(ax5, x, x, abs.(η_r - η_t); colormap = :viridis)
Colorbar(fig[4, 2], hm5; label = "|Δη|", vertical = false)

j_cut = clamp(round(Int, 0.62 / h), 1, N)          # slice through the fast blob
ax6 = Axis(fig[3, 3]; title = "Slice y = $(round(x[j_cut], digits = 2))", xlabel = "x", ylabel = "η")
lines!(ax6, x, η_t[:, j_cut]; color = :black, linewidth = 2, label = "true")
lines!(ax6, x, η_r[:, j_cut]; color = :firebrick, linewidth = 2, linestyle = :dash, label = "recovered")
axislegend(ax6; position = :rt)

out_dir = joinpath(@__DIR__, "output")
isdir(out_dir) || mkpath(out_dir)
out = joinpath(out_dir, "inverse_problem.png")
save(out, fig)
@info "Figure saved" path = out
display(fig)
