using TensorTrainNumerics
using CairoMakie
using LinearAlgebra
using Random

θ = 1.0                 # mean-reversion rate
k = 0.6                 # drift coupling  (ρ = k/θ = 0.6)
μx, μy = 2.0, -2.0      # long-term mean
σ = 1.0
D = σ^2 / 2
Θ = [θ -k; -k θ]
Σ∞ = D * inv(Θ)         # analytic stationary covariance (Lyapunov: ΘΣ + ΣΘᵀ = 2D·I)

# --- grid: 2^d points per axis on [a, b]² ------------------------------------
d = 8
N = 2^d
a, b = -6.0, 6.0
h = (b - a) / (N - 1)
xes = collect(range(a, b, N))

∂ = (1 / (2h)) * (shift(d) - (id_tto(d) - ∇(d)))   # central first derivative
∂² = -(1 / h^2) * Δ(d)                              # second derivative
idd = id_tto(d)
Mx = ttv_to_diag_tto(qtt_polynom([-μx, 1.0], d; a = a, b = b))
My = ttv_to_diag_tto(qtt_polynom([-μy, 1.0], d; a = a, b = b))

A = θ * ((∂ * Mx) ⊗ idd + idd ⊗ (∂ * My)) -
    k * (∂ ⊗ My + Mx ⊗ ∂) +
    D * (∂² ⊗ idd + idd ⊗ ∂²)

toarr(v) = qttv_to_array(QTTvector(v, 2, d, :serial))   # raw 2d-site TT -> N×N grid
mass(P) = sum(P) * h^2

gx = function_to_qtt(t -> exp(-(a + (b - a) * t)^2 / 2), d)
gy = function_to_qtt(t -> exp(-(a + (b - a) * t)^2 / 2), d)
u₀_clean = (1 / mass(toarr(gx ⊗ gy))) * (gx ⊗ gy)                  # clean product IC (t=0 panel)
Random.seed!(42)                                                  # reproducible enrichment noise
u₀ = TensorTrainNumerics.increase_ranks(gx ⊗ gy, 16; noise = 1.0e-2)      # rank-16, small noise
u₀ = (1 / mass(toarr(u₀))) * u₀

Σi = inv(Σ∞)
nrm = 1 / (2π * sqrt(det(Σ∞)))
P∞ = [nrm * exp(-0.5 * ([xi - μx, yj - μy]' * Σi * [xi - μx, yj - μy])) for xi in xes, yj in xes]

τ = 0.02
record_dt = 0.4
T = 8.0
block = round(Int, record_dt / τ)
n_blocks = round(Int, T / record_dt)

times = collect(0.0:record_dt:T)
density = Vector{Matrix{Float64}}()
errL1 = Float64[]
errL2 = Float64[]
ρhist = Float64[]

function record!(v)
    P = toarr(v)
    P ./= mass(P)
    push!(density, P)
    mx = sum(xes .* vec(sum(P, dims = 2))) * h^2
    my = sum(xes .* vec(sum(P, dims = 1))) * h^2
    vx = sum((xes .- mx) .^ 2 .* vec(sum(P, dims = 2))) * h^2
    vy = sum((xes .- my) .^ 2 .* vec(sum(P, dims = 1))) * h^2
    cov = sum((xes .- mx) .* P .* (xes .- my)') * h^2
    push!(ρhist, cov / sqrt(vx * vy))
    push!(errL1, sum(abs.(P .- P∞)) * h^2)
    return push!(errL2, sqrt(sum(abs2, P .- P∞) * h^2))
end

record!(u₀_clean)                       # t = 0: clean product Gaussian
ψ = u₀
for _ in 1:n_blocks
    global ψ = crank_nicholson_method(
        A, ψ, ψ, fill(τ, block);
        normalize = false, tt_solver = "als"
    )
    record!(ψ)
end

# final covariance vs the analytic Lyapunov solution
let P = density[end]
    mx = sum(xes .* vec(sum(P, dims = 2))) * h^2
    my = sum(xes .* vec(sum(P, dims = 1))) * h^2
    vx = sum((xes .- mx) .^ 2 .* vec(sum(P, dims = 2))) * h^2
    vy = sum((xes .- my) .^ 2 .* vec(sum(P, dims = 1))) * h^2
    cov = sum((xes .- mx) .* P .* (xes .- my)') * h^2
    @info "final state" mean = (mx, my) cov_numeric = [vx cov; cov vy] cov_analytic = Σ∞ ρ = ρhist[end] L1 = errL1[end]
end

# 1σ covariance ellipse of N(μ, Σ) for overlaying on the heatmaps
function cov_ellipse(μ, Σ; nσ = 1.0, n = 120)
    vals, vecs = eigen(Symmetric(Σ))
    pts = [μ .+ nσ .* (vecs * (sqrt.(vals) .* [cos(t), sin(t)])) for t in range(0, 2π, n)]
    return getindex.(pts, 1), getindex.(pts, 2)
end

let
    snap = [0.0, 0.4, 1.2, 8.0]
    cmax = maximum(P∞)
    ex, ey = cov_ellipse([μx, μy], Σ∞)
    fig = Figure(size = (1100, 320))
    for (k, t) in enumerate(snap)
        ax = Axis(
            fig[1, k], aspect = 1, xlabel = "x", ylabel = k == 1 ? "y" : "",
            title = "t = $t"
        )
        heatmap!(
            ax, xes, xes, density[round(Int, t / record_dt) + 1],
            colormap = :viridis, colorrange = (0, cmax)
        )
        lines!(ax, ex, ey, color = :red, linewidth = 1.5)          # analytic 1σ ellipse
        scatter!(ax, [μx], [μy], color = :red, marker = :xcross, markersize = 12)
        xlims!(ax, -4, 5)
        ylims!(ax, -5, 4)
    end
    Colorbar(
        fig[1, length(snap) + 1], limits = (0, cmax), colormap = :viridis,
        label = "P(x, y, t)"
    )
    display(fig)
end

let
    fig = Figure(size = (1000, 420))
    ax1 = Axis(
        fig[1, 1], xlabel = "t", ylabel = "‖P(·, t) − P∞‖", yscale = log10,
        title = "Convergence to the stationary distribution"
    )
    lines!(ax1, times, errL1, linewidth = 2.5, label = "L¹ error")
    lines!(ax1, times, errL2, linewidth = 2.5, label = "L² error")
    axislegend(ax1; position = :rt)

    ax2 = Axis(
        fig[1, 2], xlabel = "t", ylabel = "correlation ρ(t)",
        title = "Correlation developing from the coupling"
    )
    lines!(ax2, times, ρhist, linewidth = 2.5)
    hlines!(ax2, [k / θ], color = :black, linestyle = :dash, label = "ρ∞ = k/θ")
    axislegend(ax2; position = :rb)
    display(fig)
end
