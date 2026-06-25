using TensorTrainNumerics
using CairoMakie

θ = 1.0                # mean-reversion rate
μx, μy = 2.0, -2.0     # long-term mean (the density drifts toward this point)
σ = 1.0                # volatility
D = σ^2 / 2            # diffusion coefficient

d = 8
N = 2^d
a, b = -6.0, 6.0
h = (b - a) / (N - 1)
xes = collect(range(a, b, N))

∂x = (1 / (2h)) * (shift(d) - (id_tto(d) - ∇(d)))   # central first derivative
∂xx = -(1 / h^2) * Δ(d)                              # second derivative
idd = id_tto(d)
Mx = ttv_to_diag_tto(qtt_polynom([-μx, 1.0], d; a = a, b = b))   # diag(x - μx)
My = ttv_to_diag_tto(qtt_polynom([-μy, 1.0], d; a = a, b = b))   # diag(y - μy)

A = θ * ((∂x * Mx) ⊗ idd + idd ⊗ (∂x * My)) + D * (∂xx ⊗ idd + idd ⊗ ∂xx)

toarr(v) = qttv_to_array(QTTvector(v, 2, d, :serial))   # raw 2d-site TT -> N×N grid
mass(P) = sum(P) * h^2
gx = function_to_qtt(t -> exp(-(a + (b - a) * t)^2 / 2), d)   # samples on [0,1]
gy = function_to_qtt(t -> exp(-(a + (b - a) * t)^2 / 2), d)
u₀ = gx ⊗ gy
u₀ = (1 / mass(toarr(u₀))) * u₀

var∞ = D / θ
g1(x, m) = exp(-(x - m)^2 / (2var∞)) / sqrt(2π * var∞)
P∞ = [g1(xi, μx) * g1(yj, μy) for xi in xes, yj in xes]

τ = 0.02
record_dt = 0.4
T = 8.0
block = round(Int, record_dt / τ)
n_blocks = round(Int, T / record_dt)

times = collect(0.0:record_dt:T)
density = Vector{Matrix{Float64}}()
errL1 = Float64[]
errL2 = Float64[]

function record!(v)
    P = toarr(v)
    P ./= mass(P)
    push!(density, P)
    push!(errL1, sum(abs.(P .- P∞)) * h^2)
    return push!(errL2, sqrt(sum(abs2, P .- P∞) * h^2))
end

ψ = u₀
record!(ψ)
for _ in 1:n_blocks
    global ψ = crank_nicholson_method(
        A, ψ, ψ, fill(τ, block);
        normalize = false, tt_solver = "als"
    )
    record!(ψ)
end

# moments of the final state vs the analytic targets (mean → μ, var → σ²/2θ, cov → 0)
let P = density[end]
    mx = sum(xes .* vec(sum(P, dims = 2))) * h^2
    my = sum(xes .* vec(sum(P, dims = 1))) * h^2
    vx = sum((xes .- mx) .^ 2 .* vec(sum(P, dims = 2))) * h^2
    vy = sum((xes .- my) .^ 2 .* vec(sum(P, dims = 1))) * h^2
    cov = sum((xes .- mx) .* P .* (xes .- my)') * h^2
    @info "final state" mean = (mx, my) target_mean = (μx, μy) var = (vx, vy) target_var = var∞ cov = cov L1 = errL1[end]
end

let
    snap = [0.0, 0.4, 1.2, 8.0]
    cmax = maximum(P∞)
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
        scatter!(ax, [μx], [μy], color = :red, marker = :xcross, markersize = 14)
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
    fig = Figure(size = (760, 480))
    ax = Axis(
        fig[1, 1], xlabel = "t", ylabel = "‖P(·, t) − P∞‖", yscale = log10,
        title = "2D Ornstein–Uhlenbeck: convergence to the stationary distribution"
    )
    lines!(ax, times, errL1, linewidth = 2.5, label = "L¹ error")
    lines!(ax, times, errL2, linewidth = 2.5, label = "L² error")
    axislegend(ax; position = :rt)
    display(fig)
end
