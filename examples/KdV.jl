using TensorTrainNumerics
using CairoMakie

d     = 8       # N = 128 grid points
L     = 25.0    # domain [0, L]: wide enough that the soliton is ~ 0 at both ends
T_end = 1.0
Nt    = 250
c     = 1.0     # soliton speed  (amplitude = c/2 = 0.5)

N   = 2^d
dx  = L / (N - 1)
dt  = T_end / Nt

# Periodic centered difference operators scaled for physical spacing
D_x   = (1 / (2dx))   * ∇_c_P(d)
D_xxx = (1 / (2dx^3)) * ∇3_P(d)

x₀  = 9.0
u₀  = function_to_qtt(x -> (c/2) * sech(sqrt(c)/2 * (x - x₀))^2, d; b = L)
xes = (0:N-1) ./ (N-1) .* L
tes = (0:Nt) .* dt

sol = kdv_als(u₀, D_x, D_xxx, dt, Nt;
    max_scf = 25, scf_tol = 1e-8, max_bond = 50, verbose_steps = true)
println("max_rank = $(maximum(sol[end].ttv_rks))")

# Analytical 1-soliton at t = T_end  (travels from x₀ to x₀ + c*T_end)
u_exact = function_to_qtt(x -> (c/2) * sech(sqrt(c)/2 * (x - x₀ - c*T_end))^2, d; b = L)
err = norm(sol[end] - u_exact) / norm(u_exact)
println("relative L² error = $(round(err, sigdigits=3))")

# Build space × time solution matrix: N × (Nt+1)
U = reduce(hcat, qtt_to_function(u) for u in sol)

let
    fig = Figure(size = (900, 500))

    ax1 = Axis(fig[1, 1], xlabel = "x", ylabel = "t",
               title = "KdV soliton  (c = $c,  T = $T_end,  L = $L,  N = $N)")
    hm  = heatmap!(ax1, xes, tes, U, colormap = :viridis)
    Colorbar(fig[1, 2], hm, label = "u")

    ax2 = Axis(fig[2, 1:2], xlabel = "x", ylabel = "u(x, t)")
    lines!(ax2, xes, qtt_to_function(u_exact), label = "exact  t = $T_end",
           linewidth = 2, color = :black, linestyle = :dash)
    lines!(ax2, xes, U[:, end],               label = "CN-MALS  t = $T_end",
           linewidth = 2, color = :blue)
    lines!(ax2, xes, U[:, 1],                 label = "t = 0",
           linewidth = 2, color = :gray, linestyle = :dot)
    axislegend(ax2)

    display(fig)
end
