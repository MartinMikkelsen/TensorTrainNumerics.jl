using TensorTrainNumerics
using CairoMakie
using Logging

d = 8
N = 2^d

xes = collect(range(0.0, 1.0, length = N))

h = 1 / (N - 1)
p = 1.0
s = 0.0
v = 0.0
α = h^2 * v - 2 * p
β = p + h * s / 2
γ = p - h * s / 2

Δ = toeplitz_to_qtto(α, β, γ, d)
A = Δ ⊗ id_tto(d) + id_tto(d) ⊗ Δ

b = qtt_cos(d) ⊗ qtt_basis_vector(d, 1) + qtt_sin(d) ⊗ qtt_basis_vector(d, N)
initial_guess = rand_tt(b.ttv_dims, b.ttv_rks)

println("Solving the 2D Laplace system on a $(N) x $(N) grid")

x_mals, mals_info, x_dmrg, dmrg_info = with_logger(NullLogger()) do
    x_mals, mals_info = mals_linsolve(A, b, initial_guess; return_info = true)
    x_dmrg, dmrg_info = dmrg_linsolve(A, b, initial_guess; return_info = true)
    return x_mals, mals_info, x_dmrg, dmrg_info
end

solution_mals = reshape(qtt_to_vector(x_mals), N, N)
solution_dmrg = reshape(qtt_to_vector(x_dmrg), N, N)
solver_difference = solution_mals - solution_dmrg
rel_solver_difference = norm(solver_difference) / max(norm(solution_mals), eps())

println("MALS relative residual: $(mals_info.residual)")
println("DMRG relative residual: $(dmrg_info.residual)")
println("Relative MALS/DMRG difference: $(rel_solver_difference)")

let
    fig = Figure(size = (1500, 450))
    cmap = :roma

    ax_mals = Axis(fig[1, 1], title = "Laplace Solution (MALS)", xlabel = "x", ylabel = "y")
    hm_mals = heatmap!(ax_mals, xes, xes, solution_mals; colormap = cmap)
    Colorbar(fig[1, 2], hm_mals, label = "u(x, y)")

    ax_dmrg = Axis(fig[1, 3], title = "Laplace Solution (DMRG)", xlabel = "x", ylabel = "y")
    hm_dmrg = heatmap!(ax_dmrg, xes, xes, solution_dmrg; colormap = cmap)
    Colorbar(fig[1, 4], hm_dmrg, label = "u(x, y)")

    diff_scale = max(maximum(abs, solver_difference), eps())
    ax_diff = Axis(fig[1, 5], title = "MALS - DMRG", xlabel = "x", ylabel = "y")
    hm_diff = heatmap!(
        ax_diff,
        xes,
        xes,
        solver_difference;
        colormap = :balance,
        colorrange = (-diff_scale, diff_scale),
    )
    Colorbar(fig[1, 6], hm_diff, label = "difference")

    display(fig)
end
