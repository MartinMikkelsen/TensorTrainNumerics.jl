using TensorTrainNumerics
using CairoMakie

d = 8

xes = collect(range(0.0, 1.0, length = 2^d))

h = 1 / (2^d)
p = 1.0
s = 0.0
v = 0.0
α = h^2 * v - 2 * p
β = p + h * s / 2
γ = p - h * s / 2

Δ = toeplitz_to_qtto(α, β, γ, d)
A = Δ ⊗ id_tto(d) + id_tto(d) ⊗ Δ

b = qtt_cos(d) ⊗ qtt_basis_vector(d, 1) + qtt_sin(d) ⊗ qtt_basis_vector(d, 2^d)

initial_guess = rand_tt(b.ttv_dims, b.ttv_rks)

x_mals = mals_linsolve(A, b, initial_guess)
x_dmrg = dmrg_linsolve(A, b, initial_guess)

solution = reshape(qtt_to_function(x_mals), 2^d, 2^d)

let
    fig = Figure()
    cmap = :roma
    ax = Axis(fig[1, 1], title = "Laplace Solution", xlabel = "x", ylabel = "y")
    hm = heatmap!(ax, xes, xes, solution; colormap = cmap)
    Colorbar(fig[1, 2], hm, label = "u(x, y)")
    fig
end
