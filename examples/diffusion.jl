using TensorTrainNumerics
using CairoMakie

cores = 8

xes = collect(range(0.0,1.0,length=2^cores))

h = 1/(2^cores)
p = 1.0
s = 0.0
v = 0.0
α = h^2*v-2*p
β = p + h*s/2
γ = p-h*s/2

Δ = 1/h^2*toeplitz_to_qtto(2.0, -1, -1, cores) 
∇ = 2.0/h*toeplitz_to_qtto(3.0, 1.0, -5.0, cores)
Operator = Δ + ∇
A = Operator ⊗ id_tto(cores) + id_tto(cores) ⊗ Operator

b = qtt_cos(cores) ⊗ qtt_basis_vector(cores, 1) + qtt_sin(cores) ⊗ qtt_basis_vector(cores, 2^cores) 

initial_guess = rand_tt(b.ttv_dims, b.ttv_rks)

x_mals = mals_linsolve(A, b, initial_guess)

solution = reshape(qtt_to_function(x_mals), 2^cores, 2^cores)

let
    fig = Figure()
    cmap = :roma
    ax = Axis(fig[1, 1], title = "Diffusion Solution", xlabel = "x", ylabel = "y")
    hm = heatmap!(ax, xes, xes, solution; colormap = cmap)
    Colorbar(fig[1, 2], hm, label = "u(x, y)")
    fig
end
