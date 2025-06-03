using TensorTrainNumerics
using CairoMakie

cores = 10
a = 0.0
b = 1.0

xes = collect(range(a,b,length=2^cores))
yes = collect(range(a,b,length=2^cores))

h = (b-a)/(2^cores)
p = 1.0
s = 0.0
v = 0.0
α = h^2*v-2*p
β = p + h*s/2
γ = p-h*s/2

Δ = toeplitz_to_qtto(α, β, γ, cores) 
A = Δ ⊗ id_tto(cores) + id_tto(cores) ⊗ Δ

b = qtt_cos(cores) ⊗ qtt_basis_vector(cores, 1) + qtt_sin(cores) ⊗ qtt_basis_vector(cores, 2^cores) 

initial_guess = rand_tt(b.ttv_dims, b.ttv_rks)

x_dmrg = dmrg_linsolve(A, b, initial_guess; sweep_count=50,tol=1e-15)
solution = reshape(qtt_to_function(x_dmrg), 2^cores, 2^cores)

fig = Figure()
cmap = :roma
ax = Axis(fig[1, 1], title = "Laplace Solution", xlabel = "x", ylabel = "y")
hm = heatmap!(ax, xes, yes, solution; colormap = cmap)
Colorbar(fig[1, 2], hm, label = "u(x, y)")
fig


