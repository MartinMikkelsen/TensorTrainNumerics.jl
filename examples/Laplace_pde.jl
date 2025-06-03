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

Δ = toeplitz_to_qtto(α, β, γ, cores) 
A = Δ ⊗ id_tto(cores) + id_tto(cores) ⊗ Δ

b = qtt_cos(cores) ⊗ qtt_basis_vector(cores, 1) + qtt_sin(cores) ⊗ qtt_basis_vector(cores, 2^cores) 

initial_guess = rand_tt(b.ttv_dims, b.ttv_rks)

x_mals = mals_linsolv(A, b, initial_guess)
x_dmrg = dmrg_linsolve(A, b, initial_guess)
x_als = als_linsolv(A, b, initial_guess;
                     sweep_count = 2,
                     it_solver   = true,
                     r_itsolver  = 5000)

solution = reshape(qtt_to_function(x_mals), 2^cores, 2^cores)

fig = Figure()
cmap = :roma
ax = Axis(fig[1, 1], title = "Laplace Solution", xlabel = "x", ylabel = "y")
hm = heatmap!(ax, xes, xes, solution; colormap = cmap)
Colorbar(fig[1, 2], hm, label = "u(x, y)")
fig


