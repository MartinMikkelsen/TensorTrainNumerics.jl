using TensorTrainNumerics
using GLMakie, ColorSchemes

cores = 6

xes = collect(range(0.0,1.0,length=2^cores))

h = 1/(2^cores)
p = 1.0
s = 0.0
v = 0.0
α = h^2*v-2*p
β = p + h*s/2
γ = p-h*s/2

Δ = toeplitz_to_qtto(α, β, γ, cores) 
A = Δ ⊗ id_tto(cores) ⊗ id_tto(cores) + id_tto(cores) ⊗ Δ ⊗ id_tto(cores) + id_tto(cores) ⊗ id_tto(cores) ⊗ Δ

b = qtt_cos(cores) ⊗ qtt_basis_vector(cores, 1) ⊗ qtt_basis_vector(cores, 1) + qtt_sin(cores) ⊗ qtt_basis_vector(cores, 2^cores) ⊗ qtt_basis_vector(cores, 2^cores) 

initial_guess = rand_tt(b.ttv_dims, b.ttv_rks)

x_mals = mals_linsolv(A, b, initial_guess)

solution = reshape(qtt_to_function(x_mals), 2^cores, 2^cores, 2^cores)


@show minimum(solution)
@show maximum(solution)
@show sum(abs.(solution) .> 1e-8)

fig, ax, _ = GLMakie.volume(xes[1]..xes[end], xes[1]..xes[end], xes[1]..xes[end], solution;
    colorrange = (minimum(solution), maximum(solution)),
    colormap = :Egypt, transparency = true,
    figure = (; size = (1200, 800)),
    axis = (;
        type = Axis3,
        perspectiveness = 0.5,
        azimuth = 2.19,
        elevation = 0.57,
        aspect = (1, 1, 1)
        )
    )
fig
save("laplace3d.png", fig)

