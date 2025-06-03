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

solution = reshape(qtt_to_function(x_mals), 2^cores, 2^cores)

fig = Figure()
cmap = :roma
ax = Axis(fig[1, 1], title = "Laplace Solution", xlabel = "x", ylabel = "y")
hm = heatmap!(ax, xes, xes, solution; colormap = cmap)
Colorbar(fig[1, 2], hm, label = "u(x, y)")
fig

function square_euclidean_distance(a::TTvector{T,N}, b::TTvector{T,N}) where {T<:Number,N}
    @assert a.ttv_dims == b.ttv_dims "TT dimensions must match"
    return dot(a, a) - 2 * real(dot(b, a)) + dot(b, b)
end

function square_euclidean_distance_normalized(a::TTvector{T,N}, b::TTvector{T,N}) where {T<:Number,N}
    @assert a.ttv_dims == b.ttv_dims "TT dimensions must match"
    return 1.0 + dot(a,a) / dot(b,b) - 2.0 * real(dot(b,a))/dot(b,b)
end
