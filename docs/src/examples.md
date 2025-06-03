# Examples

Let's consider the following 2D partial differential equation 
```math
\Delta u(x,y) = f(x,y),
```
where $\Delta$ is the Laplacian operator, ``u(x,y)`` is the unknown function, and ``f(x,y)`` is a given function. In this example we assume ``f(x,y)=0`` using Dirichlet-Dirichlet boundary conditions given by ``u(0,y) = \cos(\pi y)``, ``u(1,y) = \sin(y)``.

We want to solve this equation using quantics tensor trains (QTTs). 
We start by defining the dimensions and the resolution of the grid. Lets say we want ``2^{10}`` points in each dimension, which gives us a grid of ``1024 \times 1024`` points on a ``[0,1]\times [0,1]`` grid. 

```@example Laplace
using TensorTrainNumerics
using CairoMakie

cores = 10
a = 0.0
b = 1.0
```
We follow the same convention as in [this paper](https://arxiv.org/pdf/2505.17046) where we define the finite difference operator using the following inputs
```@example Laplace
h = (b-a)/(2^cores)
p = 1.0
s = 0.0
v = 0.0
α = h^2*v-2*p
β = p + h*s/2
γ = p-h*s/2

Δ = toeplitz_to_qtto(α, β, γ, cores) 
```
To get the 2D Laplacian operator, we need to take the Kronecker product of the 1D Laplacian operator with the identity. 
```@example Laplace
A = Δ ⊗ id_tto(cores) + id_tto(cores) ⊗ Δ
```
To build the boundary vector we take the Kronecker product with the QTT basis vectors and define some random initial guess for the solution. 
```@example Laplace
b = qtt_cos(cores) ⊗ qtt_basis_vector(cores, 1) + qtt_sin(cores) ⊗ qtt_basis_vector(cores, 2^cores) 
initial_guess = rand_tt(b.ttv_dims, b.ttv_rks)
```
We solve the linear system using DMRG
```@example Laplace
x_dmrg = dmrg_linsolve(A, b, initial_guess; sweep_count=50,tol=1e-15)
```
And we reshape the solution to a 2D array for visualization
```@example Laplace

solution = reshape(qtt_to_function(x_dmrg), 2^cores, 2^cores)
xes = collect(range(0,1,length=2^cores))
yes = collect(range(0,1,length=2^cores))
fig = Figure()
cmap = :roma
ax = Axis(fig[1, 1], title = "Laplace Solution", xlabel = "x", ylabel = "y")
hm = heatmap!(ax, xes, yes, solution; colormap = cmap)
Colorbar(fig[1, 2], hm, label = "u(x, y)")
fig
```
