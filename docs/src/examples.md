# Examples

## Partial differential equations 

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

## Time-stepping

We can also solve time-dependent PDEs using the QTT framework. In this exampl we will use the explicit Euler method, the implicit Euler method and the Crank-Nicolson scheme.
```@example TimeStepping
using TensorTrainNumerics
using CairoMakie

cores = 8
h = 1/cores^2
A = h^2*toeplitz_to_qtto(-2,1.0,1.0,cores)
xes = collect(range(0.0, 1.0, 2^cores))

u₀ = qtt_sin(cores,λ=π)
init = rand_tt(u₀.ttv_dims, u₀.ttv_rks)
steps = collect(range(0.0,10.0,1000))
solution_explicit, error_explicit = euler_method(A, u₀, steps; return_error=true)

solution_implicit, rel_implicit = implicit_euler_method(A, u₀, init, steps; return_error=true)

solution_crank, rel_crank = crank_nicholson_method(A, u₀, init, steps; return_error=true, tt_solver="mals")

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "Comparison of Time-Stepping Methods")
lines!(ax, xes, qtt_to_function(solution_explicit), label = "Explicit Euler", linestyle = :solid, linewidth=3)
lines!(ax, xes, qtt_to_function(solution_implicit), label = "Implicit Euler", linestyle = :dot, linewidth=3)
lines!(ax, xes, qtt_to_function(solution_crank), label = "Crank-Nicolson", linestyle = :dash, linewidth=3)
axislegend(ax)
fig
```
And print the relative errors
```@example TimeStepping
println("Relative error for explicit Euler: ", error_explicit)
println("Relative error for implicit Euler: ", rel_implicit)
println("Relative error for Crank-Nicolson: ", rel_crank)
```