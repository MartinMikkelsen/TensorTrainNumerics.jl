# Examples

This section demonstrates key applications of Quantics Tensor Trains (QTT) for numerical computing. Each example showcases how QTT enables efficient computation on problems that would be intractable with traditional methods.

## Partial differential equations 

This example demonstrates solving a 2D Laplace equation using QTT. The key insight is that even though we're working on a 1024×1024 grid (over 1 million unknowns), QTT compression makes the problem tractable.

### Problem Setup
Let's consider the 2D Laplace equation:
```math
\Delta u(x,y) = f(x,y)
```
where $\Delta$ is the Laplacian operator, ``u(x,y)`` is the unknown function, and ``f(x,y)`` is a given function. We use ``f(x,y)=0`` with Dirichlet boundary conditions: ``u(0,y) = \cos(\pi y)`` and ``u(1,y) = \sin(y)``.

### QTT Approach
Instead of forming a million×million matrix, we:
1. **Discretize using QTT**: Represent the grid using binary indices
2. **Build operators in QTT**: Create Laplacian operators directly in QTT format
3. **Solve efficiently**: Use QTT-aware solvers that exploit the compressed structure

The solution process scales as O(d) rather than O(2^d), where d = 10 for our 1024×1024 grid. 

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

Time-dependent PDEs showcase another strength of QTT: the ability to evolve high-dimensional states efficiently over time. Here we compare three time-stepping schemes, all operating in the compressed QTT format.

### The Heat Equation
We solve the 1D heat equation:
```math
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
```
with initial condition u(x,0) = sin(πx). In QTT format, both the solution vector and the Laplacian operator are compressed, making time evolution efficient.

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

# Discrete Fourier Transform

The Fast Fourier Transform (FFT) can be represented in QTT format, enabling efficient Fourier analysis of exponentially large datasets. This is particularly useful for spectral methods and signal processing applications.

### QTT-FFT Advantages
- **Memory efficiency**: Transform vectors of size 2^20 using O(d) memory
- **Structured computation**: Exploits the recursive structure of the FFT algorithm
- **High accuracy**: Maintains numerical precision while compressing intermediate results

Based on [this paper](https://arxiv.org/pdf/2404.03182) we also have access to the discrete Fourier transform (DFT) in QTT format. Below is an esample of how to use it. You can use the `fourier_qtto` function to create a QTT representation of the Fourier transform operator where the `sign` parameter determines if its the Fourier transform or the inverse Fourier transform. 

```@example DFT
using TensorTrainNumerics   
using Random 

d = 10
N = 2^d
K = 50
sign = -1.0
normalize = true

Random.seed!(1234)
r = 12
coeffs = randn(r) .+ 1im * randn(r)

f(x) = sum(coeffs .* cispi.(2 .* (0:(r - 1)) .* x))

F = fourier_qtto(d; K = K, sign = -1.0, normalize = true)
x_qtt = function_to_qtt_uniform(f, d)
y_qtt = F * x_qtt
```

## Variational solver

You can also solve differential equations by optimizing a variational functional. Below is an example of how to use the `variational_solver` function to solve a simple differential equation based on - [OptimKit.jl](https://github.com/Jutho/OptimKit.jl) 

```@example VariationalSolver
using TensorTrainNumerics
using OptimKit
using KrylovKit

d = 6
N = 2^d
h = 1 / (N - 1)
Δ = toeplitz_to_qtto(2.0, -1.0, -1.0, d)
kappa = 0.1
A = kappa * Δ

f = qtt_sin(d, λ = π)

function fg(u::TTvector)
    û  = orthogonalize(u)           
    Au = A*û                       
    val = 0.5 * real(dot(û, Au)) - real(dot(f, û))
    grad = Au - f
    return val, grad
end

x0 = rand_tt(f.ttv_dims, f.ttv_rks)

method = GradientDescent()
x, fx, gx, numfg, normgradhistor = optimize(fg, x0, method)

relres = norm(A * x - f) / max(norm(f), eps())

x_mals = mals_linsolve(A, f, x0)
rel_residual = norm(A * x_mals - f) / max(norm(f), eps())

x_krylov, info = linsolve(A, f, x0)
relres_krylov = norm(A * x_krylov - f) / max(norm(f), eps())
```
with a relative residual of
```@example VariationalSolver
println("relative residual = ", relres)
```
Compared to the MALS solver
```@example VariationalSolver
println("relative residual MALS = ", rel_residual)
```
And the Krylov solver
```@example VariationalSolver
println("relative residual Krylov = ", relres_krylov)
```