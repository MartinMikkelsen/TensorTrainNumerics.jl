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

d = 10
a = 0.0
b = 1.0
```
We follow the same convention as in [FastandFlexible](@cite) where we define the finite difference operator using the following inputs
```@example Laplace
h = (b-a)/(2^d)
p = 1.0
s = 0.0
v = 0.0
α = h^2*v-2*p
β = p + h*s/2
γ = p-h*s/2

Δ = toeplitz_to_qtto(α, β, γ, d) 
```
To get the 2D Laplacian operator, we need to take the Kronecker product of the 1D Laplacian operator with the identity. 
```@example Laplace
A = Δ ⊗ id_tto(d) + id_tto(d) ⊗ Δ
```
To build the boundary vector we take the Kronecker product with the QTT basis vectors and define some random initial guess for the solution. 
```@example Laplace
b = qtt_cos(d) ⊗ qtt_basis_vector(d, 1) + qtt_sin(d) ⊗ qtt_basis_vector(d, 2^d) 
initial_guess = rand_tt(b.ttv_dims, b.ttv_rks)
```
We solve the linear system using DMRG
```@example Laplace
x_dmrg = dmrg_linsolve(A, b, initial_guess; sweep_count=50,tol=1e-15)
```
And we reshape the solution to a 2D array for visualization
```@example Laplace

solution = reshape(qtt_to_function(x_dmrg), 2^d, 2^d)
xes = collect(range(0,1,length=2^d))
yes = collect(range(0,1,length=2^d))
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

d = 8
h = 1/d^2
A = h^2*toeplitz_to_qtto(-2,1.0,1.0,d)
xes = collect(range(0.0, 1.0, 2^d))

u₀ = qtt_sin(d,λ=π)
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

Based on [QFT1](@cite) we also have access to the discrete Fourier transform (DFT) in QTT format. Below is an esample of how to use it. You can use the `fourier_qtto` function to create a QTT representation of the Fourier transform operator where the `sign` parameter determines if its the Fourier transform or the inverse Fourier transform. 

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

# Variational solver

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

# Interpolation

We can also do interpolation in the QTT framework:

```@example 2
using CairoMakie
using TensorTrainNumerics

f = x -> cos(1 / (x^3 + 0.01)) + sin(π * x)
num_cores = 10  
N = 150 

qtt = interpolating_qtt(f, num_cores, N)
qtt_rank_revealing = lagrange_rank_revealing(f, num_cores, N)

qtt_values = matricize(qtt, num_cores)
qtt_values_rank_revealing = matricize(qtt_rank_revealing, num_cores)

x_points = LinRange(0, 1, 2^num_cores)
original_values = f.(x_points)

fig = Figure()
ax = Axis(fig[1, 1], title="Function Approximation", xlabel="x", ylabel="f(x)")

lines!(ax, x_points, original_values, label="Original Function")
lines!(ax, x_points, qtt_values_rank_revealing, label="QTT, rank rev.", linestyle=:dash, color=:green)
lines!(ax, x_points, qtt_values, label="QTT", linestyle=:dash, color=:red)

axislegend(ax)
fig
```
We can visualize the interpolating QTT as 
```@example 2
visualize(qtt)
```
And similarly for the rank-revealing
```@example 2
visualize(qtt_rank_revealing)
```

# Functions

You can also use a low-rank representation of any trigonometric function and polynomial.

```@example 3
using TensorTrainNumerics
using CairoMakie

d = 8

A1 = qtt_exp(d)
A2 = qtt_sin(d, λ = π)
A3 = qtt_cos(d, λ = π)
A4 = qtt_polynom([0.0, 2.0, 3.0, -8.0, -5.0], d; a = 0.0, b = 1.0)


qtt_values_exponential = qtt_to_function(A1)
qtt_values_sin = qtt_to_function(A2)
qtt_values_cos = qtt_to_function(A3)
qtt_values_polynom = qtt_to_function(A4)


values_exp(x) = exp(x)
values_sin(x) = sin(x * π^2)
values_cos(x) = cos(x * π^2)
values_polynom(x) = 2 * x + 3 * x^2 - 8 * x^3 - 5 * x^4

x_points = LinRange(0, 1, 2^8)
original_values_exponential = values_exp.(x_points)
original_values_sin = values_sin.(x_points)
original_values_cos = values_cos.(x_points)
original_values_polynom = values_polynom.(x_points)

let
    fig = Figure()
    ax1 = Axis(fig[2, 2], title = "Exp Approximation", xlabel = "x", ylabel = "f(x)")
    ax2 = Axis(fig[1, 1], title = "Sin Approximation", xlabel = "x", ylabel = "f(x)")
    ax3 = Axis(fig[1, 2], title = "Cos Approximation", xlabel = "x", ylabel = "f(x)")
    ax4 = Axis(fig[2, 1], title = "Polynomial Approximation", xlabel = "x", ylabel = "f(x)")


    lines!(ax1, x_points, original_values_exponential, label = "Exponential function")
    lines!(ax1, x_points, qtt_values_exponential, label = "QTT exponential function", linestyle = :dash, color = :green)

    lines!(ax2, x_points, original_values_sin, label = "Sine function")
    lines!(ax2, x_points, qtt_values_sin, label = "QTT sine function", linestyle = :dash, color = :red)

    lines!(ax3, x_points, original_values_cos, label = "Sine function")
    lines!(ax3, x_points, qtt_values_cos, label = "QTT sine function", linestyle = :dash, color = :red)

    lines!(ax4, x_points, original_values_polynom, label = "Sine function")
    lines!(ax4, x_points, qtt_values_polynom, label = "QTT sine function", linestyle = :dash, color = :red)

    fig
end
```