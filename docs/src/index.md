# TensorTrainNumerics.jl

Tensor Train Numerics is a Julia package designed to provide efficient numerical methods for working with tensor trains (TT) and quantized tensor trains (QTT). This package offers a comprehensive set of tools for constructing, manipulating, and performing operations on tensor trains, which are useful in various scientific and engineering applications, including high-dimensional data analysis, machine learning, and computational physics.

## Key features

- Tensor Train Decomposition: Efficient algorithms for decomposing high-dimensional tensors into tensor train format, reducing computational complexity and memory usage.
- Tensor Operations: Support for basic tensor operations such as addition, multiplication, and contraction in tensor train format.
- Discrete Operators: Implementation of discrete Laplacians, gradient operators, and shift matrices in tensor train format for solving partial differential equations and other numerical problems.
- Quantized Tensor Trains: Tools for constructing and manipulating quantized tensor trains, which provide further compression and efficiency for large-scale problems.
- Iterative Solvers: Integration with iterative solvers for solving linear systems and eigenvalue problems in tensor train format.
- Visualization: Basic visualization tools for inspecting tensor train structures and their properties. 

## Getting started 

To get started with Tensor Train Numerics, you can install the package using Julia's package manager:

```Julia
using Pkg
Pkg.add("TensorTrainNumerics")
```

### Basic example

```@example 1
using TensorTrainNumerics

# Define the dimensions and ranks for the TTvector
dims = (2, 2, 2)
rks = [1, 2, 2, 1]

# Create a random TTvector
tt_vec = rand_tt(dims, rks)

# Define the dimensions and ranks for the TToperator
op_dims = (2, 2, 2)
op_rks = [1, 2, 2, 1]

# Create a random TToperator
tt_op = rand_tto(op_dims, 3)

# Perform the multiplication
result = tt_op * tt_vec

# Visualize the result

visualize(result)
```
And we can print the result
```@example 1
println(result)
```
We can also unfold this
```@example 1
matricize(result, 3)
```
### Interpolation

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

### Functions

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