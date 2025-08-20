# TensorTrainNumerics.jl

TensorTrainNumerics.jl is a Julia package for efficient numerical methods using **tensor trains (TT)** and **quantics tensor trains (QTT)**. This package enables scalable computation on high-dimensional data and functions by leveraging the low-rank structure inherent in many practical problems.

## Introduction to Quantics Tensor Trains

### Tensor Trains
A tensor train represents a high-dimensional tensor as a product of smaller tensors called "cores":
```
A[i₁, i₂, ..., iₐ] = Σ G₁[i₁, r₁] G₂[r₁, i₂, r₂] ... Gₐ[rₐ₋₁, iₐ]
```
This format can dramatically reduce storage from exponential to linear in the number of dimensions.

### Quantics Tensor Trains (QTT)
QTT extends tensor trains by using **binary index decomposition**. Instead of working with a vector of size N, we:

1. **Decompose indices**: Write N = 2^d and represent each index as d binary digits
2. **Reshape**: Transform a vector v[1:2^d] into a d-dimensional tensor T[1:2, 1:2, ..., 1:2]
3. **Apply TT**: Use tensor train decomposition on this binary tensor

**Key insight**: Functions on regular grids often have low QTT rank, enabling exponential compression.

### Why QTT Works
Many functions and operators have natural structure in binary representation:
- **Smooth functions**: Local correlations compress well
- **Differential operators**: Sparse structure translates to low rank
- **Hierarchical data**: Binary trees map naturally to QTT

This makes QTT particularly effective for:
- Solving PDEs on fine grids (2^20+ points)
- High-dimensional function approximation
- Fast transforms and convolutions

## Key features

**Quantics Tensor Trains (QTT):**
- **Function representation**: Convert continuous functions to QTT format and back
- **Built-in functions**: Pre-implemented QTT representations for trigonometric, exponential, and polynomial functions
- **PDE solving**: Efficient discretization and solution of partial differential equations
- **Automatic compression**: Functions are automatically compressed to optimal rank

**Tensor Train Operations:**
- **Decomposition**: Efficient algorithms for converting full tensors to TT format with controllable precision
- **Arithmetic**: Support for addition, multiplication, and contraction operations in TT format
- **Discrete operators**: Implementation of differential operators (Laplacians, gradients) in TT format
- **Optimization**: ALS, MALS, and DMRG algorithms for solving linear systems and eigenvalue problems

**Advanced Features:**
- **Iterative solvers**: Integration with modern iterative methods for large-scale problems
- **Visualization**: Tools for understanding TT structure, ranks, and compression efficiency
- **Memory management**: Automatic rank control and efficient storage 

## Getting started 

To get started with TensorTrainNumerics.jl, install the package using Julia's package manager:

```Julia
using Pkg
Pkg.add("TensorTrainNumerics")
```

### Your First QTT Example

Let's represent a function using QTT and see the compression in action:

```@example 1
using TensorTrainNumerics

# Represent sin(2πx) on [0,1] using QTT with 2^10 = 1024 points
d = 10
qtt_sine = qtt_sin(d, λ=2)

# Check the compression: how many parameters does this use?
println("QTT ranks: ", qtt_sine.ttv_rks)
total_params = sum(prod(size(core)) for core in qtt_sine.ttv_vec)
println("Total QTT parameters: ", total_params)
println("Original vector size: ", 2^d)
println("Compression ratio: ", 2^d / total_params)
```

The QTT representation uses far fewer parameters than storing the full vector!

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