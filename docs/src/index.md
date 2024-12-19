# TensorTrainNumerics.jl

## Work in progress

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
matricize(result)
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