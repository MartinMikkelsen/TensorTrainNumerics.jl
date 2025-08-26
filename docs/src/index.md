# TensorTrainNumerics.jl

Tensor Train Numerics is a Julia package designed to provide efficient numerical methods for working with tensor trains (TT) and quantized tensor trains (QTT). 

## Key features

- Tensor Train Decomposition: Efficient algorithms for decomposing high-dimensional tensors into tensor train format [tt_Oseledets, Schollwock_2011](@cite).
- Tensor Operations: Support for basic tensor operations such as addition, multiplication, and contraction in tensor train format [TensorOperations](@cite).
- Discrete Operators: Implementation of discrete Laplacians, gradient operators, and shift matrices in tensor train format for solving partial differential equations [kazeev2012, Khoromskij, qtt_khoromskij](@cite).
- Quantized Tensor Trains: Tools for constructing and manipulating quantized tensor trains, which provide further compression and efficiency for large-scale problems.
- Iterative Solvers: Integration with iterative solvers for solving linear systems and eigenvalue problems in tensor train format [White, Haegeman_2016, Vanderstraeten_2019, als_mals](@cite).
- The Fourier transform in QTT format and interpolation in QTT format [lindsey2023multiscale, Dolgov2012, QFT1, QFT2](@cite).
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

