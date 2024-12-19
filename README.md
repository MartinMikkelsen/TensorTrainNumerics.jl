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