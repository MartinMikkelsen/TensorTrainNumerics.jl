<div align="left">
    <picture>
      <img alt="TensorTrainNumerics.jl" src="https://raw.githubusercontent.com/MartinMikkelsen/TensorTrainNumerics.jl/main/docs/src/assets/logo.svg">
    </picture>
</div>


# TensorTrainNumerics.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://martinmikkelsen.github.io/TensorTrainNumerics.jl/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://martinmikkelsen.github.io/TensorTrainNumerics.jl/dev)
[![CI](https://github.com/MartinMikkelsen/TensorTrainNumerics.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/MartinMikkelsen/TensorTrainNumerics.jl/actions/workflows/ci.yml)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![codecov](https://codecov.io/gh/MartinMikkelsen/TensorTrainNumerics.jl/graph/badge.svg?token=p7OsfklHWr)](https://codecov.io/gh/MartinMikkelsen/TensorTrainNumerics.jl)

TensorTrainNumerics.jl is a Julia package for efficient numerical methods using **tensor trains (TT)** and **quantics tensor trains (QTT)**. This package enables scalable computation on high-dimensional data by leveraging the low-rank structure of tensor decompositions.

## What are Quantics Tensor Trains?

**Tensor Trains** are a format for representing high-dimensional tensors as a chain of smaller, interconnected tensors (called "cores"). This representation dramatically reduces memory requirements and computational complexity for many problems.

**Quantics Tensor Trains (QTT)** take this further by using a binary representation of indices. In QTT:
- Functions on grids with 2^d points are represented using only d binary indices
- Each dimension is decomposed into d binary (0/1) variables
- This enables exponential compression: a vector of size 2^20 ≈ 1 million can be efficiently represented and manipulated

### Why QTT?
- **Memory efficiency**: Store and manipulate vectors of size 2^d using only O(d) memory
- **Fast operations**: Matrix-vector products, function evaluations, and PDE solving scale as O(d) instead of O(2^d)
- **Natural for many problems**: PDEs, signal processing, and functions on regular grids benefit from this structure

## Getting started 

Install the package using Julia's package manager:

```Julia
using Pkg
Pkg.add("TensorTrainNumerics")
```

## Key features

**Quantics Tensor Trains (QTT):**
- **Function representation**: Convert functions to QTT format using `function_to_qtt()` and `qtt_to_function()`
- **Built-in functions**: Ready-to-use QTT representations for common functions (`qtt_sin`, `qtt_cos`, `qtt_exp`, polynomials)
- **PDE solving**: Efficient solution of partial differential equations using QTT-format operators
- **Interpolation**: High-accuracy function approximation with automatic rank adaptation

**Tensor Train Operations:**
- **Basic operations**: Addition, multiplication, and contraction in TT format
- **Decomposition**: Algorithms for converting full tensors to TT format with controllable accuracy
- **Discrete operators**: Laplacians, gradients, and other differential operators in TT format
- **Iterative solvers**: ALS, MALS, and DMRG algorithms for linear systems and eigenvalue problems

**Utilities:**
- **Visualization**: Tools for inspecting TT structure and ranks
- **Memory efficiency**: Automatic rank management and compression

## Quick Example

Here's how to represent and manipulate a function using QTT:

```julia
using TensorTrainNumerics

# Create QTT representation of sin(πx) on [0,1] with 2^8 = 256 points
d = 8  # Binary decomposition depth
qtt_sine = qtt_sin(d, λ=π)

# Convert back to function values
function_values = qtt_to_function(qtt_sine)

# The QTT representation is exponentially more compact:
# - Full vector: 2^8 = 256 values stored
# - QTT representation: ~8 small tensors with total size ≪ 256
```

For more examples, see the [documentation](https://martinmikkelsen.github.io/TensorTrainNumerics.jl/). 

### Acknowledgements 

Many of the features are inspired by the work of [Mi-Song Dupuy](https://github.com/msdupuy)
