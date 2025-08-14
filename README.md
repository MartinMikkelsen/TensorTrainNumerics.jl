# TensorTrainNumerics.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://martinmikkelsen.github.io/TensorTrainNumerics.jl/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://martinmikkelsen.github.io/TensorTrainNumerics.jl/dev)
[![CI](https://github.com/MartinMikkelsen/TensorTrainNumerics.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/MartinMikkelsen/TensorTrainNumerics.jl/actions/workflows/ci.yml)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![codecov](https://codecov.io/gh/MartinMikkelsen/TensorTrainNumerics.jl/graph/badge.svg?token=p7OsfklHWr)](https://codecov.io/gh/MartinMikkelsen/TensorTrainNumerics.jl)

Tensor Train Numerics is a Julia package designed to provide efficient numerical methods for working with tensor trains (TT) and quantized tensor trains (QTT). 

## Getting started 

To get started with Tensor Train Numerics, you can install the package using Julia's package manager:

```Julia
using Pkg
Pkg.add("TensorTrainNumerics")
```

## Key features

- Tensor Operations: Support for basic tensor operations such as addition, multiplication, and contraction in tensor train format.
- Tensor Train Decomposition: Algorithms for decomposing high-dimensional tensors into tensor train format, reducing computational complexity and memory usage.
- Discrete Operators: Implementation of discrete Laplacians, gradient operators, and shift matrices in tensor train format for solving partial differential equations and other numerical problems.
- Quantized Tensor Trains: Tools for constructing and manipulating quantized tensor trains, which provide further compression and efficiency for large-scale problems.
- Iterative Solvers: Integration with iterative solvers for solving linear systems and eigenvalue problems in tensor train format.
- Visualization: Basic visualization tools for inspecting tensor train structures and their properties. 

### Acknowledgements 

Many of the features are inspired by the work of [Mi-Song Dupuy](https://github.com/msdupuy)
