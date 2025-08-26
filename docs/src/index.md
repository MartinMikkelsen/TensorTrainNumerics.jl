# TensorTrainNumerics.jl

`TensorTrainNumerics.jl` is a Julia package designed to provide numerical methods for working with tensor trains (TT) and quantics tensor trains (QTT). 

There are [many packages](resources.md) for tensor decompositions and tensor network algorithms, but this package focuses on numerical methods and applications in scientific computing, such as solving high-dimensional partial differential equations (PDEs) and large-scale linear algebra problems. Thanks to [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl) this package is compatible with many Krylov methods from [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl) and optimization methods from [OptimKit.jl](https://github.com/Jutho/OptimKit.jl). So far this package includes implementations of discrete Laplace operators in the quantics tensor train format as well as iterative solvers for linear systems and eigenvalue problems. 

## Features

- Tensor Train Decomposition: Algorithms for decomposing high-dimensional tensors into tensor train format [tt_Oseledets, Schollwock_2011](@cite).
- Tensor Operations: Support for basic tensor operations such as addition, multiplication, the hadamard product in tensor train format [TensorOperations](@cite).
- Discrete Operators: Implementation of discrete Laplacians, gradient operators, and shift matrices in tensor train format for solving partial differential equations [kazeev2012, Khoromskij, qtt_khoromskij](@cite).
- Quantized Tensor Trains: Tools for constructing and manipulating quantized tensor trains, which provide further compression and efficiency for large-scale problems.
- Iterative Solvers: Integration with iterative solvers for solving linear systems and eigenvalue problems in tensor train format [White, Haegeman_2016, Vanderstraeten_2019, als_mals](@cite).
- The Fourier transform in QTT format and interpolation in QTT format [lindsey2023multiscale, Dolgov2012, QFT1, QFT2](@cite).
- Visualization: Basic visualization tools. 

## Getting started 

You can install the package using Julia's package manager:

```Julia
using Pkg
Pkg.add("TensorTrainNumerics")
```

For some examples check out the [examples page](examples.md) or the [examples folder on Github](https://github.com/MartinMikkelsen/TensorTrainNumerics.jl/tree/main/examples).