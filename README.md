<div align="left">
    <picture>
      <img alt="TensorTrainNumerics.jl" src="https://raw.githubusercontent.com/MartinMikkelsen/TensorTrainNumerics.jl/main/docs/src/assets/logo.svg">
    </picture>
</div>


# TensorTrainNumerics.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://martinmikkelsen.github.io/TensorTrainNumerics.jl/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://martinmikkelsen.github.io/TensorTrainNumerics.jl/dev)
[![CI](https://github.com/MartinMikkelsen/TensorTrainNumerics.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/MartinMikkelsen/TensorTrainNumerics.jl/actions/workflows/ci.yml)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![codecov](https://codecov.io/gh/MartinMikkelsen/TensorTrainNumerics.jl/graph/badge.svg?token=p7OsfklHWr)](https://codecov.io/gh/MartinMikkelsen/TensorTrainNumerics.jl)

Tensor Train Numerics is a Julia package designed to provide efficient numerical methods for working with tensor trains (TT) and quantized tensor trains (QTT). 

## Getting started 

To get started with Tensor Train Numerics, you can install the package using Julia's package manager:

```Julia
using Pkg
Pkg.add("TensorTrainNumerics")
```

## Quickstart

### 1. Decompose a dense tensor into TT format

```julia
using LinearAlgebra
using TensorTrainNumerics

tensor = reshape(collect(1.0:16.0), 2, 2, 2, 2)
tt = ttv_decomp(tensor; tol = 1.0e-12)

tensor_reconstructed = ttv_to_tensor(tt)
relerr = norm(tensor - tensor_reconstructed) / norm(tensor)

println("Relative error: ", relerr)
```
which returns
```julia
Relative error: 4.447195710046155e-16
```

### 2. Build a TT approximation from function evaluations with `tt_cross`

```julia
using LinearAlgebra
using TensorTrainNumerics

f(X) = vec(exp.(-sum(X .^ 2, dims = 2)))
domain = [collect(range(-1.0, 1.0, length = 8)) for _ in 1:4]

tt = tt_cross(f, domain, MaxVol(verbose = false, tol = 1.0e-8); ranks = 2)

approx = ttv_to_tensor(tt)
exact = similar(approx)
for I in CartesianIndices(exact)
    x = reshape([domain[k][I[k]] for k in 1:4], 1, :)
    exact[I] = f(x)[1]
end

relerr = norm(approx - exact) / norm(exact)
println(tt)
println("Relative error: ", relerr)
```
which returns
```julia
Relative error: 2.661496213238571e-16
```
### 3. Solve a small linear system in QTT format

```julia
using LinearAlgebra
using TensorTrainNumerics

d = 6
A = id_tto(d)
b = qtt_sin(d, λ = π)
x0 = rand_tt(b.ttv_dims, b.ttv_rks)

x = als_linsolve(A, b, x0; sweep_count = 4)

rhs = qtt_to_function(b)
sol = qtt_to_function(x)
relerr = norm(sol - rhs) / norm(rhs)

println("Relative error: ", relerr)
```
which returns 
```julia
Relative error: 4.560872651853784e-16
```
## Key features

- Tensor Train Decomposition: Algorithms for decomposing high-dimensional tensors into tensor train format 
- Tensor Operations: Support for basic tensor operations such as addition, multiplication, the hadamard product in tensor train format 
- Discrete Operators: Implementation of discrete Laplacians, gradient operators, and shift matrices in tensor train format for solving partial differential equations 
- TT-cross algorithm for approximating tensors from function calls
- Quantized Tensor Trains: Tools for constructing and manipulating quantized tensor trains, which provide further compression and efficiency for large-scale problems.
- Iterative Solvers: Integration with iterative solvers for solving linear systems and eigenvalue problems in tensor train format
- The Fourier transform in QTT format and interpolation in QTT format 
- Visualization: Basic visualization tools. 

### Acknowledgements 

Many of the features are inspired by the work of [Mi-Song Dupuy](https://github.com/msdupuy)
