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
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20514254.svg)](https://doi.org/10.5281/zenodo.20514254)

`TensorTrainNumerics.jl` is a Julia package for numerical methods in tensor train (TT) and quantics tensor train (QTT) format. It provides iterative solvers for high-dimensional linear systems and eigenvalue problems, time-stepping and TDVP methods for evolution equations, and TT-cross algorithms for black-box function approximation — all operating on compressed tensor representations that scale linearly in the number of dimensions rather than exponentially.

## Features

- **Solvers** — ALS, MALS [[1](#references)], and DMRG for linear systems [[2](#references)], non-linear (multigrid) [[3](#references)] and eigenvalue problems; adaptive rank control via SVD truncation [[4](#references)]
- **Time evolution** — single- and two-site TDVP [[5](#references)], implicit Euler, Crank–Nicolson, and Krylov exponential integrators 
- **TT-cross** — MaxVol [[6](#references)], DMRG-cross [[7](#references)], and Greedy algorithms [[8](#references)] for black-box function approximation and numerical integration [[9](#references)]
- **QTT operators** — exact low-rank representations of Laplacians, gradient operators [[10](#references)], shift matrices, and the discrete Fourier transform [[11](#references)]
- **Quantics tensor trains** — serial and interleaved multi-dimensional encodings with `QTTvector`/`QTToperator` wrappers
- **Interoperability** — compatible with [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl) and [OptimKit.jl](https://github.com/Jutho/OptimKit.jl) via [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl)


## Installation

```julia
using Pkg
Pkg.add("TensorTrainNumerics")
```

## Quick start

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
```julia
Relative error: 2.661496213238571e-16
```

### 3. Solve a linear system in QTT format

```julia
using LinearAlgebra
using TensorTrainNumerics

d = 6
A = id_tto(d)
b = qtt_sin(d, λ = π)
x0 = rand_tt(b.ttv_dims, b.ttv_rks)

x = linear_solve(A, b, x0, ALS(sweep_count = 4))

rhs = qtt_to_function(b)
sol = qtt_to_function(x)
relerr = norm(sol - rhs) / norm(rhs)

println("Relative error: ", relerr)
```
```julia
Relative error: 4.560872651853784e-16
```

For more examples including 2D PDEs, time evolution, and the QTT Fourier transform, see the [documentation](https://martinmikkelsen.github.io/TensorTrainNumerics.jl/).

### References

[1] S. Holtz, T. Rohwedder, R. Schneider, "The Alternating Linear Scheme for Tensor Optimization in the Tensor Train Format", SIAM Journal on Scientific Computing 34 (2) (2012)

[2] Oseledets, Ivan V., and Sergey V. Dolgov. "Solution of linear systems and matrix inversion in the TT-format." SIAM Journal on Scientific Computing 34.5 (2012): A2718-A2739.

[3] Lubasch, Michael, Pierre Moinier, and Dieter Jaksch. "Multigrid renormalization." Journal of Computational Physics 372 (2018): 587-602.

[4] Oseledets, Ivan V. "Tensor-train decomposition." SIAM Journal on Scientific Computing 33.5 (2011): 2295-2317.

[5] Haegeman, Jutho, et al. "Unifying time evolution and optimization with matrix product states." Physical Review B 94.16 (2016): 165116.

[6] I. Oseledets, E. Tyrtyshnikov, "TT-cross approximation for multidimensional arrays", Linear Algebra and its Applications 432 (1) (2010)

[7] Savostyanov, Dmitry, and Ivan Oseledets. "Fast adaptive interpolation of multi-dimensional arrays in tensor train format." The 2011 International Workshop on Multidimensional (nD) Systems. IEEE, 2011.

[8] Savostyanov, Dmitry V. "Quasioptimality of maximum-volume cross interpolation of tensors." Linear Algebra and its Applications 458 (2014): 217-244.

[9] Vysotsky, Lev I., Alexander V. Smirnov, and Eugene E. Tyrtyshnikov. "Tensor-train numerical integration of multivariate functions with singularities." Lobachevskii Journal of Mathematics 42.7 (2021): 1608-1621.

[10] Kazeev, Vladimir A., and Boris N. Khoromskij. "Low-rank explicit QTT representation of the Laplace operator and its inverse." SIAM journal on matrix analysis and applications 33.3 (2012): 742-758.

[11] Chen, Jielun, and Michael Lindsey. "Direct interpolative construction of the discrete Fourier transform as a matrix product operator." Applied and Computational Harmonic Analysis (2025): 101817.