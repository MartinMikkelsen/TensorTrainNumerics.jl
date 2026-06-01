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

`TensorTrainNumerics.jl` is a Julia package for numerical methods in tensor train (TT) and quantics tensor train (QTT) format. It provides iterative solvers for high-dimensional linear systems and eigenvalue problems, time-stepping and TDVP methods for evolution equations, and TT-cross algorithms for black-box function approximation — all operating on compressed tensor representations that scale linearly in the number of dimensions rather than exponentially.

## Overview

Instead of storing all $\prod_{k=1}^d n_k$ entries of a $d$-dimensional tensor, the TT format represents it as a chain of small cores whose contraction recovers the full tensor. A **TT-vector** $v \in \mathbb{K}^{n_1 \times \cdots \times n_d}$ is stored as third-order cores $A_k \in \mathbb{K}^{r_{k-1} \times n_k \times r_k}$:

$$v(i_1,\dots,i_d) = \sum_{r_1,\ldots,r_{d-1}} \prod_{k=1}^d A_k(r_{k-1}, i_k, r_k)$$

$$\begin{array}{ccccccc}
A_{1} & \;\text{---}\; r_1 \;\text{---}\; & A_{2} & \;\text{---}\; r_2 \;\text{---}\; & A_{3} & \;\text{---}\;\cdots\;\text{---}\; & A_{d} \\
| & & | & & | & & | \\
{\small i_1} & & {\small i_2} & & {\small i_3} & & {\small i_d}
\end{array}$$

A **TT-operator** $A \in \mathbb{K}^{(n_1\cdots n_d)\times(n_1\cdots n_d)}$ uses fourth-order cores $A_k \in \mathbb{K}^{r_{k-1} \times n_k \times n_k \times r_k}$ with two physical indices per site:

$$\begin{array}{ccccccc}
{\small j_1} & & {\small j_2} & & {\small j_3} & & {\small j_d} \\
| & & | & & | & & | \\
A_{1} & \;\text{---}\; r_1 \;\text{---}\; & A_{2} & \;\text{---}\; r_2 \;\text{---}\; & A_{3} & \;\text{---}\;\cdots\;\text{---}\; & A_{d} \\
| & & | & & | & & | \\
{\small i_1} & & {\small i_2} & & {\small i_3} & & {\small i_d}
\end{array}$$

Storage scales as $\mathcal{O}(dR^2n)$ instead of $\mathcal{O}(n^d)$, where $R = \max_k r_k$ is the maximum bond dimension.

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

x = als_linsolve(A, b, x0; sweep_count = 4)

rhs = qtt_to_function(b)
sol = qtt_to_function(x)
relerr = norm(sol - rhs) / norm(rhs)

println("Relative error: ", relerr)
```
```julia
Relative error: 4.560872651853784e-16
```

For more examples including 2-D PDEs, time evolution, and the QTT Fourier transform, see the [documentation](https://martinmikkelsen.github.io/TensorTrainNumerics.jl/).

## Features

- **Solvers** — ALS, MALS, and DMRG for linear systems and eigenvalue problems; adaptive rank control via SVD truncation
- **Time evolution** — single- and two-site TDVP, implicit Euler, Crank–Nicolson, and Krylov exponential integrators
- **TT-cross** — MaxVol, DMRG-cross, and Greedy algorithms for black-box function approximation and numerical integration
- **QTT operators** — exact low-rank representations of Laplacians, gradient operators, shift matrices, and the discrete Fourier transform
- **Quantics tensor trains** — serial and interleaved multi-dimensional encodings with `QTTvector`/`QTToperator` wrappers
- **Interoperability** — compatible with [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl) and [OptimKit.jl](https://github.com/Jutho/OptimKit.jl) via [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl)
