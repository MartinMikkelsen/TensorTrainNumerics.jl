# TensorTrainNumerics.jl

`TensorTrainNumerics.jl` is a Julia package for numerical computation with
tensors in the tensor-train (TT) format [tt_Oseledets, Schollwock_2011](@cite)
and its quantics variant (QTT) [Khoromskij, qtt_khoromskij](@cite). A TT
representation stores a tensor with `d` indices of size `n` using
`O(d n r²)` numbers, where `r` is the TT rank, instead of `nᵈ`. In the QTT
format a vector of length `2^d` becomes a TT with `d` binary sites, so smooth
functions on grids with billions of points can be stored and manipulated with a
few kilobytes of memory.

The package is aimed at scientific computing: solving high-dimensional or
very finely resolved linear systems, eigenvalue problems, and evolution
equations, and approximating functions from a limited number of samples. Many
tensor-network packages already exist (see [Resources](resources.md)); this
one collects TT solvers, QTT operator constructions, time integrators, and
cross interpolation behind one set of types.

## Features

- **TT and QTT data types.** [`TTvector`](@ref) and [`TToperator`](@ref), with
  multidimensional QTT wrappers [`QTTvector`](@ref) and [`QTToperator`](@ref)
  that support serial and interleaved bit orderings.
- **Decomposition and rank control.** TT-SVD decomposition of dense tensors,
  left/right canonical forms, TT rounding with a relative tolerance or a rank
  cap, and entanglement entropies across every bond.
- **Arithmetic.** Addition, scalar and operator products, inner products and
  norms, Hadamard (elementwise) and Kronecker products, and diagonal operators
  built from vectors [TensorOperations](@cite).
- **QTT functions and operators.** Low-rank QTT constructions of polynomials,
  trigonometric functions, exponentials, and Chebyshev polynomials; finite-
  difference Laplacians with Dirichlet, Neumann, and periodic boundary
  conditions and an explicit inverse Laplacian [kazeev2012](@cite); gradient
  and shift operators; prolongation operators between grids; and the discrete
  Fourier transform [lindsey2023multiscale, Dolgov2012, QFT1, QFT2](@cite).
- **Spin-chain Hamiltonians.** Ising, XY, XXZ, XXX, and XYZ Hamiltonians as
  low-rank TT operators.
- **Linear and eigenvalue solvers.** ALS and MALS [als_mals](@cite), one- and
  two-site DMRG [White](@cite), and Krylov methods from KrylovKit.jl with rank
  truncation, all through [`linear_solve`](@ref) and [`eigen_solve`](@ref).
- **Nonlinear problems.** A penalty-method ALS solver and multigrid
  renormalization [mgr](@cite) for Gross–Pitaevskii-type ground states.
- **Time evolution.** Explicit Euler, fourth-order Runge–Kutta, implicit Euler,
  and Crank–Nicolson steppers that accept any of the linear solvers, and one-
  and two-site TDVP in real and imaginary time
  [Haegeman_2016, Vanderstraeten_2019](@cite).
- **Cross interpolation.** MaxVol, DMRG-cross, and greedy TT-cross algorithms
  that build a TT from function evaluations, and high-dimensional quadrature
  built on them [SAVOSTYANOV2014217, 6076873, vysotsky2021tensor](@cite).
- **Interoperability.** `TTvector` implements the
  [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl) interface,
  so it can be used directly with
  [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl) and
  [OptimKit.jl](https://github.com/Jutho/OptimKit.jl). Package extensions add
  reverse-mode differentiation rules for ChainRulesCore.jl (and hence Zygote.jl),
  Riemannian optimization with Manopt.jl, and conversion from
  TensorCrossInterpolation.jl.

## Installation

```julia
using Pkg
Pkg.add("TensorTrainNumerics")
```

## Where to go next

- [Tensor Train Basics](theory.md) introduces the TT format and the basic
  operations.
- [Quantics Tensor Trains](qtt.md) explains the QTT encoding of functions and
  operators on grids.
- [Solvers](solvers.md) describes the linear, eigenvalue, and time-stepping
  solvers and how to choose between them.
- [Examples](examples.md) and [Advanced Examples](advanced_examples.md) solve
  complete problems.
- The API reference documents every exported function and type:
  [core types and operations](api/core.md),
  [quantics and operators](api/qtt.md), and
  [solvers and algorithms](api/solvers.md).
