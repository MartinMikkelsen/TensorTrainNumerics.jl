---
title: 'TensorTrainNumerics.jl: A Julia package for tensor train numerical methods'
tags:
  - julia
  - tensor methods
  - numerics
  - differential equations
authors:
  - name: Martin Mikkelsen
    orcid: 0009-0005-3932-6563
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: University of Copenhagen, Department of Computer Science, Denmark
   index: 1
date: 5 October 2026
bibliography: paper.bib

---

# Summary

Low-rank numerical tensor methods has gained a lot of traction recently and one of these frameworks is the tensor train formulation. There are already many software packages for tensor network methods but these focus primaily on physics applications. `TensorTrainNumerics.jl` offers a unified package for dealing with numerical problems, such as solving high-dimensional partial differential equations through a user-friendly interface using a math-like notation but doing all the tensor network operations under the hood. 

# Statement of need

`TensorTrainNumerics.jl` is a julia package for solving partial differential equation using the tensor train formulation. There is specific empathasis on the quantics tensor train representation where we provide explicit representation of different operators often used in numerics while also providing support for different solvers from the numerics community and the machine learning community. `TensorTrainNumerics.jl` is written such that the code support tensor notation and doing the tensor operations in a convinient manner. `TensorTrainNumerics.jl` also allows for easy extensions if the user quickly wants to implement a new tensor train algorithm and compare to existing algorithms. There is already a Python [@oseledets_software_ttpy] and Matlab [@oseledets_software_TT-toolbox] alternative but we provide more recent algorithms for tensor operations while having support for state-of-the-art algorithms. 

`TensorTrainNumerics.jl` was developed as a learning tool for tensor numerics which could easily be extended to benchmark new results and test different formulations. The package has already been used a publication [@FastandFlexible] and a preprint [@mikkelsen2026tensor]. This package should be a convinient way for new students wanting to dive into quantics tensor train numerical methods with existing explicit representations of operators, numerous solvers for linear systems, eigenvalue problems and time evolution. The package also provides many different examples. 

# Statement of the field
                                           
There are already several tensor train related packages

`ttpy` [@oseledets_software_ttpy] is a Python package for tensor computations allowing linear algebra in up to 100 dimensions

`TT-toolbox` is a Matlab implementation of `ttpy``

`MPSKit.jl` [@mpskitjl] is a Julia package for matrix product states and matrix product operators for (quasi) one-dimensional quantum lattices and two-dimensional statistical mechanics models

`TensorKit.jl` [@tensorkitjl] is a Julia package for large-scale tensor computations, with a hint of category theory

`scikit_tt` is a Python package for simulation and analysis of high-dimensional problems 

`MPSTime.jl` [@MPSTime2025] is a Julia package for learning the joint probability distribution of time series directly from data using matrix-product state

`ITensors.jl` is a very popular Julia package for efficient tensor network algorithms.
                                                                                    
`TensorCrossInterpolation.jl` [@nunez2025learning] is a Julia package for learning tensor networks with tensor cross interpolation
                      
# Software design

`TensorTrainNumerics.jl` is designed to match the mathematical notation used in common resources on tensor numerical methods using tensor trains [@Khoromskij] while still allowing for a common solver interface. The main building blocks are `TTVector` and `TTOperater` which correspond to a tensor train and a tensor train operator, respectively. All the relevant operations for these structs are defined and closely matches the mathematical symbol, such as ⊕ for the Hadamard product and ⋈ for the inner core product. 

The package also provides a common solver interface separates what is solved from how it is solved. The user first defines a problem object containing the operators, the discretization, and any physical constants. The user then selects an algorithm object, such as `DMRG` [@White] or `ALS` [@als_mals], which carries the method and its hyperparameters. Calling `linear_solve(problem, algorithm)` runs the method and returns the solution. This approach is consistent throughout the repository whenever more than one algorithm is available, for instance doing tensor cross interpolation. 

This package has a specific focus on the quantics tensor train representation which is a decomposition as a chain of $N$ tensors connected through some internal indices refered to as the bond dimension. The total number of tensor cores $N$ controls our discretization level by $2^N$ points. The computational complexity grows linearly with the number of tensor cores making computations on exponentially fine grids possible. `TensorTrainNumerics.jl` provides explicit QTT representation of a series of functions in the QTT format, such as $\sin(λπx)$, $\cos(λπx)$, $\exp$ and the Chebyshev polynomials.


# Research impact statement

`TensorTrainNumerics.jl` started as a tool to support the author's own research and to learn tensor train numerics, and it has since been used in two research projects. The numerical experiments on quantics tensor train solvers for differential equations with data integration in [@FastandFlexible] were carried out with the package, and the tensor train multidimensional inverse Laplace transform of [@mikkelsen2026tensor] is built on its tensor cross interpolation. The package has also received an external contribution with an example that solves the Stein-Poisson equation in lattice field theory [@butti2026variational].

The package is registered in the Julia General registry, archived on Zenodo,
and has had regular tagged releases since the first commit in July 2024. The package connects to the wider Julia ecosystem through package extensions rather than hard dependencies: 

`ChainRulesCore.jl` [@frames_white_2026_19386492] for reverse-mode automatic differentiation of tensor train
operations, `VectorInterface.jl` for use with Krylov methods [@haegeman_2026_20820534] and optimization packages that accept generic vector types, and `InterpolativeQTT.jl` for interpolative quantics tensor train constructions based on this paper [@lindsey2023multiscale].

The package is intended to lower the barrier to entry for students and researchers who are new to tensor train numerics while providing new algortihms from within the field. The repository contains several example scripts covering the heat equation, Burgers' equation, Allen-Cahn equation, Fokker-Planck equation, the Schrödinger equation, and the Gross-Pitaevskii equation. The package also contains several examples of tensor cross interpolation of high-dimensional functions.

# Usage

Here we show how to solve the 2D Poisson equation. First, we define the dimensionality, grid spacing and use the function to build the QTT representation of the finite difference Laplacian. 

```julia
using TensorTrainNumerics

d = 8; h = 1 / (2^d + 1)
Δ1d = toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
A = (1 / h^2) * (Δ1d ⊗ id_tto(d) + id_tto(d) ⊗ Δ1d)
```
We then define the right-hand side and the initial guess
```julia
b = -(1 / h^2) * qtt_sin(d; a = h, b = 1 - h) ⊗ qtt_basis_vector(d, 1)
x0 = rand_tt(b.dims, b.ranks)
```
We can then define a solver and solve the linear problem as
```julia
alg = AMEn(; tol = 1e-10, max_sweeps = 10)
x = linear_solve(A, b, x0, alg)
```
We can recover the solution on the grid as 
```julia
qttv_to_array(QTTVector(x, 2, d, :serial))
```

# AI usage disclosure

No generative AI tools were used in the initial development of this software. In recent times agentic coding has been used to improve the code quality, docstrings and provide better code coverage. Writing of this manuscript contains no generative AI.

# Acknowledgements

We acknowledge contributions from Shuhei Ohno and Louis Spatscheck.

# References