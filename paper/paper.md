---
title: 'TensorTrainNumerics.jl: A Julia package for tensor train numerical methods'
tags:
  - julia
  - tensor methods
  - tensor trains
  - quantics tensor trains
  - low-rank approximations
  - differential equations
authors:
  - name: Martin Mikkelsen
    orcid: 0009-0005-3932-6563
    affiliation: 1
affiliations:
 - name: University of Copenhagen, Department of Computer Science, Denmark
   index: 1
date: 5 October 2026
bibliography: paper.bib

---

# Summary

Many problems in scientific computing involve arrays too large to store directly, such as functions sampled on very fine grids, discretizations of partial differential equations in several dimensions, or functions of many variables. When these arrays have low-rank structure, they can be stored and manipulated in compressed form, so that such problems can be solved without ever forming the full array. `TensorTrainNumerics.jl` is a Julia package for this kind of computation. It provides compressed vectors and operators with arithmetic written in mathematical notation, explicit compressed representations of finite-difference operators [@kazeev2012] and of common functions, construction of compressed arrays from function evaluations by cross interpolation [@oseledets_TT_cross; @Dolgov_2020], and interchangeable solvers for linear systems, eigenvalue problems and time evolution.

The compression used is the tensor train (TT) format, which writes a $d$-dimensional array as a chain of small three-index tensors, so that the storage and the cost of basic linear algebra grow linearly in $d$ rather than exponentially, provided the ranks of the chain stay bounded [@tt_Oseledets]. The package puts particular emphasis on the quantics tensor train (QTT) format, which applies the same decomposition to the binary digits of a grid index: a function on a grid with $2^N$ points can then be stored and manipulated at a cost linear in $N$, again provided the ranks stay bounded [@qtt_khoromskij].

# Statement of need

Tensor train methods for numerical analysis are developed mainly in the applied mathematics literature, while the general-purpose tensor network packages in Julia, such as `ITensors.jl` and `MPSKit.jl`, focus primarily on quantum many-body physics, and their interfaces are organized around lattice models and matrix product states. `TensorTrainNumerics.jl` is aimed at researchers and students in numerical analysis and scientific computing who want to solve partial differential equations on very fine grids or in many dimensions, eigenvalue problems, and high-dimensional interpolation and integration problems in the TT and QTT formats. The package started as a tool for learning tensor train numerics, and it is organized so that a new algorithm can be added next to the existing ones and compared with them on the same problem. The repository includes runnable examples for each of these problem classes, and the package has been used in a publication [@FastandFlexible] and a preprint [@mikkelsen2026tensor].

# State of the field

The packages closest in scope are the MATLAB TT-Toolbox [@oseledets_software_TT-toolbox] and its Python counterpart `ttpy` [@oseledets_software_ttpy], which provide TT arithmetic, cross approximation, linear and eigenvalue solvers, and QTT representations of Laplace operators, and the Python package `scikit_tt` [@gelss2022scikittt], which provides linear, eigenvalue and time-evolution solvers in the TT format with a focus on high-dimensional dynamical systems. In Julia, `ITensors.jl` [@itensor] and `MPSKit.jl` [@mpskitjl], the latter built on `TensorKit.jl` [@tensorkitjl], are designed primarily for quantum many-body physics, `MPSTime.jl` [@MPSTime2025] applies matrix product states to time-series machine learning, and `TensorCrossInterpolation.jl` [@nunez2025learning] constructs tensor trains by cross interpolation.

`TensorTrainNumerics.jl` shares its numerical focus with TT-Toolbox and `scikit_tt`, and differs from them in combining three things in one Julia package: a library of QTT operators on multidimensional grids, including finite-difference, shift, Fourier and prolongation operators; solvers for linear systems, eigenvalue problems and time evolution that are interchangeable, including inside implicit time integrators; and integration with the Julia ecosystem, in particular reverse-mode automatic differentiation through tensor train operations. A separate package was warranted because no Julia package covered tensor train numerical analysis, and the existing toolboxes are written in MATLAB and Python, where combining their solvers with Julia's automatic differentiation and solver packages is not possible without rewriting them. The repository's Gross–Pitaevskii sensitivity example shows the resulting workflow: it assembles QTT operators on a sequence of grids, computes a ground state with a multigrid method [@mgr] and a penalty-based ALS solver, differentiates the ground-state energy with respect to the trap frequency and the interaction strength with `Zygote.jl`, and uses these derivatives in Newton's method to find the interaction strength that gives a prescribed energy, all in one program.

# Software design

A tensor train is stored as a plain vector of Julia arrays: the core $G_k$ of a `TTVector` is an array of size $(n_k, r_{k-1}, r_k)$, and the cores of a `TTOperator` carry an additional column index, matching the formulas in the literature [@Khoromskij]. This keeps algorithms close to their written description, lets contractions be expressed with `TensorOperations.jl`, and supports real and complex element types with the same code. Arithmetic extends Julia's own functions, such as `+`, `*`, `kron` (also written `⊗`) and `dot`, so a two-dimensional Laplacian is written as `Δ ⊗ I + I ⊗ Δ`. The cost of this choice is that indices are identified by position rather than by name, as in `ITensors.jl`; that the cores are dense, so the block-sparse structure from symmetries that physics libraries exploit is not used; and that the cores are CPU arrays.

The same QTT cores represent different functions depending on how the bits of several spatial dimensions are arranged, either one dimension after another or interleaved. The types `QTTVector` and `QTTOperator` therefore record the number of spatial dimensions, the number of bits per dimension and the bit ordering. This lets conversions to grid arrays and the assembly of multidimensional operators use the correct ordering, and lets mismatched operands be detected with `check_compat` instead of producing a silently wrong result. To avoid writing every operation twice, operations are defined once on abstract supertypes, and the QTT information is attached to the result afterwards. QTT operators and functions are constructed directly in low-rank form rather than by compressing dense arrays, so their cost does not depend on the full grid size $2^N$; these include finite-difference Laplacians [@kazeev2012], the discrete Fourier transform [@QFT1], prolongation operators between grid levels [@mgr], and functions such as $\sin(\lambda \pi x)$ and Chebyshev polynomials.

Solvers are selected by algorithm objects, such as `AMEn()` [@Dolgov_Savostyanov_2014] or `MALS()` [@als_mals], which carry the method's parameters and are passed to `linear_solve`, `eigen_solve`, `time_evolve` or `tt_cross`. Implicit time integrators take a linear solver algorithm as a parameter, so any linear solver can be used inside any implicit scheme, and a new method is added by defining a new algorithm type and a method for it, after which it can be compared with the existing solvers on the same problem. The package also integrates with other Julia packages. The TT types implement the `VectorInterface.jl` interface, so they can be used directly with Krylov [@haegeman_2026_20820534] and optimization packages that accept generic vector types. Three integrations are optional and are package extensions that load only when the corresponding packages are loaded: an extension using `ChainRulesCore.jl` [@frames_white_2026_19386492] defines reverse-mode differentiation rules for tensor train operations, so that `Zygote.jl` can differentiate through them; an extension for `Manopt.jl` [@Bergmann:2022] allows optimization directly over tensor trains; and an extension for `TensorCrossInterpolation.jl` [@nunez2025learning] converts its tensor trains to the package's format, including those produced by the interpolative QTT constructions of `InterpolativeQTT.jl` [@lindsey2023multiscale].

# Research impact statement

`TensorTrainNumerics.jl` started as a tool to support the author's own research and to learn tensor train numerics, and it has since been used in two research projects. The numerical experiments on quantics tensor train solvers for differential equations with data integration in [@FastandFlexible] were carried out with the package, and the tensor train multidimensional inverse Laplace transform of [@mikkelsen2026tensor] is built on its tensor cross interpolation. The package has also received contributions from two external contributors, including an example that solves the Stein–Poisson equation for $\phi^4$ lattice field theory following the approach of [@butti2026variational].

The package is registered in the Julia General registry and archived on Zenodo, and it has had 17 tagged releases since its first commit in July 2024. The repository contains about 40 runnable example scripts, covering the Allen–Cahn equation, Fokker–Planck equation, Schrödinger and Gross–Pitaevskii equations, ground states of spin chains, and tensor cross interpolation and integration of high-dimensional functions. Together with the explicit operator constructions, these examples are intended to lower the barrier to entry for students and researchers who are new to tensor train numerics.

# Usage

As an example, we solve Laplace's equation $\Delta u = 0$ on the unit square with $u(x, 0) = \sin(\pi x)$ and $u = 0$ on the other three sides, whose exact solution is $u(x, y) = \sin(\pi x) \sinh(\pi(1 - y)) / \sinh(\pi)$. We use the standard five-point finite-difference scheme on $2^d \times 2^d$ interior grid points with spacing $h = 1/(2^d + 1)$. The one-dimensional second-difference matrix is built directly in QTT format with `toeplitz_to_qtto`, and the two-dimensional Laplacian is assembled from it with Kronecker products:

```julia
using TensorTrainNumerics

d = 8; h = 1 / (2^d + 1)
Δ1d = toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
A = (1 / h^2) * (Δ1d ⊗ id_tto(d) + id_tto(d) ⊗ Δ1d)
```
The boundary values on $y = 0$ enter the right-hand side as the product of $\sin(\pi x)$ on the grid and the first unit vector in $y$. We start from a random tensor train with the same ranks as the right-hand side:
```julia
b = -(1 / h^2) * qtt_sin(d; a = h, b = 1 - h) ⊗ qtt_basis_vector(d, 1)
x0 = rand_tt(b.dims, b.ranks)
```
The system is solved with `AMEn`, replacing `alg` with `MALS()` or `DMRG()` selects a different solver without other changes:
```julia
alg = AMEn(; tol = 1e-10, max_sweeps = 10)
x = linear_solve(A, b, x0, alg)
u = qttv_to_array(QTTVector(x, 2, d, :serial))
```
The last line returns the solution as a $2^d \times 2^d$ array. The computed solution has maximum rank 2 and is stored in 112 numbers instead of the 65536 grid values.

# AI usage disclosure

No generative AI tools were used in the initial development of this software. After release `v1.1.1` agentic coding (Claude Code (Anthropic)) has been used to improve the code quality, docstrings and provide better code coverage. Generative AI was used to edit parts of this manuscript. All text produced with AI assistance was reviewed, revised, and verified by the author before inclusion in the code or paper.

# Acknowledgements

The author thanks Shuhei Ohno and Louis Spatscheck for their contributions.

# References