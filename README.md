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

Tensor Train Numerics is a Julia package designed to provide efficient numerical methods for working with tensor trains (TT) and quantized tensor trains (QTT). 

## Getting started 

To get started with Tensor Train Numerics, you can install the package using Julia's package manager:

```Julia
using Pkg
Pkg.add("TensorTrainNumerics")
```

## Key features

- Tensor Train Decomposition: Algorithms for decomposing high-dimensional tensors into tensor train format 
- Tensor Operations: Support for basic tensor operations such as addition, multiplication, the hadamard product in tensor train format 
- Discrete Operators: Implementation of discrete Laplacians, gradient operators, and shift matrices in tensor train format for solving partial differential equations 
- Quantized Tensor Trains: Tools for constructing and manipulating quantized tensor trains, which provide further compression and efficiency for large-scale problems.
- Iterative Solvers: Integration with iterative solvers for solving linear systems and eigenvalue problems in tensor train format
- The Fourier transform in QTT format and interpolation in QTT format 
- Visualization: Basic visualization tools. 

### Acknowledgements 

Many of the features are inspired by the work of [Mi-Song Dupuy](https://github.com/msdupuy)

### References

[1] I. Oseledets. Tensor-Train Decomposition. SIAM Journal on Scientific Computing 33, 2295–2317 (2011).
[2] U. Schollwöck. The density-matrix renormalization group in the age of matrix product states. Annals of Physics 326, 96–192 (2011).
[3] L. Devos, M. Van Damme and J. Haegeman. TensorOperations.jl (2023).
[4] V. A. Kazeev and B. N. Khoromskij. Low-rank explicit QTT representation of the Laplace operator and its inverse. SIAM Journal on Matrix Analysis and Applications 33, 742–758 (2012).
[5] B. N. Khoromskij. Tensor Numerical Methods in Scientific Computing (De Gruyter, Berlin, Boston, 2018).
[6] B. Khoromskij. O(d log N)-Quantics Approximation of N - d Tensors in High-Dimensional Numerical Modeling. Constructive Approximation 34 (2009).
[7] S. R. White. Density matrix formulation for quantum renormalization groups. Physical Review Letters 69, 2863–2866 (1992).
[8] J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken and F. Verstraete. Unifying time evolution and optimization with matrix product states. Physical Review B 94, 165116 (2016).
[9] L. Vanderstraeten, J. Haegeman and F. Verstraete. Tangent-space methods for uniform matrix product states. SciPost Physics Lecture Notes (2019).
[10] S. Holtz, T. Rohwedder and R. Schneider. The Alternating Linear Scheme for Tensor Optimization in the Tensor Train Format. SIAM Journal on Scientific Computing 34, A683–A713 (2012).
[11] M. Lindsey. Multiscale interpolative construction of quantized tensor trains, arXiv preprint arXiv:2311.12554 (2023).
[12] S. Dolgov, B. Khoromskij and D. Savostyanov. Superfast Fourier Transform Using QTT Approximation. Journal of Fourier Analysis and Applications 18, 915–953 (2012).
[13] J. Chen and M. Lindsey. Direct interpolative construction of the discrete Fourier transform as a matrix product operator (2024), arXiv:2404.03182 [quant-ph].
[14] J. Chen, E. M. Stoudenmire and S. R. White. Quantum Fourier transform has small entanglement. PRX Quantum 4, 040318 (2023).
[15] L. Arenstein, M. Mikkelsen and M. Kastoryano. Fast and Flexible Quantum-Inspired Differential Equation Solvers with Data Integration (2025), arXiv:2505.17046 [math.NA].
