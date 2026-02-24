# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run all tests:**
```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

**Run a single test file:**
```julia
julia --project=. -e 'using TensorTrainNumerics; include("test/test_tt_tools.jl")'
```

**Build documentation:**
```julia
julia --project=docs docs/make.jl
```

**Start a Julia REPL with the package loaded:**
```julia
julia --project=. -e 'using TensorTrainNumerics'
```

## Architecture

This is a Julia package for numerical methods using Tensor Train (TT) and Quantized Tensor Train (QTT) decompositions, targeting high-dimensional PDEs and large-scale linear algebra.

### Core Data Structures (`src/tt_tools.jl`)

Two central types:

- **`TTvector{T,N}`** — A TT vector with `N` cores. Each core `ttv_vec[i]` is a 3D array of shape `(n_i, r_{i-1}, r_i)` where `n_i` is the physical dimension and `r_i` are the TT ranks. The field `ttv_ot` stores orthogonality information per core (`-1` = left-orthogonal, `0` = center/root, `1` = right-orthogonal).

- **`TToperator{T,N}`** — A TT operator with `N` cores. Each core `tto_vec[i]` is a 4D array of shape `(n_i, n_i, r_{i-1}, r_i)` representing a matrix-valued core.

Both types are parametrized by element type `T <: Number` and number of dimensions `N` (a compile-time constant encoded in the type).

### Module Layout

| File | Contents |
|------|----------|
| `src/tt_tools.jl` | Core types (`TTvector`, `TToperator`), construction (`rand_tt`, `zeros_tt`, `rand_tto`, `zeros_tto`), decomposition (`ttv_decomp`, `tto_decomp`), orthogonalization, compression (`tt_compress!`), and conversion utilities (`ttv_to_tensor`, `tto_to_tensor`, `tt2qtt`) |
| `src/tt_operations.jl` | Arithmetic on TT types: `+`, `-`, `*` (scalar, operator-vector, operator-operator), `dot`, `norm`, `hadamard`, `hadamard_ttm` (TTM algorithm for element-wise product), `outer_product`, `kron`/`⊗`, `euclidean_distance` |
| `src/tt_operators.jl` | QTT-structured discrete operators: Laplacians (`Δ`, `Δ_DN`, `Δ_ND`, `Δ_NN`, `Δ_P`, `Δ⁻¹_DN`), gradient `∇`, shift matrices, Toeplitz-to-QTT conversion, identity/zero constructors |
| `src/tt_interpolations.jl` | Chebyshev-Lobatto nodes, Lagrange basis, QTT interpolation (`interpolating_qtt`, `lagrange_rank_revealing`) |
| `src/tt_cross_interpolation.jl` | TT-cross interpolation (`tt_cross`, `tt_integrate`) with algorithm variants: `MaxVol` (default), `DMRG`, `Greedy` |
| `src/tt_transformations.jl` | Fourier QTT operator (`fourier_qtto`), bit reversal (`reverse_qtt_bits`) |
| `src/qtt_tools.jl` | QTT function approximation tools: `function_to_qtt`, `qtt_to_function`, `qtt_cos`, `qtt_sin`, `qtt_exp`, `qtt_chebyshev`, QTT quadrature (`qtt_simpson`, `qtt_trapezoidal`) |
| `src/solvers/als.jl` | Alternating Linear Scheme: `als_linsolve`, `als_eigsolve`, `als_gen_eigsolv` |
| `src/solvers/mals.jl` | Modified ALS: `mals_eigsolve`, `mals_linsolve` |
| `src/solvers/dmrg.jl` | DMRG-based solvers: `dmrg_linsolve`, `dmrg_eigsolve` |
| `src/solvers/tdvp.jl` | Time-Dependent Variational Principle: `tdvp`, `tdvp2` |
| `src/solvers/euler.jl` | Time integration: `euler_method`, `implicit_euler_method`, `crank_nicholson_method`, `rk4_method` |
| `ext/TensorTrainNumericsVectorInterfaceExt/` | Extension enabling `VectorInterface.jl` compatibility for use with KrylovKit.jl and OptimKit.jl |

### Key Conventions

- **Core tensor indexing**: cores are always stored as `Array{T,3}` with layout `(physical_dim, left_rank, right_rank)` — i.e., `ttv_vec[i][s, α, β]`.
- **Operator cores**: `Array{T,4}` with layout `(row_dim, col_dim, left_rank, right_rank)` — i.e., `tto_vec[i][i, j, α, β]`.
- **Ranks**: `ttv_rks` has length `N+1` with `ttv_rks[1] = ttv_rks[N+1] = 1` (boundary conditions).
- **QTT**: Quantized Tensor Train is a TT where each physical dimension is 2 (binary). Functions like `function_to_qtt` discretize a 1D function on `2^L` grid points into an `L`-core TT. Multi-dimensional QTTs are built via `kron`.
- **Orthogonalization**: The `orthogonalize(x; i=k)` function left-orthogonalizes cores `1:k-1` and right-orthogonalizes cores `k+1:N`, making core `k` the "center". This is required before most solvers.
- **TensorOperations.jl**: Einstein summation notation via `@tensor` macros is used throughout for contraction operations.
