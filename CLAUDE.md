# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run all tests:**
```
julia --project=. -e 'using Pkg; Pkg.test()'
```

**Run a single test file:**
```
julia --project=. -e 'using TensorTrainNumerics; include("test/test_tt_tools.jl")'
```

**Start a Julia REPL with the package loaded:**
```
julia --project=. -e 'using TensorTrainNumerics'
```

**Build/instantiate dependencies:**
```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Architecture

### Core types (`src/tt_tools.jl`)

Two fundamental types:

- **`TTvector{T<:Number, M}`** — a tensor train vector. Cores stored in `ttv_vec::Vector{Array{T,3}}` with layout `(phys_dim, left_rank, right_rank)`. Also carries `ttv_dims::NTuple{M,Int}`, `ttv_rks::Vector{Int}` (length `N+1`, boundary ranks always 1), and `ttv_ot::Vector{Int}` (orthogonality info: `-1` = right-ortho, `0` = center, `+1` = left-ortho).

- **`TToperator{T<:Number, M}`** — a tensor train operator. Cores stored in `tto_vec::Vector{Array{T,4}}` with layout `(row_dim, col_dim, left_rank, right_rank)`.

Key utility functions also in `tt_tools.jl`:
- `ttv_decomp` / `tto_decomp` — hierarchical SVD decomposition from a full tensor
- `orthogonalize(tt; i=1)` — QR/LQ sweeps to orthogonalize around core `i`. Uses `LinearAlgebra.qr`/`lq` (not MatrixAlgebraKit, which causes sign-convention issues with TDVP2).
- `tt_compress!(ψ, max_bond)` — in-place truncation via alternating bond SVD sweeps
- `rand_tt`, `zeros_tt`, `zeros_tto`, `rand_tto` — constructors

### Operations (`src/tt_operations.jl`)

Arithmetic on TT types: `+`, `-`, `*` (scalar and operator-vector), `/`, `dot`, `norm`, `hadamard`, `kron`, `⊕`, `⊗`, `outer_product`. Addition doubles bond dimensions; compression is separate.

### Solvers (`src/solvers/`)

- **`als.jl`** — Alternating Linear Scheme: `als_linsolve`, `als_eigsolve`, `als_gen_eigsolv`
- **`mals.jl`** — Modified ALS: `mals_eigsolve`, `mals_linsolve`
- **`dmrg.jl`** — DMRG: `dmrg_linsolve`, `dmrg_eigsolve`
- **`tdvp.jl`** — Time-Dependent Variational Principle: `tdvp` (1-site), `tdvp2` (2-site, bond-adaptive). Uses KrylovKit for Krylov subspace time evolution.
- **`euler.jl`** — Time-stepping: `euler_method`, `implicit_euler_method`, `crank_nicholson_method`, `rk4_method`

All solvers build environment tensors (`H`, `H_b`) by contracting TT cores from both ends, then optimize core by core in sweeps.

### QTT tools (`src/qtt_tools.jl`)

Functions for constructing quantized TT representations: trigonometric functions (`qtt_cos`, `qtt_sin`, `qtt_exp`), interpolation (`qtt_chebyshev`, `qtt_trapezoidal`, `qtt_simpson`), and grid utilities. Physical dimension is 2 for QTT (binary index decomposition).

### Operators (`src/tt_operators.jl`)

Discrete differential operators in TT format: Laplacians with various boundary conditions (`Δ`, `Δ_DN`, `Δ_ND`, `Δ_NN`, `Δ_P`, `Δ⁻¹_DN`), gradient `∇`, shift matrix, identity `id_tto`. These use Toeplitz structure to achieve rank-3 TT operators.

### TT-cross interpolation (`src/tt_cross_interpolation.jl`)

Approximates a function as a TTvector from function evaluations. Entry point: `tt_cross(f, dims, alg)`. Algorithm variants: `MaxVol`, `DMRG`, `Greedy`. Uses the `Maxvol.jl` package for pivot selection.

### Extension (`ext/TensorTrainNumericsVectorInterfaceExt/`)

Implements the `VectorInterface.jl` interface for `TTvector` and `TToperator`. Every mutating vector operation calls `orthogonalize` to maintain canonical form. Loaded only when `VectorInterface` is in the environment — required by TDVP (KrylovKit uses VectorInterface).

## Key implementation notes

- **Core layout**: `(phys_dim, left_rank, right_rank)` — this is a non-standard layout; many libraries use `(left_rank, phys_dim, right_rank)`. The TDVP solver internally permutes to `(left_rank, phys_dim, right_rank)` via `_to_lsr`/`_to_slr`.
- **SVD truncation**: `_svdtrunc` is public API used by TDVP2 for bond-adaptive expansion.
- **Orthogonalization**: always use `LinearAlgebra.qr`/`lq`, not MatrixAlgebraKit, for `orthogonalize`. The Q-factor sign convention differs and causes numerical errors in TDVP2.
- **TensorOperations.jl**: contraction network notation (`@tensor`, `@tensoropt`) is used throughout for multi-index contractions.
- **Threading**: `@threads` is used in addition/copy operations; cores are independent so parallelism is safe.
