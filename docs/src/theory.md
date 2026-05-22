# Tensor Train Basics

A **tensor train** (TT) decomposes a high-dimensional array into a chain of three-index cores connected by shared bond indices. For a vector $v \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_N}$ the TT format reads

```math
v(i_1, i_2, \ldots, i_N) = G_1(i_1)\, G_2(i_2) \cdots G_N(i_N),
```

where each **core** $G_k$ is a matrix-valued function of the *physical index* $i_k \in \{1,\ldots,n_k\}$: the slice $G_k(i_k) \in \mathbb{R}^{r_{k-1} \times r_k}$ is an ordinary matrix. The boundary conditions $r_0 = r_N = 1$ make the product a scalar for every multi-index $(i_1, \ldots, i_N)$.

The integers $r_1, \ldots, r_{N-1}$ are the **bond dimensions** (also called TT-ranks). A function with low TT-rank can be stored and manipulated in $\mathcal{O}(NnR^2)$ memory instead of $\mathcal{O}(n^N)$.

A **tensor train operator** (TTO) has two physical indices per core and represents a matrix $A \in \mathbb{R}^{(n_1\cdots n_N)\times(n_1\cdots n_N)}$:

```math
A(i_1,j_1,\ldots,i_N,j_N) = W_1(i_1,j_1)\, W_2(i_2,j_2) \cdots W_N(i_N,j_N),
```

where $W_k(i_k,j_k) \in \mathbb{R}^{r_{k-1}\times r_k}$.

## Core index convention

TensorTrainNumerics.jl stores cores in **(physical, left bond, right bond)** order for vectors and **(row physical, column physical, left bond, right bond)** order for operators. Concretely, the core array at site $k$ has `size(core) == (n_k, r_{k-1}, r_k)` for a TT-vector. This differs from some tensor-network libraries that put bond indices first, but makes it natural to write `G[:, l, r]` to obtain the $l, r$ matrix slice.

## Types

```julia
abstract type AbstractTTvector{T}    end
abstract type AbstractTToperator{T}  end

mutable struct TTvector{T <: Number, N}  <: AbstractTTvector{T}
struct         TToperator{T <: Number, N} <: AbstractTToperator{T}
```

Both are parametrized by the element type `T` and the number of sites `N`.  The abstract supertypes let solvers and arithmetic accept both plain TT objects and the richer `QTTvector`/`QTToperator` wrappers described in the [QTT page](qtt.md).

## Constructing TT-vectors

```@example ttbasics
using TensorTrainNumerics

dims = (2, 2, 2, 2)    # physical dimension at each site
rks  = [1, 3, 3, 3, 1] # bond dimensions (length N+1)

v = rand_tt(dims, rks)   # random TT-vector
z = zeros_tt(dims, rks)  # zero TT-vector
```

The fields `v.ttv_dims`, `v.ttv_rks`, and `v.ttv_vec` hold the dimensions, bond dimensions, and array of cores respectively.

## Constructing TT-operators

```@example ttbasics
A = rand_tto(dims, 3)   # random TTO, max bond = 3
I = id_tto(4)           # identity on {1,…,2}^4 in TT form
```

The function `toeplitz_to_qtto(α, β, γ, d)` produces a tridiagonal Toeplitz operator — useful for finite-difference stencils:

```@example ttbasics
h = 1.0 / 2^4
L = toeplitz_to_qtto(-2.0, 1.0, 1.0, 4)   # standard Laplacian stencil
```

## Contractions and basic operations

All standard linear-algebra operations are overloaded and produce new TT objects:

| Expression | Meaning |
|---|---|
| `u + v` | TT-vector addition (bond dim doubles) |
| `u - v` | TT-vector subtraction |
| `α * v` / `v * α` | Scalar multiplication |
| `dot(u, v)` | Inner product $\langle u, v\rangle$ |
| `norm(v)` | Euclidean norm |
| `u ⊕ v` | Hadamard (element-wise) product (bond dim multiplies) |
| `A * v` | Operator–vector product (bond dim multiplies) |
| `A + B` | Operator addition |
| `A ⊗ B` | Kronecker product of operators |

```@example ttbasics
u = rand_tt(dims, rks)
w = u + v          # bond dims are now doubled
s = dot(u, v)
n = norm(v)
```

Addition and the Hadamard product grow the TT-ranks. Use `tt_compress!` to truncate them back:

```@example ttbasics
tt_compress!(w, 4)   # truncate w to max bond dimension 4 in-place
```

## Orthogonalization

A TT-vector is *left-canonical up to site k* when each core $G_1,\ldots,G_k$ has orthonormal columns (viewed as matrices of shape $n_j r_{j-1} \times r_j$). `orthogonalize` computes this decomposition:

```@example ttbasics
vL = orthogonalize(v)          # left-canonical (gauge center at site N)
vC = orthogonalize(v; i = 2)  # gauge center at site 2
```

Orthogonalization is a prerequisite for the alternating solvers and TDVP.

## Visualization

`visualize` draws the tensor network diagram of a TT-vector or TT-operator:

```@example ttbasics
visualize(v)
```

## Tensor cross interpolation

TT-cross algorithms build a TT approximation of a black-box function $f:\{1,\ldots,n\}^d\to\mathbb{R}$ without evaluating $f$ everywhere. The function interface expects a batch: `f(X)` where `X` is an $m\times d$ matrix of grid points and the return value is an $m$-vector. Three algorithms are available:

| Algorithm | Constructor | Notes |
|---|---|---|
| MaxVol | `MaxVol(tol, maxiter)` | Stable pivot selection via maximal-volume submatrices |
| DMRG-cross | `DMRG(tol, maxiter)` | Alternating left–right sweeps |
| Greedy | `Greedy(tol, maxiter)` | Fast but less robust |

```@example ttcross
using LinearAlgebra
using TensorTrainNumerics

f(X::Matrix{Float64}) = vec(sin.(sum(X, dims = 2)))

n = 8
d = 6
domain = [collect(range(0.0, π, length = n)) for _ in 1:d]

tt_mv = tt_cross(f, domain, MaxVol(tol = 1.0e-8, maxiter = 20); ranks = 4)
tt_dg = tt_cross(f, domain, DMRG(tol = 1.0e-8, maxiter = 25))
```

### Numerical integration

`tt_integrate` estimates $\int_a^b f(x)\,dx$ over a $d$-dimensional box using TT-cross to build the integrand tensor:

```@example ttcross
result = tt_integrate(f, d, lower = 0.0, upper = Float64(π); alg = MaxVol(tol = 1.0e-8))
println("∫sin(x₁+⋯+x₆) dx ≈ ", result)
```

### Reconstructing a dense tensor

For small problems you can convert the TT back to a full array:

```@example ttcross
tensor_approx = ttv_to_tensor(tt_mv)

tensor_exact = zeros(ntuple(_ -> n, d)...)
for idx in CartesianIndices(tensor_exact)
    coords = [domain[k][idx[k]] for k in 1:d]
    tensor_exact[idx] = sin(sum(coords))
end

rel_err = norm(tensor_approx .- tensor_exact) / norm(tensor_exact)
println("Relative error: ", rel_err)
```
