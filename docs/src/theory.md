# Tensor Train Basics

## The curse of dimensionality

A function sampled on a uniform grid of $n$ points per dimension requires $n^d$ numbers — the **curse of dimensionality**. Even modest grids become astronomically large: a 6-dimensional grid with $n = 256$ per axis has $256^6 \approx 4.7 \times 10^{14}$ entries, far beyond the memory of any computer.

**Tensor train** (TT) decompositions escape this by representing a $d$-dimensional array as a chain of small 3-index **cores** connected by shared **bond indices**. Many physically relevant functions — exponentials, trigonometric functions, solutions to PDEs — happen to have very low **TT-rank** (small bond dimensions), so they can be stored and processed in $\mathcal{O}(d n R^2)$ memory instead of $\mathcal{O}(n^d)$.

## TT-vectors

Instead of storing all $\prod_{k=1}^d n_k$ entries of a $d$-dimensional tensor explicitly, the TT format represents a tensor $v \in \mathbb{K}^{n_1 \times \cdots \times n_d}$ as a chain of third-order **cores** $A_k \in \mathbb{K}^{r_{k-1} \times n_k \times r_k}$, using the (left bond, physical, right bond) index convention. Here $\mathbb{K}$ is $\mathbb{R}$ or $\mathbb{C}$. The entries of $v$ are recovered by contracting the cores over all bond indices:

```math
v(i_1,\dots,i_d) = \prod_{k=1}^d A_k(:,i_k,:) = \sum_{r_1,\ldots,r_{d-1}} \prod_{k=1}^d A_k(r_{k-1}, i_k, r_k),
```

where $A_k(:,i_k,:) \in \mathbb{K}^{r_{k-1}\times r_k}$ is the matrix slice of core $A_k$ at fixed physical index $i_k$, and the boundary conditions $r_0 = r_d = 1$ ensure the contraction is a scalar.

$$\large{
\begin{array}{ccccccc}
A_{1} & \;\text{—}\; r_1 \;\text{—}\; & A_{2} & \;\text{—}\; r_2 \;\text{—}\; & A_{3} & \;\text{—}\;\cdots\;\text{—}\; & A_{d} \\
| & & | & & | & & | \\
{\small i_1} & & {\small i_2} & & {\small i_3} & & {\small i_d}
\end{array}
}$$

The maximum bond dimension $R = \max_k r_k$ controls the degree of compression. Total storage scales as

$$\sum_{k=1}^{d} r_{k-1}\, n_k\, r_k \;\leq\; d R^2 n, \qquad n = \max_k n_k,$$

compared to $\prod_{k=1}^d n_k$ for the full tensor. Since the right-hand side is linear in $d$ for fixed $R$ and $n$, the TT format avoids the curse of dimensionality whenever $R$ remains bounded as $d$ grows.

## TT-operators

A **tensor train operator** (TTO) represents a matrix $A \in \mathbb{K}^{(n_1\cdots n_d)\times(n_1\cdots n_d)}$ as a chain of fourth-order cores $A_k \in \mathbb{K}^{r_{k-1} \times n_k \times n_k \times r_k}$, using the (left bond, row physical, column physical, right bond) index convention. Each core carries two physical indices — $i_k$ for the row (output) and $j_k$ for the column (input). The matrix entries are recovered as

```math
A(i_1,j_1,\dots,i_d,j_d) = \prod_{k=1}^d A_k(:,i_k,j_k,:) = \sum_{r_1,\ldots,r_{d-1}} \prod_{k=1}^d A_k(r_{k-1},i_k,j_k,r_k),
```

where $A_k(:,i_k,j_k,:) \in \mathbb{K}^{r_{k-1}\times r_k}$ is the matrix slice at fixed row index $i_k$ and column index $j_k$.

$$\large{
\begin{array}{ccccccc}
{\small j_1} & & {\small j_2} & & {\small j_3} & & {\small j_d} \\
| & & | & & | & & | \\
A_{1} & \;\text{—}\; r_1 \;\text{—}\; & A_{2} & \;\text{—}\; r_2 \;\text{—}\; & A_{3} & \;\text{—}\;\cdots\;\text{—}\; & A_{d} \\
| & & | & & | & & | \\
{\small i_1} & & {\small i_2} & & {\small i_3} & & {\small i_d}
\end{array}
}$$

The TTO format is closed under addition and matrix-vector multiplication: if $A$ and $v$ have bond dimensions $R_A$ and $R_v$, then $Av$ has bond dimension $R_A R_v$ (before compression). Storage scales as $\mathcal{O}(d R^2 n^2)$ compared to $\mathcal{O}(n^{2d})$ for the dense matrix, making it practical to represent differential operators, Hamiltonians, and other structured matrices without ever forming a dense matrix.

## Core index convention

TensorTrainNumerics.jl stores cores in **(physical, left bond, right bond)** order for vectors and **(row physical, column physical, left bond, right bond)** order for operators. Concretely, the core array at site $k$ has `size(core) == (n_k, r_{k-1}, r_k)` for a TT-vector. This differs from some tensor-network libraries that put bond indices first, but makes it natural to write `core[:, l, r]` to obtain the matrix slice $A^{(k)}(\cdot)$ at bond indices $(l, r)$.

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

The function `toeplitz_to_qtto(α, β, γ, d)` produces a tridiagonal Toeplitz operator — the standard building block for finite-difference stencils:

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

Addition and the Hadamard product grow the TT-ranks. Use `tt_compress!` to truncate them back to a manageable size via SVD:

```@example ttbasics
tt_compress!(w, 4)   # truncate w to max bond dimension 4 in-place
```

## Orthogonalization

A TT-vector is *left-canonical up to site k* when each core $A^{(1)},\ldots,A^{(k)}$ has orthonormal columns (viewed as matrices of shape $n_j r_{j-1} \times r_j$). `orthogonalize` computes this decomposition via a sequence of QR factorizations:

```@example ttbasics
vL = orthogonalize(v)          # left-canonical (gauge center at site N)
vC = orthogonalize(v; i = 2)  # gauge center at site 2
```

Orthogonalization is a prerequisite for the alternating solvers (ALS, MALS, DMRG) and TDVP, and enables cheap norm computation: `norm(v) == norm(vC.ttv_vec[2])`.

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
