# Quantics and operators

```@meta
CurrentModule = TensorTrainNumerics
```

QTT functions and grids, differential and transfer operators, and spin-chain Hamiltonians. Unless stated otherwise, QTT functions use `2^d` grid points with site 1 holding the most significant bit.

## Quantics tensor trains

### Functions on a uniform grid

```@docs
function_to_qtt
function_to_qtt_uniform
function_to_tensor
qtt_to_function
qtt_to_vector
tensor_to_grid
matricize
qtt_polynom
qtt_sin
qtt_cos
qtt_exp
qtt_chebyshev
qtt_basis_vector
qtt_trapezoidal
gauss_chebyshev_lobatto
index_to_point
tuple_to_index
reverse_qtt_bits
```

### Multidimensional QTT

```@docs
function_to_qttv
qttv_to_array
to_qtt
to_ttv
reorder
check_compat
```

### Differential and transfer operators

```@docs
toeplitz_to_qtto
shift
∇
Δ
Δ_DN
Δ_ND
Δ_NN
Δ_P
Δ⁻¹_DN
qtt_laplacian
qtto_prolongation
qtto_constant_prolongation
qtto_linear_prolongation
qtto_to_matrix
fourier_qtto
```

## Spin-chain operators

```@docs
pauli_matrix
pauli_sum_tto
pauli_pair_sum_tto
H_μ
H_μν
heisenberg_xyz_tto
ising_tto
xxz_tto
xxx_tto
xy_tto
```
