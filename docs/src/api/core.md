# Core types and operations

```@meta
CurrentModule = TensorTrainNumerics
```

Tensor-train types, construction, arithmetic, and rank control.

## Tensor-train types

```@docs
TTvector
TToperator
AbstractTTvector
AbstractTToperator
QTTvector
QTToperator
```

## Construction and conversion

```@docs
ttv_decomp
tto_decomp
ttv_to_tensor
tto_to_tensor
tto_to_ttv
rand_tt
rand_tto
zeros_tt
zeros_tto
id_tto
r_and_d_to_rks
concatenate
Base.copy(::TTvector{T, N}) where {T <: Number, N}
```

## Arithmetic and products

```@docs
Base.:+(::TTvector{T, N}, ::TTvector{T, N}) where {T <: Number, N}
Base.:+(::TToperator{T, N}, ::TToperator{T, N}) where {T <: Number, N}
Base.:*(::S, ::TTvector{R, N}) where {S <: Number, R <: Number, N}
Base.:*(::S, ::TToperator{R, N}) where {S <: Number, R <: Number, N}
Base.:*(::TToperator{T, N}, ::TTvector{T, N}) where {T <: Number, N}
Base.:*(::TToperator{T, M}, ::TTvector{T, N}) where {T <: Number, M, N}
Base.:*(::TToperator{T, N}, ::TToperator{T, N}) where {T <: Number, N}
Base.adjoint(::TToperator{T, N}) where {T, N}
add!
dot(::TTvector{T, N}, ::TTvector{T, N}) where {T <: Number, N}
norm(::TTvector{T, N}) where {T <: Number, N}
hadamard
⊕
hadamard_ttm
outer_product
Base.kron(::TToperator{T, d1}, ::TToperator{T, d2}) where {T, d1, d2}
Base.kron(::TTvector{T, d1}, ::TTvector{T, d2}) where {T, d1, d2}
⊗
⨝
∙
ttv_to_diag_tto
euclidean_distance
euclidean_distance_normalized
```

## Canonical forms and rank truncation

```@docs
orthogonalize(::TTvector{T, N}) where {T <: Number, N}
tt_round!
tt_round
tt_compress!
entanglemententropy
visualize
```

## Package extensions

These functions have methods only when the corresponding weak dependencies are loaded.

```@docs
to_ttvector
ttvector_manifold
```
