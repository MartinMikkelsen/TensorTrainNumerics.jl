# Core types and operations

```@meta
CurrentModule = TensorTrainNumerics
```

Tensor-train types, construction, arithmetic, and rank control.

## Tensor-train types

```@docs
TTVector
TTOperator
AbstractTTVector
AbstractTTOperator
QTTVector
QTTOperator
```

## Construction and conversion

```@docs
tt_decomp
tto_decomp
tt_to_tensor
tto_to_tensor
tto_to_tt
rand_tt
rand_tto
zeros_tt
zeros_tto
id_tto
admissible_ranks
concatenate
nsites
Base.copy(::AbstractTTVector{T, N}) where {T <: Number, N}
```

## Arithmetic and products

```@docs
Base.:+(::AbstractTTVector{T, N}, ::AbstractTTVector{T, N}) where {T <: Number, N}
Base.:+(::AbstractTTOperator{T, N}, ::AbstractTTOperator{T, N}) where {T <: Number, N}
Base.:*(::S, ::AbstractTTVector{R, N}) where {S <: Number, R <: Number, N}
Base.:*(::S, ::AbstractTTOperator{R, N}) where {S <: Number, R <: Number, N}
Base.:*(::AbstractTTOperator{T, N}, ::AbstractTTVector{T, N}) where {T <: Number, N}
Base.:*(::AbstractTTOperator{T, M}, ::AbstractTTVector{T, N}) where {T <: Number, M, N}
Base.:*(::AbstractTTOperator{T, N}, ::AbstractTTOperator{T, N}) where {T <: Number, N}
Base.adjoint(::AbstractTTOperator{T, N}) where {T, N}
add!
dot(::AbstractTTVector{T, N}, ::AbstractTTVector{T, N}) where {T <: Number, N}
norm(::AbstractTTVector{T, N}) where {T <: Number, N}
hadamard
⊕
hadamard_ttm
outer_product
Base.kron(::AbstractTTOperator{T, d1}, ::AbstractTTOperator{T, d2}) where {T, d1, d2}
Base.kron(::AbstractTTVector{T, d1}, ::AbstractTTVector{T, d2}) where {T, d1, d2}
⊗
⨝
∙
tt_to_diag_tto
euclidean_distance
euclidean_distance_normalized
```

## Canonical forms and rank truncation

```@docs
orthogonalize(::AbstractTTVector{T, N}) where {T <: Number, N}
tt_round!
tt_round
tt_compress!
entanglement_entropy
visualize
```

## Package extensions

These functions have methods only when the corresponding weak dependencies are loaded.

```@docs
to_ttvector
ttvector_manifold
```
