using DifferentiationInterface

# Same element type: replace core k without any type conversion.
function _replace_core(tt::TTvector{T,N}, k::Int, new_core::AbstractArray{T,3}) where {T,N}
    TTvector{T,N}(tt.N,
                  [i == k ? new_core : tt.ttv_vec[i] for i in 1:tt.N],
                  tt.ttv_dims, tt.ttv_rks, tt.ttv_ot)
end

# Different element type S (e.g. ForwardDiff Dual numbers): promote all cores to S.
function _replace_core(tt::TTvector{T,N}, k::Int, new_core::AbstractArray{S,3}) where {T,N,S}
    TTvector{S,N}(tt.N,
                  [i == k ? Array{S,3}(new_core) : S.(tt.ttv_vec[i]) for i in 1:tt.N],
                  tt.ttv_dims, tt.ttv_rks, tt.ttv_ot)
end

"""
    tt_core_gradient(f, tt, k, backend) -> Array

Gradient of a scalar function `f(tt)` with respect to core `k` of `tt`.
Returns an array of the same shape as `tt.ttv_vec[k]`.

`backend` is any DifferentiationInterface backend.  With `AutoForwardDiff()` the
function `f` only needs to support element type promotion (e.g. ForwardDiff Dual
numbers), which works for all standard TT operations.

# Example
```julia
x = rand_tt(Float64, (2, 3, 2), [1, 2, 2, 1])
f = tt -> real(dot(tt, tt))          # norm(tt)^2
grad = tt_core_gradient(f, x, 2, AutoForwardDiff())
```
"""
function tt_core_gradient(f, tt::TTvector{T,N}, k::Int, backend) where {T,N}
    shape = size(tt.ttv_vec[k])
    g = v -> f(_replace_core(tt, k, reshape(v, shape)))
    reshape(DifferentiationInterface.gradient(g, backend, vec(copy(tt.ttv_vec[k]))), shape)
end

"""
    tt_gradient(f, tt, backend) -> Vector{Array}

Gradients of a scalar function `f(tt)` with respect to every core of `tt`.
Returns a length-`tt.N` vector of arrays, one per core.
"""
tt_gradient(f, tt::TTvector, backend) =
    [tt_core_gradient(f, tt, k, backend) for k in 1:tt.N]

"""
    tt_core_hessian(f, tt, k, backend) -> Matrix

Hessian of a scalar function `f(tt)` with respect to core `k` (vectorised).
Returns an `n × n` matrix where `n = prod(size(tt.ttv_vec[k]))`.
"""
function tt_core_hessian(f, tt::TTvector{T,N}, k::Int, backend) where {T,N}
    shape = size(tt.ttv_vec[k])
    g = v -> f(_replace_core(tt, k, reshape(v, shape)))
    DifferentiationInterface.hessian(g, backend, vec(copy(tt.ttv_vec[k])))
end

"""
    tt_core_jacobian(f, tt, k, backend) -> Matrix

Jacobian of a vector-valued function `f(tt)` with respect to core `k` of `tt`.
`f` must return an `AbstractVector`. Returns an `m × n` matrix where
`m = length(f(tt))` and `n = prod(size(tt.ttv_vec[k]))`.

# Example
```julia
x = rand_tt(Float64, (2, 2, 2), [1, 2, 2, 1])
f = tt -> [real(dot(tt, tt)), 2*real(dot(tt, tt))]
J = tt_core_jacobian(f, x, 2, AutoForwardDiff())  # 2 × n matrix
```
"""
function tt_core_jacobian(f, tt::TTvector{T,N}, k::Int, backend) where {T,N}
    shape = size(tt.ttv_vec[k])
    g = v -> f(_replace_core(tt, k, reshape(v, shape)))
    DifferentiationInterface.jacobian(g, backend, vec(copy(tt.ttv_vec[k])))
end

"""
    tt_core_curl(f, tt, k, backend) -> Matrix

Curl (antisymmetric part of the Jacobian) of a vector-valued function `f(tt)`
with respect to core `k`. `f` must return a vector of length equal to
`n = prod(size(tt.ttv_vec[k]))`. Returns the `n × n` antisymmetric matrix
`J - J'` where `J` is the Jacobian of `f` w.r.t. `vec(tt.ttv_vec[k])`.

For a 3-D vector field this reduces to the standard curl (vorticity tensor).

# Example
```julia
x  = rand_tt(Float64, (2, 2, 2), [1, 2, 2, 1])
n  = prod(size(x.ttv_vec[2]))
A  = randn(n, n)
f  = tt -> A * vec(tt.ttv_vec[2])
C  = tt_core_curl(f, x, 2, AutoForwardDiff())  # ≈ A - A'
```
"""
function tt_core_curl(f, tt::TTvector{T,N}, k::Int, backend) where {T,N}
    J = tt_core_jacobian(f, tt, k, backend)
    m, n = size(J)
    m == n || throw(DimensionMismatch(
        "curl requires output length ($m) == core size ($n)"))
    J - J'
end
