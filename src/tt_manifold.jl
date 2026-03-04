"""
    tt_manifold(dims, max_rank; field = ℝ) -> TTManifold

Construct the manifold of Tensor Train vectors with physical dimensions `dims`
and bond dimension at most `max_rank`.

Requires `ManifoldsBase.jl` to be loaded (via `using ManifoldsBase`).

The returned object is a `TTManifold <: ManifoldsBase.AbstractManifold` and is
compatible with the full Riemannian geometry ecosystem: `Manopt.jl` for
Riemannian optimization, `ManifoldDiff.jl` for automatic differentiation on
manifolds, and any other `ManifoldsBase`-based package.

# Example
```julia
using TensorTrainNumerics, ManifoldsBase
M  = tt_manifold((2, 2, 2), 4)          # 3-site, d=2, max rank 4
ψ  = rand_tt((2,2,2), [1,2,2,1])        # a point on M
X  = zero_vector(M, ψ)                  # zero tangent vector at ψ
d  = manifold_dimension(M)              # dimension of the manifold
```
"""
function tt_manifold end

export tt_manifold
