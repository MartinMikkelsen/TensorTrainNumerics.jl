module TensorTrainNumericsManifoldExt

using TensorTrainNumerics
using ManifoldsBase
using LinearAlgebra
import Random

import ManifoldsBase: inner, exp!, retract!, project!, zero_vector, zero_vector!,
    manifold_dimension, check_point, check_vector, representation_size,
    vector_transport_to!, get_vector, get_coordinates, default_retraction_method,
    allocate_result

# TensorTrainNumerics.dot and LinearAlgebra.dot are both visible; use the TT one.
const ttdot = TensorTrainNumerics.dot

# ── Type definition ────────────────────────────────────────────────────────────

"""
    TTManifold{𝔽, N} <: AbstractManifold{𝔽}

Manifold of Tensor Train vectors with physical dimensions `dims` and bond
dimension at most `max_rank`.

Points on this manifold are `TTvector` objects with `ttv_dims == dims` and all
ranks ≤ `max_rank`.  Tangent vectors at a point `ψ` are `TTvector` objects in
the tangent space `T_ψ M`, computed via the TDVP transfer-matrix projector.

# Constructor
Use [`tt_manifold`](@ref) rather than constructing `TTManifold` directly.

    M = tt_manifold((2, 2, 2), 4)

# Supported interface
- `manifold_dimension(M)` – combinatorial dimension of the manifold
- `check_point(M, ψ)` / `check_vector(M, ψ, X)` – validation
- `inner(M, ψ, X, Y)` – Euclidean inner product via TT dot
- `zero_vector(M, ψ)` / `zero_vector!(M, X, ψ)` – zero tangent vector
- `project!(M, Y, ψ, X)` – tangent space projection (TDVP projector)
- `exp!(M, q, ψ, X)` / `retract!(M, q, ψ, X, t)` – SVD retraction
- `vector_transport_to!(M, Y, ψ, X, q, ::ProjectionTransport)` – parallel transport
"""
struct TTManifold{𝔽, N} <: AbstractManifold{𝔽}
    dims::NTuple{N, Int}
    max_rank::Int
end

# Wire up the constructor stub defined in the main package
function TensorTrainNumerics.tt_manifold(
        dims::NTuple{N, Int}, max_rank::Int;
        field::F = ManifoldsBase.ℝ
    ) where {N, F <: ManifoldsBase.AbstractNumbers}
    max_rank ≥ 1 || throw(ArgumentError("max_rank must be ≥ 1, got $max_rank"))
    return TTManifold{F, N}(dims, max_rank)
end

# Convenience: accept tuple or vector of equal ranks
TensorTrainNumerics.tt_manifold(dims, max_rank::Int; kwargs...) =
    TensorTrainNumerics.tt_manifold(Tuple(dims), max_rank; kwargs...)

# ── Effective bond dimensions ──────────────────────────────────────────────────
#
# The bond dimension at each TT bond is capped by the max_rank AND by the
# maximum possible rank from either side (product of physical dims).

function _effective_ranks(dims::NTuple{N, Int}, max_rank::Int) where N
    r = ones(Int, N + 1)
    cum_left = 1
    for k in 1:N-1
        cum_left *= dims[k]
        cum_right = prod(dims[k+1:end])
        r[k + 1] = min(max_rank, cum_left, cum_right)
    end
    return r
end

# ── Manifold dimension ─────────────────────────────────────────────────────────

"""
    manifold_dimension(M::TTManifold)

Return the real dimension of the fixed-rank TT manifold.

Formula (Holtz–Rohwedder–Schneider 2012):

    dim = Σ_k r_{k-1} d_k r_k  –  Σ_{k=1}^{N-1} r_k²

where r_k are the effective bond dimensions.
"""
function manifold_dimension(M::TTManifold{𝔽, N}) where {𝔽, N}
    r = _effective_ranks(M.dims, M.max_rank)
    total  = sum(r[k] * M.dims[k] * r[k+1] for k in 1:N)
    gauges = sum(r[k]^2 for k in 2:N)
    return total - gauges
end

# ── Point and vector validation ────────────────────────────────────────────────

function check_point(M::TTManifold{𝔽, N}, p::TTvector; kwargs...) where {𝔽, N}
    p.N == N ||
        throw(DomainError(p.N, "TTvector has N=$(p.N) sites, manifold has N=$N"))
    p.ttv_dims == M.dims ||
        throw(DomainError(p.ttv_dims,
            "TTvector dims $(p.ttv_dims) ≠ manifold dims $(M.dims)"))
    all(r ≤ M.max_rank for r in p.ttv_rks[2:end-1]) ||
        throw(DomainError(p.ttv_rks,
            "TTvector has bond $(maximum(p.ttv_rks)) > max_rank=$(M.max_rank)"))
    return nothing
end

function check_vector(M::TTManifold{𝔽, N}, p::TTvector, X::TTvector; kwargs...) where {𝔽, N}
    check_point(M, p)
    X.N == N ||
        throw(DomainError(X.N, "Tangent vector has N=$(X.N) sites, manifold has N=$N"))
    X.ttv_dims == M.dims ||
        throw(DomainError(X.ttv_dims,
            "Tangent vector dims $(X.ttv_dims) ≠ manifold dims $(M.dims)"))
    return nothing
end

# ── Inner product (Euclidean in TT format) ─────────────────────────────────────

"""
    inner(M::TTManifold, ψ, X, Y)

Euclidean inner product of tangent vectors `X` and `Y` at point `ψ`:
`real(dot(X, Y))`.
"""
inner(::TTManifold, ::TTvector, X::TTvector, Y::TTvector) = real(ttdot(X, Y))

# ── Zero tangent vector ────────────────────────────────────────────────────────

"""
    zero_vector(M::TTManifold, ψ)

Return the zero tangent vector at `ψ` (a TTvector with all-zero cores and the
same rank structure as `ψ`).
"""
function zero_vector(::TTManifold, p::TTvector{T, N}) where {T, N}
    zero_cores = [zeros(T, size(c)) for c in p.ttv_vec]
    return TTvector{T, N}(p.N, zero_cores, p.ttv_dims, copy(p.ttv_rks), copy(p.ttv_ot))
end

"""
    zero_vector!(M::TTManifold, X, ψ)

Set all cores of tangent vector `X` to zero in-place.
"""
function zero_vector!(::TTManifold, X::TTvector, ::TTvector)
    for c in X.ttv_vec
        fill!(c, 0)
    end
    return X
end

# ── Tangent-space projection (TDVP transfer-matrix projector) ──────────────────
#
# Given ψ and an ambient TT vector Z, P_ψ(Z) is the component of Z in T_ψ M.
#
# Algorithm (one-site projection with center at site 1):
#   1. Orthogonalize ψ so that all cores 2,...,N are right-isometric (center=1).
#   2. Compute right transfer matrices R_k contracting ψ† with Z from the right:
#         R_{N+1} = [[1]]
#         R_k[β', β] = Σ_s Σ_{γ,γ'} Z_k[s,β',γ] R_{k+1}[γ,γ'] conj(ψ_k[s,β,γ'])
#   3. Project the core at site 1:
#         δA_1[s, 1, β] = Σ_{β'} Z_1[s, 1, β'] R_2[β', β]
#   4. Return the TTvector [δA_1, ψ_2, ..., ψ_N].
#
# This is the standard TDVP tangent-space projector restricted to the gauge
# where the orthogonality centre is at site 1.

"""
    project!(M::TTManifold, Y, ψ, Z)

Project `Z` onto the tangent space `T_ψ M` using the TDVP transfer-matrix
projector.  `Y` receives the result and must have the same rank structure as
`ψ`.
"""
function project!(M::TTManifold{𝔽, N}, Y::TTvector, p::TTvector, X::TTvector) where {𝔽, N}
    T = eltype(p)
    # Put p in right-canonical form (orthogonality centre at site 1)
    p_can = orthogonalize(p; i = 1)

    # ── Right environments: renv[k] has shape (rX_{k-1}, rψ_{k-1}) ──────────
    #   renv[k] = Σ_s Z_k[s,:,:] * renv[k+1] * conj(ψ_k[s,:,:])'
    renv = Vector{Matrix{T}}(undef, N + 1)
    renv[N + 1] = ones(T, 1, 1)

    for k in N:-1:2
        ψk = p_can.ttv_vec[k]          # (dk, rψ_{k-1}, rψ_k)
        Zk = X.ttv_vec[k]              # (dk, rZ_{k-1}, rZ_k)
        R  = renv[k + 1]               # (rZ_k, rψ_k)
        rZ_km1 = size(Zk, 2)
        rψ_km1 = size(ψk, 2)
        Rnew = zeros(T, rZ_km1, rψ_km1)
        for s in axes(ψk, 1)
            Rnew .+= Zk[s, :, :] * R * conj(ψk[s, :, :])'
        end
        renv[k] = Rnew
    end

    # ── Project core at site 1 ────────────────────────────────────────────────
    Z1  = X.ttv_vec[1]                 # (d1, rZ_0=1, rZ_1)
    d1  = M.dims[1]
    rψ1 = size(p_can.ttv_vec[1], 3)
    δA1 = zeros(T, d1, 1, rψ1)
    for s in axes(δA1, 1)
        δA1[s, :, :] = Z1[s, :, :] * renv[2]   # (1, rZ_1) @ (rZ_1, rψ_1)
    end

    # ── Build output tangent vector ───────────────────────────────────────────
    Y.ttv_vec[1] = δA1
    for k in 2:N
        Y.ttv_vec[k] = copy(p_can.ttv_vec[k])
    end
    Y.ttv_rks .= p_can.ttv_rks
    Y.ttv_ot  .= p_can.ttv_ot
    return Y
end

# Allocating variant
function ManifoldsBase.project(M::TTManifold, p::TTvector, X::TTvector)
    Y = zero_vector(M, p)
    return project!(M, Y, p, X)
end

# ── Retraction: add + orthogonalize + SVD compress ────────────────────────────
#
# This is the "projection retraction": take the ambient-space step p + t·X
# (which creates a doubled-rank TT) and compress it back to max_rank via SVD.
# It is a valid first-order retraction on the fixed-rank manifold.

"""
    retract!(M::TTManifold, q, ψ, X, t, method)

Retract `ψ` in direction `X` by step `t` using SVD-truncation back to
`max_rank`.  The result is stored in `q`.

The default (and recommended) retraction method is `ProjectionRetraction`.
"""
function retract!(
        M::TTManifold, q::TTvector, p::TTvector, X::TTvector, t::Number,
        ::ManifoldsBase.ProjectionRetraction = ManifoldsBase.ProjectionRetraction()
    )
    r = orthogonalize(p + t * X)
    tt_compress!(r, M.max_rank)
    q.ttv_vec = r.ttv_vec
    q.ttv_rks = r.ttv_rks
    q.ttv_ot  = r.ttv_ot
    return q
end

default_retraction_method(::TTManifold, ::Type{<:TTvector}) =
    ManifoldsBase.ProjectionRetraction()

# ── Exponential map (uses retraction as approximation) ────────────────────────
#
# The exact geodesic on the TT manifold requires solving an ODE (the TDVP
# equation), which is available via `tdvp`.  Here we use the SVD retraction as
# a first-order approximation that is sufficient for most optimisation purposes.
# To get the true geodesic, call `tdvp(ψ, H=X_as_operator, ...)` directly.

"""
    exp!(M::TTManifold, q, ψ, X)

Approximate exponential map using the SVD retraction.  For the exact geodesic,
use the `tdvp` integrator directly.
"""
function exp!(M::TTManifold, q::TTvector, p::TTvector, X::TTvector)
    return retract!(M, q, p, X, one(real(eltype(p))))
end

# ── Inverse retraction (approximate log map) ──────────────────────────────────

"""
    inverse_retract!(M::TTManifold, X, ψ, q, method)

Approximate inverse retraction: `X ≈ P_ψ(q - ψ)`.  Projects the TT difference
`q - ψ` onto the tangent space at `ψ`.
"""
function ManifoldsBase.inverse_retract!(
        M::TTManifold, X::TTvector, p::TTvector, q::TTvector,
        ::ManifoldsBase.LogarithmicInverseRetraction = ManifoldsBase.LogarithmicInverseRetraction()
    )
    diff = q - p
    return project!(M, X, p, diff)
end

# ── Vector transport by projection ────────────────────────────────────────────

"""
    vector_transport_to!(M::TTManifold, Y, ψ, X, q, ::ProjectionTransport)

Transport tangent vector `X` at `ψ` to the tangent space at `q` by projecting
`X` (viewed as an ambient TT vector) onto `T_q M`.
"""
function vector_transport_to!(
        M::TTManifold, Y::TTvector, ::TTvector, X::TTvector, q::TTvector,
        ::ManifoldsBase.ProjectionTransport
    )
    return project!(M, Y, q, X)
end

# ── Random point on the manifold ──────────────────────────────────────────────

"""
    rand(M::TTManifold)

Return a random point on `M` (a random TT vector with effective bond dimensions).
"""
function Base.rand(M::TTManifold{𝔽, N}) where {𝔽, N}
    r = _effective_ranks(M.dims, M.max_rank)
    return rand_tt(M.dims, r)
end

# Override allocate_result(M, rand) so ManifoldsBase doesn't try to build a
# flat Array{T, rs...} for a non-array manifold point.
function allocate_result(M::TTManifold{𝔽, N}, ::typeof(rand)) where {𝔽, N}
    r = _effective_ranks(M.dims, M.max_rank)
    return zeros_tt(Float64, M.dims, r)
end

end # module TensorTrainNumericsManifoldExt
