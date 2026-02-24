module TensorTrainNumericsChainRulesCoreExt
using TensorTrainNumerics
using ChainRulesCore
import ChainRulesCore: rrule, NoTangent, Tangent, @thunk
using TensorOperations

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers: left / right environment sweeps for TTvector dot product
#
# Core layout: ttv_vec[k] has shape (dim_k, left_rank, right_rank), i.e. [z, α, a].
# The dot product is:
#   L[0] = [[1]]
#   L[k][a, b] = Σ_{z,α,β} A_k[z,α,a] * B_k[z,β,b] * L[k-1][α,β]
# ─────────────────────────────────────────────────────────────────────────────

function _tt_left_envs(A::TTvector{T, M}, B::TTvector{T, M}) where {T, M}
    N = A.N
    Ls = Vector{Matrix{T}}(undef, N + 1)
    Ls[1] = ones(T, 1, 1)
    @inbounds for k in 1:N
        A_k = A.ttv_vec[k]
        B_k = B.ttv_vec[k]
        L_prev = Ls[k]
        @tensor L_next[a, b] := A_k[z, α, a] * B_k[z, β, b] * L_prev[α, β]
        Ls[k + 1] = L_next
    end
    return Ls
end

# Right environments, G[N+1] = fill(Δ, 1, 1), swept right-to-left.
# G[k][α,β] = Σ_{z,a,b} A_k[z,α,a] * B_k[z,β,b] * G[k+1][a,b]
function _tt_right_envs(A::TTvector{T, M}, B::TTvector{T, M}, Δ::Number) where {T, M}
    N = A.N
    ΔT = convert(T, real(Δ))
    Gs = Vector{Matrix{T}}(undef, N + 1)
    Gs[N + 1] = fill(ΔT, 1, 1)
    @inbounds for k in N:-1:1
        A_k = A.ttv_vec[k]
        B_k = B.ttv_vec[k]
        G_next = Gs[k + 1]
        @tensor G_prev[α, β] := A_k[z, α, a] * B_k[z, β, b] * G_next[a, b]
        Gs[k] = G_prev
    end
    return Gs
end

# ─────────────────────────────────────────────────────────────────────────────
# rrule for dot(A, B) — the left-right environment method
#
# Gradient of dot(A,B) w.r.t. A_k[z, α, a]:
#   Σ_{β,b} B_k[z,β,b] * G[k+1][a,b] * L[k-1][α,β]
#
# Gradient of dot(A,B) w.r.t. B_k[z, β, b]:
#   Σ_{α,a} A_k[z,α,a] * G[k+1][a,b] * L[k-1][α,β]
#
# where L[k-1] = Ls[k] and G[k+1] = Gs[k+1].
# ─────────────────────────────────────────────────────────────────────────────

function ChainRulesCore.rrule(::typeof(TensorTrainNumerics.dot),
                              A::TTvector{T, M}, B::TTvector{T, M}) where {T, M}
    Ls = _tt_left_envs(A, B)
    result = Ls[A.N + 1][1, 1]

    function dot_pullback(Δ)
        Gs = _tt_right_envs(A, B, Δ)
        N = A.N

        A_core_grads = similar(A.ttv_vec)
        B_core_grads = similar(B.ttv_vec)

        @inbounds for k in 1:N
            A_k  = A.ttv_vec[k]
            B_k  = B.ttv_vec[k]
            Lprev = Ls[k]       # left env before site k  (shape: rA[k] × rB[k])
            Gnext = Gs[k + 1]   # right env after  site k  (shape: rA[k+1] × rB[k+1])

            @tensor grad_A[z, α, a] := B_k[z, β, b] * Gnext[a, b] * Lprev[α, β]
            @tensor grad_B[z, β, b] := A_k[z, α, a] * Gnext[a, b] * Lprev[α, β]

            A_core_grads[k] = grad_A
            B_core_grads[k] = grad_B
        end

        ∂A = Tangent{typeof(A)}(; ttv_vec = A_core_grads)
        ∂B = Tangent{typeof(B)}(; ttv_vec = B_core_grads)
        return NoTangent(), ∂A, ∂B
    end

    return result, dot_pullback
end

# ─────────────────────────────────────────────────────────────────────────────
# rrule for norm(x)
#
# norm(x) = sqrt(dot(x, x))
# d(norm)/d(x_k) = (1/norm(x)) * Σ_{β,b} x_k[z,β,b] * G[k+1][a,b] * L[k-1][α,β]
#
# (the factor of 2 from d(dot(x,x))/dx = 2*(env) cancels the 1/2 from d(sqrt)/dt = 1/(2*sqrt))
# ─────────────────────────────────────────────────────────────────────────────

function ChainRulesCore.rrule(::typeof(TensorTrainNumerics.norm), x::TTvector{T, M}) where {T, M}
    n = TensorTrainNumerics.norm(x)
    function norm_pullback(Δ)
        N = x.N
        if iszero(n)
            ∂x = Tangent{typeof(x)}(; ttv_vec = zero.(x.ttv_vec))
        else
            scale = real(Δ) / n   # = Δ/n  (the 2 from d(dot(x,x))/dx and 1/(2n) from sqrt cancel)
            Ls = _tt_left_envs(x, x)
            Gs = _tt_right_envs(x, x, scale)

            core_grads = similar(x.ttv_vec)
            @inbounds for k in 1:N
                x_k   = x.ttv_vec[k]
                Lprev = Ls[k]
                Gnext = Gs[k + 1]
                # G is already scaled by (Δ/n), which combines:
                #   - 1/(2n) from d(sqrt)/d(dot(x,x))
                #   - ×2 from d(dot(x,x))/dx summing both positions (cancel)
                # Result: grad = Σ_{β,b} x_k[z,β,b] * G[k+1][a,b] * L[k-1][α,β]
                @tensor grad_k[z, α, a] := x_k[z, β, b] * Gnext[a, b] * Lprev[α, β]
                core_grads[k] = grad_k
            end
            ∂x = Tangent{typeof(x)}(; ttv_vec = core_grads)
        end
        return NoTangent(), ∂x
    end
    return n, norm_pullback
end

# ─────────────────────────────────────────────────────────────────────────────
# rrule for A * v  (TToperator × TTvector → TTvector)
#
# The forward pass produces y where:
#   y_k[i, (α_{k-1}, ν_{k-1}), (α_k, ν_k)] = Σ_j A_k[i,j,α_{k-1},α_k] * v_k[j,ν_{k-1},ν_k]
#
# (rank indices are interleaved/flattened in the stored core y.ttv_vec[k])
#
# Pullback given Δy (tangent of y, a Tangent{TTvector}):
#   ∂v_k[j, ν_prev, ν_next] = Σ_{i,α_prev,α_next} A_k[i,j,α_prev,α_next] * Δy_k[i,α_prev,ν_prev,α_next,ν_next]
#   ∂A_k[i, j, α_prev, α_next] = Σ_{ν_prev,ν_next} v_k[j,ν_prev,ν_next] * Δy_k[i,α_prev,ν_prev,α_next,ν_next]
# ─────────────────────────────────────────────────────────────────────────────

function ChainRulesCore.rrule(::typeof(*),
                              A::TToperator{T, M}, v::TTvector{T, M}) where {T, M}
    y = A * v
    function mul_pullback(Δy_tangent)
        # Extract per-core tangents for y
        Δy_cores = Δy_tangent.ttv_vec

        N = v.N
        v_core_grads = similar(v.ttv_vec)
        A_core_grads = similar(A.tto_vec)

        @inbounds for k in 1:N
            A_k  = A.tto_vec[k]            # shape: (dim_k, dim_k, tto_rks[k], tto_rks[k+1])
            v_k  = v.ttv_vec[k]            # shape: (dim_k, v_rks[k], v_rks[k+1])
            Δy_k = reshape(Δy_cores[k],    # shape: (dim_k, A.tto_rks[k], v.ttv_rks[k],
                           y.ttv_dims[k],  #                 A.tto_rks[k+1], v.ttv_rks[k+1])
                           A.tto_rks[k],
                           v.ttv_rks[k],
                           A.tto_rks[k + 1],
                           v.ttv_rks[k + 1])

            @tensor ∂v_k[j, ν_prev, ν_next] := A_k[i, j, α_prev, α_next] * Δy_k[i, α_prev, ν_prev, α_next, ν_next]
            @tensor ∂A_k[i, j, α_prev, α_next] := v_k[j, ν_prev, ν_next] * Δy_k[i, α_prev, ν_prev, α_next, ν_next]

            v_core_grads[k] = ∂v_k
            A_core_grads[k] = ∂A_k
        end

        ∂v = Tangent{typeof(v)}(; ttv_vec = v_core_grads)
        ∂A = Tangent{typeof(A)}(; tto_vec = A_core_grads)
        return NoTangent(), ∂A, ∂v
    end
    return y, mul_pullback
end

end
