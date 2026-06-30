module TensorTrainNumericsChainRulesCoreExt

using TensorTrainNumerics
using ChainRulesCore
using TensorOperations
import ChainRulesCore: rrule, NoTangent, Tangent, ZeroTangent, AbstractZero, unthunk

function _tt_left_envs(A::TTvector{T, M}, B::TTvector{T, M}) where {T, M}
    N = A.N
    Ls = Vector{Matrix{T}}(undef, N + 1)
    Ls[1] = ones(T, 1, 1)
    @inbounds for k in 1:N
        Ak = A.ttv_vec[k]
        Bk = B.ttv_vec[k]
        Lp = Ls[k]
        @tensor Ln[a, b] := conj(Ak[z, α, a]) * Bk[z, β, b] * Lp[α, β]
        Ls[k + 1] = Ln
    end
    return Ls
end

function _tt_right_envs(A::TTvector{T, M}, B::TTvector{T, M}) where {T, M}
    N = A.N
    Gs = Vector{Matrix{T}}(undef, N + 1)
    Gs[N + 1] = ones(T, 1, 1)
    @inbounds for k in N:-1:1
        Ak = A.ttv_vec[k]
        Bk = B.ttv_vec[k]
        Gn = Gs[k + 1]
        @tensor Gp[α, β] := conj(Ak[z, α, a]) * Bk[z, β, b] * Gn[a, b]
        Gs[k] = Gp
    end
    return Gs
end

function rrule(
        ::typeof(TensorTrainNumerics.dot),
        A::TTvector{T, M}, B::TTvector{T, M}
    ) where {T, M}
    Ls = _tt_left_envs(A, B)
    Ω = Ls[A.N + 1][1, 1]
    function dot_pullback(Ω̄)
        Δ = unthunk(Ω̄)
        Gs = _tt_right_envs(A, B)
        N = A.N
        Ā = Vector{Array{T, 3}}(undef, N)
        B̄ = Vector{Array{T, 3}}(undef, N)
        @inbounds for k in 1:N
            Ak = A.ttv_vec[k]
            Bk = B.ttv_vec[k]
            Lp = Ls[k]
            Gn = Gs[k + 1]
            @tensor EB[z, α, a] := Bk[z, β, b] * Lp[α, β] * Gn[a, b]
            @tensor EA[z, β, b] := conj(Ak[z, α, a]) * Lp[α, β] * Gn[a, b]
            Ā[k] = conj(Δ) .* EB
            B̄[k] = Δ .* EA
        end
        return (
            NoTangent(),
            Tangent{TTvector{T, M}}(ttv_vec = Ā),
            Tangent{TTvector{T, M}}(ttv_vec = B̄),
        )
    end
    return Ω, dot_pullback
end

function rrule(::typeof(*), H::TToperator{T, N}, ψ::TTvector{T, N}) where {T, N}
    Y = H * ψ
    function mul_pullback(Ȳraw)
        Ȳ = unthunk(Ȳraw)
        Ȳ isa AbstractZero && return (NoTangent(), NoTangent(), NoTangent())
        Ȳv = Ȳ.ttv_vec                                  # Tangent{TTvector} or TTvector
        ψ̄ = Vector{Array{T, 3}}(undef, N)
        @inbounds for k in 1:N
            Hk = H.tto_vec[k]                            # (dout, din, rHl, rHr)
            rHl = H.tto_rks[k]
            rHr = H.tto_rks[k + 1]
            rψl = ψ.ttv_rks[k]
            rψr = ψ.ttv_rks[k + 1]
            dout = size(Hk, 1)
            Yb = reshape(Ȳv[k], (dout, rHl, rψl, rHr, rψr))
            @tensor pk[j, νl, νr] := conj(Hk[i, j, αl, αr]) * Yb[i, αl, νl, αr, νr]
            ψ̄[k] = pk
        end
        return (NoTangent(), NoTangent(), Tangent{TTvector{T, N}}(ttv_vec = ψ̄))
    end
    return Y, mul_pullback
end

end
