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
            # B enters linearly, so its reverse rule conjugates the complete coefficient.
            B̄[k] = Δ .* conj.(EA)
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
        Ȳ isa AbstractZero && return (NoTangent(), ZeroTangent(), ZeroTangent())
        Ȳv = unthunk(Ȳ.ttv_vec)                         # Tangent{TTvector} or TTvector
        Ȳv isa AbstractZero && return (NoTangent(), ZeroTangent(), ZeroTangent())
        H̄ = Vector{Array{T, 4}}(undef, N)
        ψ̄ = Vector{Array{T, 3}}(undef, N)
        for k in eachindex(H.tto_vec, ψ.ttv_vec, Ȳv)
            Hk = H.tto_vec[k]                            # (dout, din, rHl, rHr)
            ψk = ψ.ttv_vec[k]
            Yk = unthunk(Ȳv[k])
            if Yk isa AbstractZero
                H̄[k], ψ̄[k] = zero(Hk), zero(ψk)
                continue
            end
            rHl = H.tto_rks[k]
            rHr = H.tto_rks[k + 1]
            rψl = ψ.ttv_rks[k]
            rψr = ψ.ttv_rks[k + 1]
            dout = size(Hk, 1)
            Yb = reshape(Yk, (dout, rHl, rψl, rHr, rψr))
            @tensor hk[i, j, αl, αr] := conj(ψk[j, νl, νr]) * Yb[i, αl, νl, αr, νr]
            @tensor pk[j, νl, νr] := conj(Hk[i, j, αl, αr]) * Yb[i, αl, νl, αr, νr]
            H̄[k] = hk
            ψ̄[k] = pk
        end
        return (
            NoTangent(),
            Tangent{TToperator{T, N}}(tto_vec = H̄),
            Tangent{TTvector{T, N}}(ttv_vec = ψ̄),
        )
    end
    return Y, mul_pullback
end

function rrule(::typeof(hadamard), x::TTvector{T, N}, y::TTvector{T, N}) where {T, N}
    z = hadamard(x, y)
    function hadamard_pullback(z̄raw)
        z̄ = unthunk(z̄raw)
        z̄ isa AbstractZero && return (NoTangent(), ZeroTangent(), ZeroTangent())
        z̄v = unthunk(z̄.ttv_vec)
        z̄v isa AbstractZero && return (NoTangent(), ZeroTangent(), ZeroTangent())
        x̄ = Vector{Array{T, 3}}(undef, N)
        ȳ = Vector{Array{T, 3}}(undef, N)
        for k in eachindex(x.ttv_vec, y.ttv_vec, z̄v)
            xk, yk = x.ttv_vec[k], y.ttv_vec[k]
            Zk = unthunk(z̄v[k])
            if Zk isa AbstractZero
                x̄[k], ȳ[k] = zero(xk), zero(yk)
                continue
            end
            n, rxl, rxr = size(xk)
            _, ryl, ryr = size(yk)
            # kron(x_slice, y_slice) makes y's bond indices vary fastest.
            Zb = reshape(Zk, (n, ryl, rxl, ryr, rxr))
            x̄[k], ȳ[k] = similar(xk), similar(yk)
            for s in axes(xk, 1)
                xs, ys = view(xk, s, :, :), view(yk, s, :, :)
                Zs = view(Zb, s, :, :, :, :)
                dx, dy = view(x̄[k], s, :, :), view(ȳ[k], s, :, :)
                @tensor dx[a, b] = conj(ys[c, d]) * Zs[c, a, d, b]
                @tensor dy[c, d] = conj(xs[a, b]) * Zs[c, a, d, b]
            end
        end
        return (
            NoTangent(),
            Tangent{TTvector{T, N}}(ttv_vec = x̄),
            Tangent{TTvector{T, N}}(ttv_vec = ȳ),
        )
    end
    return z, hadamard_pullback
end

# Cotangent cores stored under `field` of an output cotangent, or `nothing` when the
# whole cotangent is zero.
function _cotangent_cores(Ȳraw, field::Symbol)
    Ȳ = unthunk(Ȳraw)
    Ȳ isa AbstractZero && return nothing
    cores = unthunk(getproperty(Ȳ, field))
    cores isa AbstractZero && return nothing
    return cores
end

# Dense cotangent for one output core; `like` is the primal output core.
function _dense_core(Ck, like)
    Ck = unthunk(Ck)
    return Ck isa AbstractZero ? zero(like) : Ck
end

# Adjoint of the block embedding in `+`. The last two axes of every core are the rank
# axes. The first core stacks the summands along its right rank axis, the last core
# along its left rank axis, and interior cores are block diagonal, with the first
# summand in the leading block.
function _split_sum_cores(xs, ys, Zs)
    d = length(xs)
    d == 1 && return [copy(Zs[1])], [copy(Zs[1])]
    x̄, ȳ = similar(xs), similar(ys)
    for k in 1:d
        Zk = Zs[k]
        phys = ntuple(_ -> Colon(), ndims(Zk) - 2)
        rxl, rxr = size(xs[k])[(end - 1):end]
        rzl, rzr = size(Zk)[(end - 1):end]
        lx, ly = k == 1 ? (Colon(), Colon()) : (1:rxl, (rxl + 1):rzl)
        rx, ry = k == d ? (Colon(), Colon()) : (1:rxr, (rxr + 1):rzr)
        x̄[k] = Zk[phys..., lx, rx]
        ȳ[k] = Zk[phys..., ly, ry]
    end
    return x̄, ȳ
end

function rrule(::typeof(+), x::TTvector{T, N}, y::TTvector{T, N}) where {T, N}
    z = x + y
    function add_pullback(z̄raw)
        Z = _cotangent_cores(z̄raw, :ttv_vec)
        Z === nothing && return (NoTangent(), ZeroTangent(), ZeroTangent())
        Zd = [_dense_core(Z[k], z.ttv_vec[k]) for k in eachindex(z.ttv_vec)]
        x̄, ȳ = _split_sum_cores(x.ttv_vec, y.ttv_vec, Zd)
        return (
            NoTangent(),
            Tangent{TTvector{T, N}}(ttv_vec = x̄),
            Tangent{TTvector{T, N}}(ttv_vec = ȳ),
        )
    end
    return z, add_pullback
end

function rrule(::typeof(+), A::TToperator{T, N}, B::TToperator{T, N}) where {T, N}
    C = A + B
    function add_pullback(C̄raw)
        Z = _cotangent_cores(C̄raw, :tto_vec)
        Z === nothing && return (NoTangent(), ZeroTangent(), ZeroTangent())
        Zd = [_dense_core(Z[k], C.tto_vec[k]) for k in eachindex(C.tto_vec)]
        Ā, B̄ = _split_sum_cores(A.tto_vec, B.tto_vec, Zd)
        return (
            NoTangent(),
            Tangent{TToperator{T, N}}(tto_vec = Ā),
            Tangent{TToperator{T, N}}(tto_vec = B̄),
        )
    end
    return C, add_pullback
end

# Scalar multiplication scales only core `i = _scale_site(...)`. For `a == 0` the primal
# returns all-zero cores, whose value does not depend on `a`; the rule instead scales
# core `i` by zero, which represents the same zero tensor and keeps ∂(a x)/∂a = x.
function _scale_rrule(a, cores, i, ::Type{T}) where {T}
    X = [convert(Array{T, ndims(c)}, c) for c in cores]
    X[i] = convert(T, a) * X[i]
    function scale_cotangents(Z)
        Zd = [_dense_core(Z[k], X[k]) for k in eachindex(X)]
        ā = ProjectTo(a)(sum(conj.(cores[i]) .* Zd[i]))
        c̄ = [ProjectTo(cores[k])(k == i ? conj(a) .* Zd[k] : Zd[k]) for k in eachindex(cores)]
        return ā, c̄
    end
    return X, scale_cotangents
end

function rrule(::typeof(*), a::Number, x::TTvector{R, N}) where {R <: Number, N}
    T = promote_type(typeof(a), R)
    X, scale_cotangents = _scale_rrule(a, x.ttv_vec, TensorTrainNumerics._scale_site(x.ttv_ot), T)
    y = TTvector{T, N}(x.N, X, x.ttv_dims, copy(x.ttv_rks), copy(x.ttv_ot))
    function scale_pullback(ȳraw)
        Z = _cotangent_cores(ȳraw, :ttv_vec)
        Z === nothing && return (NoTangent(), ZeroTangent(), ZeroTangent())
        ā, x̄ = scale_cotangents(Z)
        return (NoTangent(), ā, Tangent{TTvector{R, N}}(ttv_vec = x̄))
    end
    return y, scale_pullback
end

function rrule(::typeof(*), a::Number, A::TToperator{R, N}) where {R <: Number, N}
    T = promote_type(typeof(a), R)
    X, scale_cotangents = _scale_rrule(a, A.tto_vec, TensorTrainNumerics._scale_site(A.tto_ot), T)
    B = TToperator{T, N}(A.N, X, A.tto_dims, copy(A.tto_rks), copy(A.tto_ot))
    function scale_pullback(B̄raw)
        Z = _cotangent_cores(B̄raw, :tto_vec)
        Z === nothing && return (NoTangent(), ZeroTangent(), ZeroTangent())
        ā, Ā = scale_cotangents(Z)
        return (NoTangent(), ā, Tangent{TToperator{R, N}}(tto_vec = Ā))
    end
    return B, scale_pullback
end

end
