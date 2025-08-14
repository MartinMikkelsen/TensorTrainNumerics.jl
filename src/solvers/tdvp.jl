using KrylovKit
using ProgressMeter
using TensorOperations
using LinearAlgebra
using TensorTrainNumerics

function _sync_ranks_from_lsr!(ψ::TTvector, A_lsr::Vector{<:AbstractArray})
    N = ψ.N
    new_rks = similar(ψ.ttv_rks)
    @inbounds for k in 1:N
        new_rks[k] = size(A_lsr[k], 1)
    end
    new_rks[N + 1] = size(A_lsr[N], 3)
    ψ.ttv_rks = new_rks
    ψ.ttv_ot .= 0
    return ψ
end

_real_or_complex_t(z) = isreal(z) ? real(z) : z

function svdtrunc(A; max_bond = max(size(A)...), truncerr = 0.0)
    F = svd(A)
    d = min(max_bond, count(F.S .>= truncerr))
    return F.U[:, 1:d], diagm(0 => F.S[1:d]), F.Vt[1:d, :]
end

_to_lsr(A) = permutedims(A, (2, 1, 3))
_to_slr(A) = permutedims(A, (2, 1, 3))

_mpo_to_asbs(M) = permutedims(M, (3, 1, 4, 2))

_dot3(X, Y) = LinearAlgebra.dot(vec(X), vec(Y))

function _applyH1_lsr(AC, FL, FR, M)
    return @tensor HAC[α, s, β] := FL[α, a, α′] * AC[α′, s′, β′] * M[a, s, b, s′] * FR[β′, b, β]
end

function _applyH0(C, FL, FR)
    return @tensor HC[α, β] := FL[α, a, α′] * C[α′, β′] * FR[β′, a, β]
end

function _update_left_env(A, M, FL)
    return @tensor FLnext[α, a, β] := FL[α′, a′, β′] * A[β′, s′, β] * M[a′, s, a, s′] * conj(A[α′, s, α])
end

function _update_right_env(A, M, FR)
    return @tensor FRprev[α, a, β] := A[α, s′, α′] * FR[α′, a′, β′] * M[a, s, a′, s′] * conj(A[β, s, β′])
end

function tdvp1sweep!(
        dt, ψ::TTvector{T, N}, H::TToperator{T, N}, F::Union{Nothing, Vector{Any}} = nothing;
        verbose::Bool = true, kwargs...
    ) where {T <: Number, N}

    Tc = (dt isa Complex || T <: Complex) ? Complex{real(T)} : T
    Nsites = ψ.N

    A_lsr = [permutedims(ψ.ttv_vec[k], (2, 1, 3)) for k in 1:Nsites]
    M_asbs = [permutedims(H.tto_vec[k], (3, 1, 4, 2)) for k in 1:Nsites]

    if F === nothing
        F = Vector{Any}(undef, Nsites + 2)
        F[1] = ones(Tc, 1, 1, 1)[:, :, :]
        F[end] = ones(Tc, 1, 1, 1)[:, :, :]
        for k in Nsites:-1:1
            F[k + 1] = _update_right_env(A_lsr[k], M_asbs[k], F[k + 2])
        end
    else
        for i in eachindex(F)
            F[i] = Tc.(F[i])
        end
    end

    AC = Tc.(A_lsr[1])

    for k in 1:(Nsites - 1)
        H1 = x -> _applyH1_lsr(x, F[k], F[k + 2], M_asbs[k])
        t1 = _real_or_complex_t(-im * dt)
        AC, _ = exponentiate(H1, t1, AC; ishermitian = true, kwargs...)
        if verbose
            E = _dot3(AC, H1(AC))
            @info "TDVP sweep:" site=k energy=real(E)
        end

        Dl, d, Dr = size(AC)
        Aqr = reshape(AC, Dl * d, Dr)
        qrf = qr(Aqr)
        r = min(size(Aqr, 1), size(Aqr, 2))
        Qthin = Matrix(qrf.Q)[:, 1:r]
        Rthin = qrf.R[1:r, :]

        AL = reshape(Qthin, Dl, d, r)
        A_lsr[k] = AL
        F[k + 1] = _update_left_env(AL, M_asbs[k], F[k])

        C = Rthin
        H0 = X -> _applyH0(X, F[k + 1], F[k + 2])
        t0 = _real_or_complex_t(+im * dt)
        C, _ = exponentiate(H0, t0, C; ishermitian = true, kwargs...)
        if verbose
            E0 = _dot3(C, H0(C))
            @info "TDVP sweep:" sites="$(k):$(k+1)" energy=real(E0)
        end

        @tensor AC[α, s, β] := C[α, γ] * A_lsr[k + 1][γ, s, β]
    end

    k = Nsites
    H1N = x -> _applyH1_lsr(x, F[k], F[k + 2], M_asbs[k])
    tN = _real_or_complex_t(-im * dt)
    AC, _ = exponentiate(H1N, tN, AC; ishermitian = true, kwargs...)
    if verbose
        E = _dot3(AC, H1N(AC))
        @info "TDVP sweep:" site=k energy=real(E)
    end

    for k in (Nsites - 1):-1:1
        Dl, d, Dr = size(AC)
        A = reshape(AC, Dl, d * Dr)
        qrf = qr(A')
        r = min(size(A, 1), size(A, 2))
        Qthin = Matrix(qrf.Q)[:, 1:r]
        Rthin = qrf.R[1:r, :]
        L = Rthin'
        A_r = reshape(Qthin', r, d, Dr)

        A_lsr[k + 1] = A_r
        F[k + 2] = _update_right_env(A_r, M_asbs[k + 1], F[k + 3])

        C = L
        H0 = X -> _applyH0(X, F[k + 1], F[k + 2])
        t0 = _real_or_complex_t(+im * dt)
        C, _ = exponentiate(H0, t0, C; ishermitian = true, kwargs...)
        if verbose
            E0 = _dot3(C, H0(C))
            @info "TDVP sweep:" sites="$(k):$(k+1)" energy=real(E0)
        end

        @tensor AC[α, s, β] := A_lsr[k][α, s, γ] * C[γ, β]

        H1k = x -> _applyH1_lsr(x, F[k], F[k + 2], M_asbs[k])
        tk = _real_or_complex_t(-im * dt)
        AC, _ = exponentiate(H1k, tk, AC; ishermitian = true, kwargs...)
        if verbose
            E = _dot3(AC, H1k(AC))
            @info "TDVP sweep:" site=k energy=real(E)
        end
    end

    A_lsr[1] = AC
    for k in 1:Nsites
        ψ.ttv_vec[k] = permutedims(A_lsr[k], (2, 1, 3))
    end
    _sync_ranks_from_lsr!(ψ, A_lsr)
    return ψ, F
end

function tdvp(
        H::TToperator,
        u₀::TTvector,
        steps::Vector{Float64};
        normalize::Bool = true,
        return_error::Bool = false,
        sweeps::Int = 1,
        carry_env::Bool = true,
        verbose::Bool = false,
        imaginary_time::Bool = false,
        kwargs...
    )
    ψ = orthogonalize(u₀)

    wants_complex = !imaginary_time
    if wants_complex && eltype(ψ) <: Real
        ψ = complex(ψ)
    end
    Hc = (wants_complex && eltype(H) <: Real) ? complex(H) : H

    ψ_prev = ψ
    F = nothing

    @showprogress for h in steps
        ψ_prev_step = ψ
        dt_eff = imaginary_time ? (+im * h) : (complex(1.0) * h)
        for s in 1:sweeps
            F_in = carry_env ? F : nothing
            ψ, F = tdvp1sweep!(dt_eff, ψ, Hc, F_in; verbose = verbose, kwargs...)
        end
        if normalize
            ψ = (1 / norm(ψ)) * ψ
        end
        ψ = orthogonalize(ψ)
        ψ_prev = ψ_prev_step
    end

    if return_error
        h = steps[end]
        if imaginary_time
            residual = (ψ - ψ_prev) * (1 / h) + (Hc * ψ)
        else
            residual = (ψ - ψ_prev) * (1 / h) + im * (Hc * ψ)
        end
        rel_error = norm(residual) / norm(ψ)
        return ψ, rel_error
    end
    return ψ
end

function _applyH2_lsr(AAC, FL, FR, M1, M2)
    return @tensor HAAC[α, s1, s2, β] := FL[α, a, α′] * AAC[α′, s1′, s2′, β′] *
        M1[a, s1, b, s1′] * M2[b, s2, c, s2′] * FR[β′, c, β]
end

function tdvp2sweep!(
        dt, ψ::TTvector{T, N}, H::TToperator{T, N}, F::Union{Nothing, Vector{Any}} = nothing;
        verbose::Bool = true, max_bond::Int = typemax(Int), truncerr::Real = 0.0, kwargs...
    ) where {T <: Number, N}

    Tc = (dt isa Complex || T <: Complex) ? Complex{real(T)} : T
    Nsites = ψ.N

    A_lsr = [permutedims(ψ.ttv_vec[k], (2, 1, 3)) for k in 1:Nsites]
    M_asbs = [permutedims(H.tto_vec[k], (3, 1, 4, 2)) for k in 1:Nsites]

    if F === nothing
        F = Vector{Any}(undef, Nsites + 2)
        F[1] = ones(Tc, 1, 1, 1)[:, :, :]
        F[end] = ones(Tc, 1, 1, 1)[:, :, :]
        for k in Nsites:-1:1
            F[k + 1] = _update_right_env(A_lsr[k], M_asbs[k], F[k + 2])
        end
    else
        for i in eachindex(F)
            F[i] = Tc.(F[i])
        end
    end

    AC = Tc.(A_lsr[1])

    for k in 1:(Nsites - 1)
        @tensor AAC[α, s1, s2, β] := AC[α, s1, γ] * A_lsr[k + 1][γ, s2, β]
        H2 = X -> _applyH2_lsr(X, F[k], F[k + 3], M_asbs[k], M_asbs[k + 1])
        t2 = _real_or_complex_t(-im * dt)
        AAC, _ = exponentiate(H2, t2, AAC; ishermitian = true, kwargs...)
        if verbose
            E = _dot3(vec(conj(AAC)), vec(H2(AAC)))
            @info "2TDVP sweep:" sites="$(k):$(k + 1)"   energy = real(E)
        end

        Dl, d1, d2, Dr = size(AAC)
        U, S, Vt = svdtrunc(
            reshape(AAC, Dl * d1, d2 * Dr);
            max_bond = max_bond, truncerr = truncerr
        )

        AL = reshape(U, Dl, d1, size(U, 2))
        A_lsr[k] = AL
        F[k + 1] = _update_left_env(AL, M_asbs[k], F[k])

        AC = reshape(S * Vt, size(S, 1), d2, Dr)
    end

    for k in (Nsites - 1):-1:1
        @tensor AAC[α, s1, s2, β] := A_lsr[k][α, s1, γ] * AC[γ, s2, β]
        H2 = X -> _applyH2_lsr(X, F[k], F[k + 3], M_asbs[k], M_asbs[k + 1])
        t2 = _real_or_complex_t(-im * dt)
        AAC, _ = exponentiate(H2, t2, AAC; ishermitian = true, kwargs...)
        if verbose
            E = _dot3(vec(conj(AAC)), vec(H2(AAC)))
            @info "2TDVP sweep:" sites="$(k):$(k+1)" energy=real(E)
        end

        Dl, d1, d2, Dr = size(AAC)
        U, S, Vt = svdtrunc(
            reshape(AAC, Dl * d1, d2 * Dr);
            max_bond = max_bond, truncerr = truncerr
        )

        AR = reshape(Vt, size(Vt, 1), d2, Dr)
        A_lsr[k + 1] = AR
        F[k + 2] = _update_right_env(AR, M_asbs[k + 1], F[k + 3])

        AC = reshape(U * S, Dl, d1, size(S, 2))
    end

    A_lsr[1] = AC
    for k in 1:Nsites
        ψ.ttv_vec[k] = permutedims(A_lsr[k], (2, 1, 3))
    end
    _sync_ranks_from_lsr!(ψ, A_lsr)
    return ψ, F
end

function tdvp2(
        H::TToperator,
        u₀::TTvector,
        steps::Vector{Float64};
        normalize::Bool = true,
        return_error::Bool = false,
        sweeps::Int = 1,
        carry_env::Bool = true,
        verbose::Bool = false,
        max_bond::Int = typemax(Int),
        truncerr::Real = 0.0,
        imaginary_time::Bool = false,
        kwargs...
    )
    ψ = orthogonalize(u₀)

    wants_complex = !imaginary_time
    if wants_complex && eltype(ψ) <: Real
        ψ = complex(ψ)
    end
    Hc = (wants_complex && eltype(H) <: Real) ? complex(H) : H

    ψ_prev = ψ
    F = nothing

    @showprogress for h in steps
        ψ_prev_step = ψ
        dt_eff = imaginary_time ? (+im * h) : (complex(1.0) * h)
        for s in 1:sweeps
            F_in = carry_env ? F : nothing
            ψ, F = tdvp2sweep!(
                dt_eff, ψ, Hc, F_in;
                verbose = verbose, max_bond = max_bond, truncerr = truncerr, kwargs...
            )
        end
        if normalize
            ψ = (1 / norm(ψ)) * ψ
        end
        ψ = orthogonalize(ψ)
        ψ_prev = ψ_prev_step
    end

    if return_error
        h = steps[end]
        if imaginary_time
            residual = (ψ - ψ_prev) * (1 / h) + (Hc * ψ)
        else
            residual = (ψ - ψ_prev) * (1 / h) + im * (Hc * ψ)
        end
        rel_error = norm(residual) / norm(ψ)
        return ψ, rel_error
    end
    return ψ
end
