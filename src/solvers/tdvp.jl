using KrylovKit
using ProgressMeter
using TensorOperations
using LinearAlgebra
using TensorTrainNumerics

function _sync_ranks_from_lsr!(ψ::TTvector, A_lsr::Vector{<:AbstractArray})
    N = ψ.N
    new_rks = similar(ψ.ttv_rks)
    @inbounds for k in 1:N
        new_rks[k] = size(A_lsr[k], 1)           # left bond of site k
    end
    new_rks[N + 1] = size(A_lsr[N], 3)             # right bond of last site
    ψ.ttv_rks = new_rks
    ψ.ttv_ot .= 0
    return ψ
end

function svdtrunc(A; truncdim = max(size(A)...), truncerr = 0.0)
    F = svd(A)
    d = min(truncdim, count(F.S .>= truncerr))
    return F.U[:, 1:d], diagm(0 => F.S[1:d]), F.Vt[1:d, :]
end

_to_lsr(A) = permutedims(A, (2, 1, 3))
_to_slr(A) = permutedims(A, (2, 1, 3))

# convert MPO core from (s_out, s_in, a, b) -> (a, s_out, b, s_in)
_mpo_to_asbs(M) = permutedims(M, (3, 1, 4, 2))

# inner product ⟨X|Y⟩ for 3-tensors
_dot3(X, Y) = LinearAlgebra.dot(vec(X), vec(Y))

# --- Effective ops in (Dl, d, Dr) layout (lsr) ---

# H • AC  (AC: Dl × d × Dr), FL: Dl×a×Dl, FR: Dr×b×Dr, M: a×d_out×b×d_in
function _applyH1_lsr(AC, FL, FR, M)
    return @tensor HAC[α, s, β] := FL[α, a, α′] * AC[α′, s′, β′] * M[a, s, b, s′] * FR[β′, b, β]
end

# H • C  (C: Dl × Dr), FL: Dl×a×Dl, FR: Dr×a×Dr
function _applyH0(C, FL, FR)
    return @tensor HC[α, β] := FL[α, a, α′] * C[α′, β′] * FR[β′, a, β]
end

# --- Environments in (Dl, d, Dr) layout (lsr) ---

# FL_{k+1} from FL_k and A_k
# FL,FR: 3-tensors with mpo bond in middle; A: Dl×d×Dr; M: a×d_out×b×d_in
function _update_left_env(A, M, FL)
    return @tensor FLnext[α, a, β] := FL[α′, a′, β′] * A[β′, s′, β] * M[a′, s, a, s′] * conj(A[α′, s, α])
end

# FR_{k} from FR_{k+1} and A_k
function _update_right_env(A, M, FR)
    return @tensor FRprev[α, a, β] := A[α, s′, α′] * FR[α′, a′, β′] * M[a, s, a′, s′] * conj(A[β, s, β′])
end

function tdvp1sweep!(
        dt, ψ::TTvector{T, N}, H::TToperator{T, N}, F::Union{Nothing, Vector{Any}} = nothing;
        verbose::Bool = true, kwargs...
    ) where {T <: Number, N}

    Tc = (dt isa Complex || T <: Complex) ? Complex{real(T)} : T
    Nsites = ψ.N

    # Work in (Dl, s, Dr) for contractions
    A_lsr = [permutedims(ψ.ttv_vec[k], (2, 1, 3)) for k in 1:Nsites]
    M_asbs = [permutedims(H.tto_vec[k], (3, 1, 4, 2)) for k in 1:Nsites]  # (a, s_out, b, s_in)

    # Build or convert environments
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

    # ---- Left -> Right ----
    for k in 1:(Nsites - 1)
        H1 = x -> _applyH1_lsr(x, F[k], F[k + 2], M_asbs[k])
        AC, _ = exponentiate(H1, -im * dt, AC; ishermitian = true, kwargs...)
        if verbose
            E = _dot3(AC, H1(AC)); 
            @info "TDVP L→R: AC site $k   energy = $E"
        end

        Dl, d, Dr = size(AC)
        Aqr = reshape(AC, Dl * d, Dr)
        qrf = qr(Aqr)           # thin QR
        r = min(size(Aqr, 1), size(Aqr, 2))
        Qthin = Matrix(qrf.Q)[:, 1:r]
        Rthin = qrf.R[1:r, :]

        AL = reshape(Qthin, Dl, d, r)   # Dl×d×r
        A_lsr[k] = AL
        F[k + 1] = _update_left_env(AL, M_asbs[k], F[k])

        C = Rthin                    # r×Dr
        H0 = X -> _applyH0(X, F[k + 1], F[k + 2])
        C, _ = exponentiate(H0, +im * dt, C; ishermitian = true, kwargs...)
        if verbose
            E0 = real.(_dot3(C, H0(C))) 
            @info "TDVP L→R: C between $k and $(k + 1)  energy = $E0"
        end

        @tensor AC[α, s, β] := C[α, γ] * A_lsr[k + 1][γ, s, β]
    end

    # last site evolve
    k = Nsites
    H1N = x -> _applyH1_lsr(x, F[k], F[k + 2], M_asbs[k])
    AC, _ = exponentiate(H1N, -im * dt, AC; ishermitian = true, kwargs...)
    if verbose
        E = _dot3(AC, H1N(AC))
        @info "TDVP L→R: AC site $k   energy = $E"
    end

    # ---- Right -> Left ----
    for k in (Nsites - 1):-1:1
        Dl, d, Dr = size(AC)
        A = reshape(AC, Dl, d * Dr)
        qrf = qr(A')                 # QR on transpose -> LQ
        r = min(size(A, 1), size(A, 2))
        Qthin = Matrix(qrf.Q)[:, 1:r]   # (d*Dr)×r
        Rthin = qrf.R[1:r, :]           # r×Dl
        L = Rthin'                      # Dl×r
        A_r = reshape(Qthin', r, d, Dr) # r×d×Dr

        A_lsr[k + 1] = A_r
        F[k + 2] = _update_right_env(A_r, M_asbs[k + 1], F[k + 3])

        C = L
        H0 = X -> _applyH0(X, F[k + 1], F[k + 2])
        C, _ = exponentiate(H0, +im * dt, C; ishermitian = true, kwargs...)
        if verbose
            E0 = _dot3(C, H0(C))
            @info "TDVP R→L: C between $k and $(k + 1)  energy = $E0"
        end

        @tensor AC[α, s, β] := A_lsr[k][α, s, γ] * C[γ, β]

        H1k = x -> _applyH1_lsr(x, F[k], F[k + 2], M_asbs[k])
        AC, _ = exponentiate(H1k, -im * dt, AC; ishermitian = true, kwargs...)
        if verbose
            E = _dot3(AC, H1k(AC))
            @info "TDVP R→L: AC site $k   energy = $E"
        end
    end

    # write back and sync ranks
    A_lsr[1] = AC
    for k in 1:Nsites
        ψ.ttv_vec[k] = permutedims(A_lsr[k], (2, 1, 3))  # back to (phys,left,right)
    end
    _sync_ranks_from_lsr!(ψ, A_lsr)
    return ψ, F
end

function tdvp_method(
        H::TToperator,
        u₀::TTvector,
        steps::Vector{Float64};
        normalize::Bool = true,
        return_error::Bool = false,
        sweeps_per_step::Int = 1,
        carry_env::Bool = true,
        verbose::Bool = false,
        imaginary_time::Bool = false,
        kwargs...
    )
    ψ = orthogonalize(u₀)

    # real-time (default) → complex amplitudes
    wants_complex = !imaginary_time
    if wants_complex && eltype(ψ) <: Real
        ψ = complex(ψ)
    end
    Hc = (wants_complex && eltype(H) <: Real) ? complex(H) : H

    ψ_prev = ψ
    F = nothing

    @showprogress for h in steps
        ψ_prev_step = ψ

        # choose dt so that sweep always uses exp(-i * dt * H):
        # real-time:  exp(-i*h H)  -> dt = h
        # imaginary:  exp(-h   H)  -> need -i*dt = -h -> dt = -i*h
        dt_eff = imaginary_time ? (-im * h) : (complex(1.0) * h)

        for s in 1:sweeps_per_step
            F_in = carry_env ? F : nothing
            ψ, F = tdvp1sweep!(dt_eff, ψ, Hc, F_in; verbose = verbose, kwargs...)
        end

        if normalize
            ψ = (1 / norm(ψ)) * ψ
        end
        ψ = orthogonalize(ψ)   # keep canonical

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
        verbose::Bool = true, truncdim::Int = typemax(Int), truncerr::Real = 0.0, kwargs...
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

    # start center on site 1
    AC = Tc.(A_lsr[1])

    # ----- L → R pass -----
    for k in 1:(Nsites - 1)
        @tensor AAC[α, s1, s2, β] := AC[α, s1, γ] * A_lsr[k + 1][γ, s2, β]

        H2 = X -> _applyH2_lsr(X, F[k], F[k + 3], M_asbs[k], M_asbs[k + 1])
        AAC, _ = exponentiate(H2, -im * dt, AAC; ishermitian = true, kwargs...)
        if verbose
            E = _dot3(vec(conj(AAC)), vec(H2(AAC)))
            @info "2TDVP L→R: sites $(k):$(k + 1)   energy = $E"
        end

        Dl, d1, d2, Dr = size(AAC)
        U, S, Vt = svdtrunc(
            reshape(AAC, Dl * d1, d2 * Dr);
            truncdim = truncdim, truncerr = truncerr
        )

        AL = reshape(U, Dl, d1, size(U, 2))          # Dl×d1×χ
        A_lsr[k] = AL
        F[k + 1] = _update_left_env(AL, M_asbs[k], F[k])

        AC = reshape(S * Vt, size(S, 1), d2, Dr)     # χ×d2×Dr
    end

    # ----- R → L pass -----
    for k in (Nsites - 1):-1:1
        @tensor AAC[α, s1, s2, β] := A_lsr[k][α, s1, γ] * AC[γ, s2, β]

        H2 = X -> _applyH2_lsr(X, F[k], F[k + 3], M_asbs[k], M_asbs[k + 1])
        AAC, _ = exponentiate(H2, -im * dt, AAC; ishermitian = true, kwargs...)
        if verbose
            E = _dot3(vec(conj(AAC)), vec(H2(AAC)))
            @info "2TDVP sweep" sites="$(k):$(k+1)" energy="E"
        end

        Dl, d1, d2, Dr = size(AAC)
        U, S, Vt = svdtrunc(
            reshape(AAC, Dl * d1, d2 * Dr);
            truncdim = truncdim, truncerr = truncerr
        )

        AR = reshape(Vt, size(Vt, 1), d2, Dr)        # χ×d2×Dr
        A_lsr[k + 1] = AR
        F[k + 2] = _update_right_env(AR, M_asbs[k + 1], F[k + 3])

        AC = reshape(U * S, Dl, d1, size(S, 2))      # Dl×d1×χ
    end

    # write back and sync ranks
    A_lsr[1] = AC
    for k in 1:Nsites
        ψ.ttv_vec[k] = permutedims(A_lsr[k], (2, 1, 3))
    end
    _sync_ranks_from_lsr!(ψ, A_lsr)
    return ψ, F
end

function tdvp2_method(
        H::TToperator,
        u₀::TTvector,
        steps::Vector{Float64};
        normalize::Bool = true,
        return_error::Bool = false,
        sweeps_per_step::Int = 1,
        carry_env::Bool = true,
        verbose::Bool = false,
        truncdim::Int = typemax(Int),
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

        # choose dt so sweep uses exp(-i * dt * H₂):
        # real-time:  exp(-i*h H₂)  -> dt = h
        # imaginary:  exp(-h   H₂)  -> need -i*dt = -h -> dt = -i*h
        dt_eff = imaginary_time ? (-im * h) : (complex(1.0) * h)

        for s in 1:sweeps_per_step
            F_in = carry_env ? F : nothing
            ψ, F = tdvp2sweep!(
                dt_eff, ψ, Hc, F_in;
                verbose = verbose, truncdim = truncdim, truncerr = truncerr, kwargs...
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
