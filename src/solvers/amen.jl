using KrylovKit
using LinearAlgebra
using TensorOperations

"""
Implementation based on
Dolgov, Sergey V., and Dmitry V. Savostyanov. "Alternating minimal energy methods for linear systems in higher dimensions." SIAM Journal on Scientific Computing 36.5 (2014): A2248-A2271,
and, for the eigenvalue problem,
Kressner, Daniel, Michael Steinlechner, and André Uschmajew. "Low-rank tensor methods with subspace correction for symmetric eigenvalue problems." SIAM Journal on Scientific Computing 36.5 (2014): A2346-A2368.
"""

# An interface is the contraction of every site on one side of a bond. Its index
# order is (test rank, operator rank, trial rank) for `y' A x` and
# (test rank, vector rank) for `y' b`. The test-side core `y` is conjugated.

function _left_interface(Φ::AbstractArray{<:Number, 3}, y, A, x)
    @tensor Φnext[γ, c, δ] := Φ[α, a, β] * conj(y[i, α, γ]) * A[i, j, a, c] * x[j, β, δ]
    return Φnext
end

function _left_interface(Φ::AbstractMatrix, y, b)
    @tensor Φnext[γ, c] := Φ[α, a] * conj(y[i, α, γ]) * b[i, a, c]
    return Φnext
end

function _right_interface(Φ::AbstractArray{<:Number, 3}, y, A, x)
    @tensor Φnext[α, a, β] := conj(y[i, α, γ]) * (A[i, j, a, c] * (x[j, β, δ] * Φ[γ, c, δ]))
    return Φnext
end

function _right_interface(Φ::AbstractMatrix, y, b)
    @tensor Φnext[α, a] := conj(y[i, α, γ]) * (b[i, a, c] * Φ[γ, c])
    return Φnext
end

# The operator core `A` between the interfaces `ΦL` and `ΦR`, applied to the core `v`.
function _local_matvec(ΦL, A, ΦR, v)
    @tensor w[i, α, γ] := ΦL[α, a, β] * v[j, β, δ] * A[i, j, a, c] * ΦR[γ, c, δ]
    return w
end

# Matrix of `_local_matvec` acting on `vec(v)`.
function _local_matrix(ΦL, A, ΦR)
    @tensor K[i, α, γ, j, β, δ] := ΦL[α, a, β] * A[i, j, a, c] * ΦR[γ, c, δ]
    return reshape(K, size(K, 1) * size(K, 2) * size(K, 3), :)
end

# The vector core `b` between the interfaces `ΦL` and `ΦR`.
function _project(ΦL, b, ΦR)
    @tensor p[i, α, γ] := ΦL[α, a] * b[i, a, c] * ΦR[γ, c]
    return p
end

# Core with orthonormal columns in the `(n * r_left, r_right)` unfolding that
# spans the columns of `c`.
function _left_orthonormal(c)
    n, rl, _ = size(c)
    return reshape(Matrix(qr(reshape(c, n * rl, :)).Q), n, rl, :)
end

struct _AMEnLinear{T}
    b::Vector{Array{T, 3}}
end

# Cores and interfaces of an AMEn solve. `x` is the iterate and `z` the
# residual approximation (empty when there is no enrichment). Each interface
# vector has one entry per bond, including the two outer bonds of rank 1.
# While site `k` is solved, entries `1:k` are left interfaces and entries
# `k+1:d+1` are right interfaces.
struct _AMEnState{T}
    x::Vector{Array{T, 3}}
    z::Vector{Array{T, 3}}
    xAx::Vector{Array{T, 3}}
    xb::Vector{Array{T, 2}}
    zAx::Vector{Array{T, 3}}
    zb::Vector{Array{T, 2}}
end

function _interfaces(::Type{T}, d::Int, ::Val{N}) where {T, N}
    Φ = Vector{Array{T, N}}(undef, d + 1)
    Φ[1] = ones(T, ntuple(_ -> 1, Val(N)))
    Φ[d + 1] = ones(T, ntuple(_ -> 1, Val(N)))
    return Φ
end

function _AMEnState(x::Vector{Array{T, 3}}, kickrank::Int) where {T}
    d = length(x)
    rz = [1; fill(kickrank, d - 1); 1]
    z = kickrank > 0 ? [randn(T, size(x[k], 1), rz[k], rz[k + 1]) for k in 1:d] : Array{T, 3}[]
    return _AMEnState{T}(
        x, z, _interfaces(T, d, Val(3)), _interfaces(T, d, Val(2)),
        _interfaces(T, d, Val(3)), _interfaces(T, d, Val(2))
    )
end

# Cores of the vector that the `xb` and `zb` interfaces are built from.
_rhs_cores(p::_AMEnLinear, x) = p.b

# Core of that vector at site `k` when the residual of the local solution `sol` is formed.
_residual_rhs(p::_AMEnLinear, k, sol, λ) = p.b[k]

_record!(::_AMEnLinear, λ, x) = nothing

_amen_showvalues(::_AMEnLinear, sweep, max_sweeps, max_rank, residual) =
    [("sweep", "$sweep/$max_sweeps"), ("largest rank", max_rank), ("residual", residual)]

function _local_linsolve(ΦL, Ak, ΦR, rhs, guess; local_solver, local_threshold, local_maxiter, local_tol)
    if _use_iterative(local_solver, length(rhs), local_threshold)
        alg = GMRES(; tol = local_tol * norm(rhs), maxiter = local_maxiter, krylovdim = min(30, length(rhs)), verbosity = 0)
        sol, _ = linsolve(v -> _local_matvec(ΦL, Ak, ΦR, v), rhs, guess, alg)
        return sol
    end
    return reshape(_local_matrix(ΦL, Ak, ΦR) \ vec(rhs), size(rhs))
end

# Solve the local problem at site `k`. Returns the new core, the relative
# residual of the current core, and the eigenvalue (`nothing` for a linear system).
function _solve_site(p::_AMEnLinear, s::_AMEnState, Ak, k; local_opts...)
    ΦL, ΦR = s.xAx[k], s.xAx[k + 1]
    rhs = _project(s.xb[k], p.b[k], s.xb[k + 1])
    norm_rhs = norm(rhs)
    # The current basis is orthogonal to `b`: the local solution is zero and the
    # relative residual is undefined, so the sweep counts as not converged.
    iszero(norm_rhs) && return zero(rhs), oftype(norm_rhs, Inf), nothing
    res = norm(_local_matvec(ΦL, Ak, ΦR, s.x[k]) - rhs) / norm_rhs
    return _local_linsolve(ΦL, Ak, ΦR, rhs, s.x[k]; local_opts...), res, nothing
end

# Rank kept at bond `k + 1` given the SVD `F` of the new core: the smallest rank
# whose local residual stays within `tol / (2√d)` of the right-hand side norm,
# or within the residual of the untruncated core if that is larger.
function _site_rank(p::_AMEnLinear, F::SVD, s::_AMEnState, Ak, k, tol, max_bond)
    ΦL, ΦR = s.xAx[k], s.xAx[k + 1]
    rhs = _project(s.xb[k], p.b[k], s.xb[k + 1])
    truncated(r) = reshape(F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :], size(rhs))
    residual(r) = norm(_local_matvec(ΦL, Ak, ΦR, truncated(r)) - rhs)
    d = length(s.x)
    rmax = min(length(F.S), max_bond)
    target = max(tol * norm(rhs) / (2 * sqrt(d)), residual(length(F.S)))
    r = _trunc_rank(F.S, tol, d, rmax)
    if residual(r) ≤ target
        while r > 1 && residual(r - 1) ≤ target
            r -= 1
        end
    else
        while r < rmax && residual(r) > target
            r += 1
        end
    end
    return r
end

# One AMEn sweep: orthogonalize from right to left, then solve from left to
# right. Returns the largest relative residual of the local problems, measured
# before each is solved. With `enrich`, the residual approximation `z` is
# updated and its directions are added to the basis of `x`.
function _amen_sweep!(s::_AMEnState{T}, problem, A::Vector; tol, max_bond, enrich::Bool, local_opts...) where {T}
    x, z = s.x, s.z
    d = length(x)
    rhs = _rhs_cores(problem, x)

    for k in d:-1:2
        _orthogonalize_right!(x, k)
        s.xAx[k] = _right_interface(s.xAx[k + 1], x[k], A[k], x[k])
        s.xb[k] = _right_interface(s.xb[k + 1], x[k], rhs[k])
        enrich || continue
        _orthogonalize_right!(z, k)
        s.zAx[k] = _right_interface(s.zAx[k + 1], z[k], A[k], x[k])
        s.zb[k] = _right_interface(s.zb[k + 1], z[k], rhs[k])
    end

    residual = zero(real(T))
    for k in 1:d
        sol, res, λ = _solve_site(problem, s, A[k], k; local_opts...)
        residual = max(residual, res)
        if k == d
            if enrich
                c = _residual_rhs(problem, k, sol, λ)
                z[k] = _project(s.zb[k], c, s.zb[k + 1]) - _local_matvec(s.zAx[k], A[k], s.zAx[k + 1], sol)
            end
            x[k] = sol
        else
            # Split the new core as `u * v`: `u` stays at site `k` and `v` moves to site `k + 1`.
            n, rl, rr = size(sol)
            F = svd(reshape(sol, n * rl, rr))
            r = _site_rank(problem, F, s, A[k], k, tol, max_bond)
            u = F.U[:, 1:r]
            v = Diagonal(F.S[1:r]) * F.Vt[1:r, :]
            if enrich
                sol = reshape(u * v, n, rl, rr)
                c = _residual_rhs(problem, k, sol, λ)
                zk = _project(s.zb[k], c, s.zb[k + 1]) - _local_matvec(s.zAx[k], A[k], s.zAx[k + 1], sol)
                z[k] = _left_orthonormal(zk)
                # Residual in the left basis of `x` and the right basis of `z`:
                # the directions that are added to bond `k + 1`.
                e = _project(s.xb[k], c, s.zb[k + 1]) - _local_matvec(s.xAx[k], A[k], s.zAx[k + 1], sol)
                Fq = qr(hcat(u, reshape(e, n * rl, :)[:, 1:min(size(e, 3), max_bond - r)]))
                u = Matrix(Fq.Q)
                v = Fq.R[:, 1:r] * v
            end
            x[k] = reshape(u, n, rl, :)
            next = x[k + 1]
            @tensor moved[i, α, γ] := v[α, β] * next[i, β, γ]
            x[k + 1] = moved
            s.xAx[k + 1] = _left_interface(s.xAx[k], x[k], A[k], x[k])
            s.xb[k + 1] = _left_interface(s.xb[k], x[k], rhs[k])
            if enrich
                s.zAx[k + 1] = _left_interface(s.zAx[k], z[k], A[k], x[k])
                s.zb[k + 1] = _left_interface(s.zb[k], z[k], rhs[k])
            end
        end
        _record!(problem, λ, x)
    end
    return residual
end

# Check the QTT metadata of `a` against `b`. A plain tensor train carries no
# metadata, so only pairs of QTT wrappers are checked.
_check_qtt(a, b) = nothing

function _amen_check(A::AbstractTToperator, x::AbstractTTvector)
    nsites(A) ≥ 2 || throw(ArgumentError("AMEn needs at least 2 cores; got $(nsites(A))"))
    A.tto_dims == x.ttv_dims || throw(
        DimensionMismatch("operator dimensions $(A.tto_dims) and vector dimensions $(x.ttv_dims) do not match")
    )
    _check_qtt(A, x)
    return nothing
end

# `x` with the wrapper type of `guess`.
_rewrap(guess::AbstractTTvector, x::TTvector) = x

# Sweep until the residual of a sweep is at most `tol`, then once more without
# enrichment. Returns the iterate and `(; converged, sweeps, residual)`, where
# `residual` is the value of the last sweep.
function _amen_solve(
        problem, A::AbstractTToperator, x0::AbstractTTvector, ::Type{T};
        tol::Real, max_sweeps::Int, max_bond::Int, kickrank::Int,
        local_solver::Symbol, local_threshold::Int, local_maxiter::Int, local_tol::Real,
        verbosity::Int, show_progress::Bool, name::String
    ) where {T}
    d = nsites(x0)
    s = _AMEnState([convert(Array{T, 3}, c) for c in x0.ttv_vec], kickrank)
    cores = [convert(Array{T, 4}, c) for c in A.tto_vec]
    local_opts = (; local_solver, local_threshold, local_maxiter, local_tol)
    progress = _solver_progress(max_sweeps, show_progress; desc = name)
    converged = false
    residual = convert(real(T), Inf)
    sweeps = 0
    for sweep in 1:max_sweeps
        final = converged
        residual = _amen_sweep!(s, problem, cores; tol, max_bond, enrich = kickrank > 0 && !final, local_opts...)
        sweeps = sweep
        converged = residual ≤ tol
        max_rank = maximum(c -> size(c, 3), s.x)
        verbosity ≥ 2 && @info name sweep max_rank residual
        next!(progress; showvalues = _amen_showvalues(problem, sweep, max_sweeps, max_rank, residual))
        converged && (final || kickrank == 0) && break
    end
    sweeps < max_sweeps && finish!(progress)
    converged || verbosity == 0 || @warn "$name did not converge" sweeps residual tol
    rks = [1; [size(c, 3) for c in s.x]]
    x = TTvector{T, d}(s.x, x0.ttv_dims, rks; orthogonality = (d, d))
    return _rewrap(x0, x), (; converged, sweeps, residual)
end

# Implementation of `linear_solve(A, b, x0, ::AMEn)`; see [`AMEn`](@ref).
function _amen_linsolve_impl(A::AbstractTToperator, b::AbstractTTvector, x0::AbstractTTvector; return_info::Bool, kwargs...)
    _amen_check(A, x0)
    _amen_check(A, b)
    _check_qtt(x0, b)
    norm_b = norm(b)
    norm_b > 0 || throw(ArgumentError("the right-hand side is zero"))
    T = float(promote_type(eltype(A), eltype(b), eltype(x0)))
    problem = _AMEnLinear([convert(Array{T, 3}, c) for c in b.ttv_vec])
    x, info = _amen_solve(problem, A, x0, T; name = "AMEn linear solve", kwargs...)
    return_info || return x
    # Orthogonalizing first avoids the cancellation in the norm of a difference of
    # tensor trains, which would otherwise limit the residual to about `√eps`.
    return x, (; residual = norm(orthogonalize(A * x - b)) / norm_b, info.converged, info.sweeps)
end

# Smallest eigenpair of a Hermitian operator. `E` and `r_hist` collect the
# eigenvalue estimate and the largest bond dimension of every micro-step.
# `scale` is the largest `‖A x‖` over the unit-norm iterates seen so far; the
# eigenvalue residual is measured against it, which stays meaningful when the
# eigenvalue is zero.
struct _AMEnEigen
    E::Vector{Float64}
    r_hist::Vector{Int}
    scale::Base.RefValue{Float64}
end

# The eigenvalue residual `A x − λ x` has the form of a linear residual with
# right-hand side `λ x`, so the vector interfaces are overlaps with `x` itself.
_rhs_cores(::_AMEnEigen, x) = x

_residual_rhs(::_AMEnEigen, k, sol, λ) = λ * sol

function _record!(p::_AMEnEigen, λ, x)
    push!(p.E, λ)
    push!(p.r_hist, maximum(c -> size(c, 3), x))
    return nothing
end

_amen_showvalues(p::_AMEnEigen, sweep, max_sweeps, max_rank, residual) =
    [("sweep", "$sweep/$max_sweeps"), ("eigenvalue", last(p.E)), ("largest rank", max_rank), ("residual", residual)]

function _local_eigmin(ΦL, Ak, ΦR, guess; local_solver, local_threshold, local_maxiter, local_tol)
    if _use_iterative(local_solver, length(guess), local_threshold)
        alg = Lanczos(; tol = local_tol, maxiter = local_maxiter, krylovdim = max(2, min(30, length(guess))), verbosity = 0)
        vals, vecs, _ = eigsolve(v -> _local_matvec(ΦL, Ak, ΦR, v), guess, 1, :SR, alg)
        return real(vals[1]), vecs[1]
    end
    F = eigen(Hermitian(_local_matrix(ΦL, Ak, ΦR)), 1:1)
    return F.values[1], reshape(F.vectors[:, 1], size(guess))
end

function _solve_site(p::_AMEnEigen, s::_AMEnState, Ak, k; local_opts...)
    ΦL, ΦR = s.xAx[k], s.xAx[k + 1]
    xk = s.x[k] / norm(s.x[k])
    Kx = _local_matvec(ΦL, Ak, ΦR, xk)
    p.scale[] = max(p.scale[], norm(Kx))
    res = norm(Kx - real(dot(xk, Kx)) * xk)
    # A nonzero residual implies `Kx ≠ 0`, so the scale is positive here.
    iszero(res) || (res /= p.scale[])
    λ, sol = _local_eigmin(ΦL, Ak, ΦR, xk; local_opts...)
    return sol, res, λ
end

_site_rank(::_AMEnEigen, F::SVD, s::_AMEnState, Ak, k, tol, max_bond) =
    _trunc_rank(F.S, tol, length(s.x), max_bond)

# Implementation of `eigen_solve(A, x0, ::AMEn)`; see [`AMEn`](@ref).
function _amen_eigsolve_impl(A::AbstractTToperator, x0::AbstractTTvector; kwargs...)
    _amen_check(A, x0)
    norm(x0) > 0 || throw(ArgumentError("the guess is zero"))
    T = float(promote_type(eltype(A), eltype(x0)))
    problem = _AMEnEigen(Float64[], Int[], Ref(0.0))
    x, _ = _amen_solve(problem, A, x0, T; name = "AMEn eigen solve", kwargs...)
    return problem.E, x, problem.r_hist
end

function eigen_solve(A::AbstractTToperator, guess::AbstractTTvector, alg::AMEn)
    _reject_unused(alg, "eigen_solve", (:return_info,), "every option except `return_info`")
    return _amen_eigsolve_impl(A, guess; _amen_options(alg)...)
end
