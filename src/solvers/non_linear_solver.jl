# Nonlinear QTT solver — Gross–Pitaevskii penalty method + multigrid renormalization (MGR),
# after M. Lubasch, P. Moinier, D. Jaksch, J. Comput. Phys. 372 (2018) 587–602
# (arXiv:1802.07259). Minimizes the discrete penalty
#     P(u) = ⟨u|A|u⟩ + (g/2) Σ_m u_m⁴ + η (⟨u|u⟩ − 1)²
# over a real TTvector by alternating site-local updates (damped Newton / nonlinear CG / SD),
# with local operators projected through swept environments (no densification). Real path only.

# ---------------------------------------------------------------------------
# Site-local objective on the vectorized core y (dense, small: n = n_i·r_{i-1}·r_i).
# Q(y) = Σ_m (Φy)_m⁴ and Qgrad(y) = ∇Q(y) are supplied as closures (dense Φ in tests,
# 4-layer environments in the sweep). g_c is the HALF-gradient: ∇P = 2 g_c.
# ---------------------------------------------------------------------------

function _nl_penalty(y::AbstractVector{T}, Aloc::AbstractMatrix{T}, Q, g::Real, η::Real) where {T <: Real}
    s = dot(y, y)
    return dot(y, Aloc * y) + (g / 2) * Q(y) + η * (s - 1)^2
end

function _nl_gradient(y::AbstractVector{T}, Aloc::AbstractMatrix{T}, Qgrad, g::Real, η::Real) where {T <: Real}
    s = dot(y, y)
    return Aloc * y .+ (g / 4) .* Qgrad(y) .+ (2η * (s - 1)) .* y
end

function _nl_newton_system(y::AbstractVector{T}, Aloc::AbstractMatrix{T}, B::AbstractMatrix{T}, Qgrad, g::Real, η::Real) where {T <: Real}
    n = length(y)
    s = dot(y, y)
    gc = Aloc * y .+ (g / 4) .* Qgrad(y) .+ (2η * (s - 1)) .* y
    H = Aloc .+ (3g) .* B .+ (4η) .* (y * y') .+ (2η * (s - 1)) .* Matrix{T}(I, n, n)
    return gc, (H .+ H') ./ 2
end

"""
One damped Newton step on the site-local penalty (paper's ν_update = 1). Levenberg–Marquardt:
grow λ (×10) until the step is a descent direction and does not increase P; `ptol` is the
stationarity floor (a step within recontraction rounding of P0 is accepted as converged).
Falls back to a backtracking steepest-descent step.
"""
function _nl_newton_step(
        x0::Vector{T}, Aloc::Matrix{T}, B::Matrix{T}, Q, Qgrad, g::Real, η::Real;
        λmax::Real = 1.0e10, maxtry::Int = 40
    ) where {T <: Real}
    gc, H = _nl_newton_system(x0, Aloc, B, Qgrad, g, η)
    P0 = _nl_penalty(x0, Aloc, Q, g, η)
    ptol = 1.0e-11 * abs(P0)
    D = Diagonal(diag(H))
    λ = 0.0
    for _ in 1:maxtry
        Δ = try
            (H .+ λ .* D) \ (-gc)
        catch e
            e isa InterruptException && rethrow()
            λ = λ == 0 ? 1.0e-6 : 10λ
            continue
        end
        if dot(gc, Δ) ≤ ptol && _nl_penalty(x0 .+ Δ, Aloc, Q, g, η) ≤ P0 + ptol
            return x0 .+ Δ
        end
        λ = λ == 0 ? 1.0e-6 : 10λ
        λ > λmax && break
    end
    return _nl_backtrack(x0, -gc, gc, Aloc, Q, g, η, P0; ptol = ptol)
end

"Backtracking-Armijo steepest-descent fallback (accepts machine-equal P via ptol)."
function _nl_backtrack(
        x::Vector{T}, p::Vector{T}, gc::Vector{T}, Aloc::Matrix{T}, Q, g::Real, η::Real, P0::Real;
        c::Real = 1.0e-4, shrink::Real = 0.5, maxbt::Int = 15, ptol::Real = 0.0
    ) where {T <: Real}
    slope = dot(gc, p)
    α = 1.0 / (1 + opnorm(Aloc) + abs(g) + η)
    for _ in 1:maxbt
        if _nl_penalty(x .+ α .* p, Aloc, Q, g, η) ≤ P0 + min(c * α * slope, 0.0) + ptol
            return x .+ α .* p
        end
        α *= shrink
    end
    return x
end

# ---------------------------------------------------------------------------
# Exact quartic line search: φ(α) = P(x + αp) is exactly quartic in α. The A- and
# norm-terms have closed forms; the interaction term I(α) = Q(x + αp) is fitted exactly
# through 5 nodes (a quartic is determined by 5 values).
# ---------------------------------------------------------------------------

function _nl_line_coeffs(x::Vector{T}, p::Vector{T}, Aloc::Matrix{T}, Q, g::Real, η::Real) where {T <: Real}
    Ax = Aloc * x
    q0 = dot(x, Ax); q1 = 2 * dot(p, Ax); q2 = dot(p, Aloc * p)
    s0 = dot(x, x); s1 = 2 * dot(p, x); s2 = dot(p, p)
    h = (1 + sqrt(s0)) / (1 + sqrt(s2))
    nodes = (-2h, -h, 0.0, h, 2h)
    vals = [Q(x .+ a .* p) for a in nodes]
    V = [a^k for a in nodes, k in 0:4]
    Ico = V \ vals
    e0 = s0 - 1
    c0 = q0 + (g / 2) * Ico[1] + η * e0^2
    c1 = q1 + (g / 2) * Ico[2] + 2η * e0 * s1
    c2 = q2 + (g / 2) * Ico[3] + η * (s1^2 + 2 * e0 * s2)
    c3 = (g / 2) * Ico[4] + 2η * s1 * s2
    c4 = (g / 2) * Ico[5] + η * s2^2
    return (c0, c1, c2, c3, c4)
end

"Real roots of the cubic co[1] + co[2]α + co[3]α² + co[4]α³ via companion-matrix eigenvalues."
function _nl_real_roots(co::NTuple{4, Float64})
    tol = 1.0e-13 * (1 + maximum(abs, co))
    n = findlast(c -> abs(c) > tol, co)
    (n === nothing || n == 1) && return Float64[]
    n == 2 && return [-co[1] / co[2]]
    d = n - 1
    p = collect(co[1:n]) ./ co[n]
    C = zeros(d, d)
    for i in 1:(d - 1)
        C[i + 1, i] = 1.0
    end
    C[:, d] .= .-p[1:d]
    return Float64[real(z) for z in eigvals(C) if abs(imag(z)) ≤ 1.0e-7 * (1 + abs(real(z)))]
end

"Backtracking-Armijo step length (fallback when the exact line search finds no descent root)."
function _nl_armijo(x::Vector{T}, p::Vector{T}, gk::Vector{T}, Aloc::Matrix{T}, Q, g::Real, η::Real; c::Real = 1.0e-4, shrink::Real = 0.5, maxbt::Int = 60) where {T <: Real}
    P0 = _nl_penalty(x, Aloc, Q, g, η)
    slope = dot(gk, p)
    α = 1.0 / (1 + opnorm(Aloc) + abs(g) + η)
    for _ in 1:maxbt
        _nl_penalty(x .+ α .* p, Aloc, Q, g, η) ≤ P0 + c * α * slope && return α
        α *= shrink
    end
    return 0.0
end

"""
ν exact-line-searched descent steps on the site-local penalty. `variant`: `:sd` (steepest
descent), `:fr` (Fletcher–Reeves), `:pr` (Polak–Ribière+, restarts on loss of descent).
"""
function _nl_descent_update(
        x0::Vector{T}, Aloc::Matrix{T}, Q, Qgrad, g::Real, η::Real;
        ν::Int = 4, variant::Symbol = :pr
    ) where {T <: Real}
    x = copy(x0)
    gk = _nl_gradient(x, Aloc, Qgrad, g, η)
    p = -gk
    for _ in 1:ν
        P0 = _nl_penalty(x, Aloc, Q, g, η)
        co = _nl_line_coeffs(x, p, Aloc, Q, g, η)
        φ = α -> co[1] + α * (co[2] + α * (co[3] + α * (co[4] + α * co[5])))
        best_α, best = 0.0, co[1]
        for a in _nl_real_roots((co[2], 2co[3], 3co[4], 4co[5]))
            fa = φ(a)
            fa < best && (best = fa; best_α = a)
        end
        # Guard against catastrophic cancellation in the exact-quartic coefficient fit
        # (co[3], co[4] are recovered from nearly-equal Q-samples when ‖p‖→0 near a
        # stationary point): a spurious large root can make the *fitted* φ(a) look like
        # a big decrease while the *true* penalty at x+a·p actually increases. Verify
        # against the true penalty before accepting; fall back to Armijo otherwise.
        if best_α == 0.0 || _nl_penalty(x .+ best_α .* p, Aloc, Q, g, η) > P0 + 1.0e-11 * abs(P0)
            best_α = _nl_armijo(x, p, gk, Aloc, Q, g, η)
        end
        x = x .+ best_α .* p
        gnew = _nl_gradient(x, Aloc, Qgrad, g, η)
        if variant === :sd
            p = -gnew
        else
            # gg == 0 means gk is exactly stationary: β would be 0/0 = NaN (and NaN
            # survives the max(·,0.0) for :pr and defeats the dot(gnew,p) ≥ 0 restart
            # guard below, since NaN ≥ 0 is false). Treat it as "no CG memory" instead.
            gg = dot(gk, gk)
            β = gg == 0 ? 0.0 :
                (variant === :fr ? dot(gnew, gnew) / gg : max(dot(gnew, gnew .- gk) / gg, 0.0))
            p = -gnew .+ β .* p
            dot(gnew, p) ≥ 0 && (p = -gnew)
        end
        gk = gnew
    end
    return x
end

# ---------------------------------------------------------------------------
# Environments. All are "pure": they contain cores strictly left/right of the
# center site (the ALS G-convention of baking in the site's operator core is NOT
# used here, because the interaction operator changes with every core update).
# Right 3-leg op envs reuse als.jl's update_H! (layout H[i] = (rA_i, r_i, r_i),
# contains cores i+1..d). 4-layer envs carry the interaction Σu⁴: four copies of
# u's cores, legs (bra, mid1, mid2, ket) — all equal to u's bond ranks.
# ---------------------------------------------------------------------------

function _nl_right_op_envs(u::TTvector{T, N}, A::TToperator{T, N}) where {T <: Real, N}
    d = u.N
    H = Vector{Array{T, 3}}(undef, d)
    H[d] = ones(T, 1, 1, 1)
    for i in d:-1:2
        H[i - 1] = zeros(T, A.tto_rks[i], u.ttv_rks[i], u.ttv_rks[i])
        update_H!(u.ttv_vec[i], A.tto_vec[i], H[i], H[i - 1])
    end
    return H
end

function _nl_op_absorb_left(x::Array{T, 3}, Acore::Array{T, 4}, L::Array{T, 3}) where {T <: Real}
    Ln = zeros(T, size(Acore, 4), size(x, 3), size(x, 3))
    @tensoropt Ln[z, α, β] = x[j, ϕ, α] * L[w, ϕ, χ] * x[k, χ, β] * Acore[j, k, w, z]
    return Ln
end

"Symmetrized site-local operator ΦᵀAΦ from left env, operator core, right env."
function _nl_effective_op(L::Array{T, 3}, Acore::Array{T, 4}, H::Array{T, 3}) where {T <: Real}
    n, rl, rr = size(Acore, 1), size(L, 2), size(H, 2)
    K = zeros(T, n * rl * rr, n * rl * rr)
    Kr = reshape(K, n, rl, rr, n, rl, rr)
    @tensoropt Kr[j, a, b, k, c, d] = L[z, a, c] * Acore[j, k, z, w] * H[w, b, d]
    return (K .+ K') ./ 2
end

"""
Absorb site x into a right 4-layer env (bra/ket layers 1-4 all read the SAME site,
so the physical index j is shared diagonally across all four copies of x — a
"hyperindex" pattern `@tensor`/`@tensoropt` cannot express in one contraction
(each index may appear at most twice). We sum explicitly over the (small)
physical dimension instead; every remaining index then appears exactly twice.
"""
function _nl_env4_absorb_right(x::Array{T, 3}, E::Array{T, 4}) where {T <: Real}
    n, rl, _ = size(x)
    En = zeros(T, rl, rl, rl, rl)
    for j in 1:n
        xj = view(x, j, :, :)
        @tensoropt tmp[α, a, b, β] := xj[α, α′] * xj[a, a′] * xj[b, b′] * xj[β, β′] * E[α′, a′, b′, β′]
        En .+= tmp
    end
    return En
end

function _nl_env4_absorb_left(x::Array{T, 3}, E::Array{T, 4}) where {T <: Real}
    n, _, rr = size(x)
    En = zeros(T, rr, rr, rr, rr)
    for j in 1:n
        xj = view(x, j, :, :)
        @tensoropt tmp[α, a, b, β] := xj[α′, α] * xj[a′, a] * xj[b′, b] * xj[β′, β] * E[α′, a′, b′, β′]
        En .+= tmp
    end
    return En
end

function _nl_right_qenvs(u::TTvector{T, N}) where {T <: Real, N}
    d = u.N
    E = Vector{Array{T, 4}}(undef, d)
    E[d] = ones(T, 1, 1, 1, 1)
    for i in d:-1:2
        E[i - 1] = _nl_env4_absorb_right(u.ttv_vec[i], E[i])
    end
    return E
end

"Exact Σ_m u_m⁴ with core y at the center site: absorb y into EL, contract with ER."
function _nl_quartic(EL::Array{T, 4}, y::Array{T, 3}, ER::Array{T, 4}) where {T <: Real}
    return dot(vec(_nl_env4_absorb_left(y, EL)), vec(ER))
end

"""
∇_y Σu⁴ as a core: 4× the 3-copy contraction (the 4-layer env is slot-symmetric).
Same hyperindex issue as the env4-absorb functions (j shared by three copies of y
plus the output) — loop over the physical index j instead of one `@tensoropt` call.
"""
function _nl_quartic_grad(EL::Array{T, 4}, y::Array{T, 3}, ER::Array{T, 4}) where {T <: Real}
    n = size(y, 1)
    G = zeros(T, size(y))
    for j in 1:n
        yj = view(y, j, :, :)
        @tensoropt g[α, α′] := EL[α, a, b, β] * yj[a, a′] * yj[b, b′] * yj[β, β′] * ER[α′, a′, b′, β′]
        G[j, :, :] .= g
    end
    return 4 .* G
end

"""
Symmetrized B = Φᵀdiag(u²)Φ frozen at core x0 (diagonal in the site's physical index).
Same hyperindex issue (j shared by two copies of x0 plus the output) — loop over j.
"""
function _nl_effective_abs2(EL::Array{T, 4}, x0::Array{T, 3}, ER::Array{T, 4}) where {T <: Real}
    n, rl, rr = size(x0)
    B = zeros(T, n * rl * rr, n * rl * rr)
    Br = reshape(B, n, rl, rr, n, rl, rr)
    for j in 1:n
        x0j = view(x0, j, :, :)
        @tensoropt Tt[α, β, α′, β′] := EL[α, a, b, β] * x0j[a, a′] * x0j[b, b′] * ER[α′, a′, b′, β′]
        Br[j, :, :, j, :, :] .= permutedims(Tt, (1, 3, 2, 4))
    end
    return (B .+ B') ./ 2
end

# ---------------------------------------------------------------------------
# Algorithm structs (style of linear_solver.jl) and the sweep driver
# ---------------------------------------------------------------------------

abstract type NonLinearSolverAlgorithm end

"""
    PenaltyALS(; local_solver=:newton, ν_local=4, η_schedule=[1e2,1e4,1e6,1e8],
                 tol=1e-6, max_sweeps=100, return_info=false, show_progress=false)

Fixed-grid nonlinear penalty solver (arXiv:1802.07259): alternating site-local minimization of
`P(u) = ⟨u|A|u⟩ + (g/2)Σu⁴ + η(⟨u|u⟩−1)²` under η-continuation. `local_solver` is `:newton`
(damped Newton, paper's default), `:cg` (Polak–Ribière+), or `:sd`; `ν_local` line-search steps
are used per site for `:cg`/`:sd`. Each η stage sweeps until the relative penalty change is
below `tol` or `max_sweeps` is hit.
"""
struct PenaltyALS <: NonLinearSolverAlgorithm
    local_solver::Symbol
    ν_local::Int
    η_schedule::Vector{Float64}
    tol::Float64
    max_sweeps::Int
    return_info::Bool
    show_progress::Bool
end

function PenaltyALS(;
        local_solver::Symbol = :newton,
        ν_local::Int = 4,
        η_schedule::Vector{Float64} = [1.0e2, 1.0e4, 1.0e6, 1.0e8],
        tol::Float64 = 1.0e-6,
        max_sweeps::Int = 100,
        return_info::Bool = false,
        show_progress::Bool = false
    )
    local_solver in (:newton, :cg, :sd) || throw(ArgumentError("local_solver must be :newton, :cg, or :sd"))
    isempty(η_schedule) && throw(ArgumentError("η_schedule must not be empty"))
    max_sweeps < 1 && throw(ArgumentError("max_sweeps must be ≥ 1"))
    ν_local < 1 && throw(ArgumentError("ν_local must be ≥ 1"))
    return PenaltyALS(local_solver, ν_local, η_schedule, tol, max_sweeps, return_info, show_progress)
end

"Site-local update: build A_loc, Q, ∇Q (and B for Newton), run the local solver, return the new core."
function _nl_site_step(
        u::TTvector{T, N}, i::Int, Acore::Array{T, 4},
        LAi::Array{T, 3}, HAi::Array{T, 3}, ELi::Array{T, 4}, ERi::Array{T, 4},
        g::Real, η::Real, alg::PenaltyALS
    ) where {T <: Real, N}
    Aloc = _nl_effective_op(LAi, Acore, HAi)
    x0core = u.ttv_vec[i]
    dims = size(x0core)
    Q = y -> _nl_quartic(ELi, reshape(y, dims), ERi)
    Qgrad = y -> vec(_nl_quartic_grad(ELi, reshape(y, dims), ERi))
    x0 = vec(x0core)
    xnew = if alg.local_solver === :newton
        B = _nl_effective_abs2(ELi, x0core, ERi)
        _nl_newton_step(x0, Aloc, B, Q, Qgrad, g, η)
    else
        variant = alg.local_solver === :cg ? :pr : :sd
        _nl_descent_update(x0, Aloc, Q, Qgrad, g, η; ν = alg.ν_local, variant = variant)
    end
    return reshape(xnew, dims)
end

"Exact penalty evaluated site-locally at site i (requires gauge + current envs at i)."
function _nl_penalty_local(
        u::TTvector{T, N}, i::Int, Acore::Array{T, 4},
        LAi::Array{T, 3}, HAi::Array{T, 3}, ELi::Array{T, 4}, ERi::Array{T, 4},
        g::Real, η::Real
    ) where {T <: Real, N}
    Aloc = _nl_effective_op(LAi, Acore, HAi)
    dims = size(u.ttv_vec[i])
    Q = y -> _nl_quartic(ELi, reshape(y, dims), ERi)
    return _nl_penalty(vec(u.ttv_vec[i]), Aloc, Q, g, η)
end

function _penalty_solve_impl(A::TToperator{T, N}, u0::TTvector{T, N}, alg::PenaltyALS; g::Real = 0.0) where {T, N}
    T <: Real || throw(ArgumentError("non_linear_solve implements the real path only; got eltype $T"))
    u0.N ≥ 2 || throw(ArgumentError("non_linear_solve needs at least 2 TT cores"))
    # Round-trip gauge (far end, then back to site 1): orthogonalize(u0) alone only
    # right-canonicalizes sites 2..d and never QR-checks site 1's own local admissibility
    # (rks[2] ≤ dims[1]·rks[1]) against what's actually reachable from the left boundary.
    # An un-rounded sum (e.g. seed + ε·padding) can carry a locally-redundant left core
    # that trips the plain (non-pivoted) QR inside right_core_move/left_core_move, which
    # assume admissible ranks. Gauging to the far end first forces the missing left QR pass.
    u = orthogonalize(orthogonalize(u0; i = u0.N); i = 1)
    d = u.N
    rks = copy(u.ttv_rks)
    # pure environments (never contain the center core)
    HA = _nl_right_op_envs(u, A)
    ER = _nl_right_qenvs(u)
    LA = Vector{Array{T, 3}}(undef, d)
    LA[1] = ones(T, 1, 1, 1)
    EL = Vector{Array{T, 4}}(undef, d)
    EL[1] = ones(T, 1, 1, 1, 1)
    η_history = Float64[]
    total_sweeps = 0
    penalty = zero(Float64)
    progress = _solver_progress(length(alg.η_schedule) * alg.max_sweeps, alg.show_progress; desc = "Nonlinear penalty ALS")
    for (stage, η) in enumerate(alg.η_schedule)
        Pprev = _nl_penalty_local(u, 1, A.tto_vec[1], LA[1], HA[1], EL[1], ER[1], g, η)
        for _ in 1:alg.max_sweeps
            for i in 1:(d - 1)                      # forward half sweep
                V = _nl_site_step(u, i, A.tto_vec[i], LA[i], HA[i], EL[i], ER[i], g, η, alg)
                u = right_core_move(u, V, i, rks)
                LA[i + 1] = _nl_op_absorb_left(u.ttv_vec[i], A.tto_vec[i], LA[i])
                EL[i + 1] = _nl_env4_absorb_left(u.ttv_vec[i], EL[i])
            end
            for i in d:-1:2                         # backward half sweep
                V = _nl_site_step(u, i, A.tto_vec[i], LA[i], HA[i], EL[i], ER[i], g, η, alg)
                u = left_core_move(u, V, i, rks)
                update_H!(u.ttv_vec[i], A.tto_vec[i], HA[i], HA[i - 1])
                ER[i - 1] = _nl_env4_absorb_right(u.ttv_vec[i], ER[i])
            end
            total_sweeps += 1
            next!(progress)
            penalty = _nl_penalty_local(u, 1, A.tto_vec[1], LA[1], HA[1], EL[1], ER[1], g, η)
            abs(penalty - Pprev) ≤ alg.tol * abs(penalty) && break
            Pprev = penalty
        end
        update!(progress, stage * alg.max_sweeps)
        push!(η_history, η)
    end
    finish!(progress)
    # η_schedule of all zeros = unconstrained minimization (no norm constraint to enforce)
    if maximum(alg.η_schedule) > 0
        s = dot(u, u)
        abs(s - 1) < 1.0e-4 || @warn "penalty did not enforce ⟨u|u⟩=1 (|s−1| = $(abs(s - 1))); extend η_schedule"
    end
    if alg.return_info
        info = (; energy = gpe_energy(A, u; g = g), penalty = penalty, sweeps = total_sweeps, η_history = η_history)
        return u, info
    end
    return u
end

"""
    non_linear_solve(A, u0; alg = PenaltyALS(), g = 0.0)
    non_linear_solve(A, u0, alg::PenaltyALS; g = 0.0)

Ground state of the discrete Gross–Pitaevskii functional
`P(u) = ⟨u|A|u⟩ + (g/2)Σ_m u_m⁴ + η(⟨u|u⟩−1)²` in TT/QTT format (arXiv:1802.07259).
`A` is the (already discretized) linear operator and `g` the discrete interaction
coefficient (`g_physical · 2^L` in the QTT convention). Real TT only. Returns the
discretely normalized minimizer, or `(u, info)` when `alg.return_info`.
"""
non_linear_solve(A::TToperator, u0::TTvector; alg::PenaltyALS = PenaltyALS(), g::Real = 0.0) =
    non_linear_solve(A, u0, alg; g = g)

non_linear_solve(A::TToperator{T, N}, u0::TTvector{T, N}, alg::PenaltyALS; g::Real = 0.0) where {T, N} =
    _penalty_solve_impl(A, u0, alg; g = g)

"""
    gpe_energy(A, u; g = 0.0)

Discrete Gross–Pitaevskii energy readout `E = ⟨u|A|u⟩/s + g·Σ_m u_m⁴/s²`, `s = ⟨u|u⟩`
(full `g`, matching the paper's Table-1 convention; the minimized functional carries `g/2`).
"""
function gpe_energy(A::TToperator{T, N}, u::TTvector{T, N}; g::Real = 0.0) where {T, N}
    s = real(dot(u, u))
    e = real(dot(u, A * u))
    w = hadamard(u, u)
    return e / s + g * real(dot(w, w)) / s^2
end

"""
    MGR(; inner = PenaltyALS(), max_rank = 8, return_info = false, show_progress = false)

Multigrid-renormalization driver (arXiv:1802.07259): solve on the coarse grid of `u0`, then
repeatedly prolong to one more QTT site (`qtto_linear_prolongation`), truncate to `max_rank`
(keeping numerically zero singular values as bond-growth room), normalize, and re-solve with
`inner`, up to the target grid.
"""
struct MGR <: NonLinearSolverAlgorithm
    inner::PenaltyALS
    max_rank::Int
    return_info::Bool
    show_progress::Bool
end

function MGR(;
        inner::PenaltyALS = PenaltyALS(),
        max_rank::Int = 8,
        return_info::Bool = false,
        show_progress::Bool = false
    )
    return MGR(inner, max_rank, return_info, show_progress)
end

"Quiet copy of the inner solver (level progress is reported by the MGR bar)."
function _mgr_inner(alg::PenaltyALS)
    return PenaltyALS(alg.local_solver, alg.ν_local, alg.η_schedule, alg.tol, alg.max_sweeps, false, false)
end

"Function barrier: one MGR level with concretely typed operator and state."
_mgr_level(A::TToperator{T, N}, u::TTvector{T, N}, inner::PenaltyALS, g::Real) where {T, N} =
    _penalty_solve_impl(A, u, inner; g = g)

"""
    non_linear_solve(A_builder, u0, alg::MGR; g_builder, target_sites)

MGR ground-state solve from the coarse grid `u0.N` up to `target_sites` QTT sites.
`A_builder(d)::TToperator` returns the discrete linear operator on `d` sites and
`g_builder(d)::Real` the discrete interaction coefficient (e.g. `g · 2^d`).
Returns `u`, or `(u, info)` with `info = (; energy, level_sites, level_energies)`.
"""
function non_linear_solve(
        A_builder::Function, u0::TTvector{T, M}, alg::MGR;
        g_builder::Function, target_sites::Int
    ) where {T, M}
    target_sites ≥ u0.N || throw(ArgumentError("target_sites ($target_sites) must be ≥ u0.N ($(u0.N))"))
    inner = _mgr_inner(alg.inner)
    levels = target_sites - u0.N + 1
    progress = _solver_progress(levels, alg.show_progress; desc = "MGR nonlinear solve")
    level_sites = Int[]
    level_energies = Float64[]
    u = _mgr_level(A_builder(u0.N), u0, inner, g_builder(u0.N))
    push!(level_sites, u0.N)
    alg.return_info && push!(level_energies, gpe_energy(A_builder(u0.N), u; g = g_builder(u0.N)))
    next!(progress)
    for d in (u0.N + 1):target_sites
        up = qtto_linear_prolongation(d - 1) * u
        tt_compress!(up, alg.max_rank)
        up = (1 / norm(up)) * up
        u = _mgr_level(A_builder(d), up, inner, g_builder(d))
        push!(level_sites, d)
        alg.return_info && push!(level_energies, gpe_energy(A_builder(d), u; g = g_builder(d)))
        next!(progress)
    end
    finish!(progress)
    if alg.return_info
        return u, (; energy = level_energies[end], level_sites = level_sites, level_energies = level_energies)
    end
    return u
end
