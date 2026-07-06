using Test
using TensorTrainNumerics
using LinearAlgebra
using Random

const TTN = TensorTrainNumerics

# ---------------------------------------------------------------------------
# Dense adapters: the local objective with an explicit isometry Φ (real path).
# P(y) = yᵀAy + (g/2)Σ(Φy)⁴ + η(yᵀy−1)²
# ---------------------------------------------------------------------------
rand_iso(m, n) = Matrix(qr(randn(m, n)).Q)[:, 1:n]
rand_sym(n) = (B = randn(n, n); (B + B') / 2)
dense_Q(Φ) = y -> sum(x -> x^4, Φ * y)
dense_Qgrad(Φ) = y -> 4 .* (Φ' * ((Φ * y) .^ 3))
dense_B(Φ, x0) = (u = Φ * x0; Φ' * (Diagonal(u .^ 2) * Φ))
dense_P(Φ, A, g, η) = y -> (s = dot(y, y); dot(y, A * y) + (g / 2) * sum(x -> x^4, Φ * y) + η * (s - 1)^2)

function fd_grad(f, x; h = 1.0e-6)
    g = similar(x)
    for i in eachindex(x)
        xp = copy(x); xm = copy(x)
        xp[i] += h; xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2h)
    end
    return g
end

fd_jac(gfun, x; h = 1.0e-6) = reduce(hcat, [begin
    xp = copy(x); xm = copy(x)
    xp[i] += h; xm[i] -= h
    (gfun(xp) - gfun(xm)) ./ (2h)
end for i in eachindex(x)])

@testset "nonlinear solver: local update math (FD-audited)" begin
    Random.seed!(21)
    # Gradient: FD of P equals 2·g_c (g_c is the half-gradient). FULL g.
    for (m, n) in ((16, 4), (8, 3)), g in (0.0, 10.0, 100.0), η in (1.0, 100.0)
        Φ = rand_iso(m, n); A = rand_sym(n); x = randn(n)
        gc = TTN._nl_gradient(x, A, dense_Qgrad(Φ), g, η)
        gf = fd_grad(dense_P(Φ, A, g, η), x)
        @test norm(2 .* gc .- gf) / norm(gf) < 1.0e-6
    end
    # Hessian: FD Jacobian of g_c equals H = A + 3gB + 4ηxxᵀ + 2η(s−1)I.
    for (m, n) in ((16, 4), (8, 3)), g in (0.0, 10.0, 100.0), η in (1.0, 1.0e4)
        Φ = rand_iso(m, n); A = rand_sym(n); x = randn(n)
        _, H = TTN._nl_newton_system(x, A, dense_B(Φ, x), dense_Qgrad(Φ), g, η)
        Hf = fd_jac(y -> TTN._nl_gradient(y, A, dense_Qgrad(Φ), g, η), x)
        @test norm(H - Hf) / norm(Hf) < 1.0e-4
        @test norm(H - H') < 1.0e-9
    end
    # Mutation rows (must stay RED-catching; reference Phase-5 table).
    Φ = rand_iso(12, 4); A = rand_sym(4); x = randn(4); g = 100.0; η = 10.0
    gf = fd_grad(dense_P(Φ, A, g, η), x)
    s = dot(x, x)
    ghalf = A * x .+ ((g / 2) / 4) .* dense_Qgrad(Φ)(x) .+ (2η * (s - 1)) .* x   # g → g/2
    @test norm(2 .* ghalf .- gf) / norm(gf) > 0.05
    _, H = TTN._nl_newton_system(x, A, dense_B(Φ, x), dense_Qgrad(Φ), g, η)
    Hf = fd_jac(y -> TTN._nl_gradient(y, A, dense_Qgrad(Φ), g, η), x)
    @test norm((H .- 3g .* dense_B(Φ, x)) - Hf) / norm(Hf) > 0.05                # omit 3gB

    # stationary start must not produce NaN β / LAPACK error (regression: final-review finding)
    Φ0 = rand_iso(12, 4); A0 = rand_sym(4)
    for variant in (:fr, :pr, :sd)
        xz = TTN._nl_descent_update(zeros(4), A0, dense_Q(Φ0), dense_Qgrad(Φ0), 10.0, 1.0; ν = 2, variant = variant)
        @test all(isfinite, xz)
    end
end

@testset "nonlinear solver: Newton and descent local solvers" begin
    # Newton: monotone descent + gradient-norm collapse.
    Random.seed!(3)
    Φ = rand_iso(16, 4); A = rand_sym(4)
    g = 50.0; η = 1.0
    Q = dense_Q(Φ); Qg = dense_Qgrad(Φ); P = dense_P(Φ, A, g, η)
    x = randn(4)
    gc0 = norm(TTN._nl_gradient(x, A, Qg, g, η))
    for _ in 1:12
        xn = TTN._nl_newton_step(x, A, dense_B(Φ, x), Q, Qg, g, η)
        @test P(xn) ≤ P(x) + 1.0e-10
        x = xn
    end
    @test norm(TTN._nl_gradient(x, A, Qg, g, η)) < gc0 / 100

    # SD / FR / PR+ all reach the same local minimum from the same start.
    Random.seed!(4)
    Φc = rand_iso(16, 4); Ac = rand_sym(4)
    Qc = dense_Q(Φc); Qgc = dense_Qgrad(Φc); Pc = dense_P(Φc, Ac, 100.0, 1.0)
    x0 = randn(4)
    Ps = Float64[]
    for variant in (:sd, :fr, :pr)
        x = copy(x0)
        for _ in 1:20
            x = TTN._nl_descent_update(x, Ac, Qc, Qgc, 100.0, 1.0; ν = 4, variant = variant)
        end
        @test norm(TTN._nl_gradient(x, Ac, Qgc, 100.0, 1.0)) < 1.0e-4
        push!(Ps, Pc(x))
    end
    @test maximum(Ps) - minimum(Ps) < 1.0e-6
end

@testset "nonlinear solver: backtracking-Armijo fallback (_nl_backtrack)" begin
    # Descent path: from a non-stationary point the SD fallback finds a step along p = −g_c
    # that satisfies Armijo sufficient decrease, so the returned point strictly lowers P.
    Random.seed!(11)
    for (g, η) in ((0.0, 1.0), (50.0, 1.0), (10.0, 10.0))
        Φ = rand_iso(16, 4); A = rand_sym(4)
        Q = dense_Q(Φ); Qg = dense_Qgrad(Φ)
        x = randn(4)
        gc = TTN._nl_gradient(x, A, Qg, g, η)
        P0 = TTN._nl_penalty(x, A, Q, g, η)
        xnew = TTN._nl_backtrack(x, -gc, gc, A, Q, g, η, P0)
        @test xnew != x                                       # a step was taken
        @test TTN._nl_penalty(xnew, A, Q, g, η) < P0          # accepted ⇒ strict decrease
    end

    # Zero direction: nothing to search, returns the input unchanged (accept by equality).
    let Φ = rand_iso(16, 4), A = rand_sym(4)
        Q = dense_Q(Φ); x = randn(4)
        gc = TTN._nl_gradient(x, A, dense_Qgrad(Φ), 5.0, 1.0)
        P0 = TTN._nl_penalty(x, A, Q, 5.0, 1.0)
        @test TTN._nl_backtrack(x, zero(x), gc, A, Q, 5.0, 1.0, P0) == x
    end

    # Give-up vs ptol: an unreachable target P0 (far below anything achievable) with ptol = 0
    # exhausts every backtracking step and returns x unchanged; a large ptol relaxes the
    # acceptance threshold so the first step is taken (the machine-equal-P path of the docstring).
    let Φ = rand_iso(16, 4), A = rand_sym(4)
        Q = dense_Q(Φ); g = 20.0; η = 1.0
        x = randn(4)
        gc = TTN._nl_gradient(x, A, dense_Qgrad(Φ), g, η)
        @test TTN._nl_backtrack(x, -gc, gc, A, Q, g, η, -1.0e18; ptol = 0.0) == x
        xacc = TTN._nl_backtrack(x, -gc, gc, A, Q, g, η, -1.0e18; ptol = 1.0e19)
        @test xacc != x
        @test all(isfinite, xacc)
    end
end

# Replace core l of u (shallow copy elsewhere); orthogonality flags reset.
function with_core(u::TTvector{T}, l::Int, c::Array{T, 3}) where {T}
    v = copy(u.ttv_vec)
    v[l] = c
    return TTvector{T, length(u.ttv_dims)}(u.N, v, u.ttv_dims, copy(u.ttv_rks), zeros(Int64, u.N))
end

# Dense environment isometry at site l by basis-column densification (u MUST be gauged at l).
function dense_phi(u::TTvector{Float64}, l::Int)
    core = u.ttv_vec[l]
    dims = size(core)
    n = prod(dims)
    Φ = zeros(2^u.N, n)
    for j in 1:n
        c = zeros(dims)
        c[j] = 1.0
        Φ[:, j] = qtt_to_function(with_core(u, l, c))
    end
    return Φ, vec(core)
end

@testset "nonlinear solver: projection gates (env == dense Φ oracle)" begin
    Random.seed!(11)
    for L in (3, 4)
        A = (4.0^L / 2) * Δ(L)
        Ad = qtto_to_matrix(A)
        u0 = rand_tt(ntuple(_ -> 2, L), 4; normalise = true)
        # matricization sanity: qtto_to_matrix matches operator application in decode order
        @test norm(qtt_to_function(A * u0) - Ad * qtt_to_function(u0)) < 1.0e-8 * norm(Ad)
        for l in 1:L
            u = orthogonalize(u0; i = l)
            Φ, x0 = dense_phi(u, l)
            @test opnorm(Φ' * Φ - I) < 1.0e-10                       # gauge ⇒ isometry
            @test norm(Φ * x0 - qtt_to_function(u)) < 1.0e-10        # decode consistency
            # environments at site l
            HA = TTN._nl_right_op_envs(u, A)
            LA = ones(1, 1, 1)
            for k in 1:(l - 1)
                LA = TTN._nl_op_absorb_left(u.ttv_vec[k], A.tto_vec[k], LA)
            end
            ER = TTN._nl_right_qenvs(u)
            EL = ones(1, 1, 1, 1)
            for k in 1:(l - 1)
                EL = TTN._nl_env4_absorb_left(u.ttv_vec[k], EL)
            end
            # A gate
            Aloc = TTN._nl_effective_op(LA, A.tto_vec[l], HA[l])
            Adense = Φ' * Ad * Φ
            @test opnorm(Aloc - (Adense + Adense') / 2) < 1.0e-8
            # B gate
            uu = Φ * x0
            Bd = Φ' * (Diagonal(uu .^ 2) * Φ)
            Bloc = TTN._nl_effective_abs2(EL, u.ttv_vec[l], ER[l])
            @test opnorm(Bloc - Bd) < 1.0e-10
            # Q and ∇Q gates + internal consistency B·x0 == ∇Q(x0)/4
            @test abs(TTN._nl_quartic(EL, u.ttv_vec[l], ER[l]) - sum(uu .^ 4)) < 1.0e-10
            Qg = vec(TTN._nl_quartic_grad(EL, u.ttv_vec[l], ER[l]))
            @test norm(Qg - 4 .* (Φ' * (uu .^ 3))) < 1.0e-10
            @test norm(Bloc * x0 - Qg ./ 4) < 1.0e-10
            # Mutation: |u| instead of u² must be caught by the B gate
            @test opnorm(Φ' * (Diagonal(abs.(uu)) * Φ) - Bd) / opnorm(Bd) > 0.05
        end
    end
end

# ---------------------------------------------------------------------------
# Dense GPE oracle (LinearAlgebra only). Discrete convention: N = 2^L points,
# h = 2^{-L}, A = (1/(2h²))·tridiag(−1,2,−1) (Dirichlet), g_eff = g·2^L,
# E = fᵀAf + g_eff·Σf⁴ on unit-2-norm f (== continuum energy, full g readout).
# Imaginary time descends the GPE functional Ê = fᵀAf + (g_eff/2)·Σf⁴ (the
# Lyapunov functional of the flow, fixed point H(f)f = μf); E is the readout.
# ---------------------------------------------------------------------------
dense_A(L) = SymTridiagonal(fill(4.0^L, 2^L), fill(-(4.0^L) / 2, 2^L - 1))

box_energy(L) = 4.0^L * 2 * sin(π / (2 * (2^L + 1)))^2   # == 4^L(1−cos(π/(2^L+1))), cancellation-free

function dense_gpe_groundstate(L::Int; g_eff::Real = 0.0, tol::Real = 1.0e-13, maxiter::Int = 100_000)
    A = dense_A(L)
    N = 2^L
    f = [sin(π * m / (N + 1)) for m in 1:N]
    f ./= norm(f)
    Ê(f) = dot(f, A * f) + (g_eff / 2) * sum(x -> x^4, f)
    E(f) = dot(f, A * f) + g_eff * sum(x -> x^4, f)
    Eprev = Ê(f)
    τ = 1.0
    for _ in 1:maxiter
        H = SymTridiagonal(A.dv .+ g_eff .* f .^ 2, A.ev)
        fn = (I + τ * H) \ f
        fn ./= norm(fn)
        En = Ê(fn)
        if En ≤ Eprev + 1.0e-13 * abs(Eprev)
            converged = abs(Eprev - En) ≤ tol * abs(En)
            f, Eprev = fn, En
            converged && break
        else
            τ /= 2
            τ < 1.0e-12 && break
        end
    end
    return f, E(f)
end

function gp_residual(L::Int, g_eff::Real, f::AbstractVector)
    H = SymTridiagonal(dense_A(L).dv .+ g_eff .* f .^ 2, dense_A(L).ev)
    Hf = H * f
    μ = dot(f, Hf)
    return norm(Hf .- μ .* f) / norm(Hf)
end

function infidelity(u::TTvector, f::AbstractVector)
    v = qtt_to_function(u)
    return 1 - abs(dot(v ./ norm(v), f ./ norm(f)))
end

@testset "nonlinear solver: dense oracle self-checks" begin
    # analytic box gates (reference O6)
    @test box_energy(6) ≈ 4.783199 atol = 1.0e-5
    @test box_energy(20) ≈ 4.934793 atol = 1.0e-5
    @test box_energy(40) ≈ π^2 / 2 atol = 1.0e-6
    # continuum-normalized GS peaks at √2 (f_c = √N·f for unit-2-norm f)
    f10 = [sin(π * m / (2^10 + 1)) for m in 1:2^10]
    f10 ./= norm(f10)
    @test maximum(sqrt(2.0^10) .* f10) ≈ sqrt(2) atol = 1.0e-3
    # g_eff = 0 imaginary time == analytic eigenvalue
    for L in (6, 8)
        _, E0 = dense_gpe_groundstate(L)
        @test E0 ≈ box_energy(L) rtol = 1.0e-8
    end
    # stationarity + monotone in g (reference O7)
    L = 8
    Eprev = -Inf
    for g in (0.0, 10.0, 100.0)
        f, E = dense_gpe_groundstate(L; g_eff = g * 2.0^L)
        @test gp_residual(L, g * 2.0^L, f) < 1.0e-6
        @test E > Eprev
        Eprev = E
    end
    # convention pin (reference §norm): g=100 box at L=12 lands near Table 1's 122.09942,
    # below it (O(1/N) box-width effect) — distinguishes continuum (~122) from discrete (~5).
    _, E12 = dense_gpe_groundstate(12; g_eff = 100.0 * 2.0^12)
    @test 121.0 < E12 < 122.09942
end

@testset "nonlinear solver: fixed-grid PenaltyALS" begin
    # readout gate: gpe_energy == dense formula on a random normalized state
    Random.seed!(32)
    for L in (3, 4, 6), g in (0.0, 100.0)
        g_eff = g * 2.0^L
        A = (4.0^L / 2) * Δ(L)
        u = rand_tt(ntuple(_ -> 2, L), 4; normalise = true)
        uu = qtt_to_function(u)
        s = dot(uu, uu)
        Ed = dot(uu, qtto_to_matrix(A) * uu) / s + g_eff * sum(uu .^ 4) / s^2
        @test abs(gpe_energy(A, u; g = g_eff) - Ed) < 1.0e-8 * (1 + abs(Ed))
    end

    # linear limit g=0 at L=6: energy and state == analytic box GS
    L = 6
    A = (4.0^L / 2) * Δ(L)
    seed = function_to_qtt(x -> sin(π * x), L)
    u0 = orthogonalize((1 / norm(seed)) * seed)
    u, info = non_linear_solve(A, u0, PenaltyALS(; return_info = true); g = 0.0)
    fbox = [sin(π * m / (2^L + 1)) for m in 1:2^L]
    @test abs(info.energy - box_energy(L)) / box_energy(L) < 1.0e-9
    @test infidelity(u, fbox) < 1.0e-10
    @test abs(dot(u, u) - 1) < 1.0e-4

    # local-solver variants agree with the dense oracle (L=5, moderate interaction)
    L = 5
    g_eff = 10.0 * 2.0^L
    A5 = (4.0^L / 2) * Δ(L)
    fd, Ed = dense_gpe_groundstate(L; g_eff = g_eff)
    seed5 = function_to_qtt(x -> sin(π * x), L)
    pad = rand_tt(ntuple(_ -> 2, L), 4; normalise = true)
    u05 = orthogonalize(seed5 + (1.0e-3 * norm(seed5)) * pad)
    u05 = (1 / norm(u05)) * u05
    Es = Float64[]
    for solver in (:newton, :cg, :sd)
        us, is = non_linear_solve(A5, u05, PenaltyALS(; local_solver = solver, return_info = true); g = g_eff)
        @test abs(is.energy - Ed) / abs(Ed) < 1.0e-4
        push!(Es, is.energy)
    end
    @test maximum(Es) - minimum(Es) < 1.0e-4 * abs(Ed)

    # complex input rejected (real path only)
    @test_throws ArgumentError non_linear_solve(complex(A), complex(u0), PenaltyALS())

    # constructor validation: degenerate configs are rejected
    @test_throws ArgumentError PenaltyALS(η_schedule = Float64[])
    @test_throws ArgumentError PenaltyALS(max_sweeps = 0)
    @test_throws ArgumentError PenaltyALS(ν_local = 0)

    # unconstrained mode (η_schedule = [0.0]): Allen–Cahn / φ⁴ domain-wall state.
    # Minimize uᵀAu + (g/2)Σu⁴ with A = (1/(2h²))·tridiag(−1,2,−1) − (1/(2ε²))I and
    # g = 1/(2ε²) (the discrete φ⁴ energy up to a constant, Dirichlet walls). No norm
    # constraint — the solver must not warn about ⟨u|u⟩ ≠ 1. Oracle: dense damped Newton
    # on the residual g_c(u) = Au + g·u³ (= ½ the discrete Allen–Cahn residual).
    let Lac = 5, ε = 0.05
        g_ac = 1 / (2 * ε^2)
        Aac_dense = SymTridiagonal(dense_A(Lac).dv .- g_ac, dense_A(Lac).ev)
        f = [tanh(m / 2^Lac / (sqrt(2) * ε)) * tanh((1 - m / 2^Lac) / (sqrt(2) * ε)) for m in 1:2^Lac]
        for _ in 1:50
            r = Aac_dense * f .+ g_ac .* f .^ 3
            J = SymTridiagonal(Aac_dense.dv .+ 3g_ac .* f .^ 2, Aac_dense.ev)
            f -= J \ r
            norm(r) < 1.0e-12 && break
        end
        E_dense = dot(f, Aac_dense * f) + (g_ac / 2) * sum(f .^ 4)
        Aac = (4.0^Lac / 2) * Δ(Lac) - g_ac * id_tto(Lac)
        seed = function_to_qtt(x -> tanh(x / (sqrt(2) * ε)) * tanh((1 - x) / (sqrt(2) * ε)), Lac)
        u0ac = orthogonalize(seed + (1.0e-3 * norm(seed)) * rand_tt(ntuple(_ -> 2, Lac), 4; normalise = true))
        alg_ac = PenaltyALS(; η_schedule = [0.0], tol = 1.0e-10, max_sweeps = 50, return_info = true)
        local uac, iac
        @test_logs match_mode = :all begin
            uac, iac = non_linear_solve(Aac, u0ac, alg_ac; g = g_ac)
        end
        E_tt = iac.penalty
        @test abs(E_tt - E_dense) / abs(E_dense) < 1.0e-8
        @test infidelity(uac, f) < 1.0e-8
    end
end

@testset "nonlinear solver: MGR driver" begin
    A_builder = d -> (4.0^d / 2) * Δ(d)
    seed3 = function_to_qtt(x -> sin(π * x), 3)
    u03 = (1 / norm(seed3)) * seed3

    # g = 0: MGR L=3→6 == analytic box GS to machine precision
    mgr0 = MGR(; max_rank = 8, return_info = true)
    u, info = non_linear_solve(A_builder, u03, mgr0; g_builder = d -> 0.0, target_sites = 6)
    fbox = [sin(π * m / (2^6 + 1)) for m in 1:2^6]
    @test abs(info.energy - box_energy(6)) / box_energy(6) < 1.0e-9
    @test infidelity(u, fbox) < 1.0e-10

    # g = 100: MGR L=3→6 == dense imaginary-time oracle
    g = 100.0
    f6, E6 = dense_gpe_groundstate(6; g_eff = g * 2.0^6)
    u6, i6 = non_linear_solve(A_builder, u03, mgr0; g_builder = d -> g * 2.0^d, target_sites = 6)
    @test abs(i6.energy - E6) / abs(E6) < 1.0e-6
    @test infidelity(u6, f6) < 1.0e-9
    @test abs(dot(u6, u6) - 1) < 1.0e-4

    # fixed-grid solve at L=6 (warm full-rank seed) reaches the same energy
    seed6 = function_to_qtt(x -> sin(π * x), 6)
    pad6 = rand_tt(ntuple(_ -> 2, 6), 8; normalise = true)
    u06 = orthogonalize(seed6 + (1.0e-3 * norm(seed6)) * pad6)
    u06 = (1 / norm(u06)) * u06
    _, ifix = non_linear_solve((4.0^6 / 2) * Δ(6), u06, PenaltyALS(; return_info = true); g = g * 2.0^6)
    @test abs(ifix.energy - E6) / abs(E6) < 1.0e-6

    # beyond the densification wall: L=10 vs the dense tridiagonal oracle
    f10, E10 = dense_gpe_groundstate(10; g_eff = g * 2.0^10)
    u10, i10 = non_linear_solve(A_builder, u03, mgr0; g_builder = d -> g * 2.0^d, target_sites = 10)
    @test abs(i10.energy - E10) / abs(E10) < 1.0e-6
    @test infidelity(u10, f10) < 1.0e-9

    # MUTATION (§norm-penalty): minimizing with bare g instead of g_eff = g·2^L must land
    # > 1% off the oracle energy when read out at full g_eff.
    ubad = non_linear_solve(A_builder, u03, MGR(; max_rank = 8); g_builder = d -> g, target_sites = 6)
    @test abs(gpe_energy((4.0^6 / 2) * Δ(6), ubad; g = g * 2.0^6) - E6) / abs(E6) > 0.01

    # χ-convergence mechanism (Figs 8/9): richer bond ⇒ smaller infidelity
    u2, _ = non_linear_solve(A_builder, u03, MGR(; max_rank = 2, return_info = true); g_builder = d -> g * 2.0^d, target_sites = 6)
    u6b, _ = non_linear_solve(A_builder, u03, MGR(; max_rank = 6, return_info = true); g_builder = d -> g * 2.0^d, target_sites = 6)
    @test infidelity(u6b, f6) < 1.0e-11
    @test infidelity(u2, f6) > infidelity(u6b, f6)
end
