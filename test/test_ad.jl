using Test
using TensorTrainNumerics
using Zygote   # loads ChainRulesCore, which activates the extension

@testset "ChainRulesCore extension loads" begin
    ext = Base.get_extension(TensorTrainNumerics, :TensorTrainNumericsChainRulesCoreExt)
    @test ext !== nothing
end

import TensorTrainNumerics: dot
using LinearAlgebra: dot as ladot   # array dot, for the directional pairing

# --- shared helpers (used by later tasks too) ---
# Rebuild a TTvector from its cores, holding metadata fixed from a template.
build_tt(cores, tmpl::TTvector{T, M}) where {T, M} =
    TTvector{eltype(cores[1]), M}(tmpl.N, cores, tmpl.ttv_dims, tmpl.ttv_rks, tmpl.ttv_ot)

flatten_cores(cores) = vcat(vec.(cores)...)

# Directional finite-difference of f at `cores` along per-core directions `dirs`.
function fd_directional(f, cores, dirs; ε = 1.0e-6)
    plus = [cores[k] .+ ε .* dirs[k] for k in eachindex(cores)]
    minus = [cores[k] .- ε .* dirs[k] for k in eachindex(cores)]
    return (f(plus) - f(minus)) / (2ε)
end

@testset "dot rrule (real) — directional FD" begin
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]
    A = rand_tt(Float64, dims, rks)
    B = rand_tt(Float64, dims, rks)
    coresA = A.ttv_vec
    f(cores) = real(dot(build_tt(cores, A), B))

    ḡ = Zygote.gradient(f, coresA)[1]               # Vector{Array{Float64,3}}
    dirs = [randn(size(c)) for c in coresA]
    ad_dd = sum(real(ladot(ḡ[k], dirs[k])) for k in eachindex(coresA))
    fd_dd = fd_directional(f, coresA, dirs)
    @test isapprox(ad_dd, fd_dd; rtol = 1.0e-5, atol = 1.0e-7)
end

@testset "operator-apply * rrule (real) — directional FD" begin
    n = 4
    H = ising_tto(n; J = -1.0, h = -0.5, interaction = :z, field = :x)
    ψ = rand_tt(Float64, ntuple(_ -> 2, n), [1, 2, 2, 2, 1])
    c = rand_tt(Float64, ntuple(_ -> 2, n), [1, 2, 2, 2, 1])
    coresψ = ψ.ttv_vec
    f(cores) = real(dot(c, H * build_tt(cores, ψ)))   # only * (and dot's B-slot) depend on ψ

    ḡ = Zygote.gradient(f, coresψ)[1]
    dirs = [randn(size(cc)) for cc in coresψ]
    ad_dd = sum(real(ladot(ḡ[k], dirs[k])) for k in eachindex(coresψ))
    fd_dd = fd_directional(f, coresψ, dirs)
    @test isapprox(ad_dd, fd_dd; rtol = 1.0e-5, atol = 1.0e-7)
end

@testset "dot rrule (complex) — Wirtinger component FD" begin
    dims = (2, 2)
    rks = [1, 2, 1]
    A = rand_tt(ComplexF64, dims, rks)
    B = rand_tt(ComplexF64, dims, rks)
    coresA = A.ttv_vec
    f(cores) = real(dot(build_tt(cores, A), B))

    ḡ = Zygote.gradient(f, coresA)[1]
    ε = 1.0e-6
    for k in eachindex(coresA), i in eachindex(coresA[k])
        er = [zeros(ComplexF64, size(c)) for c in coresA]
        er[k][i] = 1.0
        ei = [zeros(ComplexF64, size(c)) for c in coresA]
        ei[k][i] = im
        dRe = (
            f([coresA[j] .+ ε .* er[j] for j in eachindex(coresA)]) -
                f([coresA[j] .- ε .* er[j] for j in eachindex(coresA)])
        ) / (2ε)
        dIm = (
            f([coresA[j] .+ ε .* ei[j] for j in eachindex(coresA)]) -
                f([coresA[j] .- ε .* ei[j] for j in eachindex(coresA)])
        ) / (2ε)
        # Zygote convention for real f of complex z: real(g)=∂f/∂Re, imag(g)=∂f/∂Im.
        @test isapprox(real(ḡ[k][i]), dRe; rtol = 1.0e-4, atol = 1.0e-6)
        @test isapprox(imag(ḡ[k][i]), dIm; rtol = 1.0e-4, atol = 1.0e-6)
    end
end

using LinearAlgebra: norm as lanorm

@testset "Rayleigh energy gradient" begin
    # N = 1: per-core gradient equals the analytic Hilbert gradient exactly.
    H1 = (-0.5) * pauli_sum_tto(:x, 1)          # single-site h*X (ising_tto needs d≥2)
    ψ1 = rand_tt(Float64, (2,), [1, 1])
    cores1 = ψ1.ttv_vec
    E1(cores) = (
        ψ = build_tt(cores, ψ1);
        real(dot(ψ, H1 * ψ)) / real(dot(ψ, ψ))
    )
    ḡ1 = Zygote.gradient(E1, cores1)[1]
    nrm2 = real(dot(ψ1, ψ1))
    E = real(dot(ψ1, H1 * ψ1)) / nrm2
    g_analytic = (2 / nrm2) .* ((H1 * ψ1).ttv_vec[1] .- E .* ψ1.ttv_vec[1])
    @test isapprox(vec(ḡ1[1]), vec(g_analytic); rtol = 1.0e-8, atol = 1.0e-10)

    # N = 4: composed gradient matches directional FD (dot + * + constructor).
    n = 4
    H = ising_tto(n; J = -1.0, h = -0.5, interaction = :z, field = :x)
    ψ = rand_tt(Float64, ntuple(_ -> 2, n), [1, 2, 2, 2, 1])
    cores = ψ.ttv_vec
    E(cs) = (φ = build_tt(cs, ψ); real(dot(φ, H * φ)) / real(dot(φ, φ)))
    ḡ = Zygote.gradient(E, cores)[1]
    dirs = [randn(size(c)) for c in cores]
    ad_dd = sum(real(ladot(ḡ[k], dirs[k])) for k in eachindex(cores))
    fd_dd = fd_directional(E, cores, dirs)
    @test isapprox(ad_dd, fd_dd; rtol = 1.0e-5, atol = 1.0e-7)
end

@testset "AD gradient descent reaches DMRG energy" begin
    n = 10
    H = ising_tto(n; J = -1.0, h = -0.5, interaction = :z, field = :x)
    tmpl = rand_tt(ntuple(_ -> 2, n), 6)

    shapes = size.(tmpl.ttv_vec)
    offsets = cumsum([0; prod.(shapes)])
    unflatten(θ) = [reshape(θ[(offsets[k] + 1):offsets[k + 1]], shapes[k]) for k in 1:n]
    loss(θ) = (
        ψ = build_tt(unflatten(θ), tmpl);
        real(dot(ψ, H * ψ)) / real(dot(ψ, ψ))
    )

    # DMRG reference energy (essentially exact for this low-rank ground state).
    energies, ψ_dmrg, _ = dmrg_eigsolve(H, qtt_basis_vector(n, 1); sweep_schedule = [2, 4], rmax_schedule = [16, 16], tol = 1.0e-10)

    E_dmrg = energies[end]

    # Gradient descent with backtracking (guarantees monotone descent) using the
    # AD gradient.
    θ = flatten_cores(tmpl.ttv_vec)
    E0 = loss(θ)
    Eprev = E0
    α = 0.05
    for _ in 1:400
        g = Zygote.gradient(loss, θ)[1]
        θtry = θ .- α .* g
        Etry = loss(θtry)
        while Etry > Eprev && α > 1.0e-12
            α /= 2
            θtry = θ .- α .* g
            Etry = loss(θtry)
        end
        @test Etry ≤ Eprev + 1.0e-9          # monotone (non-increasing) descent
        θ = θtry
        Eprev = Etry
        α *= 1.5                            # let the step grow back between iters
    end
    @test Eprev < E0 - 1.0                  # made substantial progress
    @test Eprev > E_dmrg - 1.0e-6            # variational: never below the reference
    @test Eprev < E_dmrg + 0.2            # got reasonably close to DMRG

    # Per-core AD gradient ≈ 0 at the (exact-eigenvector) DMRG cores. Use a loss
    # built on ψ_dmrg's own shapes (its ranks may differ from the template).
    shapes_d = size.(ψ_dmrg.ttv_vec)
    offsets_d = cumsum([0; prod.(shapes_d)])
    unflatten_d(θ) = [reshape(θ[(offsets_d[k] + 1):offsets_d[k + 1]], shapes_d[k]) for k in 1:n]
    loss_d(θ) = (
        ψ = build_tt(unflatten_d(θ), ψ_dmrg);
        real(dot(ψ, H * ψ)) / real(dot(ψ, ψ))
    )
    g_dmrg = Zygote.gradient(loss_d, flatten_cores(ψ_dmrg.ttv_vec))[1]
    @test lanorm(g_dmrg) < 1.0e-4
end
