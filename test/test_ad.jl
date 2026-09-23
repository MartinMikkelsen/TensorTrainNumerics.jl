using Test
using TensorTrainNumerics
using Zygote   # loads ChainRulesCore, which activates the extension
using Random
using ChainRulesCore: rrule, Tangent, ZeroTangent, NoTangent

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

build_tto(cores, tmpl::TToperator{T, M}) where {T, M} =
    TToperator{eltype(cores[1]), M}(tmpl.N, cores, tmpl.tto_dims, tmpl.tto_rks, tmpl.tto_ot)

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

@testset "dot rrule — both complex arguments match dense pullbacks" begin
    a = ComplexF64[1 + 2im, 3 - 4im]
    b = ComplexF64[2 + im, -1 + 3im]
    A = TTvector(1, [reshape(a, 2, 1, 1)], (2,), [1, 1], [0])
    B = TTvector(1, [reshape(b, 2, 1, 1)], (2,), [1, 1], [0])
    _, tt_pullback = Zygote.pullback(
        (ac, bc) -> dot(build_tt(ac, A), build_tt(bc, B)), A.ttv_vec, B.ttv_vec
    )
    _, dense_pullback = Zygote.pullback(ladot, a, b)

    for seed in (1.0 + 0im, 1.0im, 0.3 - 0.7im)
        ga, gb = tt_pullback(seed)
        expected_a, expected_b = dense_pullback(seed)
        @test vec(ga[1]) ≈ expected_a
        @test vec(gb[1]) ≈ expected_b
    end

    f(bc) = real(dot(A, build_tt(bc, B)))
    gb = Zygote.gradient(f, B.ttv_vec)[1]
    direction = [reshape(ComplexF64[im, 0], 2, 1, 1)]
    @test real(ladot(gb[1], direction[1])) ≈ 2.0
    @test fd_directional(f, B.ttv_vec, direction) ≈ 2.0 atol = 1.0e-8
end

@testset "dot rrule — complex cotangents through TT environments" begin
    rng = Xoshiro(20260906)
    dims = (2, 3, 2)
    ranks_a = [1, 2, 2, 1]
    ranks_b = [1, 1, 2, 1]
    A = TTvector(
        3, [randn(rng, ComplexF64, dims[k], ranks_a[k], ranks_a[k + 1]) for k in 1:3],
        dims, ranks_a, zeros(Int, 3)
    )
    B = TTvector(
        3, [randn(rng, ComplexF64, dims[k], ranks_b[k], ranks_b[k + 1]) for k in 1:3],
        dims, ranks_b, zeros(Int, 3)
    )
    da = [randn(rng, ComplexF64, size(c)) for c in A.ttv_vec]
    db = [randn(rng, ComplexF64, size(c)) for c in B.ttv_vec]
    _, pullback = Zygote.pullback(
        (ac, bc) -> dot(build_tt(ac, A), build_tt(bc, B)), A.ttv_vec, B.ttv_vec
    )
    a_dense = vec(ttv_to_tensor(A))
    b_dense = vec(ttv_to_tensor(B))

    for seed in (1.0 + 0im, 1.0im, 0.3 - 0.7im)
        # Re(conj(seed) * dot(A, B)) supplies this cotangent: seed=1 and
        # seed=im are the real and imaginary objectives respectively.
        ga, gb = pullback(seed)
        fa(ac) = real(conj(seed) * ladot(vec(ttv_to_tensor(build_tt(ac, A))), b_dense))
        fb(bc) = real(conj(seed) * ladot(a_dense, vec(ttv_to_tensor(build_tt(bc, B)))))
        @test sum(real(ladot(ga[k], da[k])) for k in eachindex(da)) ≈ fd_directional(fa, A.ttv_vec, da) rtol = 1.0e-5 atol = 1.0e-7
        @test sum(real(ladot(gb[k], db[k])) for k in eachindex(db)) ≈ fd_directional(fb, B.ttv_vec, db) rtol = 1.0e-5 atol = 1.0e-7
    end
end

@testset "complex dot gradient composes with fixed operator application" begin
    rng = Xoshiro(20260907)
    dims = (2, 2, 2)
    ranks = [1, 2, 2, 1]
    H = pauli_sum_tto(:y, 3)
    ψ = TTvector(
        3, [randn(rng, ComplexF64, dims[k], ranks[k], ranks[k + 1]) for k in 1:3],
        dims, ranks, zeros(Int, 3)
    )
    c = TTvector(
        3, [randn(rng, ComplexF64, dims[k], ranks[k], ranks[k + 1]) for k in 1:3],
        dims, ranks, zeros(Int, 3)
    )
    dirs = [randn(rng, ComplexF64, size(core)) for core in ψ.ttv_vec]
    for component in (real, imag)
        f(cores) = component(dot(c, H * build_tt(cores, ψ)))
        g = Zygote.gradient(f, ψ.ttv_vec)[1]
        ad_dd = sum(real(ladot(g[k], dirs[k])) for k in eachindex(dirs))
        @test ad_dd ≈ fd_directional(f, ψ.ttv_vec, dirs) rtol = 1.0e-5 atol = 1.0e-7
    end
end

using LinearAlgebra: norm as lanorm

@testset "operator and state gradients match dense directional derivatives" begin
    rng = Xoshiro(20260917)
    for T in (Float64, ComplexF64), dims in ((3,), (2, 3, 2))
        n = length(dims)
        rh = n == 1 ? [1, 1] : [1, 2, 3, 1]
        rp = n == 1 ? [1, 1] : [1, 3, 2, 1]
        H = TToperator(n, [randn(rng, T, dims[k], dims[k], rh[k], rh[k + 1]) for k in 1:n], dims, rh, zeros(Int, n))
        ψ = TTvector(n, [randn(rng, T, dims[k], rp[k], rp[k + 1]) for k in 1:n], dims, rp, zeros(Int, n))
        c = TTvector(n, [randn(rng, T, d, 1, 1) for d in dims], dims, ones(Int, n + 1), zeros(Int, n))
        dh = [randn(rng, T, size(core)) for core in H.tto_vec]
        dp = [randn(rng, T, size(core)) for core in ψ.ttv_vec]
        cdense = vec(ttv_to_tensor(c))
        dense_output(hc, pc) = reshape(tto_to_tensor(build_tto(hc, H)), prod(dims), :) * vec(ttv_to_tensor(build_tt(pc, ψ)))
        value, pullback = Zygote.pullback(
            (hc, pc) -> dot(c, build_tto(hc, H) * build_tt(pc, ψ)), H.tto_vec, ψ.ttv_vec
        )
        @test value ≈ ladot(cdense, dense_output(H.tto_vec, ψ.ttv_vec))
        seeds = T <: Real ? (1.0, -0.7) : (1.0 + 0im, 1.0im, 0.3 - 0.7im)
        for seed in seeds
            gh, gp = pullback(seed)
            @test gh !== nothing
            if gh !== nothing
                fh(hc) = real(conj(seed) * ladot(cdense, dense_output(hc, ψ.ttv_vec)))
                @test sum(real(ladot(gh[k], dh[k])) for k in 1:n) ≈ fd_directional(fh, H.tto_vec, dh) rtol = 1.0e-5 atol = 1.0e-7
            end
            fp(pc) = real(conj(seed) * ladot(cdense, dense_output(H.tto_vec, pc)))
            @test sum(real(ladot(gp[k], dp[k])) for k in 1:n) ≈ fd_directional(fp, ψ.ttv_vec, dp) rtol = 1.0e-5 atol = 1.0e-7
        end
        Y, pullback_zero = rrule(*, H, ψ)
        for seed in (ZeroTangent(), NoTangent(), Tangent{typeof(Y)}(; ttv_vec = ZeroTangent()))
            _, gh, gp = pullback_zero(seed)
            @test gh isa ZeroTangent
            @test gp isa ZeroTangent
        end
        # Unused output cores receive individual ZeroTangent cotangents from Zygote.
        core_loss(hc, pc) = sum(abs2, (build_tto(hc, H) * build_tt(pc, ψ)).ttv_vec[1])
        gh, gp = Zygote.gradient(core_loss, H.tto_vec, ψ.ttv_vec)
        @test sum(real(ladot(gh[k], dh[k])) for k in 1:n) ≈ fd_directional(hc -> core_loss(hc, ψ.ttv_vec), H.tto_vec, dh) rtol = 1.0e-5 atol = 1.0e-7
        @test sum(real(ladot(gp[k], dp[k])) for k in 1:n) ≈ fd_directional(pc -> core_loss(H.tto_vec, pc), ψ.ttv_vec, dp) rtol = 1.0e-5 atol = 1.0e-7
        @test all(k -> iszero(gh[k]) && iszero(gp[k]), 2:n)
    end
end

@testset "Hadamard gradients match dense directional derivatives" begin
    rng = Xoshiro(20260918)
    for T in (Float64, ComplexF64)
        dims, n = (2, 3, 2), 3
        rx, ry = [1, 2, 3, 1], [1, 3, 2, 1]
        x = TTvector(n, [randn(rng, T, dims[k], rx[k], rx[k + 1]) for k in 1:n], dims, rx, zeros(Int, n))
        y = TTvector(n, [randn(rng, T, dims[k], ry[k], ry[k + 1]) for k in 1:n], dims, ry, zeros(Int, n))
        c = TTvector(n, [randn(rng, T, d, 1, 1) for d in dims], dims, ones(Int, n + 1), zeros(Int, n))
        dx = [randn(rng, T, size(core)) for core in x.ttv_vec]
        dy = [randn(rng, T, size(core)) for core in y.ttv_vec]
        cdense = vec(ttv_to_tensor(c))
        dense_output(xc, yc) = vec(ttv_to_tensor(build_tt(xc, x))) .* vec(ttv_to_tensor(build_tt(yc, y)))
        value, pullback = Zygote.pullback(
            (xc, yc) -> dot(c, hadamard(build_tt(xc, x), build_tt(yc, y))), x.ttv_vec, y.ttv_vec
        )
        @test value ≈ ladot(cdense, dense_output(x.ttv_vec, y.ttv_vec))
        seeds = T <: Real ? (1.0, -0.7) : (1.0 + 0im, 1.0im, 0.3 - 0.7im)
        for seed in seeds
            gx, gy = pullback(seed)
            fx(xc) = real(conj(seed) * ladot(cdense, dense_output(xc, y.ttv_vec)))
            fy(yc) = real(conj(seed) * ladot(cdense, dense_output(x.ttv_vec, yc)))
            @test sum(real(ladot(gx[k], dx[k])) for k in 1:n) ≈ fd_directional(fx, x.ttv_vec, dx) rtol = 1.0e-5 atol = 1.0e-7
            @test sum(real(ladot(gy[k], dy[k])) for k in 1:n) ≈ fd_directional(fy, y.ttv_vec, dy) rtol = 1.0e-5 atol = 1.0e-7
        end
        # Both occurrences must contribute when a core is reused (Cookbook Theorem 20).
        square_loss(xc) = real(dot(c, hadamard(build_tt(xc, x), build_tt(xc, x))))
        gx = only(Zygote.gradient(square_loss, x.ttv_vec))
        dense_square(xc) = real(ladot(cdense, vec(ttv_to_tensor(build_tt(xc, x))) .^ 2))
        @test sum(real(ladot(gx[k], dx[k])) for k in 1:n) ≈ fd_directional(dense_square, x.ttv_vec, dx) rtol = 1.0e-5 atol = 1.0e-7
        Y, pullback_zero = rrule(hadamard, x, y)
        for seed in (ZeroTangent(), NoTangent(), Tangent{typeof(Y)}(; ttv_vec = ZeroTangent()))
            _, gx, gy = pullback_zero(seed)
            @test gx isa ZeroTangent
            @test gy isa ZeroTangent
        end
        core_loss(xc, yc) = sum(abs2, hadamard(build_tt(xc, x), build_tt(yc, y)).ttv_vec[1])
        gx, gy = Zygote.gradient(core_loss, x.ttv_vec, y.ttv_vec)
        @test sum(real(ladot(gx[k], dx[k])) for k in 1:n) ≈ fd_directional(xc -> core_loss(xc, y.ttv_vec), x.ttv_vec, dx) rtol = 1.0e-5 atol = 1.0e-7
        @test sum(real(ladot(gy[k], dy[k])) for k in 1:n) ≈ fd_directional(yc -> core_loss(x.ttv_vec, yc), y.ttv_vec, dy) rtol = 1.0e-5 atol = 1.0e-7
        @test all(k -> iszero(gx[k]) && iszero(gy[k]), 2:n)
    end
end

# Seeded pullback of a complex scalar function g(args...) = value, checked against a
# central difference of real(conj(seed) * g) along a direction in argument `i`.
function check_seeded_fd(g, args, i, dir, grad, seed; ε = 1.0e-6)
    shifted(t) = map(j -> j == i ? (args[j] isa Number ? args[j] + t * dir : [args[j][k] .+ t .* dir[k] for k in eachindex(dir)]) : args[j], eachindex(args))
    fd = (real(conj(seed) * g(shifted(ε)...)) - real(conj(seed) * g(shifted(-ε)...))) / (2ε)
    ad = args[i] isa Number ? real(conj(grad) * dir) : sum(real(ladot(grad[k], dir[k])) for k in eachindex(dir))
    return isapprox(ad, fd; rtol = 1.0e-5, atol = 1.0e-7)
end

@testset "sum, difference, and scalar gradients match dense directional derivatives" begin
    rng = Xoshiro(20260923)
    for T in (Float64, ComplexF64), dims in ((3,), (2, 3, 2))
        n = length(dims)
        rx = n == 1 ? [1, 1] : [1, 2, 3, 1]
        ry = n == 1 ? [1, 1] : [1, 3, 2, 1]
        x = TTvector(n, [randn(rng, T, dims[k], rx[k], rx[k + 1]) for k in 1:n], dims, rx, zeros(Int, n))
        y = TTvector(n, [randn(rng, T, dims[k], ry[k], ry[k + 1]) for k in 1:n], dims, ry, zeros(Int, n))
        c = TTvector(n, [randn(rng, T, d, 1, 1) for d in dims], dims, ones(Int, n + 1), zeros(Int, n))
        cdense = vec(ttv_to_tensor(c))
        dense(cores, tmpl) = vec(ttv_to_tensor(build_tt(cores, tmpl)))
        dx = [randn(rng, T, size(core)) for core in x.ttv_vec]
        dy = [randn(rng, T, size(core)) for core in y.ttv_vec]
        complex_seeds = (1.0 + 0im, 1.0im, 0.3 - 0.7im)
        seeds = T <: Real ? (1.0, -0.7) : complex_seeds

        for (op, dense_op) in ((+, +), (-, -))
            f(xc, yc) = dot(c, op(build_tt(xc, x), build_tt(yc, y)))
            fd(xc, yc) = ladot(cdense, dense_op(dense(xc, x), dense(yc, y)))
            value, pullback = Zygote.pullback(f, x.ttv_vec, y.ttv_vec)
            @test value ≈ fd(x.ttv_vec, y.ttv_vec)
            for seed in seeds
                gx, gy = pullback(seed)
                @test check_seeded_fd(fd, (x.ttv_vec, y.ttv_vec), 1, dx, gx, seed)
                @test check_seeded_fd(fd, (x.ttv_vec, y.ttv_vec), 2, dy, gy, seed)
            end
        end

        # Scalar factors of the same, real, and complex type, including zero.
        for a in (T(0.7), 0.7, 0.3 - 0.8im, zero(T))
            for (f, fd) in (
                    ((xc, s) -> dot(c, s * build_tt(xc, x)), (xc, s) -> ladot(cdense, s .* dense(xc, x))),
                    ((xc, s) -> dot(c, build_tt(xc, x) * s), (xc, s) -> ladot(cdense, s .* dense(xc, x))),
                )
                value, pullback = Zygote.pullback(f, x.ttv_vec, a)
                @test value ≈ fd(x.ttv_vec, a)
                for seed in (value isa Complex ? complex_seeds : seeds)
                    gx, ga = pullback(seed)
                    @test eltype(gx[1]) == T
                    @test check_seeded_fd(fd, (x.ttv_vec, a), 1, dx, gx, seed)
                    for da in (a isa Real ? (1.0,) : (1.0, 1.0im))
                        @test check_seeded_fd(fd, (x.ttv_vec, a), 2, da, ga, seed)
                    end
                end
            end
        end

        # The scalar multiplies the orthogonality center, which need not be core 1.
        xo = orthogonalize(x; i = n)
        dxo = [randn(rng, T, size(core)) for core in xo.ttv_vec]
        fo(xc, s) = ladot(cdense, s .* dense(xc, xo))
        value, pullback = Zygote.pullback((xc, s) -> dot(c, s * build_tt(xc, xo)), xo.ttv_vec, T(0.7))
        @test value ≈ fo(xo.ttv_vec, T(0.7))
        gx, ga = pullback(one(T))
        @test check_seeded_fd(fo, (xo.ttv_vec, T(0.7)), 1, dxo, gx, one(T))
        @test check_seeded_fd(fo, (xo.ttv_vec, T(0.7)), 2, one(T), ga, one(T))

        # Division by a scalar composes the scalar rule with 1/a.
        a = T(1.3)
        gx, ga = Zygote.gradient((xc, s) -> real(dot(c, build_tt(xc, x) / s)), x.ttv_vec, a)
        fdiv(xc, s) = ladot(cdense, dense(xc, x) ./ s)
        @test check_seeded_fd(fdiv, (x.ttv_vec, a), 1, dx, gx, 1.0)
        @test check_seeded_fd(fdiv, (x.ttv_vec, a), 2, one(a), ga, 1.0)

        # Operator sums, differences, and scalar multiples, applied to a fixed state.
        rh = n == 1 ? [1, 1] : [1, 2, 2, 1]
        A = TToperator(n, [randn(rng, T, dims[k], dims[k], rh[k], rh[k + 1]) for k in 1:n], dims, rh, zeros(Int, n))
        B = TToperator(n, [randn(rng, T, dims[k], dims[k], rh[k], rh[k + 1]) for k in 1:n], dims, rh, zeros(Int, n))
        dA = [randn(rng, T, size(core)) for core in A.tto_vec]
        dB = [randn(rng, T, size(core)) for core in B.tto_vec]
        densemat(cores, tmpl) = reshape(tto_to_tensor(build_tto(cores, tmpl)), prod(dims), :)
        xdense = dense(x.ttv_vec, x)
        for (op, dense_op) in ((+, +), (-, -))
            f(ac, bc) = dot(c, op(build_tto(ac, A), build_tto(bc, B)) * x)
            fd(ac, bc) = ladot(cdense, dense_op(densemat(ac, A), densemat(bc, B)) * xdense)
            value, pullback = Zygote.pullback(f, A.tto_vec, B.tto_vec)
            @test value ≈ fd(A.tto_vec, B.tto_vec)
            for seed in seeds
                gA, gB = pullback(seed)
                @test check_seeded_fd(fd, (A.tto_vec, B.tto_vec), 1, dA, gA, seed)
                @test check_seeded_fd(fd, (A.tto_vec, B.tto_vec), 2, dB, gB, seed)
            end
        end
        for a in (T(0.7), 0.3 - 0.8im, zero(T))
            f(ac, s) = dot(c, (s * build_tto(ac, A)) * x)
            fd(ac, s) = ladot(cdense, s .* (densemat(ac, A) * xdense))
            value, pullback = Zygote.pullback(f, A.tto_vec, a)
            @test value ≈ fd(A.tto_vec, a)
            for seed in (value isa Complex ? complex_seeds : seeds)
                gA, ga = pullback(seed)
                @test check_seeded_fd(fd, (A.tto_vec, a), 1, dA, gA, seed)
                for da in (a isa Real ? (1.0,) : (1.0, 1.0im))
                    @test check_seeded_fd(fd, (A.tto_vec, a), 2, da, ga, seed)
                end
            end
        end

        # Zero output cotangents give zero input cotangents.
        for (fun, args) in ((+, (x, y)), (+, (A, B)), (*, (T(2), x)), (*, (T(2), A)))
            Y, pullback_zero = rrule(fun, args...)
            field = Y isa TTvector ? :ttv_vec : :tto_vec
            for seed in (ZeroTangent(), NoTangent(), Tangent{typeof(Y)}(; (field => ZeroTangent(),)...))
                _, g1, g2 = pullback_zero(seed)
                @test g1 isa ZeroTangent
                @test g2 isa ZeroTangent
            end
        end
    end
end

@testset "field reads of a differentiated TT keep the rule cotangents" begin
    rng = Xoshiro(20260924)
    dims, rks = (2, 3, 2), [1, 2, 2, 1]
    x = TTvector(3, [randn(rng, dims[k], rks[k], rks[k + 1]) for k in 1:3], dims, rks, zeros(Int, 3))
    dx = [randn(rng, size(c)) for c in x.ttv_vec]
    ad_dd(g) = sum(real(ladot(g[k], dx[k])) for k in eachindex(dx))

    # A rule (dot) and a direct read of a core of the same TTvector.
    mixed(cs) = (u = build_tt(cs, x); real(dot(u, u)) + sum(u.ttv_vec[1]))
    g = only(Zygote.gradient(mixed, x.ttv_vec))
    @test all(!isnothing, g)
    @test ad_dd(g) ≈ fd_directional(mixed, x.ttv_vec, dx) rtol = 1.0e-5 atol = 1.0e-7

    # An integer field of the same TTvector used in differentiated arithmetic.
    scaled(cs) = (u = build_tt(cs, x); real(dot(u, u)) * 2.0^u.N)
    g = only(Zygote.gradient(scaled, x.ttv_vec))
    @test g !== nothing
    @test ad_dd(g) ≈ fd_directional(scaled, x.ttv_vec, dx) rtol = 1.0e-5 atol = 1.0e-7

    # The same through a QTTvector wrapper.
    qdims, qrks = (2, 2, 2, 2), [1, 2, 3, 2, 1]
    y = TTvector(4, [randn(rng, qdims[k], qrks[k], qrks[k + 1]) for k in 1:4], qdims, qrks, zeros(Int, 4))
    dy = [randn(rng, size(c)) for c in y.ttv_vec]
    qloss(cs) = (q = QTTvector(build_tt(cs, y), 2, 2, :interleaved); real(dot(q, q)) + sum(q.ttv_vec[2]) * q.n_dims)
    g = only(Zygote.gradient(qloss, y.ttv_vec))
    @test sum(real(ladot(g[k], dy[k])) for k in eachindex(dy)) ≈ fd_directional(qloss, y.ttv_vec, dy) rtol = 1.0e-5 atol = 1.0e-7
end

@testset "coupling-constant derivatives of a Hamiltonian" begin
    n = 5
    Hzz = pauli_pair_sum_tto(:z, :z, n)
    Hx = pauli_sum_tto(:x, n)
    ψ = rand_tt(Float64, ntuple(_ -> 2, n), [1, 2, 3, 3, 2, 1])
    energy(J, h) = real(dot(ψ, (J * Hzz + h * Hx) * ψ)) / real(dot(ψ, ψ))
    gJ, gh = Zygote.gradient(energy, -1.0, -0.5)
    @test gJ ≈ real(dot(ψ, Hzz * ψ)) / real(dot(ψ, ψ))
    @test gh ≈ real(dot(ψ, Hx * ψ)) / real(dot(ψ, ψ))
end

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
