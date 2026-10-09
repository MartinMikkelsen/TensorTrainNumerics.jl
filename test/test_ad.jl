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

# Rebuild a TTVector from its cores, holding metadata fixed from a template.
build_tt(cores, tmpl::TTVector{T, M}) where {T, M} =
    TTVector{eltype(cores[1]), M}(cores, tmpl.dims, tmpl.ranks; orthogonality = tmpl.orthogonality)

build_tto(cores, tmpl::TTOperator{T, M}) where {T, M} =
    TTOperator{eltype(cores[1]), M}(cores, tmpl.row_dims, tmpl.col_dims, tmpl.ranks; orthogonality = tmpl.orthogonality)

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
    coresA = A.cores
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
    coresψ = ψ.cores
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
    coresA = A.cores
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
    A = TTVector([reshape(a, 2, 1, 1)], (2,), [1, 1])
    B = TTVector([reshape(b, 2, 1, 1)], (2,), [1, 1])
    _, tt_pullback = Zygote.pullback(
        (ac, bc) -> dot(build_tt(ac, A), build_tt(bc, B)), A.cores, B.cores
    )
    _, dense_pullback = Zygote.pullback(ladot, a, b)

    for seed in (1.0 + 0im, 1.0im, 0.3 - 0.7im)
        ga, gb = tt_pullback(seed)
        expected_a, expected_b = dense_pullback(seed)
        @test vec(ga[1]) ≈ expected_a
        @test vec(gb[1]) ≈ expected_b
    end

    f(bc) = real(dot(A, build_tt(bc, B)))
    gb = Zygote.gradient(f, B.cores)[1]
    direction = [reshape(ComplexF64[im, 0], 2, 1, 1)]
    @test real(ladot(gb[1], direction[1])) ≈ 2.0
    @test fd_directional(f, B.cores, direction) ≈ 2.0 atol = 1.0e-8
end

@testset "dot rrule — complex cotangents through TT environments" begin
    rng = Xoshiro(20260906)
    dims = (2, 3, 2)
    ranks_a = [1, 2, 2, 1]
    ranks_b = [1, 1, 2, 1]
    A = TTVector(
        [randn(rng, ComplexF64, dims[k], ranks_a[k], ranks_a[k + 1]) for k in 1:3],
        dims, ranks_a
    )
    B = TTVector(
        [randn(rng, ComplexF64, dims[k], ranks_b[k], ranks_b[k + 1]) for k in 1:3],
        dims, ranks_b
    )
    da = [randn(rng, ComplexF64, size(c)) for c in A.cores]
    db = [randn(rng, ComplexF64, size(c)) for c in B.cores]
    _, pullback = Zygote.pullback(
        (ac, bc) -> dot(build_tt(ac, A), build_tt(bc, B)), A.cores, B.cores
    )
    a_dense = vec(tt_to_tensor(A))
    b_dense = vec(tt_to_tensor(B))

    for seed in (1.0 + 0im, 1.0im, 0.3 - 0.7im)
        # Re(conj(seed) * dot(A, B)) supplies this cotangent: seed=1 and
        # seed=im are the real and imaginary objectives respectively.
        ga, gb = pullback(seed)
        fa(ac) = real(conj(seed) * ladot(vec(tt_to_tensor(build_tt(ac, A))), b_dense))
        fb(bc) = real(conj(seed) * ladot(a_dense, vec(tt_to_tensor(build_tt(bc, B)))))
        @test sum(real(ladot(ga[k], da[k])) for k in eachindex(da)) ≈ fd_directional(fa, A.cores, da) rtol = 1.0e-5 atol = 1.0e-7
        @test sum(real(ladot(gb[k], db[k])) for k in eachindex(db)) ≈ fd_directional(fb, B.cores, db) rtol = 1.0e-5 atol = 1.0e-7
    end
end

@testset "complex dot gradient composes with fixed operator application" begin
    rng = Xoshiro(20260907)
    dims = (2, 2, 2)
    ranks = [1, 2, 2, 1]
    H = pauli_sum_tto(:y, 3)
    ψ = TTVector(
        [randn(rng, ComplexF64, dims[k], ranks[k], ranks[k + 1]) for k in 1:3],
        dims, ranks
    )
    c = TTVector(
        [randn(rng, ComplexF64, dims[k], ranks[k], ranks[k + 1]) for k in 1:3],
        dims, ranks
    )
    dirs = [randn(rng, ComplexF64, size(core)) for core in ψ.cores]
    for component in (real, imag)
        f(cores) = component(dot(c, H * build_tt(cores, ψ)))
        g = Zygote.gradient(f, ψ.cores)[1]
        ad_dd = sum(real(ladot(g[k], dirs[k])) for k in eachindex(dirs))
        @test ad_dd ≈ fd_directional(f, ψ.cores, dirs) rtol = 1.0e-5 atol = 1.0e-7
    end
end

using LinearAlgebra: norm as lanorm

@testset "operator and state gradients match dense directional derivatives" begin
    rng = Xoshiro(20260917)
    for T in (Float64, ComplexF64), dims in ((3,), (2, 3, 2))
        n = length(dims)
        rh = n == 1 ? [1, 1] : [1, 2, 3, 1]
        rp = n == 1 ? [1, 1] : [1, 3, 2, 1]
        H = TTOperator([randn(rng, T, dims[k], dims[k], rh[k], rh[k + 1]) for k in 1:n], dims, rh)
        ψ = TTVector([randn(rng, T, dims[k], rp[k], rp[k + 1]) for k in 1:n], dims, rp)
        c = TTVector([randn(rng, T, d, 1, 1) for d in dims], dims, ones(Int, n + 1))
        dh = [randn(rng, T, size(core)) for core in H.cores]
        dp = [randn(rng, T, size(core)) for core in ψ.cores]
        cdense = vec(tt_to_tensor(c))
        dense_output(hc, pc) = reshape(tto_to_tensor(build_tto(hc, H)), prod(dims), :) * vec(tt_to_tensor(build_tt(pc, ψ)))
        value, pullback = Zygote.pullback(
            (hc, pc) -> dot(c, build_tto(hc, H) * build_tt(pc, ψ)), H.cores, ψ.cores
        )
        @test value ≈ ladot(cdense, dense_output(H.cores, ψ.cores))
        seeds = T <: Real ? (1.0, -0.7) : (1.0 + 0im, 1.0im, 0.3 - 0.7im)
        for seed in seeds
            gh, gp = pullback(seed)
            @test gh !== nothing
            if gh !== nothing
                fh(hc) = real(conj(seed) * ladot(cdense, dense_output(hc, ψ.cores)))
                @test sum(real(ladot(gh[k], dh[k])) for k in 1:n) ≈ fd_directional(fh, H.cores, dh) rtol = 1.0e-5 atol = 1.0e-7
            end
            fp(pc) = real(conj(seed) * ladot(cdense, dense_output(H.cores, pc)))
            @test sum(real(ladot(gp[k], dp[k])) for k in 1:n) ≈ fd_directional(fp, ψ.cores, dp) rtol = 1.0e-5 atol = 1.0e-7
        end
        Y, pullback_zero = rrule(*, H, ψ)
        for seed in (ZeroTangent(), NoTangent(), Tangent{typeof(Y)}(; ttv_vec = ZeroTangent()))
            _, gh, gp = pullback_zero(seed)
            @test gh isa ZeroTangent
            @test gp isa ZeroTangent
        end
        # Unused output cores receive individual ZeroTangent cotangents from Zygote.
        core_loss(hc, pc) = sum(abs2, (build_tto(hc, H) * build_tt(pc, ψ)).cores[1])
        gh, gp = Zygote.gradient(core_loss, H.cores, ψ.cores)
        @test sum(real(ladot(gh[k], dh[k])) for k in 1:n) ≈ fd_directional(hc -> core_loss(hc, ψ.cores), H.cores, dh) rtol = 1.0e-5 atol = 1.0e-7
        @test sum(real(ladot(gp[k], dp[k])) for k in 1:n) ≈ fd_directional(pc -> core_loss(H.cores, pc), ψ.cores, dp) rtol = 1.0e-5 atol = 1.0e-7
        @test all(k -> iszero(gh[k]) && iszero(gp[k]), 2:n)
    end
end

@testset "Hadamard gradients match dense directional derivatives" begin
    rng = Xoshiro(20260918)
    for T in (Float64, ComplexF64)
        dims, n = (2, 3, 2), 3
        rx, ry = [1, 2, 3, 1], [1, 3, 2, 1]
        x = TTVector([randn(rng, T, dims[k], rx[k], rx[k + 1]) for k in 1:n], dims, rx)
        y = TTVector([randn(rng, T, dims[k], ry[k], ry[k + 1]) for k in 1:n], dims, ry)
        c = TTVector([randn(rng, T, d, 1, 1) for d in dims], dims, ones(Int, n + 1))
        dx = [randn(rng, T, size(core)) for core in x.cores]
        dy = [randn(rng, T, size(core)) for core in y.cores]
        cdense = vec(tt_to_tensor(c))
        dense_output(xc, yc) = vec(tt_to_tensor(build_tt(xc, x))) .* vec(tt_to_tensor(build_tt(yc, y)))
        value, pullback = Zygote.pullback(
            (xc, yc) -> dot(c, hadamard(build_tt(xc, x), build_tt(yc, y))), x.cores, y.cores
        )
        @test value ≈ ladot(cdense, dense_output(x.cores, y.cores))
        seeds = T <: Real ? (1.0, -0.7) : (1.0 + 0im, 1.0im, 0.3 - 0.7im)
        for seed in seeds
            gx, gy = pullback(seed)
            fx(xc) = real(conj(seed) * ladot(cdense, dense_output(xc, y.cores)))
            fy(yc) = real(conj(seed) * ladot(cdense, dense_output(x.cores, yc)))
            @test sum(real(ladot(gx[k], dx[k])) for k in 1:n) ≈ fd_directional(fx, x.cores, dx) rtol = 1.0e-5 atol = 1.0e-7
            @test sum(real(ladot(gy[k], dy[k])) for k in 1:n) ≈ fd_directional(fy, y.cores, dy) rtol = 1.0e-5 atol = 1.0e-7
        end
        # Both occurrences must contribute when a core is reused (Cookbook Theorem 20).
        square_loss(xc) = real(dot(c, hadamard(build_tt(xc, x), build_tt(xc, x))))
        gx = only(Zygote.gradient(square_loss, x.cores))
        dense_square(xc) = real(ladot(cdense, vec(tt_to_tensor(build_tt(xc, x))) .^ 2))
        @test sum(real(ladot(gx[k], dx[k])) for k in 1:n) ≈ fd_directional(dense_square, x.cores, dx) rtol = 1.0e-5 atol = 1.0e-7
        Y, pullback_zero = rrule(hadamard, x, y)
        for seed in (ZeroTangent(), NoTangent(), Tangent{typeof(Y)}(; ttv_vec = ZeroTangent()))
            _, gx, gy = pullback_zero(seed)
            @test gx isa ZeroTangent
            @test gy isa ZeroTangent
        end
        core_loss(xc, yc) = sum(abs2, hadamard(build_tt(xc, x), build_tt(yc, y)).cores[1])
        gx, gy = Zygote.gradient(core_loss, x.cores, y.cores)
        @test sum(real(ladot(gx[k], dx[k])) for k in 1:n) ≈ fd_directional(xc -> core_loss(xc, y.cores), x.cores, dx) rtol = 1.0e-5 atol = 1.0e-7
        @test sum(real(ladot(gy[k], dy[k])) for k in 1:n) ≈ fd_directional(yc -> core_loss(x.cores, yc), y.cores, dy) rtol = 1.0e-5 atol = 1.0e-7
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
        x = TTVector([randn(rng, T, dims[k], rx[k], rx[k + 1]) for k in 1:n], dims, rx)
        y = TTVector([randn(rng, T, dims[k], ry[k], ry[k + 1]) for k in 1:n], dims, ry)
        c = TTVector([randn(rng, T, d, 1, 1) for d in dims], dims, ones(Int, n + 1))
        cdense = vec(tt_to_tensor(c))
        dense(cores, tmpl) = vec(tt_to_tensor(build_tt(cores, tmpl)))
        dx = [randn(rng, T, size(core)) for core in x.cores]
        dy = [randn(rng, T, size(core)) for core in y.cores]
        complex_seeds = (1.0 + 0im, 1.0im, 0.3 - 0.7im)
        seeds = T <: Real ? (1.0, -0.7) : complex_seeds

        for (op, dense_op) in ((+, +), (-, -))
            f(xc, yc) = dot(c, op(build_tt(xc, x), build_tt(yc, y)))
            fd(xc, yc) = ladot(cdense, dense_op(dense(xc, x), dense(yc, y)))
            value, pullback = Zygote.pullback(f, x.cores, y.cores)
            @test value ≈ fd(x.cores, y.cores)
            for seed in seeds
                gx, gy = pullback(seed)
                @test check_seeded_fd(fd, (x.cores, y.cores), 1, dx, gx, seed)
                @test check_seeded_fd(fd, (x.cores, y.cores), 2, dy, gy, seed)
            end
        end

        # Scalar factors of the same, real, and complex type, including zero.
        for a in (T(0.7), 0.7, 0.3 - 0.8im, zero(T))
            for (f, fd) in (
                    ((xc, s) -> dot(c, s * build_tt(xc, x)), (xc, s) -> ladot(cdense, s .* dense(xc, x))),
                    ((xc, s) -> dot(c, build_tt(xc, x) * s), (xc, s) -> ladot(cdense, s .* dense(xc, x))),
                )
                value, pullback = Zygote.pullback(f, x.cores, a)
                @test value ≈ fd(x.cores, a)
                for seed in (value isa Complex ? complex_seeds : seeds)
                    gx, ga = pullback(seed)
                    @test eltype(gx[1]) == T
                    @test check_seeded_fd(fd, (x.cores, a), 1, dx, gx, seed)
                    for da in (a isa Real ? (1.0,) : (1.0, 1.0im))
                        @test check_seeded_fd(fd, (x.cores, a), 2, da, ga, seed)
                    end
                end
            end
        end

        # The scalar multiplies the orthogonality center, which need not be core 1.
        xo = orthogonalize(x; center = n)
        dxo = [randn(rng, T, size(core)) for core in xo.cores]
        fo(xc, s) = ladot(cdense, s .* dense(xc, xo))
        value, pullback = Zygote.pullback((xc, s) -> dot(c, s * build_tt(xc, xo)), xo.cores, T(0.7))
        @test value ≈ fo(xo.cores, T(0.7))
        gx, ga = pullback(one(T))
        @test check_seeded_fd(fo, (xo.cores, T(0.7)), 1, dxo, gx, one(T))
        @test check_seeded_fd(fo, (xo.cores, T(0.7)), 2, one(T), ga, one(T))

        # Division by a scalar composes the scalar rule with 1/a.
        a = T(1.3)
        gx, ga = Zygote.gradient((xc, s) -> real(dot(c, build_tt(xc, x) / s)), x.cores, a)
        fdiv(xc, s) = ladot(cdense, dense(xc, x) ./ s)
        @test check_seeded_fd(fdiv, (x.cores, a), 1, dx, gx, 1.0)
        @test check_seeded_fd(fdiv, (x.cores, a), 2, one(a), ga, 1.0)

        # Operator sums, differences, and scalar multiples, applied to a fixed state.
        rh = n == 1 ? [1, 1] : [1, 2, 2, 1]
        A = TTOperator([randn(rng, T, dims[k], dims[k], rh[k], rh[k + 1]) for k in 1:n], dims, rh)
        B = TTOperator([randn(rng, T, dims[k], dims[k], rh[k], rh[k + 1]) for k in 1:n], dims, rh)
        dA = [randn(rng, T, size(core)) for core in A.cores]
        dB = [randn(rng, T, size(core)) for core in B.cores]
        densemat(cores, tmpl) = reshape(tto_to_tensor(build_tto(cores, tmpl)), prod(dims), :)
        xdense = dense(x.cores, x)
        for (op, dense_op) in ((+, +), (-, -))
            f(ac, bc) = dot(c, op(build_tto(ac, A), build_tto(bc, B)) * x)
            fd(ac, bc) = ladot(cdense, dense_op(densemat(ac, A), densemat(bc, B)) * xdense)
            value, pullback = Zygote.pullback(f, A.cores, B.cores)
            @test value ≈ fd(A.cores, B.cores)
            for seed in seeds
                gA, gB = pullback(seed)
                @test check_seeded_fd(fd, (A.cores, B.cores), 1, dA, gA, seed)
                @test check_seeded_fd(fd, (A.cores, B.cores), 2, dB, gB, seed)
            end
        end
        for a in (T(0.7), 0.3 - 0.8im, zero(T))
            f(ac, s) = dot(c, (s * build_tto(ac, A)) * x)
            fd(ac, s) = ladot(cdense, s .* (densemat(ac, A) * xdense))
            value, pullback = Zygote.pullback(f, A.cores, a)
            @test value ≈ fd(A.cores, a)
            for seed in (value isa Complex ? complex_seeds : seeds)
                gA, ga = pullback(seed)
                @test check_seeded_fd(fd, (A.cores, a), 1, dA, gA, seed)
                for da in (a isa Real ? (1.0,) : (1.0, 1.0im))
                    @test check_seeded_fd(fd, (A.cores, a), 2, da, ga, seed)
                end
            end
        end

        # Zero output cotangents give zero input cotangents.
        for (fun, args) in ((+, (x, y)), (+, (A, B)), (*, (T(2), x)), (*, (T(2), A)))
            Y, pullback_zero = rrule(fun, args...)
            field = :cores
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
    x = TTVector([randn(rng, dims[k], rks[k], rks[k + 1]) for k in 1:3], dims, rks)
    dx = [randn(rng, size(c)) for c in x.cores]
    ad_dd(g) = sum(real(ladot(g[k], dx[k])) for k in eachindex(dx))

    # A rule (dot) and a direct read of a core of the same TTVector.
    mixed(cs) = (u = build_tt(cs, x); real(dot(u, u)) + sum(u.cores[1]))
    g = only(Zygote.gradient(mixed, x.cores))
    @test all(!isnothing, g)
    @test ad_dd(g) ≈ fd_directional(mixed, x.cores, dx) rtol = 1.0e-5 atol = 1.0e-7

    # An integer field of the same TTVector used in differentiated arithmetic.
    scaled(cs) = (u = build_tt(cs, x); real(dot(u, u)) * 2.0^nsites(u))
    g = only(Zygote.gradient(scaled, x.cores))
    @test g !== nothing
    @test ad_dd(g) ≈ fd_directional(scaled, x.cores, dx) rtol = 1.0e-5 atol = 1.0e-7

    # The same through a QTTVector wrapper.
    qdims, qrks = (2, 2, 2, 2), [1, 2, 3, 2, 1]
    y = TTVector([randn(rng, qdims[k], qrks[k], qrks[k + 1]) for k in 1:4], qdims, qrks)
    dy = [randn(rng, size(c)) for c in y.cores]
    qloss(cs) = (q = QTTVector(build_tt(cs, y), 2, 2, :interleaved); real(dot(q, q)) + sum(q.cores[2]) * q.n_dims)
    g = only(Zygote.gradient(qloss, y.cores))
    @test sum(real(ladot(g[k], dy[k])) for k in eachindex(dy)) ≈ fd_directional(qloss, y.cores, dy) rtol = 1.0e-5 atol = 1.0e-7
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
    cores1 = ψ1.cores
    E1(cores) = (
        ψ = build_tt(cores, ψ1);
        real(dot(ψ, H1 * ψ)) / real(dot(ψ, ψ))
    )
    ḡ1 = Zygote.gradient(E1, cores1)[1]
    nrm2 = real(dot(ψ1, ψ1))
    E = real(dot(ψ1, H1 * ψ1)) / nrm2
    g_analytic = (2 / nrm2) .* ((H1 * ψ1).cores[1] .- E .* ψ1.cores[1])
    @test isapprox(vec(ḡ1[1]), vec(g_analytic); rtol = 1.0e-8, atol = 1.0e-10)

    # N = 4: composed gradient matches directional FD (dot + * + constructor).
    n = 4
    H = ising_tto(n; J = -1.0, h = -0.5, interaction = :z, field = :x)
    ψ = rand_tt(Float64, ntuple(_ -> 2, n), [1, 2, 2, 2, 1])
    cores = ψ.cores
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

    shapes = size.(tmpl.cores)
    offsets = cumsum([0; prod.(shapes)])
    unflatten(θ) = [reshape(θ[(offsets[k] + 1):offsets[k + 1]], shapes[k]) for k in 1:n]
    loss(θ) = (
        ψ = build_tt(unflatten(θ), tmpl);
        real(dot(ψ, H * ψ)) / real(dot(ψ, ψ))
    )

    # DMRG reference energy (essentially exact for this low-rank ground state).
    energies, ψ_dmrg, _ = dmrg_eigsolve(H, qtt_basis_vector(n, 1); max_sweeps = [1, 2], max_bond = [16, 16], trunc_tol = 1.0e-10)

    E_dmrg = energies[end]

    # Gradient descent with backtracking (guarantees monotone descent) using the
    # AD gradient.
    θ = flatten_cores(tmpl.cores)
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
    shapes_d = size.(ψ_dmrg.cores)
    offsets_d = cumsum([0; prod.(shapes_d)])
    unflatten_d(θ) = [reshape(θ[(offsets_d[k] + 1):offsets_d[k + 1]], shapes_d[k]) for k in 1:n]
    loss_d(θ) = (
        ψ = build_tt(unflatten_d(θ), ψ_dmrg);
        real(dot(ψ, H * ψ)) / real(dot(ψ, ψ))
    )
    g_dmrg = Zygote.gradient(loss_d, flatten_cores(ψ_dmrg.cores))[1]
    @test lanorm(g_dmrg) < 1.0e-4
end
