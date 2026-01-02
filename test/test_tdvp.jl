using Test
using Random
using LinearAlgebra
using TensorOperations
import TensorTrainNumerics: _sync_ranks_from_lsr!, _real_or_complex_t, _svdtrunc, _to_lsr, _to_slr, _mpo_to_asbs, _dot3, _applyH1_lsr, _applyH0, _update_left_env, _update_right_env, tdvp1sweep!, tdvp2sweep!, _applyH2_lsr
Random.seed!(42)

@testset "_sync_ranks_from_lsr!" begin
    N = 3
    dims = (2, 3, 2)
    rks = [1, 2, 2, 1]
    ψ = rand_tt(dims, rks)
    lefts = (1, 3, 5)
    rights = (4, 6, 7)
    A_lsr = [randn(lefts[k], dims[k], rights[k]) for k in 1:N]
    _sync_ranks_from_lsr!(ψ, A_lsr)
    @test ψ.ttv_rks == [lefts..., rights[end]]
    @test all(==(0), ψ.ttv_ot)
end

@testset "_real_or_complex_t" begin
    @test _real_or_complex_t(3.25) === 3.25
    @test _real_or_complex_t(2.0 + 0im) === 2.0
    z = 1.0 + 1.0im
    @test _real_or_complex_t(z) === z
end

@testset "_svdtrunc" begin
    A = randn(6, 4)
    U, S, Vt = _svdtrunc(A; max_bond = 100, truncerr = 0.0)
    @test size(U, 1) == 6 && size(Vt, 2) == 4
    @test size(S, 1) == size(S, 2) == size(U, 2) == size(Vt, 1)

    U2, S2, Vt2 = _svdtrunc(A; max_bond = 2, truncerr = 0.0)
    @test size(S2, 1) == 2
    F = svd(A)
    @test isapprox(diag(S2), F.S[1:2]; rtol = 1.0e-12, atol = 1.0e-12)

    A2 = randn(5, 5)
    F2 = svd(A2)
    thr = (F2.S[2] + F2.S[3]) / 2
    U3, S3, Vt3 = _svdtrunc(A2; max_bond = 5, truncerr = thr)
    @test size(S3, 1) == 1

    U4, S4, Vt4 = _svdtrunc(A2; max_bond = 1, truncerr = 0.0)
    @test size(S4, 1) == 1
end

@testset "_to_lsr/_to_slr" begin
    A = randn(3, 4, 5)
    Ac = randn(3, 4, 5) .+ 1im * randn(3, 4, 5)

    @test size(_to_lsr(A)) == (size(A, 2), size(A, 1), size(A, 3))
    @test _to_lsr(_to_slr(A)) == A
    @test _to_slr(_to_lsr(A)) == A
    @test _to_lsr(_to_lsr(A)) == A
    @test _to_slr(_to_slr(A)) == A

    @test _to_lsr(_to_slr(Ac)) == Ac
    @test _to_slr(_to_lsr(Ac)) == Ac
end


@testset "_mpo_to_asbs" begin
    M = randn(2, 3, 4, 5) # (s_out, s_in, a, b)
    M2 = _mpo_to_asbs(M) # (a, s_out, b, s_in)
    M_back = permutedims(M2, (2, 4, 1, 3))
    @test M_back == M
end

@testset "_dot3" begin
    X = randn(2, 3, 4) .+ 1im * randn(2, 3, 4)
    Y = randn(2, 3, 4) .+ 1im * randn(2, 3, 4)
    v1 = _dot3(X, Y)
    v2 = sum(conj.(vec(X)) .* vec(Y))
    @test isapprox(v1, v2; rtol = 1.0e-12, atol = 1.0e-12)
end

@testset "_applyH1_lsr" begin
    Dl, d_in, d_out, Dr = 2, 3, 3, 2
    a, b = 2, 2
    AC = randn(Dl, d_in, Dr) .+ 1im * randn(Dl, d_in, Dr)
    FL = randn(Dl, a, Dl) .+ 1im * randn(Dl, a, Dl)
    FR = randn(Dr, b, Dr) .+ 1im * randn(Dr, b, Dr)
    M = randn(a, d_out, b, d_in) .+ 1im * randn(a, d_out, b, d_in)

    HAC1 = _applyH1_lsr(AC, FL, FR, M)

    HAC2 = zeros(ComplexF64, Dl, d_out, Dr)
    @inbounds for α in 1:Dl, s in 1:d_out, β in 1:Dr
        z = 0.0 + 0.0im
        for ap in 1:Dl, s′ in 1:d_in, β′ in 1:Dr, aidx in 1:a, bidx in 1:b
            z += FL[α, aidx, ap] * AC[ap, s′, β′] * M[aidx, s, bidx, s′] * FR[β′, bidx, β]
        end
        HAC2[α, s, β] = z
    end
    @test isapprox(HAC1, HAC2; rtol = 1.0e-12, atol = 1.0e-12)
end

@testset "_applyH0" begin
    Dl, Dr, a = 3, 2, 4
    C = randn(Dl, Dr) .+ 1im * randn(Dl, Dr)
    FL = randn(Dl, a, Dl) .+ 1im * randn(Dl, a, Dl)
    FR = randn(Dr, a, Dr) .+ 1im * randn(Dr, a, Dr)

    HC1 = _applyH0(C, FL, FR)

    HC2 = zeros(ComplexF64, Dl, Dr)
    @inbounds for α in 1:Dl, β in 1:Dr
        z = 0.0 + 0.0im
        for ap in 1:Dl, aidx in 1:a, βp in 1:Dr
            z += FL[α, aidx, ap] * C[ap, βp] * FR[βp, aidx, β]
        end
        HC2[α, β] = z
    end
    @test isapprox(HC1, HC2; rtol = 1.0e-12, atol = 1.0e-12)
end

@testset "_update_left_env / _update_right_env shapes" begin
    Dl, d, Dr = 2, 3, 4
    a_in, a_out = 2, 5
    A = randn(Dl, d, Dr) .+ 1im * randn(Dl, d, Dr)
    FL = randn(Dl, a_in, Dl) .+ 1im * randn(Dl, a_in, Dl)
    FR = randn(Dr, a_in, Dr) .+ 1im * randn(Dr, a_in, Dr)
    M_L = randn(a_in, d, a_out, d) .+ 1im * randn(a_in, d, a_out, d)
    M_R = randn(a_out, d, a_in, d) .+ 1im * randn(a_out, d, a_in, d)

    FLnext = _update_left_env(A, M_L, FL)
    FRprev = _update_right_env(A, M_R, FR)

    @test size(FLnext) == (Dr, a_out, Dr)
    @test size(FRprev) == (Dl, a_out, Dl)
end

@testset "tdvp1sweep! with zero MPO preserves state" begin
    d = 4
    H0 = 0.0 * id_tto(d)
    u0 = qtt_sin(d, λ = π)
    ψ0 = orthogonalize(u0)
    ψ = deepcopy(ψ0)

    ψ_out, F = TensorTrainNumerics.tdvp1sweep!(0.05, ψ, H0, nothing; verbose = false)

    @test ψ_out.ttv_dims == ψ0.ttv_dims
    @test length(F) == d + 2
    @test isfinite(norm(ψ_out))
    @test norm(ψ_out - ψ0) / norm(ψ0) < 1.0e-6
end
absnorm(x::TensorTrainNumerics.TTvector) = sqrt(real(TensorTrainNumerics.dot(x, x)))

@testset "tdvp1sweep! (H = 0 ⇒ identity)" begin
    d = 4
    u0 = qtt_sin(d, λ = π)

    ψ = complex(orthogonalize(u0))
    Hc = complex(id_tto(d))                 # make MPO complex to match ψ
    H0 = (0.0 + 0.0im) * Hc                 # zero MPO with Complex element type

    ψ2, F = tdvp1sweep!(complex(0.1), ψ, H0, nothing; verbose = false)

    @test abs(absnorm(ψ2 - ψ)) / max(absnorm(ψ), eps()) < 1.0e-12
    @test length(F) == ψ.N + 2
end

absnorm(x::TensorTrainNumerics.TTvector) = sqrt(max(real(TensorTrainNumerics.dot(x, x)), 0.0))

@testset "tdvp: basic behavior" begin
    d = 4
    u0 = qtt_sin(d, λ = π)

    H0r = 0.0 * id_tto(d)
    H0c = (0.0 + 0.0im) * complex(id_tto(d))

    ψ_rt = tdvp(
        H0c, complex(u0), [0.1];
        normalize = false, sweeps = 1, carry_env = false, verbose = false, imaginary_time = false
    )
    @test eltype(ψ_rt) <: Complex

    ψ_it = tdvp(
        H0r, u0, [0.1];
        normalize = false, sweeps = 1, carry_env = false, verbose = false, imaginary_time = true
    )
    @test eltype(ψ_it) <: Real

    ψ_err, err = tdvp(
        H0c, complex(u0), [0.1];
        normalize = false, return_error = true,
        sweeps = 1, carry_env = false, verbose = false, imaginary_time = false
    )
    @test isa(err, Number)
    @test abs(real(err)) ≤ 1.0e-6

    ψ0 = complex(orthogonalize(u0))
    ψ_id = tdvp(
        H0c, ψ0, [0.1];
        normalize = false, sweeps = 1, carry_env = false, verbose = false, imaginary_time = false
    )
    rel = absnorm(ψ_id - ψ0) / max(absnorm(ψ0), eps())
    @test rel ≤ 1.0e-10

    ψ_carryT = tdvp(
        H0c, complex(u0), [0.1, 0.1];
        normalize = false, sweeps = 2, carry_env = true, verbose = false, imaginary_time = false
    )
    ψ_carryF = tdvp(
        H0c, complex(u0), [0.1, 0.1];
        normalize = false, sweeps = 2, carry_env = false, verbose = false, imaginary_time = false
    )
    rel_c = absnorm(ψ_carryT - ψ_carryF) / max(absnorm(ψ_carryF), eps())
    @test rel_c ≤ 1.0e-10
end

@testset "_applyH2_lsr" begin
    Dl, d1, d2, Dr = 2, 3, 4, 5
    AAC = randn(ComplexF64, Dl, d1, d2, Dr)

    FL = zeros(ComplexF64, Dl, 1, Dl);  for α in 1:Dl
        FL[α, 1, α] = 1
    end
    FR = zeros(ComplexF64, Dr, 1, Dr);  for β in 1:Dr
        FR[β, 1, β] = 1
    end
    M1 = zeros(ComplexF64, 1, d1, 1, d1); for s in 1:d1
        M1[1, s, 1, s] = 1
    end
    M2 = zeros(ComplexF64, 1, d2, 1, d2); for s in 1:d2
        M2[1, s, 1, s] = 1
    end

    HAAC = _applyH2_lsr(AAC, FL, FR, M1, M2)
    @test isapprox(HAAC, AAC; atol = 1.0e-12, rtol = 1.0e-12)

    X = randn(ComplexF64, Dl, d1, d2, Dr)
    Y = randn(ComplexF64, Dl, d1, d2, Dr)
    lhs = LinearAlgebra.dot(vec(conj(X)), vec(_applyH2_lsr(Y, FL, FR, M1, M2)))
    rhs = LinearAlgebra.dot(vec(conj(_applyH2_lsr(X, FL, FR, M1, M2))), vec(Y))
    @test isapprox(lhs, rhs; atol = 1.0e-12, rtol = 1.0e-12)
end

@testset "tdvp2sweep! (H = 0 ⇒ identity, no truncation)" begin
    d = 4
    u0 = qtt_sin(d, λ = π)
    ψ0 = complex(orthogonalize(u0))
    H0 = (0.0 + 0.0im) * complex(id_tto(d))

    ψ1, F1 = tdvp2sweep!(0.1im, deepcopy(ψ0), H0, nothing; verbose = false)
    @test length(F1) == ψ0.N + 2
    @test size(F1[1]) == (1, 1, 1)
    @test size(F1[end]) == (1, 1, 1)
    @test isapprox(ttv_to_tensor(ψ1), ttv_to_tensor(ψ0); atol = 1.0e-10, rtol = 1.0e-10)
end

@testset "tdvp2sweep! real-time & imaginary-time dt (H=0)" begin
    d = 4
    ψ0 = complex(orthogonalize(qtt_sin(d, λ = π)))
    H0 = (0.0 + 0.0im) * complex(id_tto(d))

    ψa, _ = tdvp2sweep!(0.05, deepcopy(ψ0), H0, nothing; verbose = false)
    ψb, _ = tdvp2sweep!(0.05im, deepcopy(ψ0), H0, nothing; verbose = false)

    @test isapprox(ttv_to_tensor(ψa), ttv_to_tensor(ψ0); atol = 1.0e-10, rtol = 1.0e-10)
    @test isapprox(ttv_to_tensor(ψb), ttv_to_tensor(ψ0); atol = 1.0e-10, rtol = 1.0e-10)
end

@testset "tdvp2sweep! respects max_bond" begin
    d = 6
    ψ0 = complex(orthogonalize(qtt_sin(d, λ = π) + qtt_sin(d, λ = 2π)))
    H0 = (0.0 + 0.0im) * complex(id_tto(d))
    mb = 2
    ψ2, _ = tdvp2sweep!(0.1im, deepcopy(ψ0), H0, nothing; verbose = false, max_bond = mb, truncerr = 0.0)
    @test maximum(ψ2.ttv_rks) ≤ mb
end

@testset "tdvp2: basic behavior" begin
    d = 6
    u0 = qtt_sin(d, λ = π)

    H0r = 0.0 * id_tto(d)
    H0c = (0.0 + 0.0im) * complex(id_tto(d))

    ψ_rt = tdvp2(
        H0c, complex(u0), [0.1];
        normalize = false, sweeps = 1, carry_env = false, verbose = false, imaginary_time = false
    )
    @test eltype(ψ_rt) <: Complex

    ψ_it = tdvp2(
        H0r, u0, [0.1];
        normalize = false, sweeps = 1, carry_env = false, verbose = false, imaginary_time = true
    )
    @test eltype(ψ_it) <: Real

    ψ_err, err = tdvp2(
        H0c, complex(u0), [0.1];
        normalize = false, return_error = true,
        sweeps = 1, carry_env = false, verbose = false, imaginary_time = false
    )
    @test isa(err, Number)
    @test abs(real(err)) ≤ 1.0e-6

    ψ0 = complex(orthogonalize(u0))
    ψ_id = tdvp2(
        H0c, ψ0, [0.1];
        normalize = false, sweeps = 1, carry_env = false, verbose = false, imaginary_time = false
    )
    rel = absnorm(ψ_id - ψ0) / max(absnorm(ψ0), eps())
    @test rel ≤ 1.0e-7

    ψ_carryT = tdvp2(
        H0c, complex(u0), [0.1, 0.1];
        normalize = false, sweeps = 2, carry_env = true, verbose = false, imaginary_time = false
    )
    ψ_carryF = tdvp2(
        H0c, complex(u0), [0.1, 0.1];
        normalize = false, sweeps = 2, carry_env = false, verbose = false, imaginary_time = false
    )
    rel_c = absnorm(ψ_carryT - ψ_carryF) / max(absnorm(ψ_carryF), eps())
    @test rel_c ≤ 1.0e-10
end

@testset "tdvp2: imaginary-time branch runs" begin
    d = 4
    ψ0 = complex(orthogonalize(qtt_sin(d, λ = π)))
    H0 = (0.0 + 0.0im) * complex(id_tto(d))
    steps = [0.02, 0.02]

    ψ_it = tdvp2(H0, ψ0, steps; normalize = false, sweeps = 2, carry_env = true, verbose = false, imaginary_time = true)
    @test absnorm(ψ_it - ψ0) / max(absnorm(ψ0), eps()) < 1.0e-12
end
