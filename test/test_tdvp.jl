using Test
using Random
using LinearAlgebra
using TensorOperations
using TensorTrainNumerics
import TensorTrainNumerics: _sync_ranks_from_lsr!, _real_or_complex_t, _svdtrunc, _to_lsr, _to_slr, _mpo_to_asbs, _dot3, _applyH1_lsr, _applyH0, _update_left_env, _update_right_env, tdvp1sweep!, tdvp2sweep!, _applyH2_lsr
Random.seed!(42)

@testset "tdvp evolves nonstationary product states for the requested time" begin
    # An on-site Hamiltonian preserves rank one, so a dense exponential is an
    # exact reference for TDVP, with no error from the fixed-rank approximation.
    for nsites in 1:3, active in unique([1, nsites])
        local_ops = [k == active ? [0.0 0.0; 0.0 1.0] : Matrix{Float64}(I, 2, 2) for k in 1:nsites]
        H = TToperator(
            nsites, [reshape(A, 2, 2, 1, 1) for A in local_ops],
            ntuple(_ -> 2, nsites), ones(Int, nsites + 1), zeros(Int, nsites)
        )
        # Site 1 is the fastest physical index in ttv_to_tensor.
        H_dense = reduce(kron, reverse(local_ops))
        initial = fill(1 / sqrt(2^nsites), ntuple(_ -> 2, nsites))
        u0 = ttv_decomp(initial)
        total_time = 0.1
        for imaginary_time in (false, true), nsteps in (1, 4)
            generator = imaginary_time ? H_dense : -im * H_dense
            expected = exp(total_time * generator) * vec(initial)
            u = tdvp(
                H, u0, fill(total_time / nsteps, nsteps);
                imaginary_time, normalize = false, show_progress = false
            )
            @test vec(ttv_to_tensor(u)) ≈ expected atol = 1.0e-11 rtol = 1.0e-11
            @test ttv_to_tensor(u0) ≈ initial
        end
    end
end

@testset "tdvp preserves complex full-rank dense evolution" begin
    H_dense = ComplexF64[
        1 im 0.2 0;
        -im 2 0.1 0.3;
        0.2 0.1 -1 -0.5im;
        0 0.3 0.5im 0.5
    ]
    H = tto_decomp(reshape(H_dense, 2, 2, 2, 2))
    initial = normalize(ComplexF64[1, 2 + im, -im, -1])
    u0 = ttv_decomp(reshape(initial, 2, 2))
    expected = exp(-0.08im * H_dense) * initial
    for steps in ([0.08], fill(0.02, 4))
        u = tdvp(H, u0, steps; normalize = false, show_progress = false)
        @test vec(ttv_to_tensor(u)) ≈ expected atol = 1.0e-10 rtol = 1.0e-10
        @test norm(u) ≈ 1.0 atol = 1.0e-11
    end
end

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
    U4, S4, Vt4 = _svdtrunc(A2; max_bond = 1, truncerr = 0.0)
    @test size(S4, 1) == 1

    A3 = Matrix(Diagonal([1.0, 0.08, 0.08]))
    _, S5, _ = _svdtrunc(A3; truncerr = 0.1)
    @test size(S5, 1) == 1
    @test :_svdtrunc ∉ names(TensorTrainNumerics)
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

function dense_relerr(x::TensorTrainNumerics.TTvector, y::TensorTrainNumerics.TTvector)
    x_dense = vec(ttv_to_tensor(x))
    y_dense = vec(ttv_to_tensor(y))
    y_norm = norm(y_dense)
    return norm(x_dense - y_dense) / max(y_norm, eps(typeof(y_norm)))
end

@testset "tdvp1sweep! (H = 0 ⇒ identity)" begin
    d = 4
    u0 = qtt_sin(d, λ = π)

    ψ = complex(orthogonalize(u0))
    ψ_ref = deepcopy(ψ)
    Hc = complex(id_tto(d))                 # make MPO complex to match ψ
    H0 = (0.0 + 0.0im) * Hc                 # zero MPO with Complex element type

    ψ2, F = tdvp1sweep!(complex(0.1), ψ, H0, nothing; verbose = false)

    @test dense_relerr(ψ2, ψ_ref) < 1.0e-12
    @test length(F) == ψ.N + 2
end

@testset "tdvp: basic behavior" begin
    d = 4
    u0 = qtt_sin(d, λ = π)

    H0r = 0.0 * id_tto(d)
    H0c = (0.0 + 0.0im) * complex(id_tto(d))

    ψ_rt = tdvp(
        H0c, complex(u0), [0.1];
        normalize = false, sweeps = 1, carry_env = false, verbose = false, imaginary_time = false,
        show_progress = false
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
    rel = dense_relerr(ψ_id, ψ0)
    @test rel ≤ 1.0e-10

    ψ_carryT = tdvp(
        H0c, complex(u0), [0.1, 0.1];
        normalize = false, sweeps = 2, carry_env = true, verbose = false, imaginary_time = false
    )
    ψ_carryF = tdvp(
        H0c, complex(u0), [0.1, 0.1];
        normalize = false, sweeps = 2, carry_env = false, verbose = false, imaginary_time = false
    )
    rel_c = dense_relerr(ψ_carryT, ψ_carryF)
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

@testset "tdvp2sweep! uses absolute singular-value truncation" begin
    spectrum = [1.0, 0.08, 0.08, 0.08]
    ψ0 = ttv_decomp(Matrix(Diagonal(spectrum)))
    H0 = zeros_tto(Float64, (4, 4), [1, 1, 1])

    ψ2, _ = tdvp2sweep!(0.1im, ψ0, H0, nothing; verbose = false, truncerr = 0.1)

    @test ψ2.ttv_rks == [1, 1, 1]
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
    rel = dense_relerr(ψ_id, ψ0)
    @test rel ≤ 1.0e-7

    ψ_carryT = tdvp2(
        H0c, complex(u0), [0.1, 0.1];
        normalize = false, sweeps = 2, carry_env = true, verbose = false, imaginary_time = false
    )
    ψ_carryF = tdvp2(
        H0c, complex(u0), [0.1, 0.1];
        normalize = false, sweeps = 2, carry_env = false, verbose = false, imaginary_time = false
    )
    rel_c = dense_relerr(ψ_carryT, ψ_carryF)
    @test rel_c ≤ 1.0e-10
end

@testset "real-time QTT TDVP accepts real inputs" begin
    d = 2
    u0 = QTTvector(qtt_sin(d), 1, d, :serial)
    H0 = QTToperator(0.0 * id_tto(d), 1, d, :serial)

    ψ1 = tdvp(H0, u0, [0.01]; normalize = false, verbose = false, show_progress = false)
    ψ2 = tdvp2(H0, u0, [0.01]; normalize = false, verbose = false, show_progress = false)

    @test ψ1 isa QTTvector{ComplexF64}
    @test ψ2 isa QTTvector{ComplexF64}
end

@testset "tdvp2: imaginary-time branch runs" begin
    d = 4
    ψ0 = complex(orthogonalize(qtt_sin(d, λ = π)))
    H0 = (0.0 + 0.0im) * complex(id_tto(d))
    steps = [0.02, 0.02]

    ψ_it = tdvp2(H0, ψ0, steps; normalize = false, sweeps = 2, carry_env = true, verbose = false, imaginary_time = true)
    ψ_it_quiet = tdvp2(H0, ψ0, steps; normalize = false, sweeps = 2, carry_env = true, verbose = false, imaginary_time = true, show_progress = false)
    @test dense_relerr(ψ_it_quiet, ψ0) < 1.0e-12
    @test dense_relerr(ψ_it, ψ0) < 1.0e-12
end

@testset "tdvp heat eigenmode with QTT" begin
    d = 4
    N = 2^d
    h = 1.0 / (N + 1)
    κ = 0.1

    Δ1d = toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    A_raw = (κ / h^2) * (Δ1d ⊗ id_tto(d) + id_tto(d) ⊗ Δ1d)
    A = QTToperator(A_raw, 2, d, :serial)

    u0_raw = qtt_sin(d; a = h, b = 1 - h) ⊗ qtt_sin(d; a = h, b = 1 - h)
    u0 = QTTvector(u0_raw, 2, d, :serial)
    λ = real(TensorTrainNumerics.dot(u0_raw, A_raw * u0_raw) / TensorTrainNumerics.dot(u0_raw, u0_raw))

    steps = fill(1.0e-3, 5)
    target = exp(λ * sum(steps)) .* qttv_to_array(u0)

    sol_tdvp = tdvp(A, u0, steps; imaginary_time = true, normalize = false, verbose = false)
    err_tdvp = norm(vec(qttv_to_array(sol_tdvp) .- target)) / norm(vec(target))
    @test err_tdvp < 1.0e-8

    sol_tdvp2 = tdvp2(
        A, u0, steps; imaginary_time = true, normalize = false,
        verbose = false, max_bond = 8, truncerr = 1.0e-12
    )
    err_tdvp2 = norm(vec(qttv_to_array(sol_tdvp2) .- target)) / norm(vec(target))
    @test err_tdvp2 < 1.0e-8
end

@testset "tdvp/tdvp2: return_error residual for λ≠0 (both time directions)" begin
    # A = 0.5·I ⇒ every state is an eigenvector (λ=0.5) and evolves exactly, so the
    # reported residual must be ≈0. Guards against (i) ψ_prev aliasing the in-place
    # sweep output and (ii) the imaginary-time residual sign.
    d = 4
    A = 0.5 * id_tto(d)
    u0 = qtt_sin(d, λ = π)
    steps = fill(1.0e-3, 5)
    for it in (false, true)
        _, e1 = tdvp(A, u0, steps; imaginary_time = it, return_error = true, normalize = false, verbose = false)
        @test e1 < 1.0e-3
        _, e2 = tdvp2(
            A, u0, steps; imaginary_time = it, return_error = true, normalize = false,
            verbose = false, max_bond = 8, truncerr = 1.0e-12
        )
        @test e2 < 1.0e-3
    end
end
