using Test
using TensorTrainNumerics
using LinearAlgebra

@testset "QTT time direction operators" begin
    S_one = qtt_lower_shift(1)
    @test qtto_to_matrix(S_one) == [0.0 0.0; 1.0 0.0]

    d = 3
    S = qtt_lower_shift(d)
    Iₜ = qtt_time_identity(d)

    S_dense = qtto_to_matrix(S)
    I_dense = qtto_to_matrix(Iₜ)
    expected = zeros(2^d, 2^d)
    for k in 2:(2^d)
        expected[k, k - 1] = 1.0
    end

    @test S_dense == expected
    @test I_dense == Matrix{Float64}(I, 2^d, 2^d)
end

@testset "SpaceTimeQTTvector slicing" begin
    time_bits = 2
    d = 3
    u1 = qtt_sin(d)
    u2 = 2.0 * qtt_sin(d)
    u3 = 3.0 * qtt_sin(d)
    u4 = 4.0 * qtt_sin(d)

    tensor = zeros(ntuple(_ -> 2, time_bits + d))
    slices = [qtt_to_vector(u1), qtt_to_vector(u2), qtt_to_vector(u3), qtt_to_vector(u4)]
    for k in 1:4
        bits = reverse(digits(k - 1, base = 2, pad = time_bits)) .+ 1
        for x in CartesianIndices(ntuple(_ -> 2, d))
            spatial_bits = Tuple(x)
            spatial_index = TensorTrainNumerics.tuple_to_index(spatial_bits)
            tensor[CartesianIndex((bits..., spatial_bits...))] = slices[k][spatial_index]
        end
    end

    st = SpaceTimeQTTvector(ttv_decomp(tensor), time_bits, 1, d, :serial)

    @test st.time_bits == time_bits
    @test st.space_n_dims == 1
    @test st.space_bits_per_dim == d
    @test st.space_ordering == :serial

    for k in 1:4
        uk = space_time_slice(st, k)
        @test uk isa QTTvector
        @test uk.n_dims == 1
        @test uk.bits_per_dim == d
        @test uk.ordering == :serial
        @test norm(qtt_to_vector(uk) - slices[k]) / norm(slices[k]) < 1.0e-10
    end

    @test_throws BoundsError space_time_slice(st, 0)
    @test_throws BoundsError space_time_slice(st, 5)
end

@testset "SpaceTimeQTTvector copy" begin
    time_bits = 2
    d = 3
    st = SpaceTimeQTTvector(
        ttv_decomp(ones(ntuple(_ -> 2, time_bits + d))),
        time_bits,
        1,
        d,
        :serial
    )
    st.ttv_vec[1][1, 1, 1] = 1.0
    st.ttv_vec[end][2, 1, 1] = 2.0

    st_copy = copy(st)

    @test st_copy isa SpaceTimeQTTvector
    @test st_copy.time_bits == st.time_bits
    @test st_copy.space_n_dims == st.space_n_dims
    @test st_copy.space_bits_per_dim == st.space_bits_per_dim
    @test st_copy.space_ordering == st.space_ordering
    @test st_copy.ttv_vec[1] !== st.ttv_vec[1]
    @test st_copy.ttv_vec[end] !== st.ttv_vec[end]

    st_copy.ttv_vec[1][1, 1, 1] = 7.0
    @test st.ttv_vec[1][1, 1, 1] != st_copy.ttv_vec[1][1, 1, 1]
end

@testset "global_crank_nicholson_method matches sequential CN" begin
    d = 3
    A_raw = -0.1 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    A = QTToperator(A_raw, 1, d, :serial)

    u0_raw = qtt_sin(d)
    u0 = QTTvector(u0_raw, 1, d, :serial)
    guess = u0
    steps = fill(0.02, 4)

    U, residual = global_crank_nicholson_method(
        A, u0, guess, steps;
        normalize = false,
        tt_solver = "krylov",
        tol = 1.0e-12,
        return_error = true
    )

    @test U isa SpaceTimeQTTvector
    @test U.time_bits == 2
    @test U.space_n_dims == 1
    @test U.space_bits_per_dim == d
    @test U.space_ordering == :serial
    @test isfinite(residual)
    @test residual < 1.0e-8

    sequential = u0
    for k in eachindex(steps)
        sequential = crank_nicholson_method(
            A, sequential, sequential, [steps[k]];
            normalize = false,
            tt_solver = "krylov",
            tol = 1.0e-12
        )
        global_slice = space_time_slice(U, k)
        rel_error = norm(qtt_to_vector(global_slice) - qtt_to_vector(sequential)) / norm(qtt_to_vector(sequential))
        @test rel_error < 1.0e-8
    end

    U2, residual2 = global_crank_nicholson_method(
        A, u0, U, steps;
        normalize = false,
        tt_solver = "krylov",
        tol = 1.0e-12,
        return_error = true
    )

    @test U2 isa SpaceTimeQTTvector
    @test isfinite(residual2)
    @test residual2 < 1.0e-8
    @test norm(qtt_to_vector(space_time_slice(U2, 1)) - qtt_to_vector(space_time_slice(U, 1))) / norm(qtt_to_vector(space_time_slice(U, 1))) < 1.0e-8
end

@testset "global_crank_nicholson_method handles minimum time QTT size" begin
    d = 3
    A = QTToperator(-0.1 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d), 1, d, :serial)
    u0 = QTTvector(qtt_sin(d), 1, d, :serial)
    steps = fill(0.02, 2)

    U = global_crank_nicholson_method(
        A, u0, u0, steps;
        normalize = false,
        tt_solver = "krylov",
        tol = 1.0e-12
    )

    sequential = u0
    for k in eachindex(steps)
        sequential = crank_nicholson_method(
            A, sequential, sequential, [steps[k]];
            normalize = false,
            tt_solver = "krylov",
            tol = 1.0e-12
        )
        global_slice = space_time_slice(U, k)
        rel_error = norm(qtt_to_vector(global_slice) - qtt_to_vector(sequential)) / norm(qtt_to_vector(sequential))
        @test rel_error < 1.0e-8
    end
end

@testset "global_crank_nicholson_method validation" begin
    d = 3
    A = QTToperator(-0.1 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d), 1, d, :serial)
    u0 = QTTvector(qtt_sin(d), 1, d, :serial)

    @test_throws ArgumentError global_crank_nicholson_method(A, u0, u0, Float64[]; normalize = false)
    @test_throws ArgumentError global_crank_nicholson_method(A, u0, u0, [0.1]; normalize = false)
    @test_throws ArgumentError global_crank_nicholson_method(A, u0, u0, [0.1, 0.2]; normalize = false)
    @test_throws ArgumentError global_crank_nicholson_method(A, u0, u0, fill(0.1, 3); normalize = false)
    @test_throws ArgumentError global_crank_nicholson_method(A, u0, u0, fill(0.1, 4); normalize = true)
    @test_throws MethodError global_crank_nicholson_method(TToperator(A), TTvector(u0), TTvector(u0), fill(0.1, 4); normalize = false)
end
