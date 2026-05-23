using Test
using TensorTrainNumerics
using LinearAlgebra

@testset "AbstractTTvector type parameter" begin
    ttv = rand_tt(fill(2, 4), 2)
    @test ttv isa AbstractTTvector
    @test eltype(ttv) == Float64
end

@testset "QTTvector struct" begin
    @test QTTvector <: AbstractTTvector
    cores = [rand(2, 1, 2), rand(2, 2, 2), rand(2, 2, 1)]
    rks   = [1, 2, 2, 1]
    dims  = (2, 2, 2)
    q = QTTvector{Float64, 3}(3, cores, dims, rks, zeros(Int, 3), 1, 3, :serial)
    @test q.n_dims == 1
    @test q.bits_per_dim == 3
    @test q.ordering == :serial
    @test q isa AbstractTTvector
end

@testset "QTToperator struct" begin
    @test QTToperator <: AbstractTToperator
    cores = [rand(2, 2, 1, 2), rand(2, 2, 2, 2), rand(2, 2, 2, 1)]
    rks   = [1, 2, 2, 1]
    dims  = (2, 2, 2)
    A = QTToperator{Float64, 3}(3, cores, dims, rks, zeros(Int, 3), 1, 3, :interleaved)
    @test A.n_dims == 1
    @test A.bits_per_dim == 3
    @test A.ordering == :interleaved
    @test A isa AbstractTToperator
end

@testset "QTTvector/QTToperator constructors and strip helpers" begin
    # Build a 2-dim, 3-bits-per-dim QTTvector (6 sites total)
    ttv = rand_tt(fill(2, 6), 2)
    q = QTTvector(ttv, 2, 3, :interleaved)
    @test q isa QTTvector{Float64, 6}
    @test q.n_dims == 2
    @test q.bits_per_dim == 3
    @test q.ordering == :interleaved
    @test q.N == 6

    # Strip back to TTvector
    ttv2 = TTvector(q)
    @test ttv2 isa TTvector{Float64, 6}
    @test ttv2.N == 6

    # Wrap constructor asserts
    @test_throws AssertionError QTTvector(ttv, 2, 4, :interleaved)  # 2*4=8 ≠ 6
    @test_throws AssertionError QTTvector(ttv, 2, 3, :bad)           # bad ordering

    # Operator wrap/strip
    tto = rand_tto(ntuple(_ -> 2, 6), 2)
    A = QTToperator(tto, 2, 3, :serial)
    @test A isa QTToperator{Float64, 6}
    @test A.ordering == :serial
    tto2 = TToperator(A)
    @test tto2 isa TToperator{Float64, 6}
end

@testset "check_compat" begin
    ttv = rand_tt(fill(2, 6), 2)
    q1 = QTTvector(ttv, 2, 3, :interleaved)
    q2 = QTTvector(ttv, 2, 3, :interleaved)
    q3 = QTTvector(ttv, 2, 3, :serial)

    @test check_compat(q1, q2) === nothing
    @test_throws AssertionError check_compat(q1, q3)  # ordering mismatch

    tto = rand_tto(ntuple(_ -> 2, 6), 2)
    A = QTToperator(tto, 2, 3, :interleaved)
    @test check_compat(A, q1) === nothing
    @test_throws AssertionError check_compat(A, q3)

    # Plain TTvector/TToperator no-ops
    @test check_compat(ttv, ttv) === nothing
end

@testset "QTTvector dispatch methods" begin
    ttv = rand_tt(fill(2, 6), 2)
    q1 = QTTvector(ttv, 2, 3, :interleaved)
    q2 = QTTvector(ttv, 2, 3, :interleaved)
    q3 = QTTvector(ttv, 2, 3, :serial)

    # copy preserves metadata
    qc = copy(q1)
    @test qc isa QTTvector
    @test qc.ordering == :interleaved

    # arithmetic returns QTTvector with same metadata
    @test (q1 + q2) isa QTTvector
    @test (q1 + q2).ordering == :interleaved
    @test (q1 - q2) isa QTTvector
    @test (2.0 * q1) isa QTTvector
    @test (q1 * 2.0) isa QTTvector

    # check_compat enforced
    @test_throws AssertionError q1 + q3

    # scalar results
    @test LinearAlgebra.dot(q1, q2) isa Number
    @test norm(q1) isa Real

    # orthogonalize
    qo = orthogonalize(q1)
    @test qo isa QTTvector
    @test qo.ordering == :interleaved
end

@testset "QTToperator dispatch methods" begin
    ttv = rand_tt(fill(2, 6), 2)
    tto = rand_tto(ntuple(_ -> 2, 6), 2)
    q = QTTvector(ttv, 2, 3, :interleaved)
    A = QTToperator(tto, 2, 3, :interleaved)
    B = QTToperator(tto, 2, 3, :interleaved)

    @test copy(A) isa QTToperator
    @test (A + B) isa QTToperator
    @test (2.0 * A) isa QTToperator

    # operator-vector product returns QTTvector
    Aq = A * q
    @test Aq isa QTTvector
    @test Aq.ordering == :interleaved
end

@testset "function_to_qttv and qttv_to_array" begin
    # 1D case: function_to_qttv with n_dims=1 should match function_to_qtt
    f1d = x -> sin(π * x[1])
    d = 4
    q_interleaved = function_to_qttv(f1d, 1, d; ordering = :interleaved)
    q_serial = function_to_qttv(f1d, 1, d; ordering = :serial)
    @test q_interleaved isa QTTvector
    @test q_serial isa QTTvector
    # Both orderings are equivalent for 1D — compare via array reconstruction
    arr_il = qttv_to_array(q_interleaved)
    arr_sr = qttv_to_array(q_serial)
    @test maximum(abs, arr_il .- arr_sr) < 1e-12

    # Round-trip: construct from f, evaluate, compare to direct grid evaluation
    n_pts = 2^d
    h = 1.0 / (n_pts - 1)
    grid = [h * i for i in 0:(n_pts - 1)]
    @test size(arr_il) == (n_pts,)
    @test maximum(abs, arr_il .- sin.(π .* grid)) < 1e-12

    # 2D case: f(x,y) = sin(πx)sin(πy)
    f2d = x -> sin(π * x[1]) * sin(π * x[2])
    bits = 3  # 8 points per dim, 6 sites total
    q2d_il = function_to_qttv(f2d, 2, bits; ordering = :interleaved)
    q2d_sr = function_to_qttv(f2d, 2, bits; ordering = :serial)
    @test q2d_il isa QTTvector
    @test q2d_sr isa QTTvector

    arr2d_il = qttv_to_array(q2d_il)
    arr2d_sr = qttv_to_array(q2d_sr)
    @test size(arr2d_il) == (2^bits, 2^bits)
    @test size(arr2d_sr) == (2^bits, 2^bits)
    # Both orderings should give the same function values
    @test maximum(abs, arr2d_il .- arr2d_sr) < 1e-12

    # Compare to direct grid evaluation
    n2 = 2^bits
    h2 = 1.0 / (n2 - 1)
    grid2 = [h2 * i for i in 0:(n2 - 1)]
    ref2d = [sin(π * x) * sin(π * y) for x in grid2, y in grid2]
    @test maximum(abs, arr2d_il .- ref2d) < 1e-12
end

@testset "reorder" begin
    # Round-trip: serial → interleaved → serial is identity (up to SVD at threshold=0)
    f2d = x -> sin(π * x[1]) * cos(π * x[2])
    bits = 3
    q_serial = function_to_qttv(f2d, 2, bits; ordering = :serial)
    q_il = reorder(q_serial, :interleaved)
    @test q_il isa QTTvector
    @test q_il.ordering == :interleaved

    q_back = reorder(q_il, :serial)
    @test q_back.ordering == :serial

    # Both should give the same function values when evaluated
    arr_serial = qttv_to_array(q_serial)
    arr_il = qttv_to_array(q_il)
    arr_back = qttv_to_array(q_back)
    @test maximum(abs, arr_serial .- arr_il) < 1e-10
    @test maximum(abs, arr_serial .- arr_back) < 1e-10

    # reorder with same ordering returns a copy
    q_same = reorder(q_serial, :serial)
    @test q_same.ordering == :serial
    @test maximum(abs, qttv_to_array(q_same) .- arr_serial) < 1e-14
end

@testset "qtt_laplacian" begin
    d = 4
    n = 2^d

    # 1D case: returns QTToperator with correct metadata
    A1_serial = qtt_laplacian(1, d; ordering = :serial, bc = :DN)
    @test A1_serial isa QTToperator
    @test A1_serial.n_dims == 1
    @test A1_serial.bits_per_dim == d
    @test A1_serial.ordering == :serial
    @test A1_serial.N == d

    A1_il = qtt_laplacian(1, d; ordering = :interleaved, bc = :DN)
    @test A1_il isa QTToperator
    @test A1_il.ordering == :interleaved

    # 2D serial case: correct metadata and dimensions
    A2s = qtt_laplacian(2, d; ordering = :serial, bc = :DD)
    @test A2s isa QTToperator
    @test A2s.n_dims == 2
    @test A2s.bits_per_dim == d
    @test A2s.ordering == :serial
    @test A2s.N == 2 * d

    # 2D interleaved case
    A2i = qtt_laplacian(2, d; ordering = :interleaved, bc = :DD)
    @test A2i isa QTToperator
    @test A2i.ordering == :interleaved
    @test A2i.N == 2 * d

    # Correctness: 2D serial Laplacian matrix matches direct Kronecker-sum reference
    h = 1.0 / (n - 1)
    M_qtt = qtto_to_matrix(TToperator(A2s))
    M1d   = qtto_to_matrix(Δ(d)) ./ h^2
    M_ref = kron(M1d, Matrix(I, n, n)) + kron(Matrix(I, n, n), M1d)
    @test norm(M_qtt - M_ref) < 1e-8

    # Correctness: serial and interleaved share the same eigenspectrum
    ev_s = sort(real(eigvals(qtto_to_matrix(TToperator(
        qtt_laplacian(2, 3; ordering = :serial, bc = :DD))))))
    ev_i = sort(real(eigvals(qtto_to_matrix(TToperator(
        qtt_laplacian(2, 3; ordering = :interleaved, bc = :DD))))))
    @test maximum(abs, ev_s .- ev_i) < 1e-8

    # BC variants with unit boundary ranks work for n_dims ≥ 2
    # (:NN has non-unit boundary ranks and is only supported for n_dims=1)
    for bc in (:DD, :DN, :ND)
        @test qtt_laplacian(2, d; bc = bc, ordering = :serial) isa QTToperator
    end

    # 1D NN case works
    @test qtt_laplacian(1, d; bc = :NN, ordering = :serial) isa QTToperator

    # 3D case
    A3 = qtt_laplacian(3, 3; ordering = :serial, bc = :DD)
    @test A3 isa QTToperator
    @test A3.n_dims == 3
    @test A3.N == 9
end

@testset "Cross-type dispatch" begin
    ttv = rand_tt(fill(2, 6), 2)
    q = QTTvector(ttv, 2, 3, :interleaved)
    tto = rand_tto(ntuple(_ -> 2, 6), 2)
    A = QTToperator(tto, 2, 3, :interleaved)

    # TToperator * QTTvector → TTvector (strips QTT metadata)
    r1 = tto * q
    @test r1 isa TTvector

    # QTToperator * TTvector → TTvector (strips QTT metadata)
    r2 = A * ttv
    @test r2 isa TTvector

    # QTTvector ± TTvector → TTvector
    @test (q + ttv) isa TTvector
    @test (q - ttv) isa TTvector
    @test (ttv + q) isa TTvector
    @test (ttv - q) isa TTvector

    # QTTvector / scalar → QTTvector (preserves metadata)
    r3 = q / 2.0
    @test r3 isa QTTvector
    @test r3.ordering == :interleaved

    # Cross-type dot products (qualify to avoid ambiguity with TensorTrainNumerics.dot)
    @test LinearAlgebra.dot(q, ttv) isa Number
    @test LinearAlgebra.dot(ttv, q) isa Number

    # TToperator ± QTToperator → TToperator
    B = rand_tto(ntuple(_ -> 2, 6), 2)
    @test (B + A) isa TToperator
    @test (B - A) isa TToperator
    @test (A + B) isa TToperator
    @test (A - B) isa TToperator
end

@testset "Base.show" begin
    ttv = rand_tt(fill(2, 6), 2)
    q   = QTTvector(ttv, 2, 3, :serial)
    tto = rand_tto(ntuple(_ -> 2, 6), 2)
    A   = QTToperator(tto, 2, 3, :interleaved)

    # Compact show
    compact_q = sprint(show, q)
    @test occursin("QTT-MPS{Float64}", compact_q)
    @test occursin("6 sites", compact_q)
    @test occursin("2d", compact_q)
    @test occursin("3bits", compact_q)
    @test occursin("serial", compact_q)

    compact_A = sprint(show, A)
    @test occursin("QTT-MPO{Float64}", compact_A)
    @test occursin("6 sites", compact_A)
    @test occursin("2d", compact_A)
    @test occursin("3bits", compact_A)
    @test occursin("interleaved", compact_A)

    # Plain-text (verbose) show
    verbose_q = sprint(show, MIME("text/plain"), q)
    @test occursin("QTT-MPS{Float64}", verbose_q)
    @test occursin("6 sites", verbose_q)
    @test occursin("2d", verbose_q)
    @test occursin("3 bits/dim", verbose_q)
    @test occursin("serial", verbose_q)
    @test occursin("Physical dims", verbose_q)
    @test occursin("Bond dims", verbose_q)
    @test occursin("Orthogonality", verbose_q)
    # Grid-points field: n_dims * 2^bits_per_dim = 2 * 8 = 16
    @test occursin("16 grid points per dim", verbose_q)

    verbose_A = sprint(show, MIME("text/plain"), A)
    @test occursin("QTT-MPO{Float64}", verbose_A)
    @test occursin("6 sites", verbose_A)
    @test occursin("2d", verbose_A)
    @test occursin("3 bits/dim", verbose_A)
    @test occursin("interleaved", verbose_A)
    @test occursin("Physical dims", verbose_A)
    @test occursin("Bond dims", verbose_A)
    @test occursin("Orthogonality", verbose_A)
    @test occursin("16 grid points per dim", verbose_A)
end

@testset "Solvers accept QTTvector" begin
    # 1D problem: 4-site QTT (bits_per_dim=4, n_dims=1)
    d = 4
    A_tto = Δ(d)
    A = QTToperator(A_tto, 1, d, :interleaved)

    x0_tt = rand_tt(fill(2, d), 2)
    x0 = QTTvector(x0_tt, 1, d, :interleaved)

    # tt_compress! preserves QTT metadata
    xc = copy(x0)
    tt_compress!(xc, 2)
    @test xc isa QTTvector
    @test xc.ordering == :interleaved

    # als_eigsolve accepts QTTvector/QTToperator and returns AbstractTTvector
    E, tt_opt = als_eigsolve(A, x0; sweep_schedule = [2])
    @test E isa Vector{Float64}
    @test tt_opt isa AbstractTTvector
end
