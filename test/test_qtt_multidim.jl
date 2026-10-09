using Test
using TensorTrainNumerics
using LinearAlgebra

@testset "AbstractTTVector type parameter" begin
    ttv = rand_tt(fill(2, 4), 2)
    @test ttv isa AbstractTTVector
    @test eltype(ttv) == Float64
end

@testset "QTTVector struct" begin
    @test QTTVector <: AbstractTTVector
    cores = [rand(2, 1, 2), rand(2, 2, 2), rand(2, 2, 1)]
    rks = [1, 2, 2, 1]
    dims = (2, 2, 2)
    q = QTTVector{Float64, 3}(cores, dims, rks, [1, 3], 1, 3, :serial)
    @test q.n_dims == 1
    @test q.bits_per_dim == 3
    @test q.ordering == :serial
    @test q isa AbstractTTVector
end

@testset "QTTOperator struct" begin
    @test QTTOperator <: AbstractTTOperator
    cores = [rand(2, 2, 1, 2), rand(2, 2, 2, 2), rand(2, 2, 2, 1)]
    rks = [1, 2, 2, 1]
    dims = (2, 2, 2)
    A = QTTOperator{Float64, 3}(cores, dims, dims, rks, [1, 3], 1, 3, :interleaved)
    @test A.n_dims == 1
    @test A.bits_per_dim == 3
    @test A.ordering == :interleaved
    @test A isa AbstractTTOperator
end

@testset "QTTVector/QTTOperator constructors and strip helpers" begin
    # Build a 2-dim, 3-bits-per-dim QTTVector (6 sites total)
    ttv = rand_tt(fill(2, 6), 2)
    q = QTTVector(ttv, 2, 3, :interleaved)
    @test q isa QTTVector{Float64, 6}
    @test q.n_dims == 2
    @test q.bits_per_dim == 3
    @test q.ordering == :interleaved
    @test nsites(q) == 6

    # Strip back to TTVector
    ttv2 = TTVector(q)
    @test ttv2 isa TTVector{Float64, 6}
    @test nsites(ttv2) == 6

    # Wrap constructor asserts
    @test_throws AssertionError QTTVector(ttv, 2, 4, :interleaved)  # 2*4=8 ≠ 6
    @test_throws AssertionError QTTVector(ttv, 2, 3, :bad)           # bad ordering

    # Operator wrap/strip
    tto = rand_tto(ntuple(_ -> 2, 6), 2)
    A = QTTOperator(tto, 2, 3, :serial)
    @test A isa QTTOperator{Float64, 6}
    @test A.ordering == :serial
    tto2 = TTOperator(A)
    @test tto2 isa TTOperator{Float64, 6}
end

@testset "check_compat" begin
    ttv = rand_tt(fill(2, 6), 2)
    q1 = QTTVector(ttv, 2, 3, :interleaved)
    q2 = QTTVector(ttv, 2, 3, :interleaved)
    q3 = QTTVector(ttv, 2, 3, :serial)

    @test check_compat(q1, q2) === nothing
    @test_throws AssertionError check_compat(q1, q3)  # ordering mismatch

    tto = rand_tto(ntuple(_ -> 2, 6), 2)
    A = QTTOperator(tto, 2, 3, :interleaved)
    @test check_compat(A, q1) === nothing
    @test_throws AssertionError check_compat(A, q3)

    # Plain TTVector/TTOperator no-ops
    @test check_compat(ttv, ttv) === nothing
end

@testset "QTTVector entanglement_entropy" begin
    bell_tensor = zeros(Float64, 2, 2)
    bell_tensor[1, 1] = inv(sqrt(2))
    bell_tensor[2, 2] = inv(sqrt(2))
    q = QTTVector(tt_decomp(bell_tensor), 1, 2, :serial)

    @test entanglement_entropy(q) ≈ [log(2)]
    @test entanglement_entropy(q; base = 2) ≈ [1.0]
end

@testset "QTTVector dispatch methods" begin
    ttv = rand_tt(fill(2, 6), 2)
    q1 = QTTVector(ttv, 2, 3, :interleaved)
    q2 = QTTVector(ttv, 2, 3, :interleaved)
    q3 = QTTVector(ttv, 2, 3, :serial)

    # copy preserves metadata
    qc = copy(q1)
    @test qc isa QTTVector
    @test qc.ordering == :interleaved

    qcomplex = complex(q1)
    @test complex(qcomplex) isa QTTVector{ComplexF64}
    @test qcomplex.n_dims == q1.n_dims
    @test qcomplex.bits_per_dim == q1.bits_per_dim
    @test qcomplex.ordering == q1.ordering

    # arithmetic returns QTTVector with same metadata
    @test (q1 + q2) isa QTTVector
    @test (q1 + q2).ordering == :interleaved
    @test (q1 - q2) isa QTTVector
    @test (2.0 * q1) isa QTTVector
    @test (q1 * 2.0) isa QTTVector

    # check_compat enforced
    @test_throws AssertionError q1 + q3

    # scalar results
    @test LinearAlgebra.dot(q1, q2) isa Number
    @test norm(q1) isa Real

    # orthogonalize
    qo = orthogonalize(q1)
    @test qo isa QTTVector
    @test qo.ordering == :interleaved
end

@testset "QTTOperator dispatch methods" begin
    ttv = rand_tt(fill(2, 6), 2)
    tto = rand_tto(ntuple(_ -> 2, 6), 2)
    q = QTTVector(ttv, 2, 3, :interleaved)
    A = QTTOperator(tto, 2, 3, :interleaved)
    B = QTTOperator(tto, 2, 3, :interleaved)

    @test copy(A) isa QTTOperator
    Acomplex = complex(A)
    @test complex(Acomplex) isa QTTOperator{ComplexF64}
    @test Acomplex.n_dims == A.n_dims
    @test Acomplex.bits_per_dim == A.bits_per_dim
    @test Acomplex.ordering == A.ordering
    @test (A + B) isa QTTOperator
    @test (2.0 * A) isa QTTOperator

    # operator-vector product returns QTTVector
    Aq = A * q
    @test Aq isa QTTVector
    @test Aq.ordering == :interleaved
end

@testset "function_to_qttv and qttv_to_array" begin
    # 1D case: function_to_qttv with n_dims=1 should match function_to_qtt
    f1d = x -> sin(π * x[1])
    d = 4
    q_interleaved = function_to_qttv(f1d, 1, d; ordering = :interleaved)
    q_serial = function_to_qttv(f1d, 1, d; ordering = :serial)
    @test q_interleaved isa QTTVector
    @test q_serial isa QTTVector
    # Both orderings are equivalent for 1D — compare via array reconstruction
    arr_il = qttv_to_array(q_interleaved)
    arr_sr = qttv_to_array(q_serial)
    @test maximum(abs, arr_il .- arr_sr) < 1.0e-12

    # Round-trip: construct from f, evaluate, compare to direct grid evaluation
    n_pts = 2^d
    h = 1.0 / (n_pts - 1)
    grid = [h * i for i in 0:(n_pts - 1)]
    @test size(arr_il) == (n_pts,)
    @test maximum(abs, arr_il .- sin.(π .* grid)) < 1.0e-12

    # 2D case: f(x,y) = sin(πx)sin(πy)
    f2d = x -> sin(π * x[1]) * sin(π * x[2])
    bits = 3  # 8 points per dim, 6 sites total
    q2d_il = function_to_qttv(f2d, 2, bits; ordering = :interleaved)
    q2d_sr = function_to_qttv(f2d, 2, bits; ordering = :serial)
    @test q2d_il isa QTTVector
    @test q2d_sr isa QTTVector

    arr2d_il = qttv_to_array(q2d_il)
    arr2d_sr = qttv_to_array(q2d_sr)
    @test size(arr2d_il) == (2^bits, 2^bits)
    @test size(arr2d_sr) == (2^bits, 2^bits)
    # Both orderings should give the same function values
    @test maximum(abs, arr2d_il .- arr2d_sr) < 1.0e-12

    # Compare to direct grid evaluation
    n2 = 2^bits
    h2 = 1.0 / (n2 - 1)
    grid2 = [h2 * i for i in 0:(n2 - 1)]
    ref2d = [sin(π * x) * sin(π * y) for x in grid2, y in grid2]
    @test maximum(abs, arr2d_il .- ref2d) < 1.0e-12
end

@testset "reorder" begin
    # Round-trip: serial → interleaved → serial is identity (up to SVD at threshold=0)
    f2d = x -> sin(π * x[1]) * cos(π * x[2])
    bits = 3
    q_serial = function_to_qttv(f2d, 2, bits; ordering = :serial)
    q_il = reorder(q_serial, :interleaved)
    @test q_il isa QTTVector
    @test q_il.ordering == :interleaved

    q_back = reorder(q_il, :serial)
    @test q_back.ordering == :serial

    # Both should give the same function values when evaluated
    arr_serial = qttv_to_array(q_serial)
    arr_il = qttv_to_array(q_il)
    arr_back = qttv_to_array(q_back)
    @test maximum(abs, arr_serial .- arr_il) < 1.0e-10
    @test maximum(abs, arr_serial .- arr_back) < 1.0e-10

    # reorder with same ordering returns a copy
    q_same = reorder(q_serial, :serial)
    @test q_same.ordering == :serial
    @test maximum(abs, qttv_to_array(q_same) .- arr_serial) < 1.0e-14
end

@testset "qtt_laplacian" begin
    d = 4
    n = 2^d

    # 1D case: returns QTTOperator with correct metadata
    A1_serial = qtt_laplacian(1, d; ordering = :serial, bc = :DN)
    @test A1_serial isa QTTOperator
    @test A1_serial.n_dims == 1
    @test A1_serial.bits_per_dim == d
    @test A1_serial.ordering == :serial
    @test nsites(A1_serial) == d

    A1_il = qtt_laplacian(1, d; ordering = :interleaved, bc = :DN)
    @test A1_il isa QTTOperator
    @test A1_il.ordering == :interleaved

    # 2D serial case: correct metadata and dimensions
    A2s = qtt_laplacian(2, d; ordering = :serial, bc = :DD)
    @test A2s isa QTTOperator
    @test A2s.n_dims == 2
    @test A2s.bits_per_dim == d
    @test A2s.ordering == :serial
    @test nsites(A2s) == 2 * d

    # 2D interleaved case
    A2i = qtt_laplacian(2, d; ordering = :interleaved, bc = :DD)
    @test A2i isa QTTOperator
    @test A2i.ordering == :interleaved
    @test nsites(A2i) == 2 * d

    # Correctness: 2D serial Laplacian matrix matches direct Kronecker-sum reference
    h = 1.0 / (n - 1)
    M_qtt = qtto_to_matrix(TTOperator(A2s))
    M1d = qtto_to_matrix(Δ(d)) ./ h^2
    M_ref = kron(M1d, Matrix(I, n, n)) + kron(Matrix(I, n, n), M1d)
    @test norm(M_qtt - M_ref) < 1.0e-8

    # Correctness: serial and interleaved share the same eigenspectrum
    ev_s = sort(
        real(
            eigvals(
                qtto_to_matrix(
                    TTOperator(
                        qtt_laplacian(2, 3; ordering = :serial, bc = :DD)
                    )
                )
            )
        )
    )
    ev_i = sort(
        real(
            eigvals(
                qtto_to_matrix(
                    TTOperator(
                        qtt_laplacian(2, 3; ordering = :interleaved, bc = :DD)
                    )
                )
            )
        )
    )
    @test maximum(abs, ev_s .- ev_i) < 1.0e-8

    # Every boundary condition composes into a multidimensional Kronecker sum.
    for bc in (:DD, :DN, :ND, :NN)
        @test qtt_laplacian(2, d; bc = bc, ordering = :serial) isa QTTOperator
    end

    # 1D NN case works
    @test qtt_laplacian(1, d; bc = :NN, ordering = :serial) isa QTTOperator

    # 3D case
    A3 = qtt_laplacian(3, 3; ordering = :serial, bc = :DD)
    @test A3 isa QTTOperator
    @test A3.n_dims == 3
    @test nsites(A3) == 9
end

@testset "NN Laplacian acts on multidimensional QTT states" begin
    d = 4
    n = 2^d
    h = 1.0 / (n - 1)
    diagonal = fill(2.0, n)
    diagonal[1] = diagonal[end] = 1.0
    L = Matrix(SymTridiagonal(diagonal, fill(-1.0, n - 1))) / h^2
    identity = Matrix{Float64}(I, n, n)
    reference = kron(L, identity) + kron(identity, L)

    for ordering in (:serial, :interleaved)
        @testset "$ordering" begin
            A = qtt_laplacian(2, d; bc = :NN, ordering)
            x = function_to_qttv(v -> sin(2v[1]) + cos(3v[2]), 2, d; ordering)
            expected = reshape(reference * vec(qttv_to_array(x)), n, n)
            y = A * x
            @test first(A.ranks) == last(A.ranks) == 1
            @test qttv_to_array(y) ≈ expected atol = 1.0e-9
            @test qttv_to_array(orthogonalize(y)) ≈ expected atol = 1.0e-9

            constant = function_to_qttv(_ -> 1.0, 2, d; ordering)
            @test norm(qttv_to_array(A * constant)) < 1.0e-9
        end
    end
end

@testset "Cross-type dispatch" begin
    ttv = rand_tt(fill(2, 6), 2)
    q = QTTVector(ttv, 2, 3, :interleaved)
    tto = rand_tto(ntuple(_ -> 2, 6), 2)
    A = QTTOperator(tto, 2, 3, :interleaved)

    # TTOperator * QTTVector → TTVector (strips QTT metadata)
    r1 = tto * q
    @test r1 isa TTVector

    # QTTOperator * TTVector → TTVector (strips QTT metadata)
    r2 = A * ttv
    @test r2 isa TTVector

    # QTTVector ± TTVector → TTVector
    @test (q + ttv) isa TTVector
    @test (q - ttv) isa TTVector
    @test (ttv + q) isa TTVector
    @test (ttv - q) isa TTVector

    # QTTVector / scalar → QTTVector (preserves metadata)
    r3 = q / 2.0
    @test r3 isa QTTVector
    @test r3.ordering == :interleaved

    # Cross-type dot products (qualify to avoid ambiguity with TensorTrainNumerics.dot)
    @test LinearAlgebra.dot(q, ttv) isa Number
    @test LinearAlgebra.dot(ttv, q) isa Number

    # TTOperator ± QTTOperator → TTOperator
    B = rand_tto(ntuple(_ -> 2, 6), 2)
    @test (B + A) isa TTOperator
    @test (B - A) isa TTOperator
    @test (A + B) isa TTOperator
    @test (A - B) isa TTOperator
end

@testset "Base.show" begin
    ttv = rand_tt(fill(2, 6), 2)
    q = QTTVector(ttv, 2, 3, :serial)
    tto = rand_tto(ntuple(_ -> 2, 6), 2)
    A = QTTOperator(tto, 2, 3, :interleaved)

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

@testset "Solvers accept QTTVector" begin
    # 1D problem: 4-site QTT (bits_per_dim=4, n_dims=1)
    d = 4
    A_tto = Δ(d)
    A = QTTOperator(A_tto, 1, d, :interleaved)

    x0_tt = rand_tt(fill(2, d), 2)
    x0 = QTTVector(x0_tt, 1, d, :interleaved)

    # tt_compress! preserves QTT metadata
    xc = copy(x0)
    tt_compress!(xc, 2)
    @test xc isa QTTVector
    @test xc.ordering == :interleaved

    # als_eigsolve accepts QTTVector/QTTOperator and returns AbstractTTVector
    E, tt_opt = als_eigsolve(A, x0; max_sweeps = 1)
    @test E isa Vector{Float64}
    @test tt_opt isa AbstractTTVector
end


@testset "function_to_qttv — 3D value accuracy, both orderings" begin
    f = x -> sin(π * x[1]) * sin(π * x[2]) * sin(π * x[3])
    bits = 3
    n = 2^bits
    h = 1.0 / (n - 1)
    grid = [h * i for i in 0:(n - 1)]
    ref = [sin(π * x) * sin(π * y) * sin(π * z) for x in grid, y in grid, z in grid]

    for ordering in (:serial, :interleaved)
        q = function_to_qttv(f, 3, bits; ordering = ordering)
        @test nsites(q) == 3 * bits
        @test q.n_dims == 3
        @test q.bits_per_dim == bits
        @test q.ordering == ordering
        arr = qttv_to_array(q)
        @test size(arr) == (n, n, n)
        @test maximum(abs, arr .- ref) < 1.0e-12
    end

    # Both orderings produce the same array values
    q_il = function_to_qttv(f, 3, bits; ordering = :interleaved)
    q_sr = function_to_qttv(f, 3, bits; ordering = :serial)
    @test maximum(abs, qttv_to_array(q_il) .- qttv_to_array(q_sr)) < 1.0e-12
end

@testset "function_to_qttv — non-separable 2D Gaussian" begin
    f = x -> exp(-10 * ((x[1] - 0.3)^2 + (x[2] - 0.7)^2))
    bits = 5
    n = 2^bits
    h = 1.0 / (n - 1)
    grid = [h * i for i in 0:(n - 1)]
    ref = [exp(-10 * ((x - 0.3)^2 + (y - 0.7)^2)) for x in grid, y in grid]

    q_il = function_to_qttv(f, 2, bits; ordering = :interleaved)
    q_sr = function_to_qttv(f, 2, bits; ordering = :serial)
    @test maximum(abs, qttv_to_array(q_il) .- ref) < 1.0e-12
    @test maximum(abs, qttv_to_array(q_sr) .- ref) < 1.0e-12
    @test maximum(abs, qttv_to_array(q_il) .- qttv_to_array(q_sr)) < 1.0e-12
end

@testset "function_to_qttv — custom interval [a, b]" begin
    a, b = -1.0, 2.0
    f = x -> sin(x[1]) * cos(x[2])
    bits = 4
    n = 2^bits
    h = (b - a) / (n - 1)
    grid = [a + h * i for i in 0:(n - 1)]
    ref = [sin(x) * cos(y) for x in grid, y in grid]

    for ordering in (:serial, :interleaved)
        q = function_to_qttv(f, 2, bits; ordering = ordering, a = a, b = b)
        @test maximum(abs, qttv_to_array(q) .- ref) < 1.0e-12
    end
end

@testset "serial/interleaved equivalence — dot, norm, arithmetic" begin
    f1 = x -> exp(-x[1]) * (1.0 + x[2])
    f2 = x -> cos(π * x[1]) * (1.0 + 2.0 * x[2])
    bits = 4

    q1_il = function_to_qttv(f1, 2, bits; ordering = :interleaved)
    q2_il = function_to_qttv(f2, 2, bits; ordering = :interleaved)
    q1_sr = function_to_qttv(f1, 2, bits; ordering = :serial)
    q2_sr = function_to_qttv(f2, 2, bits; ordering = :serial)

    arr1 = qttv_to_array(q1_il)
    arr2 = qttv_to_array(q2_il)
    dot_ref = sum(arr1 .* arr2)
    norm_ref = sqrt(sum(abs2, arr1))

    # dot matches direct array inner product, independent of ordering
    @test isapprox(TensorTrainNumerics.dot(q1_il, q2_il), dot_ref; rtol = 1.0e-10)
    @test isapprox(TensorTrainNumerics.dot(q1_sr, q2_sr), dot_ref; rtol = 1.0e-10)

    # norm matches, independent of ordering
    @test isapprox(norm(q1_il), norm_ref; rtol = 1.0e-10)
    @test isapprox(norm(q1_sr), norm_ref; rtol = 1.0e-10)

    # norm² = dot(q, q)
    @test isapprox(norm(q1_il)^2, TensorTrainNumerics.dot(q1_il, q1_il); rtol = 1.0e-10)
    @test isapprox(norm(q1_sr)^2, TensorTrainNumerics.dot(q1_sr, q1_sr); rtol = 1.0e-10)

    # addition: (q1 + q2) matches arr1 + arr2
    @test maximum(abs, qttv_to_array(q1_il + q2_il) .- (arr1 .+ arr2)) < 1.0e-12
    @test maximum(abs, qttv_to_array(q1_sr + q2_sr) .- (arr1 .+ arr2)) < 1.0e-12

    # subtraction
    @test maximum(abs, qttv_to_array(q1_il - q2_il) .- (arr1 .- arr2)) < 1.0e-12

    # scalar multiplication
    @test maximum(abs, qttv_to_array(3.5 * q1_il) .- 3.5 .* arr1) < 1.0e-12
    @test maximum(abs, qttv_to_array(q1_sr * 3.5) .- 3.5 .* arr1) < 1.0e-12
    @test maximum(abs, qttv_to_array(q1_il / 2.0) .- arr1 ./ 2.0) < 1.0e-12
end

@testset "reorder — 3D round-trip and cross-validation" begin
    f = x -> cos(π * x[1]) * sin(2π * x[2]) * exp(-x[3])
    bits = 3

    q_sr = function_to_qttv(f, 3, bits; ordering = :serial)
    q_il = function_to_qttv(f, 3, bits; ordering = :interleaved)
    arr_sr = qttv_to_array(q_sr)
    arr_il = qttv_to_array(q_il)

    # serial → interleaved via reorder matches direct interleaved construction
    q_il_r = reorder(q_sr, :interleaved)
    @test q_il_r.ordering == :interleaved
    @test q_il_r.n_dims == 3
    @test q_il_r.bits_per_dim == bits
    @test maximum(abs, qttv_to_array(q_il_r) .- arr_il) < 1.0e-10

    q_il_r_truncated = reorder(q_sr, :interleaved; threshold = 1.0e-14)
    @test q_il_r_truncated.ordering == :interleaved
    @test maximum(abs, qttv_to_array(q_il_r_truncated) .- arr_il) < 1.0e-10

    # interleaved → serial via reorder matches direct serial construction
    q_sr_r = reorder(q_il, :serial)
    @test q_sr_r.ordering == :serial
    @test maximum(abs, qttv_to_array(q_sr_r) .- arr_sr) < 1.0e-10

    # serial → interleaved → serial round-trip
    q_rt = reorder(reorder(q_sr, :interleaved), :serial)
    @test maximum(abs, qttv_to_array(q_rt) .- arr_sr) < 1.0e-10

    # norms preserved across reorder
    @test isapprox(norm(q_il_r), norm(q_sr); rtol = 1.0e-10)
end

@testset "hadamard — 2D correctness, both orderings" begin
    f1 = x -> sin(π * x[1]) * sin(π * x[2])
    f2 = x -> cos(π * x[1]) * cos(π * x[2])
    bits = 4

    for ordering in (:serial, :interleaved)
        q1 = function_to_qttv(f1, 2, bits; ordering = ordering)
        q2 = function_to_qttv(f2, 2, bits; ordering = ordering)
        h12 = hadamard(q1, q2)

        @test h12 isa QTTVector
        @test h12.ordering == ordering
        @test h12.n_dims == 2
        @test h12.bits_per_dim == bits

        arr1 = qttv_to_array(q1)
        arr2 = qttv_to_array(q2)
        @test maximum(abs, qttv_to_array(h12) .- arr1 .* arr2) < 1.0e-12
    end

    # Use identity sin²+cos²=1: hadamard(sin, sin) + hadamard(cos, cos) ≈ ones
    for ordering in (:serial, :interleaved)
        qs = function_to_qttv(x -> sin(π * x[1]) * sin(π * x[2]), 2, bits; ordering = ordering)
        qc = function_to_qttv(x -> cos(π * x[1]) * cos(π * x[2]), 2, bits; ordering = ordering)
        qss = hadamard(qs, qs)
        qcc = hadamard(qc, qc)
        arr_sum = qttv_to_array(qss + qcc)
        ref = [
            sin(π * x)^2 * sin(π * y)^2 + cos(π * x)^2 * cos(π * y)^2
                for x in range(0, 1; length = 2^bits),
                y in range(0, 1; length = 2^bits)
        ]
        @test maximum(abs, arr_sum .- ref) < 1.0e-12
    end
end

@testset "hadamard — 3D correctness, both orderings" begin
    f1 = x -> sin(π * x[1]) * sin(π * x[2]) * sin(π * x[3])
    f2 = x -> exp(-x[1] - x[2] - x[3])
    bits = 3

    for ordering in (:serial, :interleaved)
        q1 = function_to_qttv(f1, 3, bits; ordering = ordering)
        q2 = function_to_qttv(f2, 3, bits; ordering = ordering)
        h12 = hadamard(q1, q2)

        @test h12 isa QTTVector
        @test h12.ordering == ordering
        @test h12.n_dims == 3

        arr1 = qttv_to_array(q1)
        arr2 = qttv_to_array(q2)
        @test maximum(abs, qttv_to_array(h12) .- arr1 .* arr2) < 1.0e-12
    end
end

@testset "separable function has rank 1 in serial ordering" begin
    # exp(-x)*exp(-y) is a rank-1 outer product. In serial ordering (sites grouped
    # by dimension), the bond across the dimension boundary is exactly rank 1.
    f_sep = x -> exp(-x[1]) * exp(-x[2])
    bits = 6
    q = function_to_qttv(f_sep, 2, bits; ordering = :serial)
    q_c = copy(q)
    tt_compress!(q_c, 10; trunc_tol = 1.0e-12)

    # The cross-dimension bond (site `bits` → `bits+1`) should be rank 1
    @test q_c.ranks[bits + 1] == 1
    # All bonds should stay ≤ 1 (exponential is rank-1 in QTT)
    @test maximum(q_c.ranks) == 1

    # Values still correct after compression
    n = 2^bits
    h = 1.0 / (n - 1)
    grid = [h * i for i in 0:(n - 1)]
    ref = [exp(-x) * exp(-y) for x in grid, y in grid]
    @test maximum(abs, qttv_to_array(q_c) .- ref) < 1.0e-10
end

@testset "tt_compress! on QTTVector preserves metadata and accuracy" begin
    f = x -> sin(2π * x[1]) * sin(2π * x[2])
    bits = 5
    q = function_to_qttv(f, 2, bits; ordering = :interleaved)
    arr_ref = qttv_to_array(q)

    q_c = copy(q)
    tt_compress!(q_c, 8; trunc_tol = 1.0e-12)

    @test q_c isa QTTVector
    @test q_c.ordering == :interleaved
    @test q_c.n_dims == 2
    @test q_c.bits_per_dim == bits
    @test maximum(q_c.ranks) ≤ 8
    @test maximum(abs, qttv_to_array(q_c) .- arr_ref) < 1.0e-8
end

@testset "increase_ranks on QTTVector preserves metadata and values" begin
    f = x -> exp(-x[1]) * exp(-x[2])
    bits = 4
    q = function_to_qttv(f, 2, bits; ordering = :serial)
    arr_ref = qttv_to_array(q)

    q_up = TensorTrainNumerics.increase_ranks(q, 4; noise = 0.0)

    @test q_up isa QTTVector
    @test q_up.ordering == q.ordering
    @test q_up.n_dims == q.n_dims
    @test q_up.bits_per_dim == q.bits_per_dim
    @test maximum(q_up.ranks) ≤ 4
    @test maximum(q_up.ranks) > maximum(q.ranks)
    @test maximum(abs, qttv_to_array(q_up) .- arr_ref) < 1.0e-12
end

@testset "qtt_laplacian — 3D Kronecker-sum matrix correctness" begin
    d = 3
    n = 2^d
    h = 1.0 / (n - 1)
    I_n = Matrix(I, n, n)
    M1d = qtto_to_matrix(Δ(d)) ./ h^2

    A3s = qtt_laplacian(3, d; ordering = :serial, bc = :DD)
    @test A3s.n_dims == 3
    @test A3s.bits_per_dim == d
    @test nsites(A3s) == 3 * d

    M_qtt = qtto_to_matrix(TTOperator(A3s))
    M_ref = kron(M1d, kron(I_n, I_n)) +
        kron(I_n, kron(M1d, I_n)) +
        kron(I_n, kron(I_n, M1d))
    @test norm(M_qtt - M_ref) < 1.0e-6

    # serial and interleaved share the same eigenspectrum
    A3i = qtt_laplacian(3, d; ordering = :interleaved, bc = :DD)
    ev_s = sort(real(eigvals(M_qtt)))
    ev_i = sort(real(eigvals(qtto_to_matrix(TTOperator(A3i)))))
    @test maximum(abs, ev_s .- ev_i) < 1.0e-8
end

@testset "qtt_laplacian — operator action on 2D vector, both orderings" begin
    d = 3
    n = 2^d
    h = 1.0 / (n - 1)
    I_n = Matrix(I, n, n)
    M1d = qtto_to_matrix(Δ(d)) ./ h^2
    M2d = kron(M1d, I_n) + kron(I_n, M1d)

    f = x -> sin(π * x[1]) * sin(π * x[2])

    for ordering in (:serial, :interleaved)
        A = qtt_laplacian(2, d; ordering = ordering, bc = :DD)
        v = function_to_qttv(f, 2, d; ordering = ordering)

        Av = A * v
        @test Av isa QTTVector
        @test Av.ordering == ordering
        @test Av.n_dims == 2
        @test Av.bits_per_dim == d

        # Compare against dense matrix-vector product
        arr_v = qttv_to_array(v)
        ref_Av = reshape(M2d * vec(arr_v), n, n)
        @test maximum(abs, qttv_to_array(Av) .- ref_Av) < 1.0e-8
    end

    # serial and interleaved produce the same Av values
    A_sr = qtt_laplacian(2, d; ordering = :serial, bc = :DD)
    A_il = qtt_laplacian(2, d; ordering = :interleaved, bc = :DD)
    v_sr = function_to_qttv(f, 2, d; ordering = :serial)
    v_il = function_to_qttv(f, 2, d; ordering = :interleaved)
    arr_Av_sr = qttv_to_array(A_sr * v_sr)
    arr_Av_il = qttv_to_array(A_il * v_il)
    @test maximum(abs, arr_Av_sr .- arr_Av_il) < 1.0e-8
end

@testset "QTTOperator reorder — serial ↔ interleaved" begin
    d = 3
    A_sr = qtt_laplacian(2, d; ordering = :serial, bc = :DD)
    A_il = qtt_laplacian(2, d; ordering = :interleaved, bc = :DD)

    # reorder(serial → interleaved) should reproduce the directly-built interleaved operator
    A_il_r = reorder(A_sr, :interleaved)
    @test A_il_r isa QTTOperator
    @test A_il_r.ordering == :interleaved
    @test A_il_r.n_dims == A_sr.n_dims
    @test A_il_r.bits_per_dim == A_sr.bits_per_dim

    # Apply both to the same vector and check they give the same result
    f = x -> sin(π * x[1]) * cos(2π * x[2])
    v_il = function_to_qttv(f, 2, d; ordering = :interleaved)
    Av_direct = qttv_to_array(A_il * v_il)
    Av_reorder = qttv_to_array(A_il_r * v_il)
    @test maximum(abs, Av_direct .- Av_reorder) < 1.0e-8

    A_il_truncated = reorder(A_sr, :interleaved; threshold = 1.0e-14)
    @test A_il_truncated.ordering == :interleaved
    Av_truncated = qttv_to_array(A_il_truncated * v_il)
    @test maximum(abs, Av_direct .- Av_truncated) < 1.0e-8

    # Round-trip: serial → interleaved → serial
    A_sr_rt = reorder(A_il_r, :serial)
    @test A_sr_rt.ordering == :serial
    v_sr = function_to_qttv(f, 2, d; ordering = :serial)
    Av_orig = qttv_to_array(A_sr * v_sr)
    Av_rt = qttv_to_array(A_sr_rt * v_sr)
    @test maximum(abs, Av_orig .- Av_rt) < 1.0e-8
end

@testset "generic TT functions accept QTT types" begin
    x = rand_tt((2, 2, 2, 2), 2)
    y = rand_tt((2, 2, 2, 2), 2)
    q = QTTVector(x, 2, 2, :serial)
    p = QTTVector(y, 2, 2, :serial)
    A = QTTOperator(id_tto(4) + Δ(4), 2, 2, :serial)
    same_grid(r) = (r.n_dims, r.bits_per_dim, r.ordering) == (2, 2, :serial)

    @testset "values" begin
        @test tt_to_tensor(q) == tt_to_tensor(x)
        @test qtt_to_vector(q) == qtt_to_vector(x)
        @test matricize(q, 2) == matricize(x, 2)
        @test tto_to_tensor(A) == tto_to_tensor(TTOperator(A))
        @test qtto_to_matrix(A) == qtto_to_matrix(TTOperator(A))
        @test norm(q) == norm(x)
        @test dot(q, p) == dot(x, y)
        @test dot(q, y) == dot(x, y)
        @test entanglement_entropy(q) == entanglement_entropy(x)
        @test eltype(q) == Float64
        @test eltype(A) == Float64
        @test nsites(A) == 4
        @test redirect_stdout(() -> visualize(q), devnull) === nothing
    end

    @testset "results on the same sites keep the QTT metadata" begin
        for r in (
                copy(q), complex(q), orthogonalize(q; center = 2), tt_round(q), 2 * q, q * 2, q / 2, -q,
                q + p, q - p, hadamard(q, p), hadamard_ttm(q, p), A * q, rand_tt(q), q + complex(p),
                TensorTrainNumerics.increase_ranks(q, 4),
            )
            @test r isa QTTVector
            @test same_grid(r)
        end
        for R in (copy(A), complex(A), A + A, A - A, A * A, 2 * A, A', tt_to_diag_tto(q))
            @test R isa QTTOperator
            @test same_grid(R)
        end
        @test tt_to_tensor(q + p) ≈ tt_to_tensor(x + y)
        @test tt_to_tensor(A * q) ≈ tt_to_tensor(TTOperator(A) * x)
        @test tto_to_tensor(A * A) ≈ tto_to_tensor(TTOperator(A) * TTOperator(A))

        r = copy(q)
        @test tt_round!(r) === r
        @test tt_compress!(r, 2) === r
        @test add!(r, p) === r
    end

    @testset "results on other sites, or mixed with plain inputs, are plain" begin
        @test kron(q, p) isa TTVector
        @test concatenate(q, p) isa TTVector
        @test reverse_qtt_bits(q) isa TTVector
        @test outer_product(q, p) isa TTOperator
        @test q + y isa TTVector
        @test x - p isa TTVector
        @test TTOperator(A) * q isa TTVector
        @test A * x isa TTVector
        @test copy(id_tto(3)) isa TTOperator
    end

    @testset "mismatched grids are rejected" begin
        other = QTTVector(rand_tt((2, 2, 2, 2), 2), 1, 4, :serial)
        B = QTTOperator(id_tto(4), 1, 4, :serial)
        @test_throws "n_dims mismatch" A + B
        @test_throws "n_dims mismatch" A * B
        @test_throws "n_dims mismatch" hadamard_ttm(q, other)
        # The check runs before any operand is copied, converted, or contracted.
        # Copying the cores of one of these inputs alone takes more than the bound.
        big = QTTVector(rand_tt(ntuple(_ -> 2, 12), 16), 2, 6, :serial)
        other_big = QTTVector(rand_tt(ntuple(_ -> 2, 12), 16), 2, 6, :interleaved)
        bigA = QTTOperator(rand_tto(ntuple(_ -> 2, 12), 16), 2, 6, :interleaved)
        bigB = QTTOperator(rand_tto(ntuple(_ -> 2, 12), 16), 2, 6, :serial)
        cbig, cA = complex(big), complex(bigA)
        @test sum(sizeof, big.cores) > 10_000
        guarded(f) = try
            f()
        catch
            nothing
        end
        for f in (
                () -> bigA * big, () -> bigA * bigB, () -> bigA + bigB, () -> bigA - bigB,
                () -> big + other_big, () -> big - other_big, () -> hadamard(big, other_big),
                () -> cbig + other_big, () -> cbig - other_big, () -> cA + bigB, () -> cA - bigB,
                () -> cA * big, () -> cA * bigB, () -> dot(cbig, other_big),
            )
            @test_throws AssertionError f()
            guarded(f)
            @test (@allocated guarded(f)) < 10_000
        end
    end
end

@testset "Laplacian boundary conditions" begin
    for bc in (:DD, :DN, :ND, :NN, :periodic)
        @test Δ(4; bc) isa TTOperator{Float64, 4}
    end
    @test tto_to_tensor(Δ(4)) == tto_to_tensor(Δ(4; bc = :DD))
    @test_throws "`bc` must be :DD, :DN, :ND, :NN, or :periodic" Δ(4; bc = :robin)
    @test Δ⁻¹(4) isa TTOperator{Float64, 4}
    @test_throws "only available for `bc = :DN`" Δ⁻¹(4; bc = :DD)
    @test tto_to_tensor(TTOperator(qtt_laplacian(1, 4; bc = :DD))) ≈ 15.0^2 .* tto_to_tensor(Δ(4; bc = :DD))

    # A periodic grid has N distinct points per period, so h = 1/N. The first
    # Fourier mode is an eigenvector with eigenvalue 4 sin²(π/N) / h².
    for d in (4, 6)
        N = 2^d
        L = qtto_to_matrix(TTOperator(qtt_laplacian(1, d; bc = :periodic)))
        mode = sinpi.(2 .* (0:(N - 1)) ./ N)
        @test L * mode ≈ (4 * sinpi(1 / N)^2 * N^2) .* mode
        @test L * ones(N) ≈ zeros(N) atol = 1.0e-9
    end
    L = qtto_to_matrix(TTOperator(qtt_laplacian(1, 4; a = 1.0, b = 3.0, bc = :periodic)))
    @test L * sinpi.(2 .* (0:15) ./ 16) ≈ (4 * sinpi(1 / 16)^2 * 8.0^2) .* sinpi.(2 .* (0:15) ./ 16)
end
