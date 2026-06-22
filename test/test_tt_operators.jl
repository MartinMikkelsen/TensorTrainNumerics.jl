using Test
using LinearAlgebra
using TensorTrainNumerics

@testset "toeplitz_to_qtto" begin
    # Minimal mock for zeros_tto to allow testing
    struct MockTTO
        tto_vec::Vector
    end
    function zeros_tto(args...)
        # For this test, we only care about the shapes
        d = args[2]
        tto_vec = []
        # First core: (2,2,1,3)
        push!(tto_vec, zeros(2, 2, 1, 3))
        # Middle d: (2,2,3,3)
        for _ in 2:(d - 1)
            push!(tto_vec, zeros(2, 2, 3, 3))
        end
        # Last core: (2,2,3,1)
        push!(tto_vec, zeros(2, 2, 3, 1))
        return MockTTO(tto_vec)
    end

    # Test for d = 3
    d = 3
    α, β, γ = 2.0, -1.0, 0.5
    tto = TensorTrainNumerics.toeplitz_to_qtto(α, β, γ, d)
    @test length(tto.tto_vec) == d
    @test size(tto.tto_vec[1]) == (2, 2, 1, 3)
    @test size(tto.tto_vec[2]) == (2, 2, 3, 3)
    @test size(tto.tto_vec[3]) == (2, 2, 3, 1)

    # Check that the first core is filled as expected for a simple case
    # For α=1, β=0, γ=0, the first core should be identity in the first slice
    tto_id = TensorTrainNumerics.toeplitz_to_qtto(1, 0, 0, d)
    @test tto_id.tto_vec[1][1, 1, 1, 1] ≈ 1.0
    @test tto_id.tto_vec[1][2, 2, 1, 1] ≈ 1.0
    @test tto_id.tto_vec[1][1, 2, 1, 1] ≈ 0.0
    @test tto_id.tto_vec[1][2, 1, 1, 1] ≈ 0.0

    # Check that the last core is filled as expected for a simple case
    # For α=1, β=0, γ=0, the last core should be α*id in the first slice
    @test tto_id.tto_vec[d][1, 1, 1, 1] ≈ 1.0
    @test tto_id.tto_vec[d][2, 2, 1, 1] ≈ 1.0
    @test tto_id.tto_vec[d][1, 2, 1, 1] ≈ 0.0
    @test tto_id.tto_vec[d][2, 1, 1, 1] ≈ 0.0
end

@testset "id_tto" begin
    d = 3
    tto = TensorTrainNumerics.id_tto(d)

    @test typeof(tto) == TensorTrainNumerics.TToperator{Float64, 3}
    @test length(tto.tto_vec) == d
    for core in tto.tto_vec
        @test size(core) == (2, 2, 1, 1)
        @test core[:, :, 1, 1] ≈ Matrix(I, 2, 2)
    end
end

@testset "rand_tto" begin
    dims = (2, 2, 2)
    rmax = 3
    tto = TensorTrainNumerics.rand_tto(dims, rmax)

    @test typeof(tto) == TensorTrainNumerics.TToperator{Float64, 3}
    @test length(tto.tto_vec) == 3
    for (i, core) in enumerate(tto.tto_vec)
        @test size(core, 1) == dims[i]
        @test size(core, 2) == dims[i]
        @test size(core, 3) ≥ 1
        @test size(core, 4) ≥ 1
    end
end

@testset "zeros_tt (uniform)" begin
    n, d, r = 2, 3, 2
    ttv = TensorTrainNumerics.zeros_tt(n, d, r)

    @test typeof(ttv) == TensorTrainNumerics.TTvector{Float64, 3}
    @test length(ttv.ttv_vec) == d
    for core in ttv.ttv_vec
        @test all(core .== 0.0)
    end
end

@testset "ones_tt" begin
    dims = (2, 2, 2)
    ttv = TensorTrainNumerics.ones_tt(dims)

    @test typeof(ttv) == TensorTrainNumerics.TTvector{Float64, 3}
    @test length(ttv.ttv_vec) == 3
    for core in ttv.ttv_vec
        @test all(core .== 1.0)
        @test size(core) == (2, 1, 1)
    end
end

@testset "zeros_tt!" begin
    ttv = TensorTrainNumerics.ones_tt((2, 2, 2))
    TensorTrainNumerics.zeros_tt!(ttv)
    for core in ttv.ttv_vec
        @test all(core .== 0.0)
    end
end

@testset "Δ_NN" begin
    d = 6

    A = Δ_NN(d)

    function laplacian_nn_matrix(n)
        A = zeros(n, n)
        for i in 1:n
            if i == 1 || i == n
                A[i, i] = 1
            else
                A[i, i] = 2
            end
            if i > 1
                A[i, i - 1] = -1
            end
            if i < n
                A[i, i + 1] = -1
            end
        end
        return A
    end

    B = laplacian_nn_matrix(2^d)

    @test A == B
end

@testset "Δ_DD" begin
    d = 6

    A = Δ(d)

    function laplacian_dd_matrix(n)
        A = zeros(n, n)
        for i in 1:n
            if i == 1 || i == n
                A[i, i] = 2
            else
                A[i, i] = 2
            end
            if i > 1
                A[i, i - 1] = -1
            end
            if i < n
                A[i, i + 1] = -1
            end
        end
        return A
    end

    B = laplacian_dd_matrix(2^d)

    @test A == B
end

@testset "Δ_DN" begin
    d = 6

    A = Δ_DN(d)

    function laplacian_dn_matrix(n)
        A = zeros(n, n)
        for i in 1:n
            if i == 1
                A[i, i] = 2
            elseif i == n
                A[i, i] = 1
            else
                A[i, i] = 2
            end
            if i > 1
                A[i, i - 1] = -1
            end
            if i < n
                A[i, i + 1] = -1
            end
        end
        return A
    end

    B = laplacian_dn_matrix(2^d)

    @test A == B
end

@testset "Δ_ND" begin
    d = 6

    A = Δ_ND(d)

    function laplacian_nd_matrix(n)
        A = zeros(n, n)
        for i in 1:n
            if i == 1
                A[i, i] = 1
            elseif i == n
                A[i, i] = 2
            else
                A[i, i] = 2
            end
            if i > 1
                A[i, i - 1] = -1
            end
            if i < n
                A[i, i + 1] = -1
            end
        end
        return A
    end

    B = laplacian_nd_matrix(2^d)

    @test A == B
end

@testset "Δ_P" begin
    d = 6

    A = Δ_P(d)

    function laplacian_p_matrix(n)
        A = zeros(n, n)
        for i in 1:n
            A[i, i] = 2
            if i > 1
                A[i, i - 1] = -1
            end
            if i < n
                A[i, i + 1] = -1
            end
        end
        # Periodic boundary conditions
        A[1, n] = -1   # top right
        A[n, 1] = -1   # bottom left
        return A
    end

    B = laplacian_p_matrix(2^d)

    @test A == B
end

@testset "shift, ∇, Δ" begin
    # Test for d = 3
    d = 3

    # shift should call toeplitz_to_qtto(0,1,0,d)
    tto_shift = TensorTrainNumerics.shift(d)
    tto_ref = TensorTrainNumerics.toeplitz_to_qtto(0, 1, 0, d)
    @test all(size.(tto_shift.tto_vec) .== size.(tto_ref.tto_vec))
    @test all(tto_shift.tto_vec[1] .== tto_ref.tto_vec[1])
    @test all(tto_shift.tto_vec[2] .== tto_ref.tto_vec[2])
    @test all(tto_shift.tto_vec[3] .== tto_ref.tto_vec[3])

    # ∇ should call toeplitz_to_qtto(1,-1,0,d)
    tto_grad = TensorTrainNumerics.∇(d)
    tto_ref = TensorTrainNumerics.toeplitz_to_qtto(1, 0, -1, d)
    @test all(size.(tto_grad.tto_vec) .== size.(tto_ref.tto_vec))
    @test all(tto_grad.tto_vec[1] .== tto_ref.tto_vec[1])
    @test all(tto_grad.tto_vec[2] .== tto_ref.tto_vec[2])
    @test all(tto_grad.tto_vec[3] .== tto_ref.tto_vec[3])

    # Δ should call toeplitz_to_qtto(2,-1,-1,d)
    tto_lap = TensorTrainNumerics.Δ(d)
    tto_ref = TensorTrainNumerics.toeplitz_to_qtto(2, -1, -1, d)
    @test all(size.(tto_lap.tto_vec) .== size.(tto_ref.tto_vec))
    @test all(tto_lap.tto_vec[1] .== tto_ref.tto_vec[1])
    @test all(tto_lap.tto_vec[2] .== tto_ref.tto_vec[2])
    @test all(tto_lap.tto_vec[3] .== tto_ref.tto_vec[3])
end

function _qtt_from_vector(v::AbstractVector)
    d = Int(log2(length(v)))
    @assert 2^d == length(v)
    tensor = zeros(eltype(v), ntuple(_ -> 2, d))
    @inbounds for I in CartesianIndices(tensor)
        tensor[I] = v[TensorTrainNumerics.tuple_to_index(Tuple(I))]
    end
    return ttv_decomp(tensor)
end

function _dense_circulant(kernel::AbstractVector)
    n = length(kernel)
    return [kernel[mod(i - j, n) + 1] for i in 0:(n - 1), j in 0:(n - 1)]
end

function _dense_cconv(kernel::AbstractVector, x::AbstractVector)
    n = length(kernel)
    return [sum(kernel[mod(i - j, n) + 1] * x[j + 1] for j in 0:(n - 1)) for i in 0:(n - 1)]
end

function _dense_conv(kernel::AbstractVector, x::AbstractVector)
    n = length(kernel)
    y = zeros(promote_type(eltype(kernel), eltype(x)), 2n)
    @inbounds for i in 0:(n - 1), j in 0:(n - 1)
        y[i + j + 1] += kernel[i + 1] * x[j + 1]
    end
    return y
end

@testset "circulant_qtto and qtt_cconv" begin
    kernel = [1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.0, 4.0]
    x = [0.25, -1.0, 2.0, 0.5, -0.75, 1.5, 3.0, -2.0]

    kernel_qtt = _qtt_from_vector(kernel)
    x_qtt = _qtt_from_vector(x)

    C = circulant_qtto(kernel_qtt)

    @test C isa TToperator{Float64, 3}
    @test C.tto_dims == (2, 2, 2)
    @test qtto_to_matrix(C) ≈ _dense_circulant(kernel)
    @test qtt_to_vector(qtt_cconv(kernel_qtt, x_qtt)) ≈ _dense_cconv(kernel, x)

    kernel_q = QTTvector(kernel_qtt, 1, 3, :serial)
    x_q = QTTvector(x_qtt, 1, 3, :serial)
    @test circulant_qtto(kernel_q) isa QTToperator{Float64, 3}
    @test qtt_to_vector(qtt_cconv(kernel_q, x_q)) ≈ _dense_cconv(kernel, x)
end

@testset "qtt_conv" begin
    kernel = [1.0, 2.0, -1.0, 0.5]
    x = [3.0, -2.0, 0.25, 4.0]

    y_qtt = qtt_conv(_qtt_from_vector(kernel), _qtt_from_vector(x))

    @test y_qtt.N == 3
    @test y_qtt.ttv_dims == (2, 2, 2)
    @test qtt_to_vector(y_qtt) ≈ _dense_conv(kernel, x)

    y_q = qtt_conv(QTTvector(_qtt_from_vector(kernel), 1, 2, :serial),
                   QTTvector(_qtt_from_vector(x), 1, 2, :serial))
    @test y_q isa QTTvector{Float64, 3}
    @test y_q.bits_per_dim == 3
    @test qtt_to_vector(y_q) ≈ _dense_conv(kernel, x)
end

@testset "Inverse" begin

    d = 6
    A = Δ⁻¹_DN(d)

    function inv_DN(n::Int)
        @assert n ≥ 1 "n must be ≥ 1"
        G = Matrix{Float64}(undef, n, n)
        @inbounds for i in 1:n, j in 1:n
            G[i, j] = min(i, j)
        end
        return G
    end

    @test qtto_to_matrix(A) == inv_DN(2^d)

end

@testset "qtto_prolongation" begin

    d = 3
    P_qtt = qtto_prolongation(d)
    function prolongation_matrix(d::Int)
        @assert d ≥ 2 "d must be ≥ 2"
        n = 2^(d - 1)          # coarse grid size
        nf = 2n               # fine grid size
        P = zeros(Float64, nf, n)

        # first fine node (between boundary x=0 and first coarse node)
        P[1, 1] = 0.5

        # aligned fine nodes (even rows): pure injection
        @inbounds for k in 1:n
            P[2k, k] = 1.0
        end

        # interior midpoints (odd rows except the first): 1/2 on neighbors
        @inbounds for k in 1:(n - 1)
            P[2k + 1, k] += 0.5
            P[2k + 1, k + 1] += 0.5
        end

        return P
    end
    @test qtto_to_matrix(P_qtt)[1, 1] == prolongation_matrix(d)[1, 1]
    @test qtto_to_matrix(P_qtt)[1, 3] == prolongation_matrix(d)[1, 3]
    @test qtto_to_matrix(P_qtt)[1, 4] == prolongation_matrix(d)[1, 4]
    @test qtto_to_matrix(P_qtt)[2, 1] == prolongation_matrix(d)[2, 1]
end

@testset "qtto_constant_prolongation" begin
    function constant_prolongation_matrix(d::Int)
        n = 2^d
        Pmat = zeros(Float64, 2n, n)
        @inbounds for α in 0:(n - 1)
            Pmat[2α + 1, α + 1] = 1.0
            Pmat[2α + 2, α + 1] = 1.0
        end
        return Pmat
    end

    d = 3
    P = qtto_constant_prolongation(d)
    @test P isa TToperator{Float64, 4}
    @test P.N == d + 1
    @test P.tto_dims == ntuple(_ -> 2, d + 1)
    @test size(P.tto_vec[end], 2) == 1

    P_dense = constant_prolongation_matrix(d)
    for col in 1:2^d
        y = P * qtt_basis_vector(d, col)
        @test y isa TTvector{Float64, 4}
        @test y.ttv_dims == ntuple(_ -> 2, d + 1)
        @test qtt_to_function(y) ≈ P_dense[:, col]
    end

    u = function_to_qtt(x -> cos(π * x), d)
    @test qtt_to_function(P * u) ≈ P_dense * qtt_to_function(u)

    d2 = 2
    P2 = constant_prolongation_matrix(d2)
    u2 = function_to_qtt(x -> 1 + x, d2) ⊗ function_to_qtt(x -> 2 - x, d2)
    Py = id_tto(d2) ⊗ qtto_constant_prolongation(d2)
    uy = Py * u2
    @test qtt_to_function(uy) ≈ kron(Matrix(I, 2^d2, 2^d2), P2) * qtt_to_function(u2)

    Px = qtto_constant_prolongation(d2) ⊗ id_tto(d2 + 1)
    uxy = Px * uy
    @test qtt_to_function(uxy) ≈ kron(P2, Matrix(I, 2^(d2 + 1), 2^(d2 + 1))) * qtt_to_function(uy)
end

@testset "qtto_linear_prolongation" begin
    function linear_prolongation_matrix(d::Int)
        n = 2^d
        Pmat = zeros(Float64, 2n, n)
        @inbounds for α in 0:(n - 1)
            Pmat[2α + 1, α + 1] = 1.0
            Pmat[2α + 2, α + 1] += 0.5
            if α + 1 < n
                Pmat[2α + 2, α + 2] += 0.5
            end
        end
        return Pmat
    end

    d = 3
    P = qtto_linear_prolongation(d)
    @test P isa TToperator{Float64, 4}
    @test P.N == d + 1
    @test P.tto_dims == ntuple(_ -> 2, d + 1)
    @test size(P.tto_vec[end], 2) == 1

    P_dense = linear_prolongation_matrix(d)
    for col in 1:2^d
        e = qtt_basis_vector(d, col)
        y = P * e
        @test y isa TTvector{Float64, 4}
        @test y.ttv_dims == ntuple(_ -> 2, d + 1)
        @test qtt_to_function(y) ≈ P_dense[:, col]
    end

    u = function_to_qtt(x -> sin(π * x), d)
    @test qtt_to_function(P * u) ≈ P_dense * qtt_to_function(u)

    P₁ = qtto_linear_prolongation(1)
    @test qtt_to_function(P₁ * qtt_basis_vector(1, 2)) ≈ [0.0, 0.5, 1.0, 0.5]

    d2 = 2
    P2 = linear_prolongation_matrix(d2)
    u2 = function_to_qtt(x -> 1 + x, d2) ⊗ function_to_qtt(x -> 2 - x, d2)
    Py = id_tto(d2) ⊗ qtto_linear_prolongation(d2)
    uy = Py * u2
    @test qtt_to_function(uy) ≈ kron(Matrix(I, 2^d2, 2^d2), P2) * qtt_to_function(u2)

    Px = qtto_linear_prolongation(d2) ⊗ id_tto(d2 + 1)
    uxy = Px * uy
    @test qtt_to_function(uxy) ≈ kron(P2, Matrix(I, 2^(d2 + 1), 2^(d2 + 1))) * qtt_to_function(uy)
end
