using Test
using LinearAlgebra
using KrylovKit
using Random
using TensorTrainNumerics
using VectorInterface

mutable struct BlockingIdentityOperator <: AbstractTToperator
    entered::Channel{Nothing}
    release::Channel{Nothing}
    blocked::Bool
end

function Base.:*(A::BlockingIdentityOperator, x::TTvector)
    if !A.blocked
        A.blocked = true
        put!(A.entered, nothing)
        take!(A.release)
    end
    return copy(x)
end

@testset "VectorInterface TToperator zero and length" begin
    A = rand_tto((2, 3), 2)
    z = VectorInterface.zerovector(A)

    @test z isa TToperator{Float64, 2}
    @test VectorInterface.length(A) == prod(A.tto_dims)^2
    if z isa TToperator
        @test z.tto_dims == A.tto_dims
        @test z.tto_rks == A.tto_rks
        @test z.tto_rks !== A.tto_rks
        @test all(iszero, tto_to_tensor(z))
    end
end

@testset "Complex TT dot and norm" begin
    d = 4
    u = complex(qtt_sin(d, λ = π))
    w = (1im) * complex(id_tto(d)) * u
    w_dense = qtt_to_vector(w)

    @test isapprox(TensorTrainNumerics.dot(w, w), LinearAlgebra.dot(w_dense, w_dense); atol = 1.0e-12)
    @test isapprox(norm(w), norm(w_dense); atol = 1.0e-12)
end

@testset "Complex Krylov exponential action" begin
    d = 4
    u₀ = complex(qtt_sin(d, λ = π))
    A = (1im) * complex(id_tto(d))

    y, info = expintegrator(A, 0.7, u₀)

    expected = exp(0.7im) .* qtt_to_vector(u₀)
    actual = qtt_to_vector(y)

    @test info.converged == 1
    @test norm(actual - expected) / norm(expected) < 1.0e-12
end

@testset "VectorInterface add!! promotes when destination eltype is too narrow" begin
    d = 3
    y = qtt_sin(d)
    x = complex(qtt_cos(d))
    α = 0.25 + 0.5im
    β = 1.5

    z1 = VectorInterface.add!!(copy(y), x, α)
    expected1 = qtt_to_vector(y) + α * qtt_to_vector(x)
    @test z1 isa TTvector{ComplexF64}
    @test norm(qtt_to_vector(z1) - expected1) / norm(expected1) < 1.0e-12

    z2 = VectorInterface.add!!(copy(y), x, α, β)
    expected2 = β * qtt_to_vector(y) + α * qtt_to_vector(x)
    @test z2 isa TTvector{ComplexF64}
    @test norm(qtt_to_vector(z2) - expected2) / norm(expected2) < 1.0e-12
end

@testset "bounded Krylov rounding is isolated from unrelated vector operations" begin
    Random.seed!(2718)
    dims = (2, 2, 2, 2)
    rank_one = ones(Int, 5)
    entered = Channel{Nothing}(1)
    release = Channel{Nothing}(1)
    A = BlockingIdentityOperator(entered, release, false)
    b = rand_tt(dims, rank_one)
    guess = zeros_tt(Float64, dims, rank_one)
    solve_task = @async linear_solve(
        A,
        b,
        guess,
        Krylov(
            max_bond = 1,
            krylov_solver = :gmres,
            krylovdim = 3,
            maxiter = 5,
            tol = 1.0e-12,
        ),
    )

    take!(entered)
    try
        x = rand_tt(dims, [1, 2, 2, 2, 1])
        y = rand_tt(dims, [1, 2, 2, 2, 1])
        z = VectorInterface.add(x, y)
        expected = vec(ttv_to_tensor(x)) + vec(ttv_to_tensor(y))
        actual = vec(ttv_to_tensor(z))

        @test norm(actual - expected) / norm(expected) < 1.0e-12
    finally
        put!(release, nothing)
    end

    solution = fetch(solve_task)
    @test maximum(solution.ttv_rks) <= 1
end

@testset "bounded Krylov vector operations support every linear solver" begin
    Random.seed!(314)
    dims = (2, 2, 2)
    ranks = [1, 2, 2, 1]
    A = id_tto(3)
    b = rand_tt(dims, ranks)
    guess = zeros_tt(Float64, dims, ranks)
    expected = vec(ttv_to_tensor(b))

    for solver in (:cg, :gmres, :bicgstab)
        x = linear_solve(
            A,
            b,
            guess,
            Krylov(
                max_bond = 2,
                krylov_solver = solver,
                krylovdim = 4,
                maxiter = 5,
                tol = 1.0e-12,
                issymmetric = true,
                isposdef = true,
            ),
        )
        actual = vec(ttv_to_tensor(x))

        @test norm(actual - expected) / norm(expected) < 1.0e-12
        @test maximum(x.ttv_rks) <= 2
    end
end

@testset "rank-bounded mutating operations update their destination" begin
    Random.seed!(1618)
    dims = (2, 2, 2)
    ranks = [1, 2, 2, 1]
    source = rand_tt(dims, ranks)
    destination = rand_tt(dims, ranks)
    destination_before = vec(ttv_to_tensor(destination))
    source_dense = vec(ttv_to_tensor(source))
    bounded_source = TensorTrainNumerics._RankBoundedTTvector(source, 2)
    bounded_destination = TensorTrainNumerics._RankBoundedTTvector(destination, 2)

    added = VectorInterface.add!(bounded_destination, bounded_source, 0.25, 0.5)

    @test added.tt === bounded_destination.tt
    @test vec(ttv_to_tensor(bounded_destination.tt)) ≈ 0.5 * destination_before + 0.25 * source_dense
    @test maximum(bounded_destination.tt.ttv_rks) <= 2

    scaled_destination = zeros_tt(Float64, dims, ranks)
    bounded_scaled = TensorTrainNumerics._RankBoundedTTvector(scaled_destination, 2)
    scaled = VectorInterface.scale!(bounded_scaled, bounded_source, 1.5)

    @test scaled.tt === bounded_scaled.tt
    @test vec(ttv_to_tensor(bounded_scaled.tt)) ≈ 1.5 * source_dense
    @test maximum(bounded_scaled.tt.ttv_rks) <= 2
end
