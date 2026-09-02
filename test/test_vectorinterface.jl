using Test
using LinearAlgebra
using KrylovKit
using Random
using TensorTrainNumerics
using VectorInterface

_dense(x::TTvector) = vec(ttv_to_tensor(x))

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

@testset "VectorInterface TTvector zeroing and scalar metadata" begin
    Random.seed!(101)
    dims = (2, 3, 2)
    ranks = [1, 2, 2, 1]
    x = rand_tt(dims, ranks)
    x_dense = _dense(x)

    z = VectorInterface.zerovector(x)
    z_complex = VectorInterface.zerovector(x, ComplexF64)
    z_base = zero(x)

    @test z !== x
    @test z.ttv_dims == dims
    @test z.ttv_rks == ranks
    @test z.ttv_rks !== x.ttv_rks
    @test all(iszero, _dense(z))
    @test z_complex isa TTvector{ComplexF64}
    @test z_complex.ttv_dims == dims
    @test z_complex.ttv_rks == ranks
    @test z_complex.ttv_rks !== x.ttv_rks
    @test all(iszero, _dense(z_complex))
    @test all(iszero, _dense(z_base))
    @test _dense(x) == x_dense
    @test VectorInterface.length(x) == prod(dims)
    @test VectorInterface.scalartype(x) === Float64
    @test VectorInterface.scalartype(typeof(x)) === Float64
    @test VectorInterface.inner(x, x) ≈ dot(x_dense, x_dense)

    z_mutating = copy(x)
    returned_mutating = VectorInterface.zerovector!(z_mutating)
    @test returned_mutating === z_mutating
    @test all(iszero, _dense(z_mutating))

    z_maybe_mutating = copy(x)
    returned_maybe_mutating = VectorInterface.zerovector!!(z_maybe_mutating)
    @test returned_maybe_mutating === z_maybe_mutating
    @test all(iszero, _dense(z_maybe_mutating))
end

@testset "VectorInterface TTvector addition variants" begin
    Random.seed!(102)
    dims = (2, 2, 3)
    ranks = [1, 2, 2, 1]
    x = rand_tt(dims, ranks)
    y = rand_tt(dims, ranks)
    x_dense = _dense(x)
    y_dense = _dense(y)
    α, β = 0.25, -1.5

    @test _dense(VectorInterface.add(y, x)) ≈ y_dense + x_dense
    @test _dense(VectorInterface.add(y, x, α)) ≈ y_dense + α * x_dense
    @test _dense(VectorInterface.add(y, x, α, β)) ≈ β * y_dense + α * x_dense
    @test _dense(x) == x_dense
    @test _dense(y) == y_dense

    y_add = copy(y)
    returned_add = VectorInterface.add!(y_add, x)
    @test _dense(y_add) ≈ y_dense + x_dense
    @test _dense(returned_add) ≈ y_dense + x_dense

    y_weighted = copy(y)
    returned_weighted = VectorInterface.add!(y_weighted, x, α, β)
    @test _dense(y_weighted) ≈ β * y_dense + α * x_dense
    @test _dense(returned_weighted) ≈ β * y_dense + α * x_dense

    y_maybe = copy(y)
    returned_maybe = VectorInterface.add!!(y_maybe, x)
    @test _dense(y_maybe) ≈ y_dense + x_dense
    @test _dense(returned_maybe) ≈ y_dense + x_dense

    y_maybe_scaled = copy(y)
    returned_maybe_scaled = VectorInterface.add!!(y_maybe_scaled, x, α)
    @test _dense(y_maybe_scaled) ≈ y_dense + α * x_dense
    @test _dense(returned_maybe_scaled) ≈ y_dense + α * x_dense

    y_maybe_weighted = copy(y)
    returned_maybe_weighted = VectorInterface.add!!(y_maybe_weighted, x, α, β)
    @test _dense(y_maybe_weighted) ≈ β * y_dense + α * x_dense
    @test _dense(returned_maybe_weighted) ≈ β * y_dense + α * x_dense
    @test _dense(x) == x_dense
end

@testset "VectorInterface TTvector scaling variants" begin
    Random.seed!(103)
    dims = (2, 3, 2)
    ranks = [1, 2, 2, 1]
    x = rand_tt(dims, ranks)
    x_dense = _dense(x)

    scaled = VectorInterface.scale(x, -0.5)
    @test _dense(scaled) ≈ -0.5 * x_dense
    @test _dense(x) == x_dense

    scaled_mutating = copy(x)
    @test VectorInterface.scale!(scaled_mutating, 2.0) === scaled_mutating
    @test _dense(scaled_mutating) ≈ 2.0 * x_dense

    scaled_maybe_mutating = copy(x)
    returned_maybe_mutating = VectorInterface.scale!!(scaled_maybe_mutating, 0.75)
    @test _dense(scaled_maybe_mutating) ≈ 0.75 * x_dense
    @test _dense(returned_maybe_mutating) ≈ 0.75 * x_dense

    scaled_promoted = VectorInterface.scale!!(copy(x), 0.5im)
    @test scaled_promoted isa TTvector{ComplexF64}
    @test _dense(scaled_promoted) ≈ 0.5im * x_dense

    destination = rand_tt(dims, ranks)
    returned_destination = VectorInterface.scale!!(destination, x, -1.25)
    @test _dense(destination) ≈ -1.25 * x_dense
    @test _dense(returned_destination) ≈ -1.25 * x_dense

    promoted_destination = rand_tt(dims, ranks)
    promoted_destination_dense = _dense(promoted_destination)
    returned_promoted_destination = VectorInterface.scale!!(promoted_destination, x, 0.5im)
    @test returned_promoted_destination isa TTvector{ComplexF64}
    @test _dense(returned_promoted_destination) ≈ 0.5im * x_dense
    @test _dense(promoted_destination) == promoted_destination_dense
    @test _dense(x) == x_dense
end

@testset "VectorInterface TToperator addition and scalar type" begin
    Random.seed!(104)
    A = rand_tto((2, 3), 2)
    B = rand_tto((2, 3), 2)
    A_dense = tto_to_tensor(A)
    B_dense = tto_to_tensor(B)

    C = VectorInterface.add(A, B)

    @test C isa TToperator{Float64, 2}
    @test tto_to_tensor(C) ≈ A_dense + B_dense
    @test tto_to_tensor(A) == A_dense
    @test tto_to_tensor(B) == B_dense
    @test VectorInterface.scalartype(A) === Float64
    @test VectorInterface.scalartype(typeof(A)) === Float64
    @test VectorInterface.scalartype(complex(A)) === ComplexF64
end

@testset "VectorInterface TToperator zero and length" begin
    A = rand_tto((2, 3), 2)
    z = VectorInterface.zerovector(A)
    z_complex = VectorInterface.zerovector(A, ComplexF64)

    @test z isa TToperator{Float64, 2}
    @test VectorInterface.length(A) == prod(A.tto_dims)^2
    if z isa TToperator
        @test z.tto_dims == A.tto_dims
        @test z.tto_rks == A.tto_rks
        @test z.tto_rks !== A.tto_rks
        @test all(iszero, tto_to_tensor(z))
    end
    @test z_complex isa TToperator{ComplexF64, 2}
    @test z_complex.tto_dims == A.tto_dims
    @test z_complex.tto_rks == A.tto_rks
    @test z_complex.tto_rks !== A.tto_rks
    @test all(iszero, tto_to_tensor(z_complex))
end

@testset "rank-bounded VectorInterface contracts" begin
    Random.seed!(105)
    dims = (2, 2, 2)
    ranks = ones(Int, 4)
    x = rand_tt(dims, ranks)
    y = rand_tt(dims, ranks)
    x_dense = _dense(x)
    y_dense = _dense(y)
    bounded_x = TensorTrainNumerics._RankBoundedTTvector(x, 2)
    bounded_y = TensorTrainNumerics._RankBoundedTTvector(y, 2)

    z = VectorInterface.zerovector(bounded_x, ComplexF64)
    @test z.max_bond == 2
    @test eltype(z.tt) === ComplexF64
    @test all(iszero, _dense(z.tt))
    @test z.tt.ttv_rks == bounded_x.tt.ttv_rks
    @test z.tt.ttv_rks !== bounded_x.tt.ttv_rks

    z_mutating = TensorTrainNumerics._RankBoundedTTvector(copy(x), 2)
    @test VectorInterface.zerovector!(z_mutating) === z_mutating
    @test all(iszero, _dense(z_mutating.tt))

    z_maybe_mutating = TensorTrainNumerics._RankBoundedTTvector(copy(x), 2)
    @test VectorInterface.zerovector!!(z_maybe_mutating) === z_maybe_mutating
    @test all(iszero, _dense(z_maybe_mutating.tt))

    scaled = VectorInterface.scale(bounded_x, 0.5)
    @test scaled.max_bond == 2
    @test _dense(scaled.tt) ≈ 0.5 * x_dense

    scaled_mutating = TensorTrainNumerics._RankBoundedTTvector(copy(x), 2)
    @test VectorInterface.scale!(scaled_mutating, -2.0) === scaled_mutating
    @test _dense(scaled_mutating.tt) ≈ -2.0 * x_dense

    scaled_promoted = VectorInterface.scale!!(
        TensorTrainNumerics._RankBoundedTTvector(copy(x), 2), 0.5im
    )
    @test scaled_promoted.max_bond == 2
    @test eltype(scaled_promoted.tt) === ComplexF64
    @test _dense(scaled_promoted.tt) ≈ 0.5im * x_dense

    scale_destination = TensorTrainNumerics._RankBoundedTTvector(copy(y), 2)
    scale_result = VectorInterface.scale!!(scale_destination, bounded_x, 1.25)
    @test scale_result.max_bond == 2
    @test _dense(scale_destination.tt) ≈ 1.25 * x_dense
    @test _dense(scale_result.tt) ≈ 1.25 * x_dense

    promoted_scale_destination = TensorTrainNumerics._RankBoundedTTvector(copy(y), 2)
    promoted_scale_destination_dense = _dense(promoted_scale_destination.tt)
    promoted_scale_result = VectorInterface.scale!!(
        promoted_scale_destination, bounded_x, 0.5im
    )
    @test promoted_scale_result.max_bond == 2
    @test promoted_scale_result.tt isa TTvector{ComplexF64}
    @test _dense(promoted_scale_result.tt) ≈ 0.5im * x_dense
    @test _dense(promoted_scale_destination.tt) == promoted_scale_destination_dense

    added = VectorInterface.add(bounded_y, bounded_x, 0.25, -0.5)
    @test added.max_bond == 2
    @test maximum(added.tt.ttv_rks) <= 2
    @test _dense(added.tt) ≈ -0.5 * y_dense + 0.25 * x_dense

    add_destination = TensorTrainNumerics._RankBoundedTTvector(copy(y), 2)
    add_result = VectorInterface.add!!(add_destination, bounded_x, 0.25, -0.5)
    @test add_result.max_bond == 2
    @test maximum(add_result.tt.ttv_rks) <= 2
    @test _dense(add_result.tt) ≈ -0.5 * y_dense + 0.25 * x_dense

    @test VectorInterface.inner(bounded_x, bounded_y) ≈ dot(x_dense, y_dense)
    @test VectorInterface.norm(bounded_x) ≈ norm(x_dense)
    @test VectorInterface.scalartype(typeof(bounded_x)) === Float64

    incompatible = TensorTrainNumerics._RankBoundedTTvector(copy(y), 3)
    @test_throws ArgumentError VectorInterface.inner(bounded_x, incompatible)
    @test_throws ArgumentError VectorInterface.scale!(incompatible, bounded_x, 1.0)
    @test_throws ArgumentError VectorInterface.scale!!(incompatible, bounded_x, 1.0)
    @test_throws ArgumentError VectorInterface.add(bounded_x, incompatible, 1.0, 1.0)
    @test_throws ArgumentError VectorInterface.add!(bounded_x, incompatible, 1.0, 1.0)
    @test_throws ArgumentError VectorInterface.add!!(bounded_x, incompatible, 1.0, 1.0)
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

    z0 = VectorInterface.add!!(copy(y), x)
    expected0 = qtt_to_vector(y) + qtt_to_vector(x)
    @test z0 isa TTvector{ComplexF64}
    @test norm(qtt_to_vector(z0) - expected0) / norm(expected0) < 1.0e-12

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
