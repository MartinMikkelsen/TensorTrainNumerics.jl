using ManifoldsBase
using Manopt

@testset "TTVectorSpace construction" begin
    target = orthogonalize(qtt_sin(4))
    M = ttvector_manifold(target)
    @test M isa ManifoldsBase.AbstractManifold
    @test M.dims == target.ttv_dims
    @test M.ranks == target.ttv_rks
    @test M.ranks !== target.ttv_rks  # copy, not alias
end

@testset "representation_size" begin
    target = orthogonalize(qtt_sin(3))
    M = ttvector_manifold(target)
    @test ManifoldsBase.representation_size(M) == target.ttv_dims
end

@testset "default_retraction_method" begin
    target = orthogonalize(qtt_sin(3))
    M = ttvector_manifold(target)
    @test ManifoldsBase.default_retraction_method(M) isa ManifoldsBase.ProjectionRetraction
    @test ManifoldsBase.default_retraction_method(M, Float64) isa ManifoldsBase.ProjectionRetraction
end

@testset "allocate_result / zero_vector" begin
    target = orthogonalize(qtt_sin(4))
    M = ttvector_manifold(target)
    z = ManifoldsBase.zero_vector(M, target)
    @test z isa TTvector
    @test z.ttv_dims == target.ttv_dims
    @test z.ttv_rks == target.ttv_rks
    for core in z.ttv_vec
        @test all(iszero, core)
    end
end

@testset "zero_vector!" begin
    target = orthogonalize(qtt_sin(4))
    M = ttvector_manifold(target)
    X = TensorTrainNumerics.copy(target)
    ManifoldsBase.zero_vector!(M, X, target)
    for core in X.ttv_vec
        @test all(iszero, core)
    end
end

@testset "copy and copyto!" begin
    target = orthogonalize(qtt_sin(4))
    M = ttvector_manifold(target)

    p_copy = ManifoldsBase.copy(M, target)
    @test p_copy isa TTvector
    @test norm(p_copy - target) < 1e-14

    q = zeros_tt(eltype(target), target.ttv_dims, target.ttv_rks)
    ManifoldsBase.copyto!(M, q, target)
    @test norm(q - target) < 1e-14

    Y = zeros_tt(eltype(target), target.ttv_dims, target.ttv_rks)
    ManifoldsBase.copyto!(M, Y, target, target)
    @test norm(Y - target) < 1e-14
end

@testset "inner, norm, distance" begin
    p = orthogonalize(qtt_sin(4))
    q = orthogonalize(qtt_sin(4, λ = 2π))
    M = ttvector_manifold(p)

    inner_val = ManifoldsBase.inner(M, p, p, p)
    @test inner_val ≈ real(TensorTrainNumerics.dot(p, p))
    @test inner_val ≥ 0

    norm_val = ManifoldsBase.norm(M, p, p)
    @test norm_val ≈ sqrt(inner_val)

    dist = ManifoldsBase.distance(M, p, q)
    @test dist ≈ ManifoldsBase.norm(M, p, p - q)
    @test dist ≥ 0

    # distance to self is zero
    @test ManifoldsBase.distance(M, p, p) < 1e-14
end

@testset "retract_project! and retract_project_fused!" begin
    p = orthogonalize(qtt_sin(4))
    X = TensorTrainNumerics.copy(p)          # tangent = p itself for simplicity
    M = ttvector_manifold(p)

    q1 = zeros_tt(eltype(p), p.ttv_dims, p.ttv_rks)
    ManifoldsBase.retract_project!(M, q1, p, X)
    expected = orthogonalize(p + X)
    @test norm(q1 - expected) < 1e-12

    q2 = zeros_tt(eltype(p), p.ttv_dims, p.ttv_rks)
    t = 0.5
    ManifoldsBase.retract_project_fused!(M, q2, p, X, t)
    expected2 = orthogonalize(p + t * X)
    @test norm(q2 - expected2) < 1e-12

    # t=0 should give orthogonalize(p)
    q3 = zeros_tt(eltype(p), p.ttv_dims, p.ttv_rks)
    ManifoldsBase.retract_project_fused!(M, q3, p, X, 0.0)
    @test norm(q3 - orthogonalize(p)) < 1e-12
end

@testset "ManOpt gradient descent" begin
    target = orthogonalize(qtt_sin(4))
    start = zeros_tt(eltype(target), target.ttv_dims, target.ttv_rks)
    M = ttvector_manifold(target)

    cost(::typeof(M), x) = 0.5 * norm(x - target)^2
    gradient(::typeof(M), x) = x - target

    solution = gradient_descent(
        M,
        cost,
        gradient,
        start;
        stepsize = ConstantLength(1.0; type = :relative),
        stopping_criterion = StopAfterIteration(3),
        debug = [],
    )

    # Accuracy floor is ~1e-8 (retraction/projection roundoff), and whether a given
    # iteration lands exactly on the target is platform-dependent FP luck — so the
    # tolerance must sit safely above that floor rather than at 1e-10.
    @test norm(solution - target) / norm(target) < 1.0e-6
end
