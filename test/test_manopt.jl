using ManifoldsBase
using Manopt

@testset "ManOpt extension" begin
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

    @test norm(solution - target) / norm(target) < 1.0e-10
end
