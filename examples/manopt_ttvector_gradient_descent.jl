using ManifoldsBase
using Manopt
using TensorTrainNumerics

d = 6
target = qtt_sin(d)
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
    stopping_criterion = StopWhenGradientNormLess(1.0e-5) | StopAfterIteration(5),
    debug = [],
)

relative_error = norm(solution - target) / norm(target)
final_cost = cost(M, solution)

@info "ManOpt TTvector gradient descent" relative_error final_cost ranks = solution.ttv_rks
