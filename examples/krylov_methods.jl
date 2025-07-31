using TensorTrainNumerics
using CairoMakie
using KrylovKit
using OptimKit

cores = 6
h = 1 / cores^2
A = h^2 * toeplitz_to_qtto(-2, 1.0, 1.0, cores)
xes = collect(range(0.0, 1.0, 2^cores))

u₀ = qtt_sin(cores, λ = π)
init = rand_tt(u₀.ttv_dims, u₀.ttv_rks)
steps = collect(range(0.0, 10.0, 1000))

solution_krylov, info = expintegrator(A, 1.0, init,eager = true)

dims = (2, 2, 2, 2)  # a simple 4D TT shape
ranks = [1, 2, 2, 2, 1]
b = rand_tt(Float64, dims, ranks)  # the "target" vector

# 2. Starting point: zero vector
x0 = zeros_tt(Float64, dims, ranks)

function cost_with_grad(x)
    fx = 0.5 * dot(x, x)
    gx = x
    return fx, gx
end

solver = GradientDescent()

res = optimize(cost, x0, solver)

