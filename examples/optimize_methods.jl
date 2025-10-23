using TensorTrainNumerics
using CairoMakie
using KrylovKit
using OptimKit
import TensorTrainNumerics: dot
d = 6
h = 1 / d^2
A = h^2 * toeplitz_to_qtto(-2, 1.0, 1.0, d)
xes = collect(range(0.0, 1.0, 2^d))

u₀ = qtt_sin(d, λ = π)
init = rand_tt(u₀.ttv_dims, u₀.ttv_rks)
steps = collect(range(0.0, 10.0, 1000))

solution_krylov, info = expintegrator(A, 1.0, init, eager = true)

dims = (2, 2, 2, 2)
ranks = [1, 2, 2, 2, 1]

function cost_with_grad(x)
    fx = 0.5 * dot(x, x) - 0.5 * dot(u₀, x)
    gx = x
    return fx, gx
end

solver = ConjugateGradient(flavor = PolakRibiere(), verbosity = 3, gradtol = 1.0e-12)

x, fx, gx, numfg, normgradhistor = optimize(cost_with_grad, u₀, solver)
