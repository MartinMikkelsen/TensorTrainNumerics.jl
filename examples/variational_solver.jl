using TensorTrainNumerics
using OptimKit
using KrylovKit

d = 6
N = 2^d
h = 1 / (N - 1)
Δ = toeplitz_to_qtto(2.0, -1.0, -1.0, d)
kappa = 0.1
A = kappa * Δ

f = qtt_sin(d, λ = π)

function fg(u::TTvector)
    û = orthogonalize(u)
    Au = A * û
    val = 0.5 * real(dot(û, Au)) - real(dot(f, û))
    grad = Au - f
    return val, grad
end

x0 = rand_tt(f.ttv_dims, f.ttv_rks)

method = GradientDescent()
x, fx, gx, numfg, normgradhistor = optimize(fg, x0, method)

relres = norm(A * x - f) / max(norm(f), eps())
println("relative residual = ", relres)

x_mals = mals_linsolve(A, f, x0)
relres = norm(A * x_mals - f) / max(norm(f), eps())
println("relative residual = ", relres)

x_krylov, info = linsolve(A, f, x0)
relres = norm(A * x_krylov - f) / max(norm(f), eps())
println("relative residual = ", relres)
