using TensorTrainNumerics
using OptimKit
using KrylovKit

d = 6
N = 2^d
h = 1 / (N - 1)
kappa = 0.1

Δ = toeplitz_to_qtto(2.0, -1.0, -1.0, d)   # dimensionless FD Laplacian
A = (kappa / h^2) * Δ                        # scaled: A ≈ κ d²/dx²

f = qtt_sin(d, λ = π)

x0 = rand_tt(f.ttv_dims, f.ttv_rks)

relative_residual(A, x, f) = norm(A * x - f) / max(norm(f), eps())

function fg(u::TTvector)
    û = orthogonalize(u)
    Au = A * û
    val = 0.5 * real(dot(û, Au)) - real(dot(f, û))
    grad = tt_compress!(Au - f, maximum(û.ttv_rks))   # bound rank growth
    return val, grad
end

solvers = [
    ("Gradient descent (OptimKit)",
        () -> optimize(fg, x0, GradientDescent())[1]),
    ("MALS",
        () -> mals_linsolve(A, f, x0)),
    ("KrylovKit linsolve",
        () -> linsolve(A, f, x0)[1]),
]

for (label, solve) in solvers
    x = solve()
    rr = relative_residual(A, x, f)
    println("$label: relative residual = $rr")
end
