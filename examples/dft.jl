using TensorTrainNumerics
using Random

d = 10
N = 2^d
K = 50
sign = -1.0
normalize = true

Random.seed!(1234)
r = 12
coeffs = randn(r) .+ 1im * randn(r)

f(x) = sum(coeffs .* cispi.(2 .* (0:(r - 1)) .* x))

F = fourier_qtto(d; K = K, sign = -1.0, normalize = true)
x_qtt = function_to_qtt_uniform(f, d)
y_qtt = F * x_qtt

spec = matricize((y_qtt), d)
scale = sqrt(N)

@assert norm(spec[1:r] .- scale .* coeffs) / (scale * norm(coeffs)) < 1.0e-8
@assert norm(spec[(r + 1):end]) / norm(spec) < 1.0e-10
