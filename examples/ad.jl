using TensorTrainNumerics
using DifferentiationInterface, ForwardDiff

d = 8

A1 = qtt_exp(d)
A2 = qtt_sin(d, λ = π)
A3 = qtt_cos(d, λ = π)
A4 = qtt_polynom([0.0, 2.0, 3.0, -8.0, -5.0], d; a = 0.0, b = 1.0)

normsq(tt) = real(TensorTrainNumerics.dot(tt, tt))

grads = tt_gradient(normsq, A1, AutoForwardDiff())

