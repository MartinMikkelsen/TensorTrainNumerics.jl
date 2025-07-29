using TensorTrainNumerics
using CairoMakie
using KrylovKit

d = 2

A = Δ(d)
x = qtt_sin(d, λ = π)

exponentiate(A,0.1,x,tol = 1e-2,verbosity=3)
linsolve(A, x, verbosity=3)