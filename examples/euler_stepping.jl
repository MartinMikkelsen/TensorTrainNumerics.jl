using TensorTrainNumerics
using CairoMakie

cores = 8
h = 1/cores^2
A = h^2*toeplitz_to_qtto(-2,1.0,1.0,cores)

u₀ = qtt_sin(cores,λ=π)
init = rand_tt(u₀.ttv_dims, u₀.ttv_rks)
steps = collect(range(0.0,10.0,1000))
solution, rel_err = euler_method(A, u₀, steps; return_error=true)
Q = qtt_to_function(solution)
xes = collect(range(0.0, 1.0, 2^cores))

plot(xes,Q)

solution, rel_err = implicit_euler_method(A, u₀, init, steps; return_error=true)
plot(xes,qtt_to_function(solution))
