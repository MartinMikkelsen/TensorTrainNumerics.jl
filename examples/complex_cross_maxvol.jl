using LinearAlgebra
using Random
using TensorTrainNumerics

println("Complex-domain TT cross (MaxVol)")

d = 3
n = 7
base = collect(range(0.0, 1.2, length = n))
imag_axis = collect(range(-0.4, 0.4, length = n))
domain = [base .+ im .* imag_axis for _ in 1:d]

f(X::AbstractMatrix{<:Number}) = vec(exp.(X[:, 1] .+ 0.6 .* X[:, 2] .- 0.2 .* X[:, 3]))

Random.seed!(20260)
alg = MaxVol(verbose = true, tol = 1.0e-8, maxiter = 20, rmax = 40, kickrank = 2)
tt = tt_cross(f, domain, alg; ranks = 2, val_size = 1200)
println("TT ranks: ", tt.ttv_rks)

Random.seed!(20261)
nsamp = 400
idx = hcat([rand(1:n, nsamp) for _ in 1:d]...)
Xcheck = hcat([domain[k][idx[:, k]] for k in 1:d]...)
y = f(Xcheck)
yhat = TensorTrainNumerics._evaluate_tt(tt.ttv_vec, idx, d)

rel = norm(y .- yhat) / max(norm(y), 1.0e-14)
println("Validation relative L2 error: ", rel)
