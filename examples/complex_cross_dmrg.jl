using LinearAlgebra
using Random
using TensorTrainNumerics

println("Complex-domain TT cross (DMRG)")

d = 4
n = 6
t = collect(range(-1.0, 1.0, length = n))
domain = [0.4 .* (k .* t) .+ im .* (0.3 .* t) for k in 1:d]

function f(X::AbstractMatrix{<:Number})
    z = X[:, 1] .+ 1.7 .* X[:, 2]
    return vec(sin.(z) .* exp.(-0.4 .* X[:, 3]) .+ 0.3 ./ (1 .+ X[:, 4]))
end

Random.seed!(30260)
alg = DMRG(verbose = true, tol = 1.0e-8, maxiter = 18, rmax = 50)
tt = tt_cross(f, domain, alg; ranks = 2, val_size = 1200)
println("TT ranks: ", tt.ttv_rks)

Random.seed!(30261)
nsamp = 400
idx = hcat([rand(1:n, nsamp) for _ in 1:d]...)
Xcheck = hcat([domain[k][idx[:, k]] for k in 1:d]...)
y = f(Xcheck)
yhat = TensorTrainNumerics._evaluate_tt(tt.ttv_vec, idx, d)

rel = norm(y .- yhat) / max(norm(y), 1.0e-14)
println("Validation relative L2 error: ", rel)
