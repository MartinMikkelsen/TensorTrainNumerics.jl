using LinearAlgebra
using Random
using TensorTrainNumerics

println("Complex-domain 5D Wishart Laplace transform via TT cross")

d = 5
nu = d + 2
p = nu / 2

Sigma = [
    1.0  0.3  0.2  0.1  0.18
    0.3  1.2  0.25 0.15 0.22
    0.2  0.25 0.9  0.2  0.28
    0.1  0.15 0.2  1.1  0.19
    0.18 0.22 0.28 0.19 1.05
]

@assert isposdef(Sigma)
println("Eigenvalues of Sigma: ", round.(eigvals(Sigma); sigdigits = 4))
println("Condition number: ", round(cond(Sigma); sigdigits = 4))

sigma = 2.0 .* Sigma

f_tilde_wishart(s::AbstractVector, sigma::AbstractMatrix, p::Real) =
    det(Matrix{ComplexF64}(I, length(s), length(s)) + sigma * Diagonal(s))^(-p)

function f_tilde_tt(X::AbstractMatrix{<:Number})
    y = Vector{ComplexF64}(undef, size(X, 1))
    for i in 1:size(X, 1)
        y[i] = f_tilde_wishart(@view(X[i, :]), sigma, p)
    end
    return y
end

n = 6
re_axis = collect(range(0.0, 1.0, length = n))
im_axis = collect(range(-0.35, 0.35, length = n))
domain = [re_axis .+ im .* im_axis for _ in 1:d]

Random.seed!(40260)
alg = MaxVol(verbose = true, tol = 6.0e-6, maxiter = 25, rmax = 70, kickrank = 2)
tt = tt_cross(f_tilde_tt, domain, alg; ranks = 2, val_size = 2000)
println("TT ranks: ", tt.ttv_rks)

Random.seed!(40261)
nsamp = 500
idx = hcat([rand(1:n, nsamp) for _ in 1:d]...)
Xcheck = hcat([domain[k][idx[:, k]] for k in 1:d]...)
y = f_tilde_tt(Xcheck)
yhat = TensorTrainNumerics._evaluate_tt(tt.ttv_vec, idx, d)

rel = norm(y .- yhat) / max(norm(y), 1.0e-14)
println("Validation relative L2 error: ", rel)
