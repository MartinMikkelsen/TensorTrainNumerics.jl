using LinearAlgebra
using Random
using Statistics
using TensorTrainNumerics

# Parameters
d = 5
nu = d + 2
p = nu / 2

Σ = [
    1.0  0.3  0.2  0.1  0.18
    0.3  1.2  0.25 0.15 0.22
    0.2  0.25 0.9  0.2  0.28
    0.1  0.15 0.2  1.1  0.19
    0.18 0.22 0.28 0.19 1.05
]

@assert isposdef(Σ)
println("Eigenvalues of Σ: ", round.(eigvals(Σ); sigdigits = 4))
println("Condition number: ", round(cond(Σ); sigdigits = 4))

Σ = 2.0 .* Σ

f_tilde_wishart(s::AbstractVector, Σ::AbstractMatrix, p::Real) =
    det(Matrix{Float64}(I, length(s), length(s)) + Σ * Diagonal(s))^(-p)

f_tilde_w(s1, s2, s3, s4, s5) = f_tilde_wishart([s1, s2, s3, s4, s5], Σ, p)

function f_tilde_tt(X::AbstractMatrix{<:Real})
    y = Vector{Float64}(undef, size(X, 1))
    for i in 1:size(X, 1)
        y[i] = f_tilde_wishart(@view(X[i, :]), Σ, p)
    end
    return y
end

n = 12
domain = [collect(range(0.0, 10.0, length = n)) for _ in 1:d]

Random.seed!(2026)
alg = DMRG(verbose = true, tol = 1.0e-12)
tt = tt_cross(f_tilde_tt, domain, alg)

println("TT ranks: ", tt.ttv_rks)

Random.seed!(2027)
ncheck = 500
idx = hcat([rand(1:n, ncheck) for _ in 1:d]...)
Xcheck = hcat([domain[k][idx[:, k]] for k in 1:d]...)
ytrue = f_tilde_tt(Xcheck)
yhat = TensorTrainNumerics._evaluate_tt(tt.ttv_vec, idx, d)

rel_l2 = norm(ytrue .- yhat) / max(norm(ytrue), 1.0e-14)
pointwise_rel = abs.(ytrue .- yhat) ./ max.(abs.(ytrue), 1.0e-14)

println("Validation rel-L2 error: ", rel_l2)
println("Pointwise relative error (median / max): ", median(pointwise_rel), " / ", maximum(pointwise_rel))
