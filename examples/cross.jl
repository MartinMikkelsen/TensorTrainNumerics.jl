using LinearAlgebra
using Random
using CairoMakie
using TensorTrainNumerics

function sin_6d(coords::Matrix{Float64})
    return vec(sin.(sum(coords, dims = 2)))
end

n = 8
d = 6

domain = [collect(range(0.0, π, length = n)) for _ in 1:d]

tt = tt_cross(sin_6d, domain; ranks_tt = 12, eps = 1.0e-15, max_iter = 50, verbose = true)



println("1. Simple 1D integral: ∫₀¹ x² dx = 1/3")
f1(x) = x[:, 1] .^ 2
result1 = tt_integrate(f1, [(0.0, 1.0)]; nquad = 8, verbose = false)
println("   Result: $result1")
println("   Exact:  $(1 / 3)\n")

println("2. 2D integral: ∫₀¹∫₀¹ xy dx dy = 1/4")
f2(x) = x[:, 1] .* x[:, 2]
result2 = tt_integrate(f2, [(0.0, 1.0), (0.0, 1.0)]; nquad = 8, verbose = false)
println("   Result: $result2")
println("   Exact:  $(1 / 4)\n")

f3(x) = sin.(sum(x, dims = 2))

result3 = tt_integrate(f3, 6, (0.0, 1.0); nquad = 16, verbose = false)
exact3 = imag((exp(im) - 1)^6 / im^6)


println("4. High-dimensional Gaussian: ∫[-5,5]^d exp(-||x||²) dx")
for d in [10, 20, 50]
    f(x) = exp.(-sum(x .^ 2, dims = 2))
    result = tt_integrate(f, d, (-5.0, 5.0); nquad = 32, ranks_tt = 2, eps = 1.0e-10, verbose = false)
    exact = π^(d / 2)
    println("   d=$d: Result=$(round(result, digits = 8)), Exact=$(round(exact, digits = 8)), RelErr=$(abs(result - exact) / exact)")
end

function Q(x)
    r2 = sum(x .^ 2, dims = 2)
    s = sum(x, dims = 2)
    return vec(1.0e3 .* cos.(10 .* r2) .* exp.(-s .^ 4 ./ 1.0e3))
end

result5 = tt_integrate(Q, 10, (-1.0, 1.0); nquad = 50, verbose = true, eps = 1.0e-12)

tt_maxvol = tt_cross(sin_6d, domain, MaxVol(tol=1e-8, maxiter=20, verbose=true); ranks=4);

tt_dmrg = tt_cross(sin_6d, domain, DMRG(tol=1e-8, maxiter=20, verbose=true); ranks=4);

tt_greedy = tt_cross(sin_6d, domain, Greedy(tol=1e-8, verbose=true, maxiter=100));


println("MaxVol ranks: ", tt_maxvol.ttv_rks)
println("DMRG ranks: ", tt_dmrg.ttv_rks)
println("Greedy ranks: ", tt_dmrg.ttv_rks)


println("\nResulting TT ranks: $(tt_dmrg.ttv_rks)")

println("\nConverting TT back to full tensor...")
tensor_approx = ttv_to_tensor(tt_dmrg);

println("Building reference tensor...")
tensor_exact = zeros(Float64, ntuple(_ -> n, d));
for idx in CartesianIndices(tensor_exact)
    coords = [domain[k][idx[k]] for k in 1:d]
    tensor_exact[idx] = sin(sum(coords))
end

error = norm(tensor_approx - tensor_exact) / norm(tensor_exact)
println("\nRelative error: $error")

max_error = maximum(abs.(tensor_approx - tensor_exact))
println("Maximum absolute error: $max_error")

println("\nSpot checks at random indices:")
for _ in 1:5
    idx = Tuple(rand(1:n) for _ in 1:d)
    coords = [domain[k][idx[k]] for k in 1:d]
    exact_val = sin(sum(coords))
    approx_val = tensor_approx[idx...]
    println("  Index $idx: exact=$(round(exact_val, digits = 8)), approx=$(round(approx_val, digits = 8)), diff=$(abs(exact_val - approx_val))")
end