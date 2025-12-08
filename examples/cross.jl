using LinearAlgebra
using Random
using Maxvol
using CairoMakie

function sin_6d(coords::Matrix{Float64})
    return vec(sin.(sum(coords, dims = 2)))
end

n = 8
d = 6

domain = [collect(range(0.0, π, length = n)) for _ in 1:d]

tt = tt_cross(sin_6d, domain; ranks_tt = 12, eps = 1.0e-15, max_iter = 50, verbose = true)

println("\nResulting TT ranks: $(tt.ttv_rks)")

println("\nConverting TT back to full tensor...")
tensor_approx = ttv_to_tensor(tt);

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

function Hilbert_tensor(x,y,z,t,w)
    return 1/(x+y+z+t+w)
end

function Hilbert_wrapper(coords::Matrix{Float64})
    npts = size(coords, 1)
    vals = Vector{Float64}(undef, npts)
    @inbounds for i in 1:npts
        x = coords[i,1]
        y = coords[i,2]
        z = coords[i,3]
        t = coords[i,4]
        w = coords[i,5]
        vals[i] = 1/(x + y + z + t + w)
    end
    return vals
end

domain = [collect(range(1.0, 33.0, length = 33)) for _ in 1:5];

t = tt_cross(Hilbert_wrapper,domain, verbose=true)

t_dense = ttv_to_tensor(t);