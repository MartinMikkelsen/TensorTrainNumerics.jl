using TensorTrainNumerics
using InterpolativeQTT
import TensorCrossInterpolation as TCI
using LinearAlgebra

# Regularized 3D Coulomb potential — non-separable, relevant in quantum chemistry
# and N-body simulations. The ε offset removes the singularity at the origin.
ε = 0.01
f(x, y, z) = 1 / sqrt((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2 + ε)

numbits = 8    # 2^8 = 256 grid points per dimension
degree  = 4    # local Chebyshev polynomial degree

tt_tci = interpolatesinglescale(f, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), numbits, degree)
println("  TCI bond dims: ", TCI.linkdims(tt_tci))

# Convert to a TensorTrainNumerics TTvector
tt = to_ttvector(tt_tci)

g(x) = x == 0.0 ? 0.0 : 1/x
tci_multiscale = interpolatemultiscale(g, 0.0, 1.0, 12, 25, [0.0])

tt_multiscale = to_ttvector(tci_multiscale)

function eval_ttvector(tt::TTvector, idx::Vector{Int})
    v = tt.ttv_vec[1][idx[1], 1, :]        # shape (r₁,)
    for k in 2:tt.N
        M = tt.ttv_vec[k][idx[k], :, :]    # shape (r_{k-1}, r_k)
        v = M' * v                         # (r_k, r_{k-1}) * (r_{k-1},) → (r_k,)
    end
    return v[1]
end

println("\nAccuracy check (TCI vs TTvector at 20 random binary indices):")
max_err = 0.0
for _ in 1:20
    idx = [rand(1:8) for _ in 1:numbits]
    err = abs(tt_tci(idx) - eval_ttvector(tt, idx))
    global max_err = max(max_err, err)
end
println("  max |TCI − TTN|: ", max_err)


println("\nAccuracy check (TCI vs TTvector at 20 random binary indices):")
max_err = 0.0
for _ in 1:20
    idx = [rand(1:12) for _ in 1:12]
    err = abs(tci_multiscale(idx) - eval_ttvector(tt_multiscale, idx))
    global max_err = max(max_err, err)
end
println("  max |TCI − TTN|: ", max_err)

# Demonstrate compression: the TTvector can be further truncated using TTN tools.
# The fused QTT already achieves low rank; tt_compress! can tighten it further.
tt_c = copy(tt)
tt_compress!(tt_c, 50; truncerr = 1e-10, sweeps = 5)
println("\nAfter tt_compress! (max_bond=15, tol=1e-10):")
println("  compressed bond dims: ", tt_c.ttv_rks)

# Check that compression preserved accuracy
max_err_c = 0.0
for _ in 1:100
    idx = [rand(1:8) for _ in 1:numbits]
    err = abs(tt_tci(idx) - eval_ttvector(tt_c, idx))
    global max_err_c = max(max_err_c, err)
end
println("  max |TCI − compressed TTN| over 100 samples: ", max_err_c)
