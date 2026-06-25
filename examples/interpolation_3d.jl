using TensorTrainNumerics
using InterpolativeQTT
import TensorCrossInterpolation as TCI
using LinearAlgebra

function eval_tt(tt::TTvector, idx::Vector{Int})
    v = tt.ttv_vec[1][idx[1], 1, :]
    for k in 2:tt.N
        v = tt.ttv_vec[k][idx[k], :, :]' * v
    end
    return v[1]
end

ε = 0.01
_f3d(x, y, z) = 1 / sqrt((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2 + ε)
f3d(x, y, z) = _f3d(x, y, z)
f3d(c::AbstractVector) = _f3d(c...)

numbits = 6
degree = 4

tt_tci = interpolatesinglescale(f3d, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), numbits, degree)
println("TCI bond dims:   ", TCI.linkdims(tt_tci))

tt_fused = to_ttvector(tt_tci)

# Accuracy check at random binary multi-indices
max_err = maximum(1:100) do _
    idx = [rand(1:8) for _ in 1:numbits]
    abs(tt_tci(idx) - eval_tt(tt_fused, idx))
end
println("max |TCI − TTN| (100 samples): ", max_err)

ttv_split = to_qtt(tt_fused, [[2, 2, 2] for _ in 1:numbits])
q_interleaved = QTTvector(ttv_split, 3, numbits, :interleaved)

q_serial = reorder(q_interleaved, :serial)

arr_il = qttv_to_array(q_interleaved)
arr_sr = qttv_to_array(q_serial)
println("max |interleaved − serial|: ", maximum(abs, arr_il .- arr_sr))

q_native_il = function_to_qttv(f3d, 3, numbits; ordering = :interleaved)
q_native_sr = function_to_qttv(f3d, 3, numbits; ordering = :serial)

h = 1.0 / (2^numbits - 1)
arr_native = qttv_to_array(q_native_sr)
err_native = 0.0
for _ in 1:500
    ix, iy, iz = rand(0:(2^numbits - 1)), rand(0:(2^numbits - 1)), rand(0:(2^numbits - 1))
    x, y, z = ix * h, iy * h, iz * h
    global err_native = max(err_native, abs(arr_native[ix + 1, iy + 1, iz + 1] - f3d(x, y, z)))
end
println("max |native − f3d| at equispaced grid pts: ", err_native)

err_tci_self = 0.0
for _ in 1:500
    idx = [rand(1:8) for _ in 1:numbits]
    global err_tci_self = max(err_tci_self, abs(tt_tci(idx) - eval_tt(tt_fused, idx)))
end
println("max |TCI − to_ttvector| at TCI grid pts:   ", err_tci_self)

println("Bond dims — TCI-converted serial: ", q_serial.ttv_rks)
println("Bond dims — native serial:        ", q_native_sr.ttv_rks)

g1d(x) = x == 0.0 ? 0.0 : 1 / x

n_levels = 10
local_degree = 8
tci_ms = interpolatemultiscale(g1d, 0.0, 1.0, n_levels, local_degree, [0.0])

sites_ms = TCI.sitetensors(tci_ms)
phys_dims_ms = [size(s, 2) for s in sites_ms]
println("TCI bond dims:   ", TCI.linkdims(tci_ms))
println("Physical dims:   ", phys_dims_ms, "  (binary: one refinement bit per level)")

tt_ms = to_ttvector(tci_ms)
println("TTvector N=$(tt_ms.N), bonds=$(tt_ms.ttv_rks)")

max_err_ms = maximum(1:100) do _
    idx = [rand(1:d) for d in tt_ms.ttv_dims]
    abs(tci_ms(idx) - eval_tt(tt_ms, idx))
end
println("max |TCI − TTN| (100 samples): ", max_err_ms)

# Compress the multiscale TTvector
tt_ms_c = copy(tt_ms)
tt_compress!(tt_ms_c, 20; truncerr = 1.0e-8, sweeps = 5)
println("\nAfter tt_compress! (max_bond=20, tol=1e-8):")
println("  Bond dims: ", tt_ms_c.ttv_rks)

max_err_c = maximum(1:100) do _
    idx = [rand(1:d) for d in tt_ms.ttv_dims]
    abs(tci_ms(idx) - eval_tt(tt_ms_c, idx))
end
println("  max |TCI − compressed| (100 samples): ", max_err_c)
