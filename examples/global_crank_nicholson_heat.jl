using LinearAlgebra
using TensorTrainNumerics

# 2D heat equation on (0, 1)^2 with zero Dirichlet boundaries:
#
#     u_t = κ Δu,
#     u(0, x, y) = sin(πx) sin(πy).
#
# The global Crank-Nicholson solver returns a single compressed space-time QTT
# object containing all unknown time slices u_1, ..., u_Nt.

bits_per_dim = 6              # 2^3 = 8 interior points per spatial dimension
N = 2^bits_per_dim
h = 1.0 / (N + 1)
κ = 0.1

time_bits = 6                 # Nt = 2^2 = 4 unknown time slices
Nt = 2^time_bits
Tfinal = 0.01
τ = Tfinal / (Nt)
steps = fill(τ, Nt)

xes = h .* (1:N)

Δ1d = toeplitz_to_qtto(-2.0, 1.0, 1.0, bits_per_dim)
A_raw = (κ / h^2) * (Δ1d ⊗ id_tto(bits_per_dim) + id_tto(bits_per_dim) ⊗ Δ1d)
A = QTToperator(A_raw, 2, bits_per_dim, :serial)

u0_raw = qtt_sin(bits_per_dim; a = h, b = 1 - h) ⊗
         qtt_sin(bits_per_dim; a = h, b = 1 - h)
u0 = QTTvector(u0_raw, 2, bits_per_dim, :serial)

U, residual = global_crank_nicholson_method(
    A,
    u0,
    u0,
    steps;
    tt_solver = "als",
    max_bond = 16,
    normalize = false,
    return_error = true
)

λ1d = -(4κ / h^2) * sin(π / (2(N + 1)))^2
λ2d = 2λ1d
cn_factor = (1 + τ * λ2d / 2) / (1 - τ * λ2d / 2)

function discrete_cn_heat_mode(xes, amplification)
    return [sin(π * x) * sin(π * y) * amplification for x in xes, y in xes]
end

println("2D global Crank-Nicholson heat equation example")
println("grid: $(N) × $(N), time slices: $Nt, residual: $residual")
println()
println(rpad("k", 4), rpad("t", 12), rpad("slice rank", 12), "vs discrete CN mode")

for k in 1:Nt
    uk = space_time_slice(U, k)
    u_global = qttv_to_array(uk)
    u_reference = discrete_cn_heat_mode(xes, cn_factor^k)
    rel_error = norm(vec(u_global .- u_reference)) / norm(vec(u_reference))

    println(
        rpad(k, 4),
        rpad(round(k * τ; digits = 5), 12),
        rpad(maximum(uk.ttv_rks), 12),
        rel_error
    )
end
