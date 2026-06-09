using LinearAlgebra
using TensorTrainNumerics

function heat_problem(d::Int; κ::Float64 = 0.1)
    N = 2^d
    h = 1.0 / (N + 1)
    xes = h .* (1:N)

    Δ1d = toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    A_raw = (κ / h^2) * (Δ1d ⊗ id_tto(d) + id_tto(d) ⊗ Δ1d)
    A = QTToperator(A_raw, 2, d, :serial)

    u0_raw = qtt_sin(d; a = h, b = 1 - h) ⊗ qtt_sin(d; a = h, b = 1 - h)
    u0 = QTTvector(u0_raw, 2, d, :serial)
    return A, u0, xes
end

function prolong_serial_2d(u::QTTvector; max_bond::Int = 16, truncerr::Float64 = 1.0e-12)
    @assert u.n_dims == 2 "Only 2D QTTvectors are supported"
    @assert u.ordering == :serial "Only serial QTT ordering is supported"

    d = u.bits_per_dim
    Py = id_tto(d) ⊗ qtto_linear_prolongation(d)
    uy = Py * TTvector(u)

    Px = qtto_linear_prolongation(d) ⊗ id_tto(d + 1)
    uf = QTTvector(Px * uy, 2, d + 1, :serial)
    return tt_compress!(uf, max_bond; truncerr = truncerr, sweeps = 2)
end

function exact_heat_mode(xes, κ::Float64, T::Float64)
    return [sin(π * x) * sin(π * y) * exp(-2κ * π^2 * T) for x in xes, y in xes]
end

κ = 0.1
d_coarse = 4
d_fine = d_coarse + 1
dt = 5.0e-3
T_coarse = 5.0e-3
T_fine = 5.0e-3
T_total = T_coarse + T_fine

A_coarse, u0_coarse, _ = heat_problem(d_coarse; κ = κ)
A_fine, u0_fine, xes_fine = heat_problem(d_fine; κ = κ)

coarse_solution = tdvp2(
    A_coarse,
    u0_coarse,
    fill(dt, round(Int, T_coarse / dt));
    imaginary_time = true,
    normalize = false,
    max_bond = 12,
    truncerr = 1.0e-12,
    verbose = false
)

prolonged = prolong_serial_2d(coarse_solution; max_bond = 16, truncerr = 1.0e-12)
# The interior Dirichlet grids i/(2^d+1) are not nested across d.
# This diagnostic separates grid-transfer error from TDVP continuation error.
prolonged_initial = prolong_serial_2d(u0_coarse; max_bond = 16, truncerr = 1.0e-12)

multilevel_solution = tdvp2(
    A_fine,
    prolonged,
    fill(dt, round(Int, T_fine / dt));
    imaginary_time = true,
    normalize = false,
    max_bond = 16,
    truncerr = 1.0e-12,
    verbose = false
)

direct_solution = tdvp2(
    A_fine,
    u0_fine,
    fill(dt, round(Int, T_total / dt));
    imaginary_time = true,
    normalize = false,
    max_bond = 16,
    truncerr = 1.0e-12,
    verbose = false
)

u_multilevel = qttv_to_array(multilevel_solution)
u_direct = qttv_to_array(direct_solution)
u0_transfer = qttv_to_array(prolonged_initial)
u0_direct = qttv_to_array(u0_fine)
u_exact = exact_heat_mode(xes_fine, κ, T_total)

relerr_initial_transfer = norm(vec(u0_transfer .- u0_direct)) / norm(vec(u0_direct))
relerr_multilevel = norm(vec(u_multilevel .- u_exact)) / norm(vec(u_exact))
relerr_direct = norm(vec(u_direct .- u_exact)) / norm(vec(u_exact))
relerr_vs_direct = norm(vec(u_multilevel .- u_direct)) / norm(vec(u_direct))

@info "2D heat equation with QTT prolongation" d_coarse=d_coarse d_fine=d_fine T_total=T_total
@info "Relative errors" initial_transfer=relerr_initial_transfer multilevel_vs_exact=relerr_multilevel direct_vs_exact=relerr_direct multilevel_vs_direct=relerr_vs_direct
@info "Ranks" coarse_rank=maximum(coarse_solution.ttv_rks) prolonged_rank=maximum(prolonged.ttv_rks) multilevel_rank=maximum(multilevel_solution.ttv_rks) direct_rank=maximum(direct_solution.ttv_rks)
