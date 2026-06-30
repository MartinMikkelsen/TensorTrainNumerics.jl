using LinearAlgebra
using ProgressMeter
using TensorTrainNumerics
using CairoMakie

d = 20
max_bond = 5
g_values = collect(0.0:0.1:2.0)

function pauli_product_tto(factors, d)
    T = promote_type((eltype(pauli_matrix(axis)) for (_, axis) in factors)...)
    id = Matrix{T}(I, 2, 2)
    factor_map = Dict(factors)
    dims = ntuple(_ -> 2, d)
    cores = Vector{Array{T, 4}}(undef, d)

    for site in 1:d
        local_matrix = haskey(factor_map, site) ? convert.(T, pauli_matrix(factor_map[site])) : id
        cores[site] = reshape(local_matrix, 2, 2, 1, 1)
    end
    return TToperator{T, d}(d, cores, dims, ones(Int, d + 1), zeros(Int, d))
end

function periodic_transverse_field_ising_tto(d, g)
    # H = -sum_i sigma_z(i) sigma_z(i+1) - g sum_i sigma_x(i),
    # with the closing bond sigma_z(d) sigma_z(1).
    zz_open = pauli_pair_sum_tto(:z, :z, d)
    zz_boundary = pauli_product_tto([1 => :z, d => :z], d)
    return (-1.0) * (zz_open + zz_boundary) + (-g) * pauli_sum_tto(:x, d)
end

function z_magnetization(state)
    d = state.N
    probabilities = abs2.(qtt_to_function(state))
    probabilities ./= sum(probabilities)

    magnetization = 0.0
    for (basis_index, probability) in enumerate(probabilities)
        bits = basis_index - 1
        spin_sum = 0.0
        for site in 1:d
            bit = (bits >> (d - site)) & 1
            spin_sum += iszero(bit) ? 1.0 : -1.0
        end
        magnetization += probability * spin_sum / d
    end
    return abs(magnetization)
end

function ground_state(H, initial_state; max_bond)
    energies, state, rank_history = dmrg_eigsolve(
        H,
        initial_state;
        sweep_schedule = [2, 4],
        rmax_schedule = [max_bond, max_bond],
        tol = 1.0e-10,
    )
    return energies[end], state, rank_history[end]
end

function magnetization_sweep(d, g_values; max_bond)
    state = qtt_basis_vector(d, 1)

    magnetization = zeros(Float64, length(g_values))
    energies = zeros(Float64, length(g_values))
    ranks = zeros(Int, length(g_values))

    @showprogress for (i, g) in collect(enumerate(g_values))
        H = periodic_transverse_field_ising_tto(d, g)
        energies[i], state, ranks[i] = ground_state(H, state; max_bond = max_bond)
        magnetization[i] = z_magnetization(state)
    end
    return magnetization, energies, ranks
end

magnetization, energies, ranks = magnetization_sweep(d, g_values; max_bond = max_bond)

println("Transverse-field Ising magnetization sweep: sites=$d max_bond=$max_bond min_energy=$(minimum(energies)) max_rank=$(maximum(ranks))")

fig = Figure(size = (800, 560))
ax = Axis(
    fig[1, 1];
    xlabel = "g",
    ylabel = "M",
    title = "Magnetization",
)
scatter!(ax, g_values, magnetization; markersize = 12, label = "χ=$max_bond")
axislegend(ax; position = :rt)
xlims!(ax, minimum(g_values) - 0.05, maximum(g_values) + 0.05)
ylims!(ax, -0.03, 1.05)

display(fig)

# ---------------------------------------------------------------------------
# Variational ground state with OptimKit (analytic gradient)
# ---------------------------------------------------------------------------
# This mirrors the ITensors + OptimKit pattern, but with TensorTrainNumerics
# types. There is no Zygote AD here: `dot` writes in place, so reverse-mode AD
# cannot trace through it. Instead, TTvectors implement the VectorInterface
# vector space that OptimKit needs, and we supply the *analytic* gradient of the
# Rayleigh quotient
#
#     E(ψ) = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩,        ∇E = (2 / ⟨ψ|ψ⟩) (Hψ − E ψ),
#
# whose minimiser is the ground state. ∇E is orthogonal to ψ, so the scale of ψ
# is preserved along the flow and the Rayleigh quotient is well behaved.

using Random
using OptimKit
import TensorTrainNumerics: dot   # disambiguate from LinearAlgebra.dot

n = 10
J = 1.0
h = 0.5
opt_bond = 16   # working bond dimension for the variational search

# H = -J Σ Zᵢ Zᵢ₊₁ - h Σ Xᵢ  (open chain; signs match the ITensors example).
H_ising = ising_tto(n; J = -J, h = -h, interaction = :z, field = :x)

# Energy and its gradient. `bond > 0` truncates the gradient, since TT `*` and
# `+` grow the bond dimension (rank(Hψ) = rank(H)·rank(ψ)); `bond = 0` leaves it
# exact, which we use for reporting the true gradient norm.
function energy_and_gradient(ψ; bond = 0)
    Hψ = H_ising * ψ
    nrm2 = real(dot(ψ, ψ))
    E = real(dot(ψ, Hψ)) / nrm2
    grad = (2.0 / nrm2) * (Hψ - E * ψ)
    bond > 0 && (grad = tt_compress!(grad, bond))
    return E, grad
end

loss_and_grad(ψ) = energy_and_gradient(ψ; bond = opt_bond)
report(ψ) = (vals = energy_and_gradient(ψ); (vals[1], norm(vals[2])))

Random.seed!(1)
ψ0_ad = rand_tt(ntuple(_ -> 2, n), opt_bond; normalise = true)

optimizer = LBFGS(; maxiter = 100, verbosity = 1)

# `KRYLOV_ROUND_RANK` caps the rank growth from OptimKit's repeated vector
# additions (the line search and the L-BFGS history), exactly as the package's
# Krylov time-steppers do.
old_round = TensorTrainNumerics.KRYLOV_ROUND_RANK[]
TensorTrainNumerics.KRYLOV_ROUND_RANK[] = opt_bond
ψ_ad, E_lbfgs, g_final, numfg, normgradhistory = try
    optimize(loss_and_grad, ψ0_ad, optimizer)
finally
    TensorTrainNumerics.KRYLOV_ROUND_RANK[] = old_round
end

# DMRG reference for the same Hamiltonian (reuses the helper defined above).
E_dmrg, ψ_dmrg, _ = ground_state(H_ising, qtt_basis_vector(n, 1); max_bond = opt_bond)

println("Variational (LBFGS) vs DMRG ground state: sites=$n J=$J h=$h E_lbfgs=$E_lbfgs E_dmrg=$E_dmrg gap=$(abs(E_lbfgs - E_dmrg))")

# (energy, ‖∇E‖) for the initial guess, the optimized state, and the DMRG state.
@show report(ψ0_ad)
@show report(ψ_ad)
@show report(ψ_dmrg)

# ---------------------------------------------------------------------------
# Same variational ground state, but with Zygote autodiff instead of the
# analytic gradient. We optimise over the *cores* (flattened), exactly like the
# ITensorMPS example optimises over its Vector{ITensor}. Loading Zygote activates
# TensorTrainNumerics' ChainRulesCore extension (rrules for `dot` and `*`), so
# Zygote can differentiate through the rebuilt TTvector. Unlike the analytic
# gradient above (a Hilbert-space vector), the AD gradient is per-core, so the
# optimisation lives in parameter space — the geometry that matches OptimKit's
# core-wise pairing here.
using Zygote

shapes_ad = size.(ψ0_ad.ttv_vec)
offsets_ad = cumsum([0; prod.(shapes_ad)])
unflatten_ad(θ) = [reshape(θ[(offsets_ad[k] + 1):offsets_ad[k + 1]], shapes_ad[k]) for k in 1:n]
rebuild_ad(θ) = TTvector{Float64, n}(n, unflatten_ad(θ), ψ0_ad.ttv_dims, ψ0_ad.ttv_rks, ψ0_ad.ttv_ot)
loss_ad(θ) = (ψ = rebuild_ad(θ); real(dot(ψ, H_ising * ψ)) / real(dot(ψ, ψ)))

θ0 = vcat(vec.(ψ0_ad.ttv_vec)...)
zygote_loss_and_grad(θ) = (loss_ad(θ), Zygote.gradient(loss_ad, θ)[1])

θ_ad, E_zygote, _, _, _ = optimize(zygote_loss_and_grad, θ0, LBFGS(; maxiter = 100, verbosity = 0))

println("Zygote-AD (LBFGS over cores) vs analytic vs DMRG: E_zygote=$E_zygote E_lbfgs=$E_lbfgs E_dmrg=$E_dmrg")
