using LinearAlgebra
using Logging
using Random
using TensorTrainNumerics
using CairoMakie 

Random.seed!(1234)

d = 8
H = heisenberg_xyz_tto(d; jx = 2.0, jy = 0.8, jz = 1.2, λ = 0.5, field = :z)

x0_ranks = vcat(1, fill(2, d - 1), 1)
x0 = rand_tt(eltype(H), H.tto_dims, x0_ranks; normalise = true)

energies, ground_state, rank_history = with_logger(NullLogger()) do
    dmrg_eigsolve(H, x0; sweep_schedule = [4, 8], rmax_schedule = [4, 8])
end

H_dense = qtto_to_matrix(H)
exact_ground_energy = first(eigvals(Hermitian(H_dense)))
dmrg_ground_energy = energies[end]
energy_error = abs(dmrg_ground_energy - exact_ground_energy)
ground_entropy = entanglemententropy(ground_state; base = 2)

@info "Heisenberg XYZ ground state" sites=d hamiltonian_rank=maximum(H.tto_rks) state_rank=maximum(ground_state.ttv_rks) dmrg_ground_energy exact_ground_energy energy_error final_sweep_rank=rank_history[end]

ψ = qtt_to_function(ground_state)
probabilities = abs2.(ψ)
probabilities ./= sum(probabilities)
basis_indices = 0:(length(probabilities) - 1)
bond_indices = 1:(d - 1)

neel_bits = [isodd(k) ? 0 : 1 for k in 1:d]
neel_position = 1 + sum(bit * 2^(d - k) for (k, bit) in enumerate(neel_bits))
initial_state = qtt_basis_vector(d, neel_position)

dt = 0.02
nsteps = 25
times = collect(0:dt:(nsteps * dt))

function entropy_trajectory(H, initial_state, dt, nsteps; max_bond = 8, truncerr = 1.0e-10)
    d = initial_state.N
    state = initial_state
    entropy_history = zeros(Float64, nsteps + 1, d - 1)
    rank_history = zeros(Int, nsteps + 1)
    entropy_history[1, :] .= entanglemententropy(state; base = 2)
    rank_history[1] = maximum(state.ttv_rks)

    for step in 1:nsteps
        state = tdvp2(H, state, [dt]; normalize = true, sweeps = 1, max_bond = max_bond, truncerr = truncerr, verbose = false)
        entropy_history[step + 1, :] .= entanglemententropy(state; base = 2)
        rank_history[step + 1] = maximum(state.ttv_rks)
    end
    return entropy_history, rank_history
end

entropy_history, time_rank_history = entropy_trajectory(H, initial_state, dt, nsteps; max_bond = 8)

@info "Real-time Heisenberg entropy growth" dt nsteps max_time=times[end] max_entropy=maximum(entropy_history) final_rank=time_rank_history[end]

fig = Figure(size = (1100, 760))

ax_prob = Axis(fig[1, 1], xlabel = "basis index", ylabel = "|ψ|²", title = "Ground-state")
barplot!(ax_prob, basis_indices, probabilities)

ax_entropy = Axis(fig[1, 2], xlabel = "bond index", ylabel = "S (bits)", title = "Ground-state entanglement")
scatterlines!(ax_entropy, bond_indices, ground_entropy; marker = :circle, linewidth = 3)

ax_time = Axis(fig[2, 1:2], xlabel = "time", ylabel = "bond index", title = "TDVP real-time entanglement growth")
hm = heatmap!(ax_time, times, bond_indices, entropy_history)
Colorbar(fig[2, 3], hm, label = "S (bits)")

display(fig)