using LinearAlgebra
using Logging
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
    energies, state, rank_history = with_logger(NullLogger()) do
        dmrg_eigsolve(
            H,
            initial_state;
            sweep_schedule = [2, 4],
            rmax_schedule = [max_bond, max_bond],
            tol = 1.0e-10,
        )
    end
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

@info "Transverse-field Ising magnetization sweep" sites=d max_bond=max_bond min_energy=minimum(energies) max_rank=maximum(ranks)

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
