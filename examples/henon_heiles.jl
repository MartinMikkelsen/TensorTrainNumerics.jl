
using LinearAlgebra
using TensorTrainNumerics
using CairoMakie

"""
C. Lubich, I. V. Oseledets, B. Vandereycken, "Time integration of tensor trains", SIAM J. Numer. Anal. 53(2):917–941, 2015, arXiv:1407.2042
"""

function henon_heiles_hamiltonian(n; λ = 0.111803)
    n >= 2 || throw(ArgumentError("use at least two oscillator basis states"))
    padded_n = n + 6
    position = Matrix(SymTridiagonal(zeros(padded_n), sqrt.((1:(padded_n - 1)) ./ 2)))
    q = position[1:n, 1:n]
    q2 = (position^2)[1:n, 1:n]
    q3 = (position^3)[1:n, 1:n]
    h0 = Diagonal((0:(n - 1)) .+ 0.5)
    id = Matrix{Float64}(I, n, n)

    # Three separable terms give an MPO with bond rank three.
    left = zeros(n, n, 1, 3)
    right = zeros(n, n, 3, 1)
    left[:, :, 1, 1] = h0
    left[:, :, 1, 2] = λ * q2
    left[:, :, 1, 3] = id
    right[:, :, 1, 1] = id
    right[:, :, 2, 1] = q
    right[:, :, 3, 1] = h0 - λ * q3 / 3
    return TToperator(2, [left, right], (n, n), [1, 3, 1], [0, 0])
end

function henon_heiles_trajectory(stepper, H, initial, reference, times, eigensystem; kwargs...)
    ψ = initial
    ψ0 = vec(ttv_to_tensor(reference))
    coefficients = eigensystem.vectors' * ψ0
    initial_energy = real(dot(coefficients, eigensystem.values .* coefficients))
    correlation = zeros(ComplexF64, length(times))
    state_error = zeros(length(times))
    norm_drift = zeros(length(times))
    energy_drift = zeros(length(times))
    ranks = zeros(Int, length(times))
    dt = Float64(step(times))

    for k in eachindex(times)
        if k > 1
            ψ = stepper(
                H, ψ, [dt]; normalize = false, sweeps = 1,
                verbose = false, show_progress = false, tol = 1.0e-12, kwargs...
            )
        end
        # Dense diagnostics are affordable here (only n² amplitudes).
        state = vec(ttv_to_tensor(ψ))
        exact = eigensystem.vectors * (coefficients .* cis.(-times[k] .* eigensystem.values))
        spectral_state = eigensystem.vectors' * state
        correlation[k] = dot(ψ0, state)
        state_error[k] = norm(state - exact)
        norm_drift[k] = abs(norm(state) - 1)
        energy = real(dot(spectral_state, eigensystem.values .* spectral_state)) / sum(abs2, state)
        energy_drift[k] = abs(energy - initial_energy)
        ranks[k] = maximum(ψ.ttv_rks)
    end
    return (; correlation, state_error, norm_drift, energy_drift, ranks, final_state = ψ)
end

function henon_heiles_example(; n = 16, λ = 0.111803, q0 = 0.7, fixed_rank = min(8, n), dt = 0.1, nsteps = 2048)
    1 <= fixed_rank <= n || throw(ArgumentError("fixed_rank must lie between 1 and n"))
    dt > 0 && nsteps >= 1 || throw(ArgumentError("dt and nsteps must be positive"))
    H = complex(henon_heiles_hamiltonian(n; λ))
    # A displaced oscillator ground state |α⟩, with α = q0/√2 and zero momentum.
    α = q0 / sqrt(2)
    packet = zeros(ComplexF64, n)
    packet[1] = exp(-α^2 / 2)
    for k in 2:n
        packet[k] = packet[k - 1] * α / sqrt(k - 1)
    end
    normalize!(packet) # Normalize the initial truncated packet only.
    ψ0 = TTvector(2, [reshape(copy(packet), n, 1, 1) for _ in 1:2], (n, n), [1, 1, 1], [0, 0])
    # Exact zero-padding gives fixed-rank TDVP room to develop entanglement.
    fixed_initial = fixed_rank == 1 ? copy(ψ0) : TensorTrainNumerics.increase_ranks(ψ0, fixed_rank; noise = 0.0)
    times = range(0.0; step = Float64(dt), length = nsteps + 1)
    eigensystem = eigen(Hermitian(reshape(tto_to_tensor(H), n^2, n^2)))
    fixed = henon_heiles_trajectory(tdvp, H, fixed_initial, ψ0, times, eigensystem)
    # With only two TT sites, tdvp2 evolves the whole pair; its remaining errors
    # come from Krylov exponentiation and SVD truncation, rather than splitting.
    adaptive = henon_heiles_trajectory(tdvp2, H, ψ0, ψ0, times, eigensystem; max_bond = n, truncerr = 1.0e-12)
    weights = abs2.(eigensystem.vectors' * vec(ttv_to_tensor(ψ0)))
    energies = eigensystem.values
    exact_correlation = [sum(weights .* cis.(-t .* energies)) for t in times]
    return (; times, fixed, adaptive, exact_correlation, energies, weights, n, λ)
end

function henon_heiles_spectrum(times, correlation, energies)
    # Re ∫₀ᵀ C(t) cos²(πt/2T) exp(+iEt) dt / π, using the trapezoidal rule.
    # The positive sign maps C(t) = exp(-iE₀t) to a peak at E₀.
    window = cospi.(times ./ (2last(times))) .^ 2
    window[1] /= 2
    window[end] /= 2
    weighted = window .* correlation
    dt = step(times)
    return [dt / π * real(sum(c * cis(E * t) for (t, c) in zip(times, weighted))) for E in energies]
end

function plot_henon_heiles(result)
    (; times, fixed, adaptive, exact_correlation, energies, weights) = result
    fig = Figure(size = (1200, 760))
    Label(fig[0, 1:3], "Hénon–Heiles · two modes · $(result.n) oscillator states per mode", fontsize = 22)
    ax_corr = Axis(fig[1, 1], xlabel = "time", ylabel = "Re ⟨ψ(0)|ψ(t)⟩", title = "Wavepacket autocorrelation")
    ax_spectrum = Axis(fig[1, 2], xlabel = "energy", ylabel = "windowed spectrum", title = "Spectrum (T = $(round(last(times); digits = 1)))")
    ax_rank = Axis(fig[1, 3], xlabel = "time", ylabel = "bond rank", title = "Bond rank (early evolution)")
    ax_error = Axis(fig[2, 1], xlabel = "time", ylabel = "‖ψ − ψexact‖", title = "Error in the finite oscillator basis", yscale = log10)
    ax_norm = Axis(fig[2, 2], xlabel = "time", ylabel = "|‖ψ‖ − 1|", title = "Norm drift (no renormalization)", yscale = log10, yticks = 10.0 .^ (-16:0), ytickformat = "{:.0e}")
    ax_energy = Axis(fig[2, 3], xlabel = "time", ylabel = "|⟨H⟩ − ⟨H⟩₀|", title = "Energy drift", yscale = log10)

    # Show early oscillations clearly; use the entire trajectory for the spectrum.
    visible = times .<= min(30.0, last(times))
    early = times .<= min(10.0, last(times))
    spectral_grid = range(0.0, 6.0; length = 1001)
    for (trajectory, label, color) in ((fixed, "tdvp (fixed rank)", :dodgerblue), (adaptive, "tdvp2 (adaptive rank)", :darkorange))
        lines!(ax_corr, times[visible], real.(trajectory.correlation[visible]); label, color, linewidth = 2)
        lines!(ax_spectrum, spectral_grid, henon_heiles_spectrum(times, trajectory.correlation, spectral_grid); color, linewidth = 2)
        stairs!(ax_rank, times[early], trajectory.ranks[early]; color, linewidth = 2)
        # A plotting floor keeps exact zeros visible on logarithmic axes.
        for (ax, values) in ((ax_error, trajectory.state_error), (ax_norm, trajectory.norm_drift), (ax_energy, trajectory.energy_drift))
            lines!(ax, times, max.(values, 1.0e-16); color, linewidth = 2)
        end
    end
    lines!(ax_corr, times[visible], real.(exact_correlation[visible]); label = "dense reference", color = :black, linestyle = :dash)
    lines!(ax_spectrum, spectral_grid, henon_heiles_spectrum(times, exact_correlation, spectral_grid); color = :black, linestyle = :dash)
    # Only mark eigenvalues carrying appreciable initial-state spectral weight.
    vlines!(ax_spectrum, energies[weights .> 0.01maximum(weights)]; color = (:black, 0.2), linestyle = :dot)
    xlims!(ax_spectrum, 0, 6)
    Legend(fig[3, 1:3], ax_corr; orientation = :horizontal, framevisible = false)
    return fig
end

result = henon_heiles_example(; n = 16, fixed_rank = 8, dt = 0.1, nsteps = 2048)
for (method, trajectory) in (("tdvp", result.fixed), ("tdvp2", result.adaptive))
    @info "Hénon–Heiles evolution" method max_state_error = maximum(trajectory.state_error) max_norm_drift = maximum(trajectory.norm_drift) max_energy_drift = maximum(trajectory.energy_drift) max_rank = maximum(trajectory.ranks)
end
@info "Spectrum" duration = last(result.times) approximate_energy_resolution = 2π / last(result.times)
display(plot_henon_heiles(result))
