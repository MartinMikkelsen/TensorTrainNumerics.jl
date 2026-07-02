using LinearAlgebra
using Printf
using Random
using TensorTrainNumerics
using CairoMakie

# 2D Ornstein-Uhlenbeck / Fokker-Planck equation on [a, b]^2:
#
#     ∂t P = θ ∂x((x - μx)P) + θ ∂y((y - μy)P)
#            + D(∂xx P + ∂yy P),
#
# with an initially centered product Gaussian. The global Crank-Nicholson solve
# returns one compressed space-time QTT object containing all time slices.

θ = 1.0
μx, μy = 2.0, -2.0
σ = 1.0
D = σ^2 / 2

bits_per_dim = 7              # 2^5 = 32 grid points per spatial dimension
time_bits = 7                 # Nt = 2^5 = 32 unknown time slices
Nt = 2^time_bits
Tfinal = 0.5
τ = Tfinal / Nt
steps = fill(τ, Nt)

a, b = -6.0, 6.0
N = 2^bits_per_dim
h = (b - a) / (N - 1)
xes = collect(range(a, b, N))

∂ = (1 / (2h)) * (shift(bits_per_dim) - (id_tto(bits_per_dim) - ∇(bits_per_dim)))
∂² = -(1 / h^2) * Δ(bits_per_dim)
I1 = id_tto(bits_per_dim)
Mx = ttv_to_diag_tto(qtt_polynom([-μx, 1.0], bits_per_dim; a = a, b = b))
My = ttv_to_diag_tto(qtt_polynom([-μy, 1.0], bits_per_dim; a = a, b = b))

A_raw = θ * ((∂ * Mx) ⊗ I1 + I1 ⊗ (∂ * My)) +
        D * (∂² ⊗ I1 + I1 ⊗ ∂²)
A = QTToperator(A_raw, 2, bits_per_dim, :serial)

mass(P) = sum(P) * h^2

gx = function_to_qtt(t -> exp(-0.5 * (a + (b - a) * t)^2), bits_per_dim)
gy = function_to_qtt(t -> exp(-0.5 * (a + (b - a) * t)^2), bits_per_dim)
u0 = QTTvector(gx ⊗ gy, 2, bits_per_dim, :serial)
u0 = (1 / mass(qttv_to_array(u0))) * u0

# ALS keeps the ranks of the starting guess. Build an explicit space-time guess
# so the time direction can have nontrivial ranks.
Random.seed!(42)
time_guess = TensorTrainNumerics.ones_tt(Float64, ntuple(_ -> 2, time_bits))
space_time_guess = time_guess ⊗ TTvector(u0)
guess = SpaceTimeQTTvector(
    TensorTrainNumerics.increase_ranks(space_time_guess, 16; noise = 1.0e-5),
    time_bits,
    2,
    bits_per_dim,
    :serial
)

U, residual = global_crank_nicholson_method(
    A,
    u0,
    guess,
    steps;
    tt_solver = "als",
    max_bond = 25,
    sweep_count = 8,
    normalize = false,
    return_error = true
)

var∞ = D / θ

function analytic_ou_density(xes, θ, μx, μy, var∞, t)
    decay = exp(-θ * t)
    mx = μx * (1 - decay)
    my = μy * (1 - decay)
    var_t = var∞ + (1 - var∞) * exp(-2θ * t)
    norm_const = 1 / (2π * var_t)
    return [
        norm_const * exp(-0.5 * ((x - mx)^2 + (y - my)^2) / var_t)
        for x in xes, y in xes
    ]
end

function moments(P, xes, h)
    mx = sum(xes .* vec(sum(P, dims = 2))) * h^2
    my = sum(xes .* vec(sum(P, dims = 1))) * h^2
    return mx, my
end

println("2D global Crank-Nicholson Ornstein equation example")
println("grid: $(N) × $(N), time slices: $Nt, residual: $residual")
println()
@printf("%-4s %-10s %-10s %-13s %-13s %s\n", "k", "t", "rank", "mass", "mean error", "vs analytic OU")

for k in 1:Nt
    uk = space_time_slice(U, k)
    P = qttv_to_array(uk)
    numerical_mass = mass(P)
    P ./= numerical_mass

    t = k * τ
    Pref = analytic_ou_density(xes, θ, μx, μy, var∞, t)
    Pref ./= mass(Pref)

    mx, my = moments(P, xes, h)
    target_mx = μx * (1 - exp(-θ * t))
    target_my = μy * (1 - exp(-θ * t))
    mean_error = norm([mx - target_mx, my - target_my])
    rel_error = norm(vec(P .- Pref)) / norm(vec(Pref))

    @printf(
        "%-4d %-10.5f %-10d %-13.6e %-13.6e %.6e\n",
        k,
        t,
        maximum(uk.ttv_rks),
        numerical_mass,
        mean_error,
        rel_error
    )
end

let
    snaps = unique(round.(Int, range(1, Nt, length = 4)))
    P_num = Vector{Matrix{Float64}}(undef, length(snaps))
    P_ref = Vector{Matrix{Float64}}(undef, length(snaps))
    ts = Vector{Float64}(undef, Nt)
    rel_errors = Vector{Float64}(undef, Nt)
    for k in 1:Nt
        P = qttv_to_array(space_time_slice(U, k))
        P ./= mass(P)
        Pref = analytic_ou_density(xes, θ, μx, μy, var∞, k * τ)
        Pref ./= mass(Pref)
        ts[k] = k * τ
        rel_errors[k] = norm(vec(P .- Pref)) / norm(vec(Pref))
        j = findfirst(==(k), snaps)
        if j !== nothing
            P_num[j] = P
            P_ref[j] = Pref
        end
    end

    p_lims = extrema(reduce(vcat, [vec(S) for S in vcat(P_num, P_ref)]))

    fig = Figure(size = (1180, 760))
    hmP = nothing
    hmR = nothing
    for (j, k) in enumerate(snaps)
        t = k * τ
        axP = Axis(
            fig[1, j],
            aspect = 1,
            ylabel = j == 1 ? "y" : "",
            title = "P, t = $(round(t; digits = 4))"
        )
        hmP = heatmap!(axP, xes, xes, P_num[j]; colormap = :viridis, colorrange = p_lims)

        axR = Axis(
            fig[2, j],
            aspect = 1,
            xlabel = "x",
            ylabel = j == 1 ? "y" : "",
            title = "analytic OU"
        )
        hmR = heatmap!(axR, xes, xes, P_ref[j]; colormap = :viridis, colorrange = p_lims)
    end
    Colorbar(fig[1, length(snaps) + 1], hmP, label = "P")
    Colorbar(fig[2, length(snaps) + 1], hmR, label = "P")

    axE = Axis(
        fig[3, 1:length(snaps)],
        xlabel = "t",
        ylabel = "relative error",
        title = "relative error vs analytic OU"
    )
    lines!(axE, ts, rel_errors; color = :crimson)
    scatter!(
        axE,
        [k * τ for k in snaps],
        [rel_errors[k] for k in snaps];
        color = :black,
        markersize = 9
    )

    outpath = joinpath(@__DIR__, "global_crank_nicholson_ornstein2d.png")
    save(outpath, fig)
    println("saved plot: $outpath")
end
