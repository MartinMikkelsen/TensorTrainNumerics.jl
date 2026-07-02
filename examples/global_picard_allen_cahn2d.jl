using LinearAlgebra
using Printf
using CairoMakie
using TensorTrainNumerics
using InterpolativeQTT
import TensorCrossInterpolation as TCI

# 2D forced Allen-Cahn equation on (0, 1)^2 with zero Dirichlet boundaries:
#
#     ∂ₜu = ε²Δu + u - u³ + ϕ(t, x, y).
#
# We manufacture a smooth exact trajectory with several spatial frequencies,
#
#     u★(t, x, y) = Σₘ αₘ exp(-ωₘt) sin(pₘπx) sin(qₘπy),
#
# and choose ϕ so that u★ solves the semi-discrete equation with the same
# second-difference Laplacian used below. At each Picard step we invert the
# current fused space-time IQTT into Lindsey's Chebyshev cell representation
# and build the nonlinear source
#
#     𝒢ⁿ(t, x, y) = ϕ(t, x, y) - (Ũⁿ(t, x, y))^3
#
# by Lindsey's interpolative QTT method, then solve one global Crank-Nicholson
# block system with ALS.

Rₓ = 6
Rₜ = Rₓ
Nₓ = 2^Rₓ
Nₜ = 2^Rₜ

h = 1.0 / (Nₓ + 1)
x⃗ = collect(range(h, 1 - h, Nₓ))

ε = 0.04
T = 0.10
τ = T / Nₜ

𝓜 = (
    (; α = 0.50, ω = 1.0, p = 1, q = 1),
    (; α = 0.25, ω = 1.7, p = 2, q = 3),
    (; α = 0.15, ω = 2.3, p = 5, q = 2),
)

λΔ(p, q) = -(4.0 / h^2) * (
    sin(p * π / (2(Nₓ + 1)))^2 +
        sin(q * π / (2(Nₓ + 1)))^2
)
c𝒢(m) = -m.ω - ε^2 * λΔ(m.p, m.q) - 1.0

p = 6
εᵢ = 1.0e-8
rₘₐₓ = 75
nₛ = 24
nₚ = 5
qᵢ = 1

uₘ(m, t, x, y) = m.α * exp(-m.ω * t) * sinpi(m.p * x) * sinpi(m.q * y)
u★(t, x, y) = sum(uₘ(m, t, x, y) for m in 𝓜)
𝒢★(t, x, y) = sum(c𝒢(m) * uₘ(m, t, x, y) for m in 𝓜)
ϕ(t, x, y) = 𝒢★(t, x, y) + u★(t, x, y)^3

unit_clamp(z) = clamp(float(z), 0.0, 1.0)
ξ_to_x(ξ) = h + unit_clamp(ξ) * (1 - 2h)
t̂_to_t(t̂) = τ + unit_clamp(t̂) * (T - τ)

Δ₁ = toeplitz_to_qtto(-2.0, 1.0, 1.0, Rₓ)
I₁ = id_tto(Rₓ)
I₂ = I₁ ⊗ I₁

A_raw = (ε^2 / h^2) * (Δ₁ ⊗ I₁ + I₁ ⊗ Δ₁) + I₂
A = QTToperator(A_raw, 2, Rₓ, :serial)

space_mode(m) = m.α * (
    qtt_sin(Rₓ; a = h, b = 1 - h, λ = m.p) ⊗
        qtt_sin(Rₓ; a = h, b = 1 - h, λ = m.q)
)

u₀_raw = reduce(+, space_mode(m) for m in 𝓜)
u₀ = QTTvector(u₀_raw, 2, Rₓ, :serial)

function spacetime_array(U::SpaceTimeQTTvector)
    Nₜ = 2^U.time_bits
    Nₓ = 2^U.space_bits_per_dim
    values = Array{Float64}(undef, Nₜ, Nₓ, Nₓ)
    for k in 1:Nₜ
        values[k, :, :] .= qttv_to_array(space_time_slice(U, k))
    end
    return values
end

function exact_array(u★, Nₜ::Int, τ::Real, x⃗)
    values = Array{Float64}(undef, Nₜ, length(x⃗), length(x⃗))
    for k in 1:Nₜ, i in eachindex(x⃗), j in eachindex(x⃗)
        values[k, i, j] = u★(k * τ, x⃗[i], x⃗[j])
    end
    return values
end

function iqtt_to_spacetime(tt_iqtt, Rₜ::Int, Rₓ::Int; εᵢ::Real)
    tt_fused = to_ttvector(tt_iqtt)
    tt_split = to_qtt(tt_fused, [[2, 2, 2] for _ in 1:Rₜ]; threshold = εᵢ)
    q_interleaved = QTTvector(tt_split, 3, Rₜ, :interleaved)
    q_serial = reorder(q_interleaved, :serial; threshold = εᵢ)

    return SpaceTimeQTTvector(TTvector(q_serial), Rₜ, 2, Rₓ, :serial)
end

function build_iqtt(
        𝒇,
        Rₜ::Int,
        Rₓ::Int;
        p::Int,
        εᵢ::Real,
        rₘₐₓ::Int,
        upper_order::Symbol = :txy
    )
    @assert Rₜ == Rₓ "InterpolativeQTT currently uses one bit depth for every variable in this example"
    βₜ = 2.0^Rₜ / (2.0^Rₜ - 1.0)
    βₓ = 2.0^Rₓ / (2.0^Rₓ - 1.0)
    upper = upper_order == :txy ? (βₜ, βₓ, βₓ) :
        upper_order == :yxt ? (βₓ, βₓ, βₜ) :
        throw(ArgumentError("upper_order must be :txy or :yxt"))
    return InterpolativeQTT.interpolatesinglescale(
        𝒇,
        (0.0, 0.0, 0.0),
        upper,
        Rₜ,
        p;
        tolerance = εᵢ,
        maxbonddim = rₘₐₓ
    )
end

function spacetime_to_iqtt(U::SpaceTimeQTTvector; εᵢ::Real)
    @assert U.time_bits == U.space_bits_per_dim "fused IQTT conversion currently requires equal time and space bit depths"
    @assert U.space_n_dims == 2 "this example expects one time dimension and two spatial dimensions"
    q_serial = QTTvector(TTvector(U), 3, U.time_bits, :serial)
    q_interleaved = reorder(q_serial, :interleaved; threshold = εᵢ)
    tt_fused = to_ttv(TTvector(q_interleaved), fill(3, U.time_bits))
    return to_tci_tensortrain(tt_fused)
end

function iqtt_space_time(𝒇, Rₜ::Int, Rₓ::Int; p::Int, εᵢ::Real, rₘₐₓ::Int)
    # InterpolativeQTT's fused multivariate TT is reversed after binary splitting:
    # argument 1 becomes the last serial QTT variable. Use (η, ξ, t̂) here so
    # SpaceTimeQTTvector sees dimensions ordered as (t̂, ξ, η).
    𝒇ᵣ(η, ξ, t̂) = 𝒇(t̂, ξ, η)

    tt_iqtt = build_iqtt(
        𝒇ᵣ,
        Rₜ,
        Rₓ;
        p = p,
        εᵢ = εᵢ,
        rₘₐₓ = rₘₐₓ,
        upper_order = :yxt
    )

    return tt_iqtt, iqtt_to_spacetime(tt_iqtt, Rₜ, Rₓ; εᵢ = εᵢ)
end

function morton_cell_flat(cell_ixs, K_out::Int)
    row = 0
    ndims = length(cell_ixs)
    for bit_position in 0:(K_out - 1)
        for n in 1:ndims
            bit = (cell_ixs[n] >> bit_position) & 1
            row |= bit << (ndims * bit_position + (n - 1))
        end
    end
    return row + 1
end

function cell_and_local(ζ::Real, cells_per_dim::Int)
    z = unit_clamp(ζ)
    if z == 1.0
        return cells_per_dim - 1, 1.0
    end
    scaled = z * cells_per_dim
    cell = floor(Int, scaled)
    return cell, scaled - cell
end

function inverted_iqtt_evaluator(tt_iqtt, P; q::Int)
    result = InterpolativeQTT.invertqtt(tt_iqtt, P; q = q)
    R = length(tt_iqtt)
    K_out = R - q
    S = result[K_out]
    nᵦ = length(P.grid)
    cells_per_dim = 2^K_out
    β = 2.0^R / (2.0^R - 1.0)

    function eval_iqtt(t̂, ξ, η)
        # Converted space-time tensors fuse dimensions in Lindsey/TCI order
        # (η, ξ, t̂). The QTT bits represent dyadic coordinates n/2ᴿ, while
        # TensorTrainNumerics endpoint grids use n/(2ᴿ-1), hence ζ ↦ ζ/β.
        iᵧ, θᵧ = cell_and_local(unit_clamp(η) / β, cells_per_dim)
        iₓ, θₓ = cell_and_local(unit_clamp(ξ) / β, cells_per_dim)
        iₜ, θₜ = cell_and_local(unit_clamp(t̂) / β, cells_per_dim)
        row = morton_cell_flat((iᵧ, iₓ, iₜ), K_out)

        out = 0.0
        for βᵧ in 0:(nᵦ - 1), βₓ in 0:(nᵦ - 1), βₜ in 0:(nᵦ - 1)
            col = βᵧ + 1 + nᵦ * βₓ + nᵦ^2 * βₜ
            out += P(βᵧ, θᵧ) * P(βₓ, θₓ) * P(βₜ, θₜ) * S[row, col]
        end
        return out
    end

    return eval_iqtt
end

function evaluator_grid_error(U::SpaceTimeQTTvector, U_iqtt, P; q::Int)
    Û = inverted_iqtt_evaluator(U_iqtt, P; q = q)
    U_dense = spacetime_array(U)
    ks = unique(round.(Int, range(1, Nₜ, length = 4)))
    is = unique(round.(Int, range(1, Nₓ, length = 4)))
    max_abs = 0.0
    ref_abs = 0.0
    for k in ks, i in is, j in is
        t̂ = (k - 1) / (Nₜ - 1)
        ξ = (i - 1) / (Nₓ - 1)
        η = (j - 1) / (Nₓ - 1)
        ref = U_dense[k, i, j]
        max_abs = max(max_abs, abs(Û(t̂, ξ, η) - ref))
        ref_abs = max(ref_abs, abs(ref))
    end
    return max_abs / max(ref_abs, eps())
end

function nonlinear_source_qtt(U_iqtt, P; p::Int, εᵢ::Real, rₘₐₓ::Int, q::Int)
    Û = inverted_iqtt_evaluator(U_iqtt, P; q = q)
    𝒢ⁿ_unit(t̂, ξ, η) =
        ϕ(t̂_to_t(t̂), ξ_to_x(ξ), ξ_to_x(η)) - Û(t̂, ξ, η)^3
    return iqtt_space_time(𝒢ⁿ_unit, Rₜ, Rₓ; p = p, εᵢ = εᵢ, rₘₐₓ = rₘₐₓ)
end

function compress(tt::TTvector, rₘₐₓ::Int, εᵢ::Real)
    out = copy(tt)
    return tt_compress!(out, rₘₐₓ; truncerr = εᵢ, sweeps = 2)
end

function global_blocks(A::QTToperator, τ::Float64, Rₜ::Int)
    A_tt = TensorTrainNumerics._convert_tto_eltype(Float64, TToperator(A))
    Iₓ = id_tto(Float64, A.N)
    L = Iₓ - (τ / 2) * A_tt
    R = Iₓ + (τ / 2) * A_tt

    Iₜ = qtt_time_identity(Float64, Rₜ)
    Sₜ = qtt_lower_shift(Float64, Rₜ)
    𝒜 = (Iₜ ⊗ L) - (Sₜ ⊗ R)
    return 𝒜, R, Sₜ, Iₓ
end

function initial_source_tt(rₘₐₓ::Int, εᵢ::Real)
    𝒢₀_raw = reduce(+, c𝒢(m) * space_mode(m) for m in 𝓜)
    return compress(𝒢₀_raw, rₘₐₓ, εᵢ)
end

function rhs_cn(
        R::TToperator,
        Sₜ::TToperator,
        Iₓ::TToperator,
        u₀::QTTvector,
        𝒢₀::TTvector,
        𝒢_unknown::SpaceTimeQTTvector,
        τ::Float64,
        rₘₐₓ::Int,
        εᵢ::Real
    )
    Rₜ = 𝒢_unknown.time_bits
    e₁ = TensorTrainNumerics._convert_ttv_eltype(Float64, qtt_basis_vector(Rₜ, 1))
    u₀_tt = TensorTrainNumerics._convert_ttv_eltype(Float64, TTvector(u₀))
    𝒢_tt = TensorTrainNumerics._convert_ttv_eltype(Float64, TTvector(𝒢_unknown))

    linear_rhs = e₁ ⊗ (R * u₀_tt)
    previous_source = (e₁ ⊗ 𝒢₀) + ((Sₜ ⊗ Iₓ) * 𝒢_tt)
    source_rhs = (τ / 2) * (𝒢_tt + previous_source)

    return compress(linear_rhs + source_rhs, rₘₐₓ, εᵢ)
end

wrap(tt::TTvector, Rₜ::Int, Rₓ::Int) =
    SpaceTimeQTTvector(tt, Rₜ, 2, Rₓ, :serial)

als_guess(tt::TTvector) = copy(tt)

relerr(a, b) = norm(vec(a .- b)) / max(norm(vec(b)), eps())

function initial_spacetime_guess()
    terms = [
        qtt_exp(Rₜ; a = τ, b = T, α = -0.5 * m.ω) ⊗ space_mode(m)
        for m in 𝓜
    ]
    initial_st = compress(reduce(+, terms), rₘₐₓ, εᵢ)
    return wrap(initial_st, Rₜ, Rₓ)
end

function initial_iqtt_guess()
    u_guess(t, x, y) = sum(
        m.α * exp(-0.5 * m.ω * t) * sinpi(m.p * x) * sinpi(m.q * y)
            for m in 𝓜
    )
    u_guess_reversed(η, ξ, t̂) = u_guess(t̂_to_t(t̂), ξ_to_x(ξ), ξ_to_x(η))
    return build_iqtt(
        u_guess_reversed,
        Rₜ,
        Rₓ;
        p = p,
        εᵢ = εᵢ,
        rₘₐₓ = rₘₐₓ,
        upper_order = :yxt
    )
end

Pᵢ = InterpolativeQTT.getChebyshevGrid(p)
U = initial_spacetime_guess()
U_iqtt = initial_iqtt_guess()

U★ = exact_array(u★, Nₜ, τ, x⃗)
𝒜, R, Sₜ, Iₓ = global_blocks(A, τ, Rₜ)
𝒢₀ = initial_source_tt(rₘₐₓ, εᵢ)

# --- diagnostic: pure CN-discretization residual --------------------------
# Bypass Picard, the ALS solve, and the Û³ reconstruction by plugging the
# analytic exact solution and exact source 𝒢★ = ϕ - u★³ = Σ c𝒢(m) uₘ (both
# rank ≤ 3, full decay rate -ωₘ) into the assembled global-CN system.
#   ~1e-5 or below  → u★ solves the system; the 1.7e-3 floor is the numerical
#                     solve (fixed-rank ALS) → raise nₛ / try "krylov".
#   ~1.7e-3         → u★ is NOT the system's solution; the floor is in the RHS
#                     assembly (source term / boundary row / a scale factor).
let
    U★_terms = [qtt_exp(Rₜ; a = τ, b = T, α = -m.ω) ⊗ space_mode(m) for m in 𝓜]
    U★_tt = compress(reduce(+, U★_terms), rₘₐₓ, εᵢ)

    G★_terms = [
        c𝒢(m) * (qtt_exp(Rₜ; a = τ, b = T, α = -m.ω) ⊗ space_mode(m))
        for m in 𝓜
    ]
    G★_st = wrap(compress(reduce(+, G★_terms), rₘₐₓ, εᵢ), Rₜ, Rₓ)

    rhs★ = rhs_cn(R, Sₜ, Iₓ, u₀, 𝒢₀, G★_st, τ, rₘₐₓ, εᵢ)
    resid★ = norm(𝒜 * U★_tt - rhs★) / max(norm(rhs★), eps())
    @printf("diagnostic: exact-solution CN residual (rank %d) = %.6e\n\n",
        maximum(U★_tt.ttv_rks), resid★)
end
# --------------------------------------------------------------------------

println("2D+time forced Allen-Cahn global CN nonlinear Picard test")
println("exact: u★(t,x,y) = Σₘ αₘ exp(-ωₘt) sin(pₘπx) sin(qₘπy)")
println("grid: $(Nₓ) × $(Nₓ), time slices: $Nₜ, ε: $ε, τ: $τ")
println("source: ϕ = ∂ₜu★ - ε²Δₕu★ - u★ + u★³")
println("Picard source: 𝒢ⁿ = ϕ - (Ũⁿ)^3 rebuilt by invertqtt + InterpolativeQTT")
println("local solver: rank-preserving ALS")
println("initial guess: structured multi-mode guess with half-rate time decays")
println("initial fused IQTT rank: $(TCI.rank(U_iqtt))")
println()
@printf(
    "%-6s %-11s %-14s %-14s %-12s %-12s %-14s %-14s\n",
    "picard",
    "src rank",
    "linear res",
    "update",
    "max rank",
    "eval err",
    "space-time err",
    "final err"
)

let U_iter = U, U_iqtt_iter = U_iqtt
    for n in 1:nₚ
        _, 𝒢_unknown = nonlinear_source_qtt(
            U_iqtt_iter,
            Pᵢ;
            p = p,
            εᵢ = εᵢ,
            rₘₐₓ = rₘₐₓ,
            q = qᵢ
        )
        rᵣₕₛ = max(rₘₐₓ, maximum(𝒢_unknown.ttv_rks))
        rhs = rhs_cn(
            R,
            Sₜ,
            Iₓ,
            u₀,
            𝒢₀,
            𝒢_unknown,
            τ,
            rᵣₕₛ,
            εᵢ
        )

        guess_tt = als_guess(TTvector(U_iter))
        solved_tt = TensorTrainNumerics._global_tt_linsolve(
            𝒜,
            rhs,
            guess_tt,
            "als",
            rᵣₕₛ;
            sweep_count = nₛ
        )
        solved_tt = compress(solved_tt, rᵣₕₛ, εᵢ)
        U_next = wrap(solved_tt, Rₜ, Rₓ)

        linear_residual = norm(𝒜 * solved_tt - rhs) / max(norm(rhs), eps())
        update = norm(TTvector(U_next) - TTvector(U_iter)) / max(norm(TTvector(U_next)), eps())
        U_dense = spacetime_array(U_next)
        space_time_error = relerr(U_dense, U★)
        final_error = relerr(U_dense[Nₜ, :, :], U★[Nₜ, :, :])
        U_iqtt_next = spacetime_to_iqtt(U_next; εᵢ = εᵢ)
        eval_error = evaluator_grid_error(U_next, U_iqtt_next, Pᵢ; q = qᵢ)

        @printf(
            "%-6d %-11d %-14.6e %-14.6e %-12d %-12.6e %-14.6e %-14.6e\n",
            n,
            maximum(𝒢_unknown.ttv_rks),
            linear_residual,
            update,
            maximum(U_next.ttv_rks),
            eval_error,
            space_time_error,
            final_error
        )

        U_iter = U_next
        U_iqtt_iter = U_iqtt_next
    end
    global U = U_iter
    global U_iqtt = U_iqtt_iter
end

_, 𝒢_final = nonlinear_source_qtt(
    U_iqtt,
    Pᵢ;
    p = p,
    εᵢ = εᵢ,
    rₘₐₓ = rₘₐₓ,
    q = qᵢ
)
rᵣₕₛ_final = max(rₘₐₓ, maximum(𝒢_final.ttv_rks))
rhs_final = rhs_cn(R, Sₜ, Iₓ, u₀, 𝒢₀, 𝒢_final, τ, rᵣₕₛ_final, εᵢ)
final_residual = norm(𝒜 * TTvector(U) - rhs_final) / max(norm(rhs_final), eps())
U_dense = spacetime_array(U)
final_slice = U_dense[Nₜ, :, :]
space_time_error = relerr(U_dense, U★)
final_error = relerr(final_slice, U★[Nₜ, :, :])

println()
println("final global CN residual: $final_residual")
println("space-time relative error vs u★: $space_time_error")
println("final-slice relative error vs u★: $final_error")
println("final slice rank: $(maximum(space_time_slice(U, Nₜ).ttv_rks))")
println("final slice min/max: $(minimum(final_slice)) / $(maximum(final_slice))")

let
    snap = unique(round.(Int, range(1, Nₜ, length = 4)))
    U_num = [U_dense[k, :, :] for k in snap]
    U_ref = [U★[k, :, :] for k in snap]
    E = [U_num[j] .- U_ref[j] for j in eachindex(snap)]

    u_lims = extrema(reduce(vcat, [vec(S) for S in vcat(U_num, U_ref)]))
    e_abs = maximum(abs, reduce(vcat, vec.(E)))
    e_lims = (-e_abs, e_abs)

    fig = Figure(size = (1180, 760))
    hmᵤ = nothing
    hm★ = nothing
    hmₑ = nothing
    for (j, k) in enumerate(snap)
        t = k * τ
        axᵤ = Axis(
            fig[1, j],
            aspect = 1,
            xlabel = "",
            ylabel = j == 1 ? "y" : "",
            title = "u, t = $(round(t; digits = 4))"
        )
        hmᵤ = heatmap!(axᵤ, x⃗, x⃗, U_num[j]; colormap = :viridis, colorrange = u_lims)

        ax★ = Axis(
            fig[2, j],
            aspect = 1,
            xlabel = "",
            ylabel = j == 1 ? "y" : "",
            title = "u★"
        )
        hm★ = heatmap!(ax★, x⃗, x⃗, U_ref[j]; colormap = :viridis, colorrange = u_lims)

        axₑ = Axis(
            fig[3, j],
            aspect = 1,
            xlabel = "x",
            ylabel = j == 1 ? "y" : "",
            title = "u - u★"
        )
        hmₑ = heatmap!(axₑ, x⃗, x⃗, E[j]; colormap = :balance, colorrange = e_lims)
    end
    Colorbar(fig[1, length(snap) + 1], hmᵤ, label = "u")
    Colorbar(fig[2, length(snap) + 1], hm★, label = "u★")
    Colorbar(fig[3, length(snap) + 1], hmₑ, label = "u - u★")

    outpath = joinpath(@__DIR__, "global_picard_allen_cahn2d.png")
    save(outpath, fig)
    println("saved plot: $outpath")
end
