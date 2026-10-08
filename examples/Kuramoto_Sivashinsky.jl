using TensorTrainNumerics
using CairoMakie
using LinearAlgebra
using OptimKit
using Zygote
using InterpolativeQTT
import TensorTrainNumerics.increase_ranks as increase_ranks

Lₓ, Lₜ = 6, 5
ℓ, T = 32π, 5.0
u₀(x) = cos(x / 16) * (1 + sin(x / 16))
χ = 8
ε = 1.0e-2
maxiter = 8000

h, Δt = ℓ / 2^Lₓ, T / 2^Lₜ
d = Lₓ + Lₜ

# QTT of the initial condition on the periodic grid x = j/2^Lₓ, by Chebyshev interpolation of the given degree.
degree = 16
û₀ = to_ttvector(interpolatesinglescale(x -> u₀(ℓ * x), 0.0, 1.0, Lₓ, degree))

# Periodic shift (Su)ₖ = u_{k+1 mod 2^L}: the open shift plus the corner entry |2^L−1⟩⟨0|.
function periodic_shift(L)
    corner = zeros_tto(ntuple(_ -> 2, L), ones(Int, L + 1))
    for k in 1:L
        corner.tto_vec[k][2, 1, 1, 1] = 1.0
    end
    return shift(L) + corner
end

𝟙ₓ, 𝟙ₜ = id_tto(Lₓ), id_tto(Lₜ)
S = periodic_shift(Lₓ)

∂ₜ = (1 / Δt) * (𝟙ₓ ⊗ ∇(Lₜ))
∂ₓ = tt_round!((1 / 2h) * (S - S') ⊗ 𝟙ₜ; trunc_tol = 1.0e-8)
∂ₓₓ = (-1 / h^2) * Δ_P(Lₓ) ⊗ 𝟙ₜ
∂ₓₓₓₓ = (1 / h^4) * (Δ_P(Lₓ) ∙ Δ_P(Lₓ)) ⊗ 𝟙ₜ
𝒜 = tt_round!(∂ₜ + ∂ₓₓ + ∂ₓₓₓₓ; trunc_tol = 1.0e-8)

f = (û₀ ⊗ qtt_basis_vector(Lₜ, 1)) / Δt

R(u) = 𝒜 * u + (u ⊕ (∂ₓ * u)) - f

# Split the flat vector θ into TT cores of sizes (dims[k], rks[k], rks[k+1]).
function unpack(θ, dims, rks)
    lengths = [dims[k] * rks[k] * rks[k + 1] for k in eachindex(dims)]
    offsets = cumsum(lengths) .- lengths
    cores = map(eachindex(dims)) do k
        reshape(θ[offsets[k] .+ (1:lengths[k])], dims[k], rks[k], rks[k + 1])
    end
    return TTvector(length(dims), cores, dims, rks, zeros(Int, length(dims)))
end

pack(u) = reduce(vcat, vec.(u.ttv_vec))

# Scaled mean-square residual of the TT with cores θ, and its gradient.
function fg(θ, dims, rks)
    loss(θ) = (r = R(unpack(θ, dims, rks)); real(r ⋅ r) * Δt^2 / 2^d)
    return loss(θ), only(Zygote.gradient(loss, θ))
end

𝟏ₜ = TTvector(Lₜ, [ones(2, 1, 1) for _ in 1:Lₜ], ntuple(_ -> 2, Lₜ), ones(Int, Lₜ + 1), zeros(Int, Lₜ))
u = increase_ranks(û₀ ⊗ 𝟏ₜ, χ; noise = ε)

θ, loss, = optimize(θ -> fg(θ, u.ttv_dims, u.ttv_rks), pack(u), LBFGS(100; maxiter, gradtol = 1.0e-8, verbosity = 0))
u = unpack(θ, u.ttv_dims, u.ttv_rks)
@info "space-time solve, χ = $χ" mean_square_residual = loss

# Rows of the reshaped vector are x (x bits are most significant), columns are t.
U = reshape(qtt_to_vector(u), 2^Lₜ, 2^Lₓ)'
x = (0:(2^Lₓ - 1)) .* h
t = (1:(2^Lₜ)) .* Δt

fig = Figure(size = (600, 400))
ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "u(x, t)", title = "Kuramoto–Sivashinsky, QTT space-time solve (χ = $χ)")
for k in (1, 2^Lₜ)
    lines!(ax, x, U[:, k]; label = "t = $(round(t[k]; digits = 2))")
end
axislegend(ax; position = :lb)
display(fig)
