using TensorTrainNumerics
using CairoMakie
using LinearAlgebra: opnorm, eigvals, Hermitian

bits = 8; N = 2^bits; lo, hi = -4.0, 4.0
h = (hi - lo) / (N - 1); q = 1.0
max_bond = 8; truncerr = 1.0e-8

X = ttv_to_diag_tto(qtt_polynom([0.0, 1.0], bits; a = lo, b = hi))   # diag(x)
D = (1 / (2h)) * (shift(bits) - (id_tto(bits) - ∇(bits)))            # central ∂ₓ
a = (1 / sqrt(2)) * (X + D); adag = (1 / sqrt(2)) * (X - D)          # ladder operators
I = id_tto(bits)

a1, a2, a3 = a ⊗ I ⊗ I, I ⊗ a ⊗ I, I ⊗ I ⊗ a                         # annihilation, modes 1-3
ad1, ad2, ad3 = adag ⊗ I ⊗ I, I ⊗ adag ⊗ I, I ⊗ I ⊗ adag             # creation, modes 1-3
A = ad1 * a1 + ad2 * a2 + ad3 * a3                                   # Σ n̂ᵢ  (dissipative)
C = sqrt(q / 2) * (
    ad2 * ad3 * a1 - ad1 * a3 * a2 + ad1 * ad3 * a2 -
        ad2 * a3 * a1 - 2.0 * (ad1 * ad2 * a3 - ad3 * a2 * a1)
)          # divergence-free drift
G = (-1.0) * A + C                                                   # generator

nrm(ψ) = TensorTrainNumerics.norm(ψ); ip(x, y) = TensorTrainNumerics.dot(x, y)
gaussian(c) = (g = function_to_qtt(t -> exp(-0.5 * (lo + (hi - lo) * t - c)^2), bits); (1 / nrm(g)) * g)

ground = tt_compress!(gaussian(0.0), max_bond; truncerr = truncerr)             # vacuum |0⟩
excited = tt_compress!(sqrt(q / 2) * (adag * ground), max_bond; truncerr = truncerr)  # ∝ |1⟩
ψ0 = tt_compress!(excited ⊗ ground ⊗ ground, max_bond; truncerr = truncerr)
readout = tt_compress!(gaussian(0.8) ⊗ gaussian(-0.4) ⊗ gaussian(0.2), max_bond; truncerr = truncerr)

let
    Am, Adm = qtto_to_matrix(a), qtto_to_matrix(adag)
    Dm, Idm, nm = qtto_to_matrix(D), qtto_to_matrix(I), qtto_to_matrix(adag * a)
    rv() = (z = rand_tt(a1.tto_dims, 4); (1 / nrm(z)) * z); x, y = rv(), rv()
    @assert opnorm(Adm - Am') / opnorm(Am) < 1.0e-10                         # a† = aᴴ
    @assert opnorm(Dm + Dm') / opnorm(Dm) < 1.0e-10                          # Dᵀ = -D
    @assert opnorm(nm - nm') / opnorm(nm) < 1.0e-10                          # n̂ = a†a Hermitian
    @assert minimum(eigvals(Hermitian(nm))) > -1.0e-10                       # n̂ ⪰ 0
    @assert abs(ip(x, A * y) - ip(A * x, y)) / (nrm(A * x) * nrm(y)) < 1.0e-8 # A = Aᴴ  (dissipative)
    @assert abs(ip(x, C * x)) / (nrm(x) * nrm(C * x)) < 1.0e-8                # C = -Cᴴ (transport)
    @assert nrm(a1 * (ad2 * x) - ad2 * (a1 * x)) / nrm(a1 * (ad2 * x)) < 1.0e-8 # [a₁,a₂†] = 0
    @info "operator algebra (exact identities ✓)" canonical_comm = opnorm(Am * Adm - Adm * Am - Idm) vacuum = nrm(a * ground) overlap_01 = abs(ip(ground, excited))
end

τstep = 0.001; record_dt = 0.002; T = 0.02
blk = round(Int, record_dt / τstep); nblk = round(Int, T / record_dt)
times = collect(0.0:record_dt:T)

snaps = TTvector[]; overlap = Float64[]; mass = Float64[]
function record!(ψ)
    push!(snaps, copy(ψ)); push!(overlap, ip(readout, ψ)); return push!(mass, nrm(ψ))
end

ψ = ψ0; record!(ψ)
for _ in 1:nblk
    global ψ = crank_nicholson_method(G, ψ, ψ, fill(τstep, blk); normalize = false, tt_solver = "als", max_bond = max_bond, sweep_count = 5)
    global ψ = tt_compress!(ψ, max_bond; truncerr = truncerr)
    record!(ψ)
end

@info "second-quantized KE final" bits T overlap = overlap[end] mass = mass[end] rank = maximum(ψ.ttv_rks)

let
    mid = cld(N, 2)
    slice(i) = (i - 1) * N^2 + (mid - 1) * N + mid
    xes = collect(range(lo, hi, N))
    xslice(ψ) = qtt_to_function(ψ)[slice.(1:N)]
    snap = unique(round.(Int, range(1, length(snaps), length = 5)))

    fig = Figure(size = (760, 480))
    ax = Axis(
        fig[1, 1], xlabel = "x₁", ylabel = "ψ(x₁, x₂≈0, x₃≈0)",
        title = "Second-quantized Kolmogorov  (coordinate basis, bits=$bits)"
    )
    for k in snap
        lines!(ax, xes, xslice(snaps[k]), linewidth = 2, label = "t = $(round(times[k], digits = 3))")
    end
    axislegend(ax; position = :rt)
    display(fig)
end

let
    fig = Figure(size = (1000, 420))
    ax1 = Axis(
        fig[1, 1], xlabel = "t", ylabel = "⟨readout, ψ⟩",
        title = "Readout overlap  ⟨g(0.8)⊗g(-0.4)⊗g(0.2), ψ⟩"
    )
    lines!(ax1, times, overlap, linewidth = 2.5)

    ax2 = Axis(
        fig[1, 2], xlabel = "t", ylabel = "‖ψ‖",
        title = "Norm decays under the dissipative -A"
    )
    lines!(ax2, times, mass, linewidth = 2.5)
    display(fig)
end
