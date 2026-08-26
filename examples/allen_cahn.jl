using TensorTrainNumerics
using CairoMakie
using LinearAlgebra

ε = 0.02
L0, L = 6, 12
χ = 12
g_ac = 1 / (2 * ε^2)

A_builder(d) = (4.0^d / 2) * Δ(d) - g_ac * id_tto(d)

alg = PenaltyALS(; local_solver = :newton, η_schedule = [0.0], tol = 1.0e-10, max_sweeps = 60)

# discrete φ⁴ energy readout: E = h·(uᵀAu + (g/2)Σu⁴) + N·h/(4ε²)
function phi4_energy(u::TTvector, d::Int)
    h = 2.0^(-d)
    w = hadamard(u, u)
    return h * (dot(u, A_builder(d) * u) + (g_ac / 2) * dot(w, w)) + 2^d * h / (4 * ε^2)
end

wall_profile(x) = tanh(x / (sqrt(2) * ε)) * tanh((1 - x) / (sqrt(2) * ε))

# coarse solve, then multigrid: prolong → truncate to χ → re-solve
function allen_cahn_mgr(L0::Int, L::Int, χ::Int)
    seed = function_to_qtt(wall_profile, L0)
    u = orthogonalize(seed + (1.0e-3 * norm(seed)) * rand_tt(ntuple(_ -> 2, L0), 4; normalise = true))
    u = non_linear_solve(A_builder(L0), u, alg; g = g_ac)
    println("level d = $L0:  E = $(phi4_energy(u, L0))")
    for d in (L0 + 1):L
        u = qtto_linear_prolongation(d - 1) * u
        tt_compress!(u, χ)
        u = non_linear_solve(A_builder(d), u, alg; g = g_ac)
        println("level d = $d:  E = $(phi4_energy(u, d))")
    end
    return u
end

u = allen_cahn_mgr(L0, L, χ)
E = phi4_energy(u, L)
println("E(L=$L, χ=$χ) = $E   [2 walls, analytic 2√2/(3ε) = $(2 * sqrt(2) / (3ε))]")

# dense damped-Newton reference on the same 2^L grid (tridiagonal, O(N) per step)
let N = 2^L
    A_dense = SymTridiagonal(fill(4.0^L - g_ac, N), fill(-(4.0^L) / 2, N - 1))
    f = [wall_profile(m / N) for m in 1:N]
    for _ in 1:100
        r = A_dense * f .+ g_ac .* f .^ 3
        norm(r) < 1.0e-11 * g_ac && break
        f -= SymTridiagonal(A_dense.dv .+ 3g_ac .* f .^ 2, A_dense.ev) \ r
    end
    E_dense = 2.0^(-L) * (dot(f, A_dense * f) + (g_ac / 2) * sum(f .^ 4)) + 1 / (4 * ε^2)
    v = qtt_to_function(u)
    println("vs dense Newton:  ΔE/E = $(abs(E - E_dense) / abs(E_dense)),  max|u − u_dense| = $(maximum(abs.(v - f)))")
end

# plot: full domain-wall state + zoom on the left wall layer vs the analytic tanh kink
x = (1:(2^L)) ./ 2^L
v = qtt_to_function(u)
zoom = x .≤ 8 * sqrt(2) * ε

fig = Figure(size = (900, 400))
ax1 = Axis(
    fig[1, 1]; xlabel = "x", ylabel = "u(x)",
    title = "Allen–Cahn domain-wall state (QTT multigrid, L = $L, ε = $ε)"
)
lines!(ax1, x, v; label = "QTT (E = $(round(E; digits = 4)))")
lines!(ax1, x, wall_profile.(x); linestyle = :dash, label = "tanh product ansatz")
axislegend(ax1; position = :cb)
ax2 = Axis(fig[1, 2]; xlabel = "x", ylabel = "u(x)", title = "left wall layer (width √2·ε)")
lines!(ax2, x[zoom], v[zoom]; label = "QTT")
lines!(ax2, x[zoom], tanh.(x[zoom] ./ (sqrt(2) * ε)); linestyle = :dash, label = "tanh(x/√2ε)")
axislegend(ax2; position = :rb)
display(fig)
