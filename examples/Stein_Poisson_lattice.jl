# Stein–Poisson equation for 0+1D φ⁴ theory in multi-dimensional QTT format
#
# Background: P. Butti et al., arXiv:2609.28142.
# Free theory (λ = 0): the exact solution is linear, u = (K⁻¹e_{t0})·φ, with TT rank 2.
# Interacting theory (λ > 0): the solution is nonlinear and its TT ranks grow.
# The solution gives the transport field f = ∇u of the variance-reduced correlator estimator
#     X_t = (φ_t0 + ∇·f − ∇S·f) φ_t + f_t,
# which is unbiased for any f; it is compared with the standard estimator φ_t0 φ_t on Monte Carlo samples.

using TensorTrainNumerics
using CairoMakie
using Random, Statistics
import LinearAlgebra as LA

m2, a = 1.0, 0.5
N, t0 = 3, 2                      # time slices, source slice
n, L = 5, 6.0                     # digits per slice, field range [−L, L]
M = 2^n
x = collect(range(-L, L; length = M))
h = x[2] - x[1]

# Polynomial in the field variable of one slice, as a QTT
poly(coef) = qtt_polynomial(coef, n; a = -L, b = L)

# Dense single-slice matrix → QTT operator
function mat_to_qtto(A)
    T = zeros(ntuple(_ -> 2, 2n))
    for t in CartesianIndices(T)
        i = Tuple(t)
        T[t] = A[tuple_to_index(i[1:n]), tuple_to_index(i[(n + 1):end])]
    end
    return tt_round(tto_decomp(T); trunc_tol = 1.0e-13)
end

# Derivatives with linear-extrapolation ghost points: linear functions are differentiated
# exactly everywhere, so the free-theory solution is reproduced exactly.
D2 = zeros(M, M)
D1 = zeros(M, M)
for j in 2:(M - 1)
    D2[j, j - 1], D2[j, j], D2[j, j + 1] = 1 / h^2, -2 / h^2, 1 / h^2
    D1[j, j - 1], D1[j, j + 1] = -1 / (2h), 1 / (2h)
end
D1[1, 1], D1[1, 2], D1[M, M - 1], D1[M, M] = -1 / h, 1 / h, -1 / h, 1 / h

D2q, D1q = mat_to_qtto(D2), mat_to_qtto(D1)
xq, oneq = poly([0.0, 1.0]), poly([1.0])
Xq = tt_to_diag_tto(xq)
@assert qtto_to_matrix(D1q) ≈ D1 && qtt_to_vector(xq) ≈ x

embed(ops) = reduce(⊗, [get(ops, t, id_tto(n)) for t in 1:N])   # operator acting on chosen slices
embed_vec(vs) = reduce(⊗, [get(vs, t, oneq) for t in 1:N])      # product of single-slice functions

# L = Σ_t ∂²_t − Σ_t (∂_t S) ∂_t,  ∂_t S = (n_t/a + am²)φ_t − (φ_{t−1} + φ_{t+1})/a + 4aλφ_t³
function generator(λ)
    Lop = embed(Dict(1 => D2q))
    for t in 2:N
        Lop = tt_round(Lop + embed(Dict(t => D2q)); trunc_tol = 1.0e-12)
    end
    for t in 1:N
        κ = ((t > 1) + (t < N)) / a + a * m2
        drift = tt_round(tt_to_diag_tto(poly([0.0, κ, 0.0, 4a * λ])) * D1q; trunc_tol = 1.0e-13)
        Lop = tt_round(Lop + (-1.0) * embed(Dict(t => drift)); trunc_tol = 1.0e-12)
        t > 1 && (Lop = tt_round(Lop + (1 / a) * embed(Dict(t - 1 => Xq, t => D1q)); trunc_tol = 1.0e-12))
        t < N && (Lop = tt_round(Lop + (1 / a) * embed(Dict(t => D1q, t + 1 => Xq)); trunc_tol = 1.0e-12))
    end
    return Lop
end

# Solves L u = −g by summing implicit Euler steps of ∂_τ w = L w, w(0) = g. The sum stops
# only when the relative increment is below `tol` and the Poisson residual
# ‖L u + g‖ / ‖g‖ is below `rtol`, since the increments also shrink
# when the rank cap or the inner solver limits the accuracy of each step.
function solve_poisson(Lop, g; Δτ = 2.0, kmax = 200, tol = 1.0e-6, rtol = 1.0e-6, max_bond = 50, trunc_tol = 1.0e-10)
    alg = ImplicitEuler(; linear_solver = AMEn(; local_solver = :direct, max_bond, tol, show_progress = true), show_progress = false)
    ones_all = embed_vec(Dict())
    nall = dot(ones_all, ones_all)
    w, w_raw, u = g, g, nothing
    increment = Inf
    for k in 1:kmax
        w_raw = time_evolve(Lop, w, [Δτ], alg; guess = w_raw)
        w = tt_round(w_raw - (dot(w_raw, ones_all) / nall) * ones_all; trunc_tol)
        u = u === nothing ? Δτ * w : tt_round(u + Δτ * w; trunc_tol)
        increment = Δτ * norm(w) / norm(u)
        if increment < tol
            # Orthogonalizing first avoids the cancellation in the norm of a sum of
            # tensor trains, which would otherwise limit the residual to about √eps.
            residual = norm(orthogonalize(Lop * u + g)) / norm(g)
            residual < rtol || continue
            @info "converged" steps = k residual
            return u
        end
    end
    error("no convergence after $kmax steps: relative increment $increment, tol = $tol")
end

# --- Monte Carlo comparison ---
# f = ∇u at arbitrary φ: per-slice tables of TT blocks, multilinear interpolation between grid
# points. ∇·f is the exact derivative of the interpolant, which keeps the estimator unbiased.
function tables(F)
    return map(1:N) do t
        cores = F.cores[((t - 1) * n + 1):(t * n)]
        r = size(cores[1], 2)
        B = [Matrix{Float64}(LA.I, r, r)]
        for G in cores
            B = [b * G[i, :, :] for b in B for i in 1:2]
        end
        B
    end
end

function interp(tab, j, w, dax = 0)
    v = ones(1, 1)
    for t in 1:N
        lo, hi = tab[t][j[t] + 1], tab[t][j[t] + 2]
        v = v * (t == dax ? (hi - lo) / h : (1 - w[t]) * lo + w[t] * hi)
    end
    return v[1, 1]
end

V(z, λ) = m2 * z^2 / 2 + λ * z^4
gradS(φ, λ) = [
    a * (m2 * φ[t] + 4λ * φ[t]^3) + (t > 1 ? (φ[t] - φ[t - 1]) / a : 0.0) + (t < N ? (φ[t] - φ[t + 1]) / a : 0.0)
        for t in 1:N
]

function sample(λ, nsamp; δ = 0.8, therm = 2_000, every = 2, rng = Xoshiro(1))
    Sloc(φ, t, z) = a * V(z, λ) + (t > 1 ? (z - φ[t - 1])^2 / (2a) : 0.0) + (t < N ? (φ[t + 1] - z)^2 / (2a) : 0.0)
    φ = zeros(N)
    out = zeros(nsamp, N)
    for s in 1:(therm + every * nsamp)
        for t in 1:N
            z = φ[t] + δ * (2rand(rng) - 1)
            dS = Sloc(φ, t, z) - Sloc(φ, t, φ[t])
            (dS <= 0 || rand(rng) < exp(-dS)) && (φ[t] = z)
        end
        s > therm && (s - therm) % every == 0 && (out[(s - therm) ÷ every, :] .= φ)
    end
    return out
end

# Exact ⟨φ_t φ_t0⟩ of the open chain from the transfer matrix
function exact_correlator(λ; ng = 601)
    xg = collect(range(-L, L; length = ng))
    dx = xg[2] - xg[1]
    T = [dx * exp(-(p - q)^2 / (2a) - a * (V(p, λ) + V(q, λ)) / 2) for p in xg, q in xg]
    b = sqrt(dx) .* exp.(-a .* V.(xg, λ) ./ 2)
    X = LA.Diagonal(xg)
    Z = b' * T^(N - 1) * b
    return [(b' * T^(min(t, t0) - 1) * X * T^abs(t - t0) * X * T^(N - max(t, t0)) * b) / Z for t in 1:N]
end

block_err(v; nb = 20) = std([mean(c) for c in Iterators.partition(v, length(v) ÷ nb)][1:nb]) / sqrt(nb)

function estimators(u, λ; nsamp = 20_000)
    tab = [tables(tt_round(embed(Dict(t => D1q)) * u; trunc_tol = 1.0e-12)) for t in 1:N]
    Φ = sample(λ, nsamp)
    naive = Φ[:, t0] .* Φ
    improved = similar(naive)
    for s in axes(Φ, 1)
        φ = Φ[s, :]
        pos = (φ .+ L) ./ h
        j = clamp.(floor.(Int, pos), 0, M - 2)
        w = pos .- j
        f = [interp(tab[t], j, w) for t in 1:N]
        weight = φ[t0] + sum(interp(tab[t], j, w, t) for t in 1:N) - sum(gradS(φ, λ) .* f)
        improved[s, :] .= weight .* φ .+ f
    end
    return naive, improved
end

g = embed_vec(Dict(t0 => xq))
ranks = Dict{Float64, Vector{Int}}()
solutions = Dict{Float64, Array{Float64, N}}()
correlators = Dict{Float64, Any}()
for λ in (0.0, 1.0)
    local u, naive, improved, C
    Lop = generator(λ)
    seconds = @elapsed u = solve_poisson(Lop, g)
    ranks[λ] = tt_round(u; trunc_tol = 1.0e-8).ranks
    @info "TT ranks of u" λ ranks = ranks[λ]
    λ == 1 && @info "solve time" λ seconds
    # `qtt_to_vector` orders the grid points with slice 1 most significant, so the
    # reshaped array has slice N along its first dimension.
    solutions[λ] = permutedims(reshape(qtt_to_vector(u), ntuple(_ -> M, N)), N:-1:1)
    if λ == 0
        K = LA.SymTridiagonal([((t > 1) + (t < N)) / a + a * m2 for t in 1:N], fill(-1 / a, N - 1))
        c = K \ [t == t0 ? 1.0 : 0.0 for t in 1:N]
        u_exact = tt_round(sum(c[t] * embed_vec(Dict(t => xq)) for t in 1:N); trunc_tol = 1.0e-12)
        err = norm(orthogonalize(u - u_exact)) / norm(u_exact)
        @info "relative error to exact solution (K⁻¹e_t0)·φ" err
        @assert maximum(ranks[λ]) <= 2 && err < 1.0e-6
    end

    naive, improved = estimators(u, λ)
    C = exact_correlator(λ)
    correlators[λ] = (; C, naive, improved)
    println("\nλ = $λ: ⟨φ_t φ_$t0⟩, exact vs. standard MC vs. Stein–Poisson")
    for t in 1:N
        println(
            "  t = $t   exact ", round(C[t], sigdigits = 5),
            "   MC ", round(mean(naive[:, t]), sigdigits = 5), " ± ", round(block_err(naive[:, t]), sigdigits = 2),
            "   SP ", round(mean(improved[:, t]), sigdigits = 5), " ± ", round(block_err(improved[:, t]), sigdigits = 2),
            "   variance ratio ", round(var(improved[:, t]) / var(naive[:, t]), sigdigits = 3)
        )
    end
end

let
    fig = Figure(size = (900, 760))
    ax = Axis(
        fig[1, 1:4]; xlabel = "bond", ylabel = "rank of u",
        title = "Stein-Poisson solution in QTT format"
    )
    for λ in (0.0, 1.0)
        scatterlines!(ax, collect(0:(N * n)), ranks[λ]; label = "λ = $λ")
    end
    vlines!(ax, [n * t for t in 1:(N - 1)]; color = :gray, linestyle = :dash)
    axislegend(ax; position = :lt)

    # u on the plane spanned by the source slice and one neighbor; the other slices
    # are fixed at the grid point closest to zero.
    tn = t0 == 1 ? 2 : t0 - 1
    mid = M ÷ 2 + 1
    for (col, λ) in enumerate((0.0, 1.0))
        plane = solutions[λ][ntuple(t -> t in (t0, tn) ? Colon() : mid, N)...]
        t0 < tn || (plane = permutedims(plane))
        umax = maximum(abs, plane)
        axh = Axis(
            fig[2, 2col - 1]; xlabel = rich("φ", subscript("$t0")), ylabel = rich("φ", subscript("$tn")), aspect = 1,
            title = "u at λ = $λ"
        )
        hm = heatmap!(axh, x, x, plane; colormap = :balance, colorrange = (-umax, umax))
        Colorbar(fig[2, 2col], hm)
    end
    display(fig)
end

# Correlator: exact, standard MC and Stein–Poisson estimator (negative noisy means are not shown)
let
    fig = Figure(size = (900, 400))
    for (col, λ) in enumerate((0.0, 1.0))
        (; C, naive, improved) = correlators[λ]
        ax = Axis(
            fig[1, col]; yscale = log10, xlabel = "t",
            ylabel = rich("⟨φ", subscript("t"), "φ", subscript("$t0"), "⟩"), title = "λ = $λ"
        )
        lines!(ax, 1:N, C; color = :black, label = "exact")
        for (X, label, color) in ((naive, "standard MC", :tomato), (improved, "Stein-Poisson", :dodgerblue))
            m = vec(mean(X; dims = 1))
            e = [block_err(X[:, t]) for t in 1:N]
            ok = findall(m .> 0)
            errorbars!(ax, ok, m[ok], min.(e[ok], 0.99 .* m[ok]), e[ok]; color)
            scatter!(ax, ok, m[ok]; color, label)
        end
        col == 1 && axislegend(ax; position = :lb)
    end
    display(fig)
end