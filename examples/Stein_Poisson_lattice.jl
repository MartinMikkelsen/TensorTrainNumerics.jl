# Stein–Poisson equation for 0+1D φ⁴ theory in multi-dimensional QTT format
#
# Background: P. Butti et al., arXiv:2609.28142.
# Free theory (λ = 0): the exact solution is linear, u = (K⁻¹e_{t0})·φ, with TT rank 2.
# Interacting theory (λ > 0): the solution is nonlinear and its TT ranks grow.

using TensorTrainNumerics
using CairoMakie
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

g = embed_vec(Dict(t0 => xq))
ranks = Dict{Float64, Vector{Int}}()
solutions = Dict{Float64, Array{Float64, N}}()
for λ in (0.0, 1.0)
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
