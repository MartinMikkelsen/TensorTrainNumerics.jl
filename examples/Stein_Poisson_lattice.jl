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
n, L = 5, 5.0                     # digits per slice, field range [−L, L]
M = 2^n
x = collect(range(-L, L; length = M))
h = x[2] - x[1]

# Dense single-slice objects → QTT
compress(A; tol = 1.0e-12) = TensorTrainNumerics.ttv_to_tto(tt_round(tto_to_ttv(A); trunc_tol = tol))

function vec_to_qtt(v)
    T = zeros(ntuple(_ -> 2, n))
    for t in CartesianIndices(T)
        T[t] = v[tuple_to_index(Tuple(t))]
    end
    return tt_round(ttv_decomp(T); trunc_tol = 1.0e-14)
end

function mat_to_qtto(A)
    T = zeros(ntuple(_ -> 2, 2n))
    for t in CartesianIndices(T)
        i = Tuple(t)
        T[t] = A[tuple_to_index(i[1:n]), tuple_to_index(i[(n + 1):end])]
    end
    return compress(tto_decomp(T); tol = 1.0e-13)
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

D2q, D1q, Xq = mat_to_qtto(D2), mat_to_qtto(D1), mat_to_qtto(LA.diagm(x))
xq, oneq = vec_to_qtt(x), vec_to_qtt(ones(M))
@assert qtto_to_matrix(D1q) ≈ D1 && qtt_to_vector(xq) ≈ x

embed(ops) = reduce(⊗, [get(ops, t, id_tto(n)) for t in 1:N])   # operator acting on chosen slices
embed_vec(vs) = reduce(⊗, [get(vs, t, oneq) for t in 1:N])      # product of single-slice functions

# L = Σ_t ∂²_t − Σ_t (∂_t S) ∂_t,  ∂_t S = (n_t/a + am²)φ_t − (φ_{t−1} + φ_{t+1})/a + 4aλφ_t³
function generator(λ)
    Lop = embed(Dict(1 => D2q))
    for t in 2:N
        Lop = compress(Lop + embed(Dict(t => D2q)))
    end
    for t in 1:N
        κ = ((t > 1) + (t < N)) / a + a * m2
        drift = mat_to_qtto(LA.Diagonal(κ .* x .+ 4a * λ .* x .^ 3) * D1)
        Lop = compress(Lop + (-1.0) * embed(Dict(t => drift)))
        t > 1 && (Lop = compress(Lop + (1 / a) * embed(Dict(t - 1 => Xq, t => D1q))))
        t < N && (Lop = compress(Lop + (1 / a) * embed(Dict(t => D1q, t + 1 => Xq))))
    end
    return Lop
end

function solve_poisson(Lop, g; Δτ = 2.0, kmax = 200, tol = 1.0e-8, max_bond = 20, trunc_tol = 1.0e-10)
    alg = MALS(max_sweeps = 4, max_bond = max_bond, trunc_tol = trunc_tol, show_progress = false)
    ones_all = embed_vec(Dict())
    nall = dot(ones_all, ones_all)
    w, w_raw, u = g, g, nothing
    for k in 1:kmax
        w_raw = implicit_euler_method(Lop, w, w_raw, [Δτ]; alg = alg, show_progress = false)
        w = tt_round(w_raw - (dot(w_raw, ones_all) / nall) * ones_all; trunc_tol = trunc_tol)
        u = u === nothing ? Δτ * w : tt_round(u + Δτ * w; trunc_tol = trunc_tol)
        if Δτ * norm(w) / norm(u) < tol
            println("  converged after $k steps")
            return u
        end
    end
    @warn "not converged"
    return u
end

g = embed_vec(Dict(t0 => xq))
ranks = Dict{Float64, Vector{Int}}()
for λ in (0.0, 1.0)
    println("λ = $λ")
    u = solve_poisson(generator(λ), g)
    ranks[λ] = tt_round(u; trunc_tol = 1.0e-8).ttv_rks
    println("  TT ranks of u: ", ranks[λ])
    if λ == 0
        K = LA.SymTridiagonal([((t > 1) + (t < N)) / a + a * m2 for t in 1:N], fill(-1 / a, N - 1))
        c = K \ [t == t0 ? 1.0 : 0.0 for t in 1:N]
        u_exact = tt_round(sum(c[t] * embed_vec(Dict(t => xq)) for t in 1:N); trunc_tol = 1.0e-12)
        err = norm(u - u_exact) / norm(u_exact)
        println("  relative error to exact solution (K⁻¹e_t0)·φ: ", err)
        @assert maximum(ranks[λ]) <= 2 && err < 1.0e-6
    end
end

let
    fig = Figure(size = (650, 380))
    ax = Axis(
        fig[1, 1]; xlabel = "bond", ylabel = "rank of u",
        title = "Stein–Poisson solution in QTT format"
    )
    for λ in (0.0, 1.0)
        scatterlines!(ax, collect(0:(N * n)), ranks[λ]; label = "λ = $λ")
    end
    vlines!(ax, [n * t for t in 1:(N - 1)]; color = :gray, linestyle = :dash)
    axislegend(ax; position = :lt)
    display(fig)
end
