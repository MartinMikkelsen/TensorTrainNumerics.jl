using TensorTrainNumerics
using LinearAlgebra
using Random
using CairoMakie

# Compare ALS, MALS, DMRG, and AMEn on the 2D Poisson problem −Δu = 1 with
# Dirichlet boundary conditions on a 2^L × 2^L grid in QTT format. The operator
# carries the 1/h² scaling, so its condition number grows like 4^L. Each solver
# is run with increasing effort, and the residual is plotted against wall time.

function poisson_2d(L)
    h = 1 / (2^L + 1)
    Δ1 = (1 / h^2) * toeplitz_to_qtto(2.0, -1.0, -1.0, L)
    A = Δ1 ⊗ id_tto(L) + id_tto(L) ⊗ Δ1
    dims = ntuple(_ -> 2, 2L)
    b = TTvector{Float64, 2L}([ones(2, 1, 1) for _ in 1:2L], dims, ones(Int, 2L + 1))
    return A, b, dims
end

# Algorithm objects of increasing effort. ALS keeps the ranks of its guess, so
# it starts from rank `als_rank`; the other solvers start from rank 1.
als_rank = 12
solvers = [
    ("ALS", [ALS(; max_sweeps, show_progress = false) for max_sweeps in (2, 4, 8)], als_rank),
    ("MALS", [MALS(; max_sweeps, trunc_tol = 1.0e-8, max_bond = 40, show_progress = false) for max_sweeps in (2, 4, 8)], 1),
    ("DMRG", [DMRG(; max_sweeps, trunc_tol = 1.0e-8, max_bond = 40, show_progress = false) for max_sweeps in (2, 4, 8)], 1),
    ("AMEn", [AMEn(; tol, max_sweeps = 30, show_progress = false) for tol in (1.0e-2, 1.0e-4, 1.0e-6)], 1),
]

# Pairs `(seconds, residual)` of every solver at every effort level.
function compare(L)
    A, b, dims = poisson_2d(L)
    residual(x) = norm(orthogonalize(A * x - b)) / norm(b)
    results = Dict{String, Vector{NTuple{2, Float64}}}()
    for (solver, algs, rank) in solvers
        results[solver] = map(algs) do alg
            Random.seed!(1)
            x0 = rand_tt(dims, rank)
            seconds = @elapsed x = linear_solve(A, b, x0, alg)
            @info "2D Poisson, L = $L" solver residual = residual(x) max_rank = maximum(x.ttv_rks) seconds
            (seconds, residual(x))
        end
    end
    return results
end

# A smaller problem first, so that the timings below exclude compilation. It
# is large enough to reach the iterative local solvers.
compare(6)

L = 8
results = compare(L)

fig = Figure(size = (800, 500))
ax = Axis(
    fig[1, 1];
    xlabel = "time (s)", ylabel = "‖A x − b‖ / ‖b‖", xscale = log10, yscale = log10,
    title = "2D Poisson, $(2^L) × $(2^L) grid"
)
for (solver, _, _) in solvers
    scatterlines!(ax, first.(results[solver]), last.(results[solver]); label = solver, linewidth = 2)
end
axislegend(ax; position = :lb)
display(fig)
