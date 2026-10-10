using LinearAlgebra
using TensorTrainNumerics
using OptimKit
using CairoMakie

# Ground state of [-½∇² + (x²+y²)/2 + g|ψ|²]ψ = μψ on a square with
# zero Dirichlet boundary conditions and ∫|ψ|² dx dy = 1. Inspired by https://arxiv.org/abs/2507.04279.

d = 5                          # QTT levels per dimension, N = 2ᵈ points
domain = 6.0
g = 100.0                      # interaction strength
max_bond = 16                  # maximum TT rank of the state
trunc_tol = 1.0e-12            # truncation tolerance of intermediate products
dt = 0.02                      # imaginary-time step
tol = 1.0e-5                   # stationary residual ‖H[u]u − μu‖
max_steps = 1000

N = 2^d
h = 2domain / (N + 1)
a, b = -domain + h, domain - h

V = tt_to_diag_tto(qtt_polynomial([0.0, 0.0, 0.5], d; a, b))
H1 = (1 / (2h^2)) * Δ(d) + V
H0 = H1 ⊗ id_tto(d) + id_tto(d) ⊗ H1

# u is Euclidean-normalized: ψ(xᵢ,yⱼ) = uᵢⱼ/h, hence g_eff = g/h².
g_eff = g / h^2

gaussian = function_to_qtt(t -> exp(-0.5 * (a + (b - a) * t)^2), d)
u0 = tt_round!(gaussian ⊗ gaussian; trunc_tol)
u0 /= norm(u0)

# Energy of u/‖u‖ and its gradient with respect to u.
function energy_and_gradient(u)
    n = norm(u)
    v = u / n
    ρ = tt_round!(hadamard(v, v); trunc_tol)
    Hv = tt_round!(H0 * v + g_eff * hadamard(ρ, v); trunc_tol)
    μ = dot(v, Hv)
    E = μ - (g_eff / 2) * dot(ρ, ρ)
    grad = (2 / n) * tt_round!(Hv - μ * v; trunc_tol)
    return E, grad
end

function measure(u)
    E, grad = energy_and_gradient(u)
    return (; seconds = (time_ns() - started) / 1.0e9, energy = E, residual = norm(u) * norm(grad) / 2)
end

function tdvp_step(u)
    ρ = tt_round!(hadamard(u, u); trunc_tol)
    H = H0 + g_eff * tt_to_diag_tto(ρ)
    # tdvp2 evolves exp(+dt*A) in imaginary time, hence -H.
    return tdvp2(
        -H, u, [dt]; imaginary_time = true, normalize = true,
        max_bond, trunc_tol, tol = 1.0e-12, show_progress = true
    )
end

function truncate_and_record!(u, E, grad, iteration)
    u = tt_round!(u; max_bond, trunc_tol)
    u /= norm(u)
    push!(history_optim, measure(u))
    return u, energy_and_gradient(u)...
end

gradient_descent(maxiter) = OptimKit.optimize(
    energy_and_gradient, copy(u0),
    OptimKit.GradientDescent(; maxiter, gradtol = 2tol, verbosity = 0);
    (finalize!) = truncate_and_record!
)

started = time_ns()
history_optim = [measure(u0)]
tdvp_step(u0)
gradient_descent(1)

started = time_ns()
u_tdvp = copy(u0)
history_tdvp = [measure(u_tdvp)]
for _ in 1:max_steps
    last(history_tdvp).residual <= tol && break
    global u_tdvp = tdvp_step(u_tdvp)
    push!(history_tdvp, measure(u_tdvp))
end

started = time_ns()
history_optim = [measure(u0)]
gradient_descent(max_steps)

fig = Figure()
ax = Axis(
    fig[1, 1]; title = "Gross–Pitaevskii ground state · g = $g · $N × $N grid",
    xlabel = "elapsed seconds", ylabel = "‖H[u]u − μu‖", yscale = log10
)
for (label, history) in (("TDVP2", history_tdvp), ("OptimKit gradient descent", history_optim))
    final = last(history)
    @info label final.energy final.residual steps = length(history) - 1 final.seconds
    lines!(ax, getproperty.(history, :seconds), getproperty.(history, :residual); label)
end
hlines!(ax, [tol]; color = :gray, linestyle = :dash, label = "tolerance")
axislegend(ax)
display(fig)
