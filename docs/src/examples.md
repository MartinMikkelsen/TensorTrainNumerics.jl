# Examples

## 2D Laplace equation

Solve $\Delta u = 0$ on the unit square with $u(x,0) = \sin(\pi x)$ and zero on the other three sides. The exact solution is $u(x,y) = \sin(\pi x)\,\sinh(\pi(1-y))/\sinh(\pi)$.

The discretisation uses $N = 2^d$ interior grid points per dimension (spacing $h = 1/(N+1)$). The standard Toeplitz stencil is applied to interior points only: missing boundary values equal zero and drop out of every stencil equation.

```@example Laplace
using TensorTrainNumerics
using CairoMakie

d   = 6
N   = 2^d
h   = 1.0 / (N + 1)
xes = h .* (1:N)

# Operator: (1/h²)(Δ₁D⊗I + I⊗Δ₁D), Δ₁D = tridiag(−2,1,1)
Δ1d   = toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
A_raw = (1/h^2) * (Δ1d ⊗ id_tto(d) + id_tto(d) ⊗ Δ1d)
A     = QTToperator(A_raw, 2, d, :serial)

# Bottom BC u(x,0)=sin(πx) contributes −sin(πxᵢ)/h² to the first y-row.
b_raw = -(1/h^2) * qtt_sin(d; a = h, b = 1 - h) ⊗ qtt_basis_vector(d, 1)
b     = QTTvector(b_raw, 2, d, :serial)

x0    = QTTvector(rand_tt(b_raw.ttv_dims, b_raw.ttv_rks), 2, d, :serial)
x_sol = dmrg_linsolve(A, b, x0; sweep_count = 30, tol = 1e-12)
```

Compare the numerical solution to the exact result:

```@example Laplace
sol     = qttv_to_array(x_sol)
u_exact = [sin(π * xi) * sinh(π * (1 - yi)) / sinh(π) for xi in xes, yi in xes]

fig = Figure(size = (900, 350))
ax1 = Axis(fig[1, 1], title = "DMRG solution", xlabel = "x", ylabel = "y")
ax2 = Axis(fig[1, 2], title = "Exact solution", xlabel = "x", ylabel = "y")
ax3 = Axis(fig[1, 3], title = "|error|", xlabel = "x", ylabel = "y")
hm1 = heatmap!(ax1, xes, xes, sol;                    colormap = :roma)
hm2 = heatmap!(ax2, xes, xes, u_exact;                colormap = :roma)
hm3 = heatmap!(ax3, xes, xes, abs.(sol .- u_exact);   colormap = :viridis)
Colorbar(fig[1, 4], hm1, label = "u(x, y)")
fig
```

---

## 2D heat equation with TDVP

Solve the 2D heat equation $u_t = \kappa \Delta u$ using two-site TDVP with imaginary-time evolution.

```@example heat2d
using TensorTrainNumerics
using CairoMakie

d = 8
κ = 0.1
h = 1.0 / 2^d
p = 1.0; s = 0.0; v = 0.0
α = h^2 * v - 2 * p
β = p + h * s / 2
γ = p - h * s / 2

Δ1d = toeplitz_to_qtto(α, β, γ, d)
I1d = id_tto(d)
A   = κ * ((I1d ⊗ Δ1d) + (Δ1d ⊗ I1d))

u0    = qtt_sin(d, λ = 1 / π) ⊗ qtt_cos(d, λ = 1 / π)
dt    = 1e-2
steps = fill(dt, 500)

sol = tdvp2(A, u0, steps; imaginary_time = true, sweeps = 2, truncerr = 1e-3)
```

```@example heat2d
solution = reshape(qtt_to_function(sol), 2^d, 2^d)
x = range(0, 1, length = 2^d)
y = range(0, 1, length = 2^d)

fig = Figure()
ax  = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "TDVP heat equation")
hm  = heatmap!(ax, x, y, solution)
Colorbar(fig[1, 2], hm, label = "u(x, y)")
fig
```

---

## Time-stepping comparison

Compare the explicit Euler, implicit Euler, and Crank–Nicolson schemes on the 1D diffusion equation.

```@example TimeStepping
using TensorTrainNumerics
using CairoMakie

d = 8
h = 1.0 / 2^d
A = h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)

u0   = qtt_sin(d, λ = π)
init = rand_tt(u0.ttv_dims, u0.ttv_rks)

steps = collect(range(0.0, 10.0, 1000))

solution_explicit, error_explicit = euler_method(A, u0, steps; return_error = true)
solution_implicit, rel_implicit   = implicit_euler_method(A, u0, init, steps; return_error = true)
solution_crank,    rel_crank      = crank_nicholson_method(A, u0, init, steps;
    return_error = true, tt_solver = "mals")

xes = collect(range(0, 1, 2^d))
fig = Figure()
ax  = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "Time-stepping methods")
lines!(ax, xes, qtt_to_function(solution_explicit), label = "Explicit Euler",  linestyle = :solid, linewidth = 3)
lines!(ax, xes, qtt_to_function(solution_implicit), label = "Implicit Euler",  linestyle = :dot,   linewidth = 3)
lines!(ax, xes, qtt_to_function(solution_crank),    label = "Crank–Nicolson", linestyle = :dash,   linewidth = 3)
axislegend(ax)
fig
```

Relative errors at the final time step:

```@example TimeStepping
println("Explicit Euler:  ", error_explicit)
println("Implicit Euler:  ", rel_implicit)
println("Crank–Nicolson:  ", rel_crank)
```

---

## Discrete Fourier transform

Apply the QTT Fourier operator to a sparse-spectrum signal and verify recovery of the Fourier coefficients.

```@example DFT
using TensorTrainNumerics
using Random

d = 10
N = 2^d

Random.seed!(1234)
r = 12
coeffs = randn(r) .+ 1im * randn(r)
f(x) = sum(coeffs .* cispi.(2 .* (0:(r-1)) .* x))

F     = fourier_qtto(d; K = 50, sign = -1.0, normalize = true)
x_qtt = function_to_qtt_uniform(f, d)
y_qtt = F * x_qtt

spec  = matricize(y_qtt, d)
scale = sqrt(N)

println("Spectral recovery error: ", norm(spec[1:r] .- scale .* coeffs) / (scale * norm(coeffs)))
```

---

## TT-cross approximation

Approximate a 6-dimensional function via TT-cross without evaluating on the full $n^6$ grid.

```@example TTcross
using LinearAlgebra
using TensorTrainNumerics

f(X::Matrix{Float64}) = vec(sin.(sum(X, dims = 2)))

n = 8
d = 6
domain = [collect(range(0.0, π, length = n)) for _ in 1:d]

tt_mv    = tt_cross(f, domain, MaxVol(tol = 1.0e-8, maxiter = 20, verbose = true); ranks = 4)
tt_dmrg  = tt_cross(f, domain, DMRG(tol = 1.0e-8,  maxiter = 25, verbose = true); ranks = 4)
tt_greedy = tt_cross(f, domain, Greedy(tol = 1.0e-12, verbose = true, maxiter = 100))
```

Verify accuracy against the full reference tensor:

```@example TTcross
tensor_approx = ttv_to_tensor(tt_mv)

tensor_exact = zeros(Float64, ntuple(_ -> n, d)...)
for idx in CartesianIndices(tensor_exact)
    coords = [domain[k][idx[k]] for k in 1:d]
    tensor_exact[idx] = sin(sum(coords))
end

rel_err = norm(tensor_approx .- tensor_exact) / norm(tensor_exact)
println("Relative error: ", rel_err)
println("Maximum absolute error: ", maximum(abs.(tensor_approx .- tensor_exact)))
```

---

## Variational solver for Burgers' equation

Minimize the residual functional for one time step of Burgers' equation $u_t + \frac{1}{2}(u^2)_x = \nu u_{xx}$ using gradient descent in TT format.

```@example VariationalSolver
using TensorTrainNumerics
using OptimKit
using CairoMakie
import TensorTrainNumerics: dot, orthogonalize

orth2(x) = orthogonalize(orthogonalize(x; i = 1); i = x.N)

d  = 6
ν  = 0.01
N  = 2^d
dx = 1.0 / N
dt = 1e-3

Dx  = (1 / dx) * ∇(d)
Dxx = (1 / dx^2) * Δ_DN(d)

u0    = qtt_sin(d, λ = π / 2)
x0    = orth2(u0)
v_ref = Ref(u0)

function burgers_residual(u)
    t1 = (u - v_ref[]) * (1 / dt)
    nl = 0.5 * orth2(Dx * (u ⊕ u))
    R  = t1 + nl + ν * (Dxx * u)
    return orth2(R)
end

function burgers_cost_grad(u)
    R = burgers_residual(u)
    J = 0.5 * dx * dt * dot(R, R)
    g = (1 / dt) * R + ν * (Dxx * R)
    Dxu = orth2(Dx * u)
    g  += orth2(Dxu ⊕ R) + orth2(Dx * orth2(u ⊕ R))
    return J, orth2((dx * dt) * g)
end

solver = GradientDescent(verbosity = 0, gradtol = 1e-6)

v_ref[] = x0
for _ in 1:150
    x, _, _, _, _ = optimize(burgers_cost_grad, v_ref[], solver)
    v_ref[] = orth2(x)
end
```

```@example VariationalSolver
vals_init  = qtt_to_function(x0)
vals_final = qtt_to_function(v_ref[])
xes = (1:N) ./ N

fig = Figure()
ax  = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "Variational Burgers")
lines!(ax, xes, vals_init,  label = "initial", linewidth = 2)
lines!(ax, xes, vals_final, label = "final",   linewidth = 2)
axislegend(ax)
fig
```
