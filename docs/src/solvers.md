# Solver Guide

TensorTrainNumerics.jl provides four families of iterative solvers for problems in tensor-train format: **ALS**, **MALS**, **DMRG**, and **TDVP**. In addition, three time-stepping methods are available for evolution problems.

All solvers operate on `AbstractTTvector` and `AbstractTToperator` inputs, so they accept both plain `TTvector`/`TToperator` and the `QTTvector`/`QTToperator` wrappers transparently.

---

## ALS, MALS, DMRG — alternating sweep solvers

These three solvers address linear systems $Ax = b$ and eigenvalue problems $Ax = \lambda x$ by sweeping back and forth over TT sites and updating one (ALS) or two (MALS, DMRG) cores at a time by solving a small dense local problem. They differ in how bond dimensions are managed.

| Property | ALS | MALS | DMRG |
|---|---|---|---|
| Bond dimensions | Fixed | Adaptive (SVD) | Adaptive (SVD) |
| Local problem | Single-site | Two-site | Two-site |
| Memory per sweep | Low | Moderate | Moderate–high |
| Convergence | Moderate | Often faster | Often fastest |
| Linear solve | `als_linsolve` | `mals_linsolve` | `dmrg_linsolve` |
| Eigenvalue solve | `als_eigsolve` | `mals_eigsolve` | `dmrg_eigsolve` |

### ALS

ALS holds the bond dimensions fixed and updates one core per step. It converges reliably when a good initial rank is provided, and has the lowest memory footprint.

```@example als
using TensorTrainNumerics

d = 6
dims = ntuple(_ -> 2, d)
A = rand_tto(dims, 3)
b = rand_tt(dims, [1; fill(3, d - 1); 1])
x0 = rand_tt(dims, [1; fill(2, d - 1); 1])

x_als = als_linsolve(A, b, x0; sweep_count = 4)
```

For eigenvalue problems use `als_eigsolve`:

```@example als
E, x_eig = als_eigsolve(A, x0; sweep_schedule = [4])
println("Lowest eigenvalue: ", E[end])
```

### MALS

MALS updates two adjacent cores simultaneously, then SVD-truncates the merged core to control rank growth. This allows the bond dimensions to adapt automatically.

```@example mals
using TensorTrainNumerics

d = 6
dims = ntuple(_ -> 2, d)
A = rand_tto(dims, 3)
b = rand_tt(dims, [1; fill(3, d - 1); 1])
x0 = rand_tt(dims, [1; fill(2, d - 1); 1])

x_mals = mals_linsolve(A, b, x0; tol = 1e-10)
E_mals, x_eig_mals = mals_eigsolve(A, x0; sweep_schedule = [4])
```

### DMRG

DMRG uses the same two-site update as MALS but includes richer local subspace expansion strategies that accelerate convergence, especially for eigenvalue problems. The `rmax_schedule` controls the maximum bond dimension at each sweep stage.

```@example dmrg
using TensorTrainNumerics

d = 4
dims = ntuple(_ -> 2, d)
A = rand_tto(dims, 3)
b = rand_tt(dims, [1; fill(2, d - 1); 1])
x0 = rand_tt(dims, [1; fill(2, d - 1); 1])

x_dmrg = dmrg_linsolve(A, b, x0; sweep_count = 20, tol = 1e-12)

sweep_schedule = [2, 4, 8]
rmax_schedule  = [2, 3, 4]
E_dmrg, x_eig, r_hist = dmrg_eigsolve(A, x0;
    sweep_schedule = sweep_schedule,
    rmax_schedule  = rmax_schedule,
    tol = 1e-12)

println("Lowest eigenvalue: ", E_dmrg[end])
println("Rank history: ", r_hist)
```

---

## TDVP — time-dependent variational principle

TDVP evolves a TT-vector under the equation $\dot{u} = A u$ while keeping the state on the TT manifold of fixed (or bounded) rank. Two variants are available:

- **`tdvp`** — single-site TDVP, fixed rank, lower cost per step.
- **`tdvp2`** — two-site TDVP with SVD truncation, adaptive rank.

Both support **real-time** evolution (default) and **imaginary-time** evolution (`imaginary_time = true`), which acts as a variational ground-state finder by computing $e^{-A\tau} u_0 / \|e^{-A\tau} u_0\|$.

```@example tdvp
using TensorTrainNumerics
using CairoMakie

d = 8
h = 1.0 / (2^d - 1)
A = h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)

u0 = qtt_sin(d, λ = π)
dt = 1e-2
steps = fill(dt, 500)

sol_tdvp  = tdvp(A, u0, steps;  imaginary_time = true, sweeps = 4)
sol_tdvp2 = tdvp2(A, u0, steps; imaginary_time = true, sweeps = 2, max_bond = 8)

xes = LinRange(0, 1, 2^d)
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "TDVP imaginary-time evolution")
lines!(ax, xes, qtt_to_function(sol_tdvp),  label = "TDVP",  linewidth = 2)
lines!(ax, xes, qtt_to_function(sol_tdvp2), label = "TDVP2", linewidth = 2, linestyle = :dash)
axislegend(ax)
fig
```

---

## Time-stepping methods

For the parabolic problem $u_t = A u$, $u(0) = u_0$, three classical time-stepping schemes are provided. Each returns the evolved TT-vector and optionally a relative-error history.

| Function | Scheme | Stability |
|---|---|---|
| `euler_method` | Explicit (forward) Euler | Conditionally stable, $\Delta t < 2/\|A\|$ |
| `implicit_euler_method` | Implicit (backward) Euler | Unconditionally stable |
| `crank_nicholson_method` | Crank–Nicolson | Unconditionally stable, second-order |
| `expintegrator` | Krylov exponential integrator | Exact up to Krylov tolerance |

```@example timestep
using TensorTrainNumerics
using CairoMakie
using KrylovKit

d = 8
N = 2^d
h = 1.0 / (N - 1)
xes = LinRange(0, 1, N)

A    = h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
u0   = qtt_sin(d, λ = π)
init = rand_tt(u0.ttv_dims, u0.ttv_rks)

steps = collect(range(0.0, 5.0, 500))

sol_impl, err_impl  = implicit_euler_method(A, u0, init, steps;
    return_error = true, normalize = false)
sol_cn, err_cn      = crank_nicholson_method(A, u0, init, steps;
    return_error = true, tt_solver = "mals", normalize = false)
sol_krylov, _       = expintegrator(A, last(steps), u0)

fig = Figure()
ax  = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "Time-stepping comparison")
lines!(ax, xes, qtt_to_function(sol_impl),   label = "Implicit Euler",  linestyle = :dot,  linewidth = 3)
lines!(ax, xes, qtt_to_function(sol_cn),     label = "Crank–Nicolson", linestyle = :dash, linewidth = 3)
lines!(ax, xes, qtt_to_function(sol_krylov), label = "Krylov exp.",    linestyle = :solid, linewidth = 3)
axislegend(ax)
fig
```

---

## Choosing a solver

**ALS** is the right starting point when you already know a good rank and want low memory use.

**MALS or DMRG** are better when the target rank is unknown: they grow bonds during sweeps and SVD-truncate them down, so they self-tune. DMRG is often the fastest to converge for eigenvalue problems.

**TDVP** is the method of choice for time evolution: it respects the TT manifold geometry and avoids the rank blowup that naive time-stepping causes.

**Exponential integrators** give the most accurate result for diffusion-type problems at large time steps, at the cost of Krylov subspace construction per step.
