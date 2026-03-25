# Lorenz QTT Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `examples/Lorenz.jl` — a self-contained example that evolves a Gaussian probability density on a 3D QTT grid under the Lorenz Liouville equation using TCI and `rk4_method`, then renders the result as a 3D isosurface in GLMakie.

**Architecture:** Sequential QTT with 3d cores (d x-bits, d y-bits, d z-bits). TCI builds the initial Gaussian. The Liouville operator is assembled from coordinate QTTs (`function_to_qtt` + `⊗ ones`), diagonal operators (`ttv_to_diag_tto`), and scaled finite-difference derivatives (`∇(d) ⊗ id ⊗ id`). `rk4_method` evolves the state; GLMakie renders the final density with `volume!`.

**Tech Stack:** TensorTrainNumerics.jl, GLMakie.jl. No new package dependencies — both are already in the project environment.

**Note on commits:** Do NOT commit autonomously. Leave commits for the user.

**Spec:** `docs/superpowers/specs/2026-03-24-lorenz-qtt-example-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `examples/Lorenz.jl` | Create | Entire example: parameters, TCI initial condition, operator, evolution, visualization |

No library files are touched. This is a standalone script.

---

## Variable Name Convention (important — avoid Julia clashes)

| Quantity | Variable name | Reason |
|----------|--------------|--------|
| Lorenz σ | `σ_L` | Avoid clash with Gaussian spread |
| Lorenz ρ | `ρ_L` | `ρ` is the density function |
| Lorenz β | `β_L` | `β` used elsewhere |
| Gaussian spread | `σ_x`, `σ_y`, `σ_z` | |
| Number of QTT bits | `d` | |
| Grid size per dim | `N = 2^d` | |

---

## Task 1: File Scaffold + Grid Parameters

**Files:**
- Create: `examples/Lorenz.jl`

- [ ] **Step 1.1: Create the file with imports and physical parameters**

```julia
using TensorTrainNumerics
using GLMakie

# ── Lorenz parameters ────────────────────────────────────────────────────────
const σ_L = 10.0
const ρ_L = 28.0
const β_L = 8/3

# ── QTT grid ─────────────────────────────────────────────────────────────────
const d = 5          # bits per physical dimension; set to 6 for higher resolution
const N = 2^d        # grid points per axis (32 for d=5)

# Physical domain — contains the Lorenz attractor
const xmin, xmax = -25.0, 25.0
const ymin, ymax = -30.0, 30.0
const zmin, zmax =   0.0, 60.0

# Grid spacings for finite-difference scaling
const h_x = (xmax - xmin) / (N - 1)
const h_y = (ymax - ymin) / (N - 1)
const h_z = (zmax - zmin) / (N - 1)

println("Grid: $(N)³ = $(N^3) points  |  h_x=$(round(h_x,digits=3))  h_y=$(round(h_y,digits=3))  h_z=$(round(h_z,digits=3))")
```

- [ ] **Step 1.2: Run and verify output**

```bash
cd /Users/pzb464/Documents/TensorTrainNumerics.jl
julia --project=. examples/Lorenz.jl
```

Expected output (d=5):
```
Grid: 32³ = 32768 points  |  h_x=1.613  h_y=1.935  h_z=1.935
```

---

## Task 2: Initial Condition via TCI

**Files:**
- Modify: `examples/Lorenz.jl` (append)

TCI is called with a **binary domain** (`[1,2]` per dimension × `3d` total dimensions).
Each row of the input matrix `bits` encodes one sample as a multi-index of 1s and 2s.
`index_to_point(t; L)` maps such a tuple to a physical coordinate in `[0, L]`, so the
physical coordinate is `a + index_to_point(bits[s, k_range]; L = b - a)`.

- [ ] **Step 2.1: Append initial condition block**

```julia
# ── Initial Gaussian density (via TCI on binary QTT domain) ──────────────────
const x₀, y₀, z₀ = 1.0, 1.0, 1.0    # center (near the classical Lorenz start)
const σ_x, σ_y, σ_z = 2.0, 2.0, 2.0  # Gaussian spread

# Binary domain: 3d dimensions each with values {1, 2}
domain_ic = [collect(1:2) for _ in 1:(3d)]

function f_gauss(bits)
    # bits: n_samples × 3d matrix of 1s and 2s
    out = zeros(size(bits, 1))
    for s in axes(bits, 1)
        x = xmin + index_to_point(Int.(bits[s, 1:d]);      L = xmax - xmin)
        y = ymin + index_to_point(Int.(bits[s, d+1:2d]);   L = ymax - ymin)
        z = zmin + index_to_point(Int.(bits[s, 2d+1:3d]);  L = zmax - zmin)
        out[s] = exp(-((x - x₀)^2/σ_x^2 + (y - y₀)^2/σ_y^2 + (z - z₀)^2/σ_z^2) / 2)
    end
    return out
end

const max_bond = 12

println("Building initial condition via TCI...")
u₀ = tt_cross(f_gauss, domain_ic, MaxVol(tol=1e-12, verbose=false))
u₀ = tt_compress!(u₀, max_bond)

println("u₀: N=$(u₀.N) cores, dims=$(u₀.ttv_dims), ranks=$(u₀.ttv_rks)")
println("u₀ norm = $(norm(u₀))")
```

- [ ] **Step 2.2: Run and sanity-check**

```bash
julia --project=. examples/Lorenz.jl
```

Expected output (approximate):
```
u₀: N=15 cores, dims=(2, 2, ..., 2), ranks=(1, ..., 1)
u₀ norm = <positive finite number>
```

Key checks:
- `u₀.N` should equal `3*d` (15 for d=5)
- All `ttv_dims` entries should be 2 (binary QTT cores)
- Norm should be positive and finite (≈ a few units)
- Ranks should be low (≤ max_bond); for a separable Gaussian, all interior bonds will be 1

---

## Task 3: Coordinate QTTs and Diagonal Operators

**Files:**
- Modify: `examples/Lorenz.jl` (append)

Build the variable-coefficient parts of the Liouville operator.
Each Lorenz velocity component `f_k(x,y,z)` becomes a diagonal TToperator via:
1. Express as a TTvector over the 3D QTT grid using `function_to_qtt` + `⊗ ones_tt`
2. Convert to a diagonal TToperator with `ttv_to_diag_tto`

The Lorenz velocity components are all degree-2 polynomials — low rank in QTT.

- [ ] **Step 3.1: Append coordinate QTT and diagonal operator block**

```julia
# ── Coordinate QTTs ───────────────────────────────────────────────────────────
# ones_d: a d-core all-ones TTvector (used to extend 1D QTTs to 3D via ⊗)
ones_d = ones_tt(ntuple(_ -> 2, d))

# 1D coordinate QTTs on their respective physical domains
qtt_x = function_to_qtt(x -> x, d; a = xmin, b = xmax)
qtt_y = function_to_qtt(y -> y, d; a = ymin, b = ymax)
qtt_z = function_to_qtt(z -> z, d; a = zmin, b = zmax)

# Extend to full 3D QTT (3d cores) by tensoring with ones for the other dimensions
qtt_x3 = qtt_x ⊗ ones_d ⊗ ones_d   # f(ix,iy,iz) = x_grid[ix]
qtt_y3 = ones_d ⊗ qtt_y ⊗ ones_d   # f(ix,iy,iz) = y_grid[iy]
qtt_z3 = ones_d ⊗ ones_d ⊗ qtt_z   # f(ix,iy,iz) = z_grid[iz]
qtt_xz = qtt_x  ⊗ ones_d ⊗ qtt_z   # f(ix,iy,iz) = x*z  (multiplicatively separable)
qtt_xy = qtt_x  ⊗ qtt_y  ⊗ ones_d  # f(ix,iy,iz) = x*y

# Variable-coefficient diagonal TToperators — one per velocity component
# C1: coefficient of ∂ρ/∂x  →  σ_L*(y − x)
C1 = ttv_to_diag_tto(σ_L * qtt_y3 - σ_L * qtt_x3)
# C2: coefficient of ∂ρ/∂y  →  ρ_L*x − y − x*z
C2 = ttv_to_diag_tto(ρ_L * qtt_x3 - qtt_y3 - qtt_xz)
# C3: coefficient of ∂ρ/∂z  →  x*y − β_L*z
C3 = ttv_to_diag_tto(qtt_xy - β_L * qtt_z3)

println("Diagonal operators built:")
println("  C1 ranks: $(C1.tto_rks)")
println("  C2 ranks: $(C2.tto_rks)")
println("  C3 ranks: $(C3.tto_rks)")
```

- [ ] **Step 3.2: Run and verify operator dimensions**

```bash
julia --project=. examples/Lorenz.jl
```

Expected output:
```
Diagonal operators built:
  C1 ranks: (1, ..., 1)   # low rank — σ(y-x) is rank-2 in QTT
  C2 ranks: (1, ..., 1)   # similarly low
  C3 ranks: (1, ..., 1)
```

All three operators should have `N = 3d` (15) cores each of physical dimension 2×2.

---

## Task 4: Gradient Operators and Full Liouville Operator

**Files:**
- Modify: `examples/Lorenz.jl` (append)

`∇(d)` = `toeplitz_to_qtto(1, 0, -1, d)`: a d-core backward-difference stencil
representing `u[i] - u[i-1]` (unnormalised). Dividing by `h` gives the finite-difference
derivative approximation.

For the 3D sequential QTT, the x-derivative acts only on the first d cores:
`D_x = (1/h_x) * (∇(d) ⊗ id_tto(d) ⊗ id_tto(d))`

The full Liouville operator:
`L = −C1·D_x − C2·D_y − C3·D_z + (σ_L + 1 + β_L) · I_{3d}`

The `(σ_L+1+β_L)ρ` source term comes from `−∇·f = −(−σ_L − 1 − β_L) = +(σ_L+1+β_L)`
(Lorenz flow is volume-contracting: divergence = −σ_L − 1 − β_L < 0).

- [ ] **Step 4.1: Append gradient operator and full L block**

```julia
# ── Derivative (gradient) operators ──────────────────────────────────────────
I_d = id_tto(d)

# Scaled backward-difference operators, one per physical dimension
D_x = (1 / h_x) * (∇(d) ⊗ I_d  ⊗ I_d)
D_y = (1 / h_y) * (I_d  ⊗ ∇(d) ⊗ I_d)
D_z = (1 / h_z) * (I_d  ⊗ I_d  ⊗ ∇(d))

# ── Full Liouville operator ───────────────────────────────────────────────────
# ∂ρ/∂t = L ρ
# L = −σ(y−x)∂/∂x − (ρx−y−xz)∂/∂y − (xy−βz)∂/∂z + (σ+1+β)I
println("Assembling Liouville operator...")
L = -(C1 * D_x) - (C2 * D_y) - (C3 * D_z) + (σ_L + 1 + β_L) * id_tto(3d)

println("L: N=$(L.N) cores, ranks=$(extrema(L.tto_rks))")

# Quick sanity check: L applied to u₀ should give a finite-norm result
Lu₀ = L * u₀
println("L*u₀ norm = $(norm(Lu₀))  (should be finite and non-zero)")
```

- [ ] **Step 4.2: Run and verify**

```bash
julia --project=. examples/Lorenz.jl
```

Expected:
```
Assembling Liouville operator...
L: N=15 cores, ranks=(1, <max_rank>)
L*u₀ norm = <positive finite number>
```

Key checks:
- `L.N` == `3d` (15)
- `L * u₀` has finite norm — confirms operator and state types are compatible
- No `MethodError` or dimension mismatches

---

## Task 5: Time Evolution

**Files:**
- Modify: `examples/Lorenz.jl` (append)

`rk4_method(A, u₀, steps, max_bond; normalize)` — internally applies `tt_compress!(state, max_bond)` at each RK4 sub-step to control rank growth. `normalize=false` preserves the physical density norm (which should be conserved by the Liouville equation).

CFL check: `dt * max(|velocity|) / h ≈ 0.01 * 55 / 1.6 ≈ 0.34 < 2.83` (RK4 stability limit for advection) — stable.

- [ ] **Step 5.1: Append time evolution block**

```julia
# ── Time evolution ────────────────────────────────────────────────────────────
const dt     = 0.01
const nsteps = 500      # T = 5.0 Lorenz time units
                        # Increase to ~2000 for T=20 to see full attractor shape
const steps  = fill(dt, nsteps)

println("Running RK4 for $(nsteps) steps (T = $(nsteps*dt))...")
solution = rk4_method(L, u₀, steps, max_bond; normalize=false)

println("solution ranks: $(solution.ttv_rks)")
println("solution norm  = $(norm(solution))  (should be close to initial norm=$(norm(u₀)))")
```

- [ ] **Step 5.2: Run and verify (this step takes ~minutes for d=5)**

```bash
julia --project=. examples/Lorenz.jl
```

Expected:
```
Running RK4 for 500 steps (T = 5.0)...
[progress bar from @showprogress inside rk4_method]
solution ranks: (1, ..., max_bond, ...)
solution norm  = <value within ~10x of initial norm>
```

Key checks:
- No `InexactError` or `NaN` in ranks
- Norm should stay in a reasonable range (not explode to Inf or collapse to 0)
- Bond dimensions should be bounded by `max_bond`

---

## Task 6: Visualization

**Files:**
- Modify: `examples/Lorenz.jl` (append)

The QTT output vector index for sequential [x,y,z] layout maps physical point `(px,py,pz)`
to position `(px−1)·N² + (py−1)·N + pz`. Julia's column-major `reshape(v, N, N, N)` gives
`arr[iz, iy, ix]`, so `permutedims(arr, [3,2,1])` is required to get `ρ[ix,iy,iz]` for
GLMakie's `volume!(ax, x_grid, y_grid, z_grid, data)` convention.

- [ ] **Step 6.1: Append visualization block**

```julia
# ── Visualization ─────────────────────────────────────────────────────────────
# Recover the physical 3D density array from the QTT
ρ_raw   = reshape(qtt_to_vector(solution), N, N, N)
# qtt_to_vector with sequential [x,y,z] QTT gives arr[iz, iy, ix] in column-major
# permutedims corrects to ρ_3d[ix, iy, iz] = ρ(x_grid[ix], y_grid[iy], z_grid[iz])
ρ_3d    = permutedims(ρ_raw, [3, 2, 1])

# Clip negatives (numerical dispersion artifact from backward differences) and normalise
ρ_3d    = max.(ρ_3d, 0.0)
ρ_3d  ./= maximum(ρ_3d)

x_grid = collect(range(xmin, xmax, length=N))
y_grid = collect(range(ymin, ymax, length=N))
z_grid = collect(range(zmin, zmax, length=N))

set_theme!(theme_black())

fig = Figure(size=(900, 700))
ax  = Axis3(fig[1, 1];
    xlabel = "x", ylabel = "y", zlabel = "z",
    title  = "Lorenz Phase-Space Density — QTT (d=$(d), T=$(nsteps*dt))",
    protrusions = (0, 0, 0, 0),
    viewmode    = :fit,
    limits      = (xmin, xmax, ymin, ymax, zmin, zmax))

volume!(ax, x_grid, y_grid, z_grid, ρ_3d;
    algorithm    = :iso,
    isorange     = 0.05,
    isovalue     = 0.3,      # 30% of peak; lower to ~0.05 for T>10 as density spreads
    colormap     = :inferno,
    transparency = true)

display(fig)
save("lorenz_density.png", fig)
println("Saved lorenz_density.png")
```

- [ ] **Step 6.2: Run the complete example**

```bash
julia --project=. examples/Lorenz.jl
```

Expected: a GLMakie window opens showing a 3D isosurface, and `lorenz_density.png` is saved. The shape at T=5 will be an elongated Gaussian blob beginning to curve along the attractor's unstable manifold. At T=20 (nsteps=2000) the characteristic two-lobe butterfly should be visible.

- [ ] **Step 6.3: Tune visualization if needed**

If the isosurface is invisible (density too spread), lower `isovalue` to 0.1 or 0.05.
If the density is entirely in one spot (short time), increase `nsteps`.
The `max_bond` can be increased (e.g. 20) for more accurate evolution at the cost of memory.

---

## Complete File Reference

The complete `examples/Lorenz.jl` results from appending the code blocks from Tasks 1–6 in order. No additional changes are needed beyond what each task step specifies.

<!--

```julia
using TensorTrainNumerics
using GLMakie

# ── Lorenz parameters ────────────────────────────────────────────────────────
σ_L = 10.0
ρ_L = 28.0
β_L = 8/3

# ── QTT grid ─────────────────────────────────────────────────────────────────
d = 5
N = 2^d

xmin, xmax = -25.0, 25.0
ymin, ymax = -30.0, 30.0
zmin, zmax =   0.0, 60.0

h_x = (xmax - xmin) / (N - 1)
h_y = (ymax - ymin) / (N - 1)
h_z = (zmax - zmin) / (N - 1)

println("Grid: $(N)³ = $(N^3) points  |  h_x=$(round(h_x,digits=3))  h_y=$(round(h_y,digits=3))  h_z=$(round(h_z,digits=3))")

# ── Initial Gaussian density (via TCI on binary QTT domain) ──────────────────
x₀, y₀, z₀ = 1.0, 1.0, 1.0
σ_x, σ_y, σ_z = 2.0, 2.0, 2.0

domain_ic = [collect(1:2) for _ in 1:(3d)]

function f_gauss(bits)
    out = zeros(size(bits, 1))
    for s in axes(bits, 1)
        x = xmin + index_to_point(Int.(bits[s, 1:d]);      L = xmax - xmin)
        y = ymin + index_to_point(Int.(bits[s, d+1:2d]);   L = ymax - ymin)
        z = zmin + index_to_point(Int.(bits[s, 2d+1:3d]);  L = zmax - zmin)
        out[s] = exp(-((x - x₀)^2/σ_x^2 + (y - y₀)^2/σ_y^2 + (z - z₀)^2/σ_z^2) / 2)
    end
    return out
end

const max_bond = 12

println("Building initial condition via TCI...")
u₀ = tt_cross(f_gauss, domain_ic, MaxVol(tol=1e-12, verbose=false))
u₀ = tt_compress!(u₀, max_bond)
println("u₀: N=$(u₀.N) cores, ranks=$(u₀.ttv_rks), norm=$(norm(u₀))")

# ── Coordinate QTTs ───────────────────────────────────────────────────────────
ones_d = ones_tt(ntuple(_ -> 2, d))

qtt_x = function_to_qtt(x -> x, d; a = xmin, b = xmax)
qtt_y = function_to_qtt(y -> y, d; a = ymin, b = ymax)
qtt_z = function_to_qtt(z -> z, d; a = zmin, b = zmax)

qtt_x3 = qtt_x ⊗ ones_d ⊗ ones_d
qtt_y3 = ones_d ⊗ qtt_y ⊗ ones_d
qtt_z3 = ones_d ⊗ ones_d ⊗ qtt_z
qtt_xz = qtt_x  ⊗ ones_d ⊗ qtt_z
qtt_xy = qtt_x  ⊗ qtt_y  ⊗ ones_d

C1 = ttv_to_diag_tto(σ_L * qtt_y3 - σ_L * qtt_x3)
C2 = ttv_to_diag_tto(ρ_L * qtt_x3 - qtt_y3 - qtt_xz)
C3 = ttv_to_diag_tto(qtt_xy - β_L * qtt_z3)

# ── Derivative operators and Liouville operator ───────────────────────────────
I_d = id_tto(d)
D_x = (1 / h_x) * (∇(d) ⊗ I_d  ⊗ I_d)
D_y = (1 / h_y) * (I_d  ⊗ ∇(d) ⊗ I_d)
D_z = (1 / h_z) * (I_d  ⊗ I_d  ⊗ ∇(d))

println("Assembling Liouville operator...")
L = -(C1 * D_x) - (C2 * D_y) - (C3 * D_z) + (σ_L + 1 + β_L) * id_tto(3d)
println("L: N=$(L.N) cores, rank range=$(extrema(L.tto_rks))")

# ── Time evolution ────────────────────────────────────────────────────────────
const dt     = 0.01
const nsteps = 500
const steps  = fill(dt, nsteps)

println("Running RK4 for $(nsteps) steps (T = $(nsteps*dt))...")
solution = rk4_method(L, u₀, steps, max_bond; normalize=false)
println("Done. solution norm=$(norm(solution))")

# ── Visualization ─────────────────────────────────────────────────────────────
ρ_raw = reshape(qtt_to_vector(solution), N, N, N)
ρ_3d  = permutedims(ρ_raw, [3, 2, 1])
ρ_3d  = max.(ρ_3d, 0.0)
ρ_3d ./= maximum(ρ_3d)

x_grid = collect(range(xmin, xmax, length=N))
y_grid = collect(range(ymin, ymax, length=N))
z_grid = collect(range(zmin, zmax, length=N))

set_theme!(theme_black())
fig = Figure(size=(900, 700))
ax  = Axis3(fig[1, 1];
    xlabel = "x", ylabel = "y", zlabel = "z",
    title  = "Lorenz Phase-Space Density — QTT (d=$(d), T=$(nsteps*dt))",
    protrusions = (0, 0, 0, 0),
    viewmode    = :fit,
    limits      = (xmin, xmax, ymin, ymax, zmin, zmax))

volume!(ax, x_grid, y_grid, z_grid, ρ_3d;
    algorithm    = :iso,
    isorange     = 0.05,
    isovalue     = 0.3,
    colormap     = :inferno,
    transparency = true)

display(fig)
save("lorenz_density.png", fig)
println("Saved lorenz_density.png")
```
-->

---

## Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Negative density values (numerical dispersion from backward differences) | `max.(ρ_3d, 0.0)` clip before visualization |
| Density norm drifts at long times | Check `norm(solution)` after evolution; acceptable within ~2× of `norm(u₀)` |
| Bond dims hit `max_bond` too early | Increase `max_bond` to 20; watch memory usage |
| CFL instability (large velocities near domain boundary) | Domain chosen so attractor is interior; reduce `dt` to 0.005 if instability seen |
| Operator assembly is slow | All TToperator operations are one-time; only `rk4_method` loop is repeated |
| `isovalue=0.3` shows nothing at T=5 | Lower to 0.1; if still empty, increase `nsteps` to 1000 |
