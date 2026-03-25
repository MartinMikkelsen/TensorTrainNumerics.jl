# Lorenz QTT Example Design

**Date:** 2026-03-24
**File to create:** `examples/Lorenz.jl`
**Goal:** Showcase phase-space density evolution of the Lorenz system using quantics tensor trains (QTT), tensor cross interpolation (TCI), and `rk4_method`, with a 3D GLMakie density visualization.

---

## Overview

Rather than integrating a single Lorenz trajectory, we evolve a probability density
`ρ(x, y, z, t)` in 3D phase space. The Liouville equation governing this density is
**linear** in `ρ`, even though the Lorenz flow is nonlinear — so `rk4_method` applies
directly. TCI is used to build the initial Gaussian density as a QTT. The final density
is rendered in 3D using GLMakie's volume rendering to produce a butterfly-shaped
density cloud tracing the Lorenz attractor.

### Physics

Lorenz system: `dx/dt = σ(y−x)`, `dy/dt = ρx − y − xz`, `dz/dt = xy − βz`
Standard parameters: `σ = 10`, `ρ = 28`, `β = 8/3`

Liouville equation:
```
∂ρ/∂t = −σ(y−x)∂ρ/∂x − (ρx − y − xz)∂ρ/∂y − (xy − βz)∂ρ/∂z + (σ+1+β)ρ
```
The `(σ+1+β)ρ` term arises from `−∇·f = σ+1+β` (the Lorenz flow is volume-contracting).

---

## Section 1: Grid & QTT Layout

**Layout:** Sequential QTT with `3d` total cores.

```
Cores 1..d    → x-dimension (d bits, most to least significant)
Cores d+1..2d → y-dimension
Cores 2d+1..3d→ z-dimension
```

- Default `d = 5` (32 pts/axis, 32³ ≈ 32k grid points); increase to `d = 6` for publication quality
- Physical domain: `x ∈ [−25, 25]`, `y ∈ [−30, 30]`, `z ∈ [0, 60]`
- Grid spacings: `h_x = 50/(2^d − 1)`, `h_y = 60/(2^d − 1)`, `h_z = 60/(2^d − 1)`

**Recovering the 3D array:**
Tracing `qtt_to_vector` with sequential [x,y,z] cores: the output vector index for
physical grid point `(p_x, p_y, p_z)` is `(p_x−1)·N² + (p_y−1)·N + p_z` (N=2^d).
Julia's column-major `reshape(v, N, N, N)` therefore gives `arr[iz, iy, ix]`.
A `permutedims` is needed to recover `ρ[ix, iy, iz]`:
```julia
ρ_raw = reshape(qtt_to_vector(solution), 2^d, 2^d, 2^d)
ρ_3d  = permutedims(ρ_raw, [3, 2, 1])   # now ρ_3d[ix, iy, iz] = ρ(x_ix, y_iy, z_iz)
```

---

## Section 2: Initial Condition via TCI

A 3D Gaussian blob built directly as a `3d`-core QTT using `tt_cross` on a binary domain:

```julia
domain = [collect(1:2) for _ in 1:(3d)]  # 3d binary "bit" dimensions

function f_gauss(bits)  # bits: n_samples × 3d matrix of 1s and 2s
    out = zeros(size(bits, 1))
    for s in axes(bits, 1)
        x = xmin + index_to_point(Int.(bits[s, 1:d]);     L = xmax - xmin)
        y = ymin + index_to_point(Int.(bits[s, d+1:2d]);  L = ymax - ymin)
        z = zmin + index_to_point(Int.(bits[s, 2d+1:3d]); L = zmax - zmin)
        out[s] = exp(-((x-x₀)^2/σ_x^2 + (y-y₀)^2/σ_y^2 + (z-z₀)^2/σ_z^2) / 2)
    end
    return out
end

u₀ = tt_cross(f_gauss, domain, MaxVol(tol=1e-12, verbose=false))
u₀ = tt_compress!(u₀, max_bond)   # compress after TCI
```

- Initial center: `(x₀, y₀, z₀) = (1.0, 1.0, 1.0)`, spread `σ_x = σ_y = σ_z = 2.0`
- The Gaussian is rank-1 separable, so TCI converges immediately; the scaffolding
  generalises to any non-separable initial density

---

## Section 3: Liouville Operator

All components built from existing TensorTrainNumerics building blocks:

```julia
# 1D coordinate QTTs extended to 3D via ⊗ with ones
ones_d  = ones_tt(ntuple(_ -> 2, d))
qtt_x   = function_to_qtt(x -> x, d; a=xmin, b=xmax)
qtt_y   = function_to_qtt(y -> y, d; a=ymin, b=ymax)
qtt_z   = function_to_qtt(z -> z, d; a=zmin, b=zmax)

qtt_x3  = qtt_x ⊗ ones_d ⊗ ones_d    # value = x(ix), independent of iy, iz
qtt_y3  = ones_d ⊗ qtt_y ⊗ ones_d
qtt_z3  = ones_d ⊗ ones_d ⊗ qtt_z
qtt_xz  = qtt_x ⊗ ones_d ⊗ qtt_z     # value = x(ix)*z(iz)  (rank-1)
qtt_xy  = qtt_x ⊗ qtt_y  ⊗ ones_d    # value = x(ix)*y(iy)  (rank-1)

# Variable-coefficient diagonal operators
C1 = ttv_to_diag_tto(σ*qtt_y3 - σ*qtt_x3)                  # σ(y−x)
C2 = ttv_to_diag_tto(ρ_p*qtt_x3 - qtt_y3 - qtt_xz)         # ρx−y−xz
C3 = ttv_to_diag_tto(qtt_xy - β_p*qtt_z3)                   # xy−βz

# Scaled backward-difference gradient operators
I_d  = id_tto(d)
D_x  = (1/h_x) * (∇(d) ⊗ I_d  ⊗ I_d)
D_y  = (1/h_y) * (I_d  ⊗ ∇(d) ⊗ I_d)
D_z  = (1/h_z) * (I_d  ⊗ I_d  ⊗ ∇(d))

# Full Liouville operator (3d-core TToperator)
L = -(C1*D_x) - (C2*D_y) - (C3*D_z) + (σ + 1 + β_p)*id_tto(3d)
```

- `∇(d)` = `toeplitz_to_qtto(1, 0, −1, d)`: backward-difference stencil, scaled by `1/h`
- Zero-flux boundary conditions are implicit (density assumed negligible at domain boundary)
- Operator bond dimension is O(20–30) after construction — acceptable, L is fixed

---

## Section 4: Time Evolution

```julia
dt       = 0.01
nsteps   = 500        # T = 5.0 Lorenz time units
steps    = fill(dt, nsteps)
max_bond = 12

solution = rk4_method(L, u₀, steps, max_bond; normalize=false)
```

- `normalize=false`: density norm is physically conserved by the Liouville equation
- `rk4_method` calls `tt_compress!(state, max_bond)` internally at each RK4 stage —
  no additional compression needed on the operator or during the loop
- Increase `nsteps` to ~2000 (T = 20) to see the full butterfly attractor shape emerge

---

## Section 5: Visualization

```julia
using GLMakie

ρ_raw   = reshape(qtt_to_vector(solution), 2^d, 2^d, 2^d)
ρ_final = permutedims(ρ_raw, [3, 2, 1])   # ρ_final[ix,iy,iz] = ρ(x_ix, y_iy, z_iz)
ρ_final ./= maximum(ρ_final)   # normalize to [0, 1] for rendering

x_grid = collect(range(xmin, xmax, length=2^d))
y_grid = collect(range(ymin, ymax, length=2^d))
z_grid = collect(range(zmin, zmax, length=2^d))

set_theme!(theme_black())
fig = Figure(size=(900, 700))
ax  = Axis3(fig[1, 1],
    xlabel="x", ylabel="y", zlabel="z",
    title="Lorenz Attractor — Phase-Space Density (QTT)",
    limits=(xmin, xmax, ymin, ymax, zmin, zmax))

volume!(ax, x_grid, y_grid, z_grid, ρ_final;
    algorithm=:iso, isorange=0.05, isovalue=0.3,
    colormap=:inferno, transparency=true)

display(fig)
```

- `isovalue=0.3`: isosurface at 30% of peak density traces the density support
- `:inferno` colormap on dark background — matches the reference Lorenz style
- Adjust `isovalue` downward (e.g. 0.1) once the density spreads to fill the attractor

---

## Parameters Summary

| Parameter     | Value         | Notes                                      |
|---------------|---------------|--------------------------------------------|
| `d`           | 5             | Bits per dimension; 6 for higher res       |
| `σ`, `ρ`, `β` | 10, 28, 8/3   | Standard Lorenz parameters                 |
| Domain x      | [−25, 25]     | Contains the attractor                     |
| Domain y      | [−30, 30]     |                                            |
| Domain z      | [0, 60]       |                                            |
| `x₀,y₀,z₀`   | 1.0, 1.0, 1.0 | Initial density center                     |
| `σ_x,σ_y,σ_z`| 2.0           | Initial Gaussian spread                    |
| `dt`          | 0.01          | Time step                                  |
| `nsteps`      | 500           | T = 5.0; increase to 2000 for full attractor|
| `max_bond`    | 12            | QTT rank cap in rk4_method                 |
| TCI tol       | 1e-12         | MaxVol tolerance for initial condition     |
