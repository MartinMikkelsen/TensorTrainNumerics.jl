# 1D Residual Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add residual diagnostics to the 1D analytical nonlinear benchmarks so paper tables can report error, residual, rank, and runtime together.

**Architecture:** Keep diagnostics in `examples/nonlinear_benchmark_utils.jl`. Compute dense diagnostic residuals from QTT outputs without changing the nonlinear solvers: KdV uses the final implicit-Euler PDE residual, and 1D GPE uses the nonlinear eigen residual plus final SCF chemical-potential change.

**Tech Stack:** Julia, TensorTrainNumerics.jl, InterpolativeQTT benchmark helpers.

---

### Task 1: Add Failing Residual Tests

**Files:**
- Modify: `test/test_nonlinear_examples.jl`

- [x] **Step 1: Assert KdV residual metrics exist**

Inside the 1D KdV solver comparison test, assert each result exposes:

```julia
@test isfinite(result.metrics.final_pde_residual_norm)
@test isfinite(result.metrics.final_pde_relative_residual)
@test result.metrics.final_pde_relative_residual < 1.0
```

- [x] **Step 2: Assert GPE residual metrics exist**

Inside the 1D GPE solver comparison test, assert each result exposes:

```julia
@test isfinite(result.metrics.final_nonlinear_residual_norm)
@test isfinite(result.metrics.final_nonlinear_relative_residual)
@test result.metrics.final_nonlinear_relative_residual < 1.0
@test isfinite(result.metrics.final_mu_step_change)
```

- [x] **Step 3: Run red test**

Run:

```bash
julia --project=. test/test_nonlinear_examples.jl
```

Expected: fail because the new residual metric fields do not exist.

### Task 2: Implement Dense KdV Residual Metrics

**Files:**
- Modify: `examples/nonlinear_benchmark_utils.jl`

- [x] **Step 1: Add `_relative_residual_metrics` helper**

Add a helper that returns residual norm, scale, and relative residual:

```julia
function _relative_residual_metrics(residual, terms...)
    scale = sum(norm, terms)
    return (
        residual_norm = Float64(norm(residual)),
        residual_scale = Float64(scale),
        relative_residual = Float64(norm(residual) / max(scale, eps(Float64))),
    )
end
```

- [x] **Step 2: Add `_kdv_final_pde_residual_metrics`**

For the last KdV step, compute:

```julia
time_term = (u_final - u_prev) / dt
nonlinear_term = 6 .* u_final .* (D_x * u_final)
dispersion_term = D_xxx * u_final
residual = time_term + nonlinear_term + dispersion_term
```

using dense vectors reconstructed from the QTT states/operators.

- [x] **Step 3: Attach KdV residual fields to each comparison result**

Each result gets:

```julia
final_pde_residual_norm
final_pde_residual_scale
final_pde_relative_residual
```

### Task 3: Implement Dense 1D GPE Residual Metrics

**Files:**
- Modify: `examples/nonlinear_benchmark_utils.jl`

- [x] **Step 1: Add `_gpe_1d_nonlinear_residual_metrics`**

For final state `ψ`, compute:

```julia
residual = H_lin * ψ + g * abs2(ψ) * ψ - μ * ψ
```

dense-vector-wise, normalized by the sum of term norms.

- [x] **Step 2: Track final SCF chemical-potential change**

For each nonlinear solve, store:

```julia
abs(μ_hist[end] - μ_hist[end - 1]) / max(abs(μ_hist[end]), eps(Float64))
```

using `NaN` only if a solver returns a single history entry.

- [x] **Step 3: Attach GPE residual fields to each comparison result**

Each result gets:

```julia
final_nonlinear_residual_norm
final_nonlinear_residual_scale
final_nonlinear_relative_residual
final_mu_step_change
```

### Task 4: Verification

**Files:**
- No additional files.

- [x] **Step 1: Run focused tests**

Run:

```bash
julia --project=. test/test_nonlinear_examples.jl
julia --project=. test/test_nonlinear.jl
```

Expected: pass.

- [x] **Step 2: Run full test suite**

Run:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: pass.
