# 1D Nonlinear Solver Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 1D analytical benchmark helpers that compare InterpolativeQTT nonlinear ALS/MALS with dense local micro-solves and KrylovKit local micro-solves.

**Architecture:** Keep the nonlinear method unchanged and expose Krylov as a benchmark variant through existing `it_solver=true` local eigensolver hooks. Start with 1D KdV, where the final soliton is known analytically, and 1D GPE, where the linear chemical potential has a continuum analytical reference.

**Tech Stack:** Julia, TensorTrainNumerics.jl, InterpolativeQTT, KrylovKit local eigensolver support.

---

### Task 1: Tests For 1D Solver Comparison Benchmarks

**Files:**
- Modify: `test/test_nonlinear_examples.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write failing tests for KdV and GPE comparison helpers**

Add tests that call:

```julia
kdv_1d_solver_comparison_benchmark(;
    d = 5,
    T_end = 0.05,
    Nt = 3,
    methods = (:als_direct, :als_krylov, :mals_direct, :mals_krylov),
    max_scf = 3,
    scf_tol = 1.0e-6,
    max_bond = 20,
    projection_degree = 8,
    projection_tolerance = 1.0e-9,
)
```

and:

```julia
gpe_1d_solver_comparison_benchmark(;
    L = 5,
    κ = 50.0,
    g_vals = [0.0, 25.0],
    methods = (:als_direct, :als_krylov, :mals_direct, :mals_krylov),
    random_rank = 4,
    linear_sweeps = 6,
    nonlinear_sweeps = 3,
    mals_rmax = 8,
    projection_degree = 8,
    projection_tolerance = 1.0e-9,
)
```

Assert each benchmark returns four result entries, every result has an InterpolativeQTT label, finite runtime, bounded rank, and a finite analytical error metric.

- [ ] **Step 2: Run tests red**

Run:

```bash
julia --project=. test/test_nonlinear_examples.jl
```

Expected: failure with `UndefVarError: kdv_1d_solver_comparison_benchmark not defined`.

### Task 2: Implement Benchmark Variant Dispatch

**Files:**
- Modify: `examples/nonlinear_benchmark_utils.jl`

- [ ] **Step 1: Add `_solver_variant_label` and `_solver_variant_kind`**

Define mapping:

```julia
_solver_variant_label(:als_direct) = "InterpolativeQTT-SCF-ALS"
_solver_variant_label(:als_krylov) = "InterpolativeQTT-SCF-ALS-Krylov"
_solver_variant_label(:mals_direct) = "InterpolativeQTT-SCF-MALS"
_solver_variant_label(:mals_krylov) = "InterpolativeQTT-SCF-MALS-Krylov"
```

and validate unknown methods with `ArgumentError`.

- [ ] **Step 2: Implement `kdv_1d_solver_comparison_benchmark`**

Reuse `kdv_soliton_benchmark` for each method. Map `:als_direct` and `:als_krylov` to `method=:als`; map `:mals_direct` and `:mals_krylov` to `method=:mals`. Pass `it_solver=true` for the Krylov variants once `kdv_soliton_benchmark` accepts that option.

- [ ] **Step 3: Implement `gpe_1d_solver_comparison_benchmark`**

Build the 1D GPE Hamiltonian, solve the linear problem once, then warm-start each nonlinear variant through `g_vals`. Use `nonlinear_als_eigsolve` for ALS variants and `nonlinear_mals_eigsolve` for MALS variants. Pass `it_solver=true` for local Krylov variants.

### Task 3: Add Krylov Flags To KdV Wrapper

**Files:**
- Modify: `src/solvers/kdv.jl`
- Modify: `examples/nonlinear_benchmark_utils.jl`

- [ ] **Step 1: Thread `it_solver` into KdV ALS wrappers**

Add `it_solver::Bool=false`, `itslv_thresh::Int=256`, `linsolv_maxiter::Int=200`, and `linsolv_tol::Float64=1.0e-8` to `kdv_als_step` / `kdv_als`, passing the values into `als_linsolve`.

- [ ] **Step 2: Keep MALS KdV direct for now if nonsymmetric Krylov is unavailable**

If the current nonsymmetric KdV MALS local solve has no KrylovKit path, make `:mals_krylov` explicit in the returned result as unsupported with `success=false` and an explanatory message. If a local Krylov path exists, wire it through.

### Task 4: Green Tests And Verification

**Files:**
- Modify as needed from prior tasks.

- [ ] **Step 1: Run focused tests**

Run:

```bash
julia --project=. test/test_nonlinear_examples.jl
julia --project=. test/test_nonlinear.jl
```

Expected: pass.

- [ ] **Step 2: Run full tests**

Run:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: pass.
