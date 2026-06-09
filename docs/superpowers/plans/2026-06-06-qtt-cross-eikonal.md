# QTT-Cross Eikonal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a TT-first variable-speed Godunov Eikonal solver that can target grids such as `2^10 x 2^10` without dense solution grids or dense Jacobians.

**Architecture:** Keep the existing dense Godunov solver as the validation baseline. Add a new QTT-cross path that samples nonlinear residuals and active-set coefficient fields through pointwise TT evaluation, builds frozen Newton Jacobians from structured QTT upwind operators and diagonal QTT coefficient operators, and solves corrections with MALS.

**Tech Stack:** Julia, existing `TTvector`/`QTTvector`/`QTToperator` types, existing `tt_cross` DMRG/MaxVol algorithms, existing nonsymmetric MALS micro-solver.

---

### Task 1: Tests For TT-Only Nonlinear Pieces

**Files:**
- Modify: `test/test_eikonal_godunov.jl`
- Modify: `test/runtests.jl`

- [ ] Add tests that require:
  - pointwise QTT evaluation from bit indices;
  - TT-cross construction of a 2D speed function as a serial `QTTvector`;
  - structured 2D upwind difference operators matching dense stencils for small `d`;
  - TT-cross residual matching the dense Godunov residual for small `d`;
  - a `d=8` smoke solve returning a QTT vector without forming dense Jacobian matrices.

- [ ] Run `test/test_eikonal_godunov.jl` and verify the new tests fail because the new public functions are missing.

### Task 2: QTT-Cross Sampling Utilities

**Files:**
- Create: `src/solvers/eikonal_qtt_godunov.jl`
- Modify: `src/TensorTrainNumerics.jl`

- [ ] Implement serial 2D bit/index conversion helpers.
- [ ] Implement pointwise QTT evaluation helpers using existing TT cores.
- [ ] Implement `eikonal_speed_cross_qtt_2d`.
- [ ] Export the new public helper.
- [ ] Run the tests added in Task 1 until the sampling and speed-cross tests pass.

### Task 3: Structured Upwind Operators

**Files:**
- Modify: `src/solvers/eikonal_qtt_godunov.jl`

- [ ] Implement 1D backward and forward upwind derivative QTTOs using existing `∇`, `shift`, and `id_tto`.
- [ ] Lift them to serial 2D `QTToperator`s with Kronecker products.
- [ ] Implement `eikonal_upwind_operators_qtt_2d`.
- [ ] Run the operator tests until they pass.

### Task 4: TT-Cross Residual And Active Coefficients

**Files:**
- Modify: `src/solvers/eikonal_qtt_godunov.jl`

- [ ] Implement `eikonal_godunov_residual_cross_qtt_2d`.
- [ ] Implement `eikonal_godunov_coefficients_cross_qtt_2d`.
- [ ] Implement `eikonal_godunov_jacobian_cross_qtt_2d` as a sum of diagonal coefficient operators times structured upwind operators.
- [ ] Run residual/Jacobian tests until they pass.

### Task 5: TT-Cross Newton-MALS Solver

**Files:**
- Modify: `src/solvers/eikonal_qtt_godunov.jl`
- Modify: `examples/nonlinear_benchmark_utils.jl`

- [ ] Implement `eikonal_godunov_mals_2d_qtt`.
- [ ] Use a TT-cross boundary-distance initializer and optional speed continuation.
- [ ] Measure residual norm in TT form.
- [ ] Add benchmark helper for the QTT-cross path.
- [ ] Run targeted tests and example smoke checks.

### Task 6: Verification

**Files:**
- Modify: `test/runtests.jl`

- [ ] Run `test/test_eikonal_godunov.jl`.
- [ ] Run `test/test_nonlinear_examples.jl` if benchmark helpers change.
- [ ] Run full `Pkg.test()`.
- [ ] Report residuals, ranks, and whether `d=10` is usable with the current rank/cross settings.

