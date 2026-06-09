# Interpolative Nonlinear Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first reusable primitive for the new paper direction: project a nonlinear coefficient of a QTT iterate by inverting to Lindsey's multiresolution interpolation values, applying a pointwise nonlinear map, and rebuilding with InterpolativeQTT.

**Architecture:** Keep InterpolativeQTT-dependent code in the existing package extension. The core package declares generic entry points; the extension implements `interpolative_qtt`, `invert_interpolative_qtt`, and `project_nonlinearity` for 1D TT/QTT vectors. Tests verify round-trip inversion and the nonlinear projection `sin(2πx) -> sin(2πx)^2`.

**Tech Stack:** Julia, TensorTrainNumerics `TTvector`, InterpolativeQTT `interpolatesinglescale`/`invertqtt`, TensorCrossInterpolation `TensorTrain`, existing package extension mechanism.

---

### Task 1: Red Tests For The Projection Primitive

**Files:**
- Create: `test/test_interpolative_nonlinear.jl`
- Modify: `test/runtests.jl`

- [ ] Add a test that builds a 1D QTT with `interpolative_qtt`.
- [ ] Add a test that inverts the QTT with `invert_interpolative_qtt` and checks Chebyshev-Lobatto values.
- [ ] Add a test that projects `u -> u^2` through `project_nonlinearity` and compares dyadic values against `sin(2πx)^2`.
- [ ] Run `test/test_interpolative_nonlinear.jl` and verify it fails because the functions are undefined.

### Task 2: Core Generic Functions

**Files:**
- Modify: `src/TensorTrainNumerics.jl`

- [ ] Export and declare empty generic functions:
  - `interpolative_qtt`
  - `invert_interpolative_qtt`
  - `project_nonlinearity`

### Task 3: Extension Implementation

**Files:**
- Modify: `ext/TensorTrainNumericsInterpolativeQTTExt/TensorTrainNumericsInterpolativeQTTExt.jl`

- [ ] Implement conversion from TensorTrainNumerics `TTvector` to TensorCrossInterpolation `TensorTrain`.
- [ ] Implement `interpolative_qtt(f, bits; degree, tolerance, maxbonddim, a, b)` for 1D.
- [ ] Implement `invert_interpolative_qtt(u; degree, q)` using `InterpolativeQTT.invertqtt`.
- [ ] Implement a Chebyshev-Lobatto interpolation evaluator over the finest returned inversion table.
- [ ] Implement `project_nonlinearity(u, Φ; ...)` as `invert -> pointwise nonlinear evaluator -> interpolative_qtt`.
- [ ] Run targeted tests until they pass.

### Task 4: Verification

**Files:**
- Test-only unless failures expose small integration bugs.

- [ ] Run `test/test_interpolative_nonlinear.jl`.
- [ ] Run `test/test_interpolations.jl` to ensure the existing extension behavior still works.
- [ ] Report results and the next recommended PDE-level test.

