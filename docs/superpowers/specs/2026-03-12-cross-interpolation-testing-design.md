# Cross Interpolation: Testing & Code Quality Design

**Date:** 2026-03-12
**Scope:** `src/tt_cross_interpolation.jl` and `test/test_tt_cross_interpolation.jl`
**Goals:** (1) systematic mathematical accuracy tests, (2) targeted audit and fix of complex-domain code paths, (3) minor secondary performance improvements.

---

## Background

`tt_cross_interpolation.jl` implements three TT cross-approximation algorithms — MaxVol, Greedy, and DMRG — plus `tt_integrate`. The existing test suite covers struct construction and a handful of smoke/accuracy tests, but:

- MaxVol and Greedy lack full-tensor accuracy verification.
- Complex-domain paths in MaxVol and Greedy are undertested ("tests pass but I don't trust them").
- DMRG is considered more trustworthy; it already handles the complex fallback from MaxVol.

---

## Section 1: Systematic Test Matrix

### Verification Strategy

For small grids (physical dimension ≤ 8 per mode, N ≤ 5 modes), reconstruct the full tensor via `ttv_to_tensor(tt)` and compare against `f` evaluated exhaustively over the grid using broadcasting. This provides exact ground truth rather than sampling-based error estimates.

Assertion for each test:
```julia
relerr = norm(ttv_to_tensor(tt) - exact) / max(norm(exact), eps())
@test relerr < tol
```

### Test Cases (all three algorithms: MaxVol, Greedy, DMRG)

| Class | Function | Grid | Expected rank | Purpose |
|---|---|---|---|---|
| Rank-1 separable, real | `f(x) = ∏ sin(xₖ)` | 6×6×6×6 | 1 | Any error = clear algorithm bug |
| Rank-1 separable, complex | `f(x) = ∏ exp(i·xₖ)` | 5×5×5 | 1 | Isolates complex path bugs |
| Low-rank polynomial, real | `f(x) = (x₁+x₂+x₃)²` | 8×8×8 | ≤3 | Known exact TT rank |
| Low-rank polynomial, complex | `f(x) = x₁·x₂·x₃` on complex grid | 5×5×5 | 1 | Complex grid, separable |
| Smooth Gaussian, real | `f(x) = exp(-‖x‖²)` | 8×8×8×8 | small | Standard benchmark |
| Smooth, complex | `f(x) = exp(i·∑xₖ²)` | 6×6×6 | small | Full complex path |

### Tolerances

- Rank-1 separable: `tol = 1e-8` (should reach near machine precision).
- Low-rank polynomial: `tol = 1e-6`.
- Smooth: `tol = 1e-4`.

### Test Organization

Tests are added to the existing `@testset "tt_cross"` block in `test/test_tt_cross_interpolation.jl`. Each function class becomes a nested `@testset`. Within each, a helper loops over algorithms to avoid repetition:

```julia
for (alg_name, alg) in [("MaxVol", MaxVol(...)), ("Greedy", Greedy(...)), ("DMRG", DMRG(...))]
    @testset "$alg_name" begin
        tt = tt_cross(f, domain, alg)
        relerr = norm(ttv_to_tensor(tt) - exact) / max(norm(exact), eps())
        @test relerr < tol
    end
end
```

---

## Section 2: Complex-Domain Audit & Fixes

### Suspected Bug Sites

**MaxVol — `maxvol!` on complex Q:**

At lines 238–239 and 265–266, `qr(V)` produces a complex unitary Q, which is then passed to `maxvol!`. The `Maxvol.jl` package may assume real input. Action: check `maxvol!` source/docs; if it doesn't handle complex, wrap with `real.(Q)` only when the imaginary part is negligible, or apply `maxvol!` to `[real(Q); imag(Q)]` stacked vertically (a standard workaround).

**Greedy — `dot` conjugates first argument:**

At lines 455 and 477, `LinearAlgebra.dot(a, b)` computes `∑ conj(aᵢ)·bᵢ`, not `∑ aᵢ·bᵢ`. The Greedy cross algorithm (TT-CROSS skeleton approximation) expects a plain bilinear product, not a sesquilinear inner product. For real inputs these are identical, masking the bug. Fix: replace with `sum(cre1[tind1[j], :] .* cre2[:, tind2[j]])` and similarly for the `alpha` computation.

**DMRG:** No complex issues identified. `svd`, `qr` on complex matrices are correct in Julia's LinearAlgebra. No changes needed.

### Audit Steps

1. Read `Maxvol.jl` source to confirm whether `maxvol!` handles complex matrices.
2. If not: implement and test the stacked-real workaround for the MaxVol cross path.
3. Replace `dot` with explicit element-wise products in Greedy (lines 455, 477).
4. Re-run the new complex test cases to confirm fixes.

---

## Section 3: Code Quality & Performance Improvements

### `_evaluate_tt` vectorization

Current implementation (lines 128–142) loops over each point with repeated `1×1` matrix multiplications, allocating per-point. Rewrite to process all P points simultaneously:

```julia
function _evaluate_tt(cores, indices, N)
    T = eltype(cores[1])
    n_points = size(indices, 1)
    # state: (n_points,) vector of partial products as row vectors
    state = ones(T, n_points, 1)
    for d in 1:N
        # gather core slices for each point: (n_points, r_left, r_right)
        r_r = size(cores[d], 3)
        slices = cores[d][indices[:, d], :, :]  # (n_points, r_left, r_r)
        # batched matmul: state (n_points, 1, r_l) × slices (n_points, r_l, r_r)
        state = reshape(sum(reshape(state, n_points, 1, :) .* slices, dims=2), n_points, r_r)
    end
    return vec(state)
end
```

This eliminates per-point allocation and is amenable to vectorization by the Julia compiler.

### `_build_fiber_indices` allocation

The inner `vcat(left_idx, [i], right_idx)` (line 170) allocates a new vector for every fiber. Replace with a pre-allocated `indices` buffer written in-place using sliced assignment — the outer `indices` matrix already exists, so just fill rows directly without the intermediate `vcat`.

### `_svdtrunc` consolidation

The local `_svdtrunc` in `tt_cross_interpolation.jl` (lines 144–161) duplicates logic from `tt_tools.jl` but uses plain `svd` instead of MatrixAlgebraKit's `svd_compact`. Add a comment clarifying the reason (cross file needs to handle complex matrices; MAK's compact SVD should also handle complex — verify and consolidate if safe, otherwise document the intentional separation).

---

## Out of Scope

- No changes to public API (`tt_cross`, `tt_integrate` signatures unchanged).
- No refactoring of algorithm sweep logic.
- No new exports.
- No changes to `tt_integrate` or the Gauss-Legendre quadrature.

---

## Success Criteria

1. All new test cases pass at stated tolerances for all three algorithms.
2. Complex separable functions (rank-1) achieve `relerr < 1e-8` for all algorithms.
3. No regression in existing tests.
4. `_evaluate_tt` rewrite passes all tests and is measurably faster on a benchmark (informal).
