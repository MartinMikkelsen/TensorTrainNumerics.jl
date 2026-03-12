# Cross Interpolation: Testing & Code Quality Design

**Date:** 2026-03-12
**Scope:** `src/tt_cross_interpolation.jl` and `test/test_tt_cross_interpolation.jl`
**Goals:** (1) systematic mathematical accuracy tests, (2) targeted audit and fix of complex-domain code paths, (3) minor secondary performance improvements.

---

## Background

`tt_cross_interpolation.jl` implements three TT cross-approximation algorithms — MaxVol, Greedy, and DMRG — plus `tt_integrate`. The existing test suite covers struct construction and a handful of smoke/accuracy tests, but:

- MaxVol and Greedy lack full-tensor accuracy verification.
- Complex-domain paths in MaxVol and Greedy are undertested ("tests pass but I don't trust them").
- DMRG is considered more trustworthy.

**Important dispatch note:** When `tt_cross(..., MaxVol(...))` is called with a complex-valued function, the existing code at lines 193–204 immediately redirects to `tt_cross(..., DMRG(...))`. This means complex-valued tests targeting MaxVol actually exercise DMRG under the hood. Tests must account for this — either assert on the dispatch explicitly, or target DMRG directly for complex cases.

---

## Section 1: Systematic Test Matrix

### Verification Strategy

For small grids (physical dimension ≤ 8 per mode, N ≤ 5 modes), reconstruct the full tensor via `ttv_to_tensor(tt)` and compare against `f` evaluated exhaustively over the grid using broadcasting. This provides exact ground truth rather than sampling-based error estimates.

Assertion for each test:
```julia
relerr = norm(ttv_to_tensor(tt) - exact) / max(norm(exact), eps())
@test relerr < tol
```

### Real-valued test cases (all three algorithms: MaxVol, Greedy, DMRG)

| Class | Function | Grid | Expected rank | Tolerance |
|---|---|---|---|---|
| Rank-1 separable | `f(x) = ∏ sin(xₖ)` | 6×6×6×6 | 1 | 1e-8 |
| Low-rank polynomial | `f(x) = (x₁+x₂+x₃)²` | 8×8×8 | ≤3 | 1e-6 |
| Smooth Gaussian | `f(x) = exp(-‖x‖²)` | 8×8×8×8 | small | 1e-4 |

### Complex-valued test cases

Complex-valued inputs are handled differently per algorithm:
- **MaxVol**: silently redirects to DMRG (line 193–204), so complex MaxVol tests assert on the DMRG code path.
- **Greedy**: has its own complex path (suspected `dot` bug — see Section 2).
- **DMRG**: native complex support via Julia's LinearAlgebra (`svd`, `qr`) and `maxvol!` (which supports `ComplexF32`/`ComplexF64` natively).

| Class | Function | Grid | Algorithms | Tolerance |
|---|---|---|---|---|
| Rank-1 separable | `f(x) = ∏ exp(i·xₖ)`, grid in `[0,1]` | 5×5×5 | Greedy, DMRG | 1e-8 |
| Low-rank separable | `f(x) = x₁·x₂·x₃`, grid `collect(range(1.0+0.5im, 2.0+1.0im, 5))` per dim | 5×5×5 | Greedy, DMRG | 1e-8 |
| Smooth complex | `f(x) = exp(i·∑xₖ²)`, grid in `[0,1]` | 6×6×6 | Greedy, DMRG | 1e-4 |

Note: MaxVol is excluded from complex test cases since it dispatches to DMRG; the DMRG row already covers that code path.

### Test organization

Tests are added to the existing `@testset "tt_cross"` block. Each function class becomes a nested `@testset`. A helper loops over algorithms for the real-valued cases:

```julia
for (alg_name, alg) in [("MaxVol", MaxVol(...)), ("Greedy", Greedy(...)), ("DMRG", DMRG(...))]
    @testset "$alg_name" begin
        tt = tt_cross(f, domain, alg)
        relerr = norm(ttv_to_tensor(tt) - exact) / max(norm(exact), eps())
        @test relerr < tol
    end
end
```

For complex cases, only Greedy and DMRG appear in the loop.

---

## Section 2: Complex-Domain Audit & Fixes

### MaxVol — no fix needed

`maxvol!` from `Maxvol.jl` is defined with type constraint `T <: Union{Float32, Float64, ComplexF32, ComplexF64}` and handles complex matrices natively. The `qr(V)` → `Matrix(first(qr(V)))` path correctly produces a `ComplexF64` Q matrix that `maxvol!` can handle. No changes needed for MaxVol.

The existing dispatch to DMRG for complex-valued targets (line 193–204) is an intentional stability choice, not a workaround for a `maxvol!` limitation.

### Greedy — `dot` conjugation bug

At two sites, `LinearAlgebra.dot(a, b)` computes `∑ conj(aᵢ)·bᵢ` (sesquilinear), but the Greedy cross skeleton approximation requires a plain bilinear product `∑ aᵢ·bᵢ`. For real inputs these are identical, masking the bug.

**Line 455:**
```julia
# Current (wrong for complex):
cry_approx = [LinearAlgebra.dot(cre1[tind1[j], :], cre2[:, tind2[j]]) for j in 1:testsz]
# Fix:
cry_approx = [sum(cre1[tind1[j], :] .* cre2[:, tind2[j]]) for j in 1:testsz]
```

**Line 477:**
```julia
# Current (wrong for complex):
alpha = cre1_new[imax1, 1] - LinearAlgebra.dot(vec(erow * uold), lold * ecol)
# Fix:
alpha = cre1_new[imax1, 1] - sum(vec(erow * uold) .* vec(lold * ecol))
```
Note: at this code site, `ecol = cre1_new[ilocl[i+1], 1]` where `ilocl[i+1]` is always a length-1 vector (each bond holds a single pivot during Greedy rank growth). So `lold * ecol` is a length-`Rs[i+1]` vector and `vec(erow * uold)` is also length-`Rs[i+1]`; the element-wise `sum(.*)` correctly produces a scalar. If this path were ever reached with multiple pivots per bond, the fix would need to be the scalar `(vec(erow * uold))' * vec(lold * ecol)` (bilinear, not Hermitian).

### DMRG — no fix needed

`svd` and `qr` on complex matrices are correct in Julia's LinearAlgebra. `maxvol!` supports `ComplexF32`/`ComplexF64` natively. No changes needed.

### Audit steps

1. Apply the two Greedy `dot` fixes above.
2. Run the new complex separable test cases for Greedy to confirm the fix.
3. Verify the existing complex test in `test_tt_cross_interpolation.jl` (lines 213–240) still passes; if it previously passed via DMRG fallback (Greedy stalled → DMRG), confirm the fix allows Greedy to succeed directly.

---

## Section 3: Code Quality & Performance Improvements

### `_evaluate_tt` vectorization

Current implementation (lines 128–142) loops over each point with per-point `1×1` matrix multiplications. Rewrite to process all P points simultaneously. Note: cores have layout `(phys_dim, left_rank, right_rank)`, so indexing `cores[d][indices[:, d], :, :]` correctly gathers slices as `(n_points, r_left, r_right)`.

```julia
function _evaluate_tt(cores, indices, N)
    T = eltype(cores[1])
    n_points = size(indices, 1)
    state = ones(T, n_points, 1)  # (n_points, r_right) after each step
    for d in 1:N
        r_r = size(cores[d], 3)
        slices = cores[d][indices[:, d], :, :]  # (n_points, r_left, r_right)
        # contract state (n_points, 1, r_left) with slices (n_points, r_left, r_right)
        state = reshape(sum(reshape(state, n_points, 1, :) .* slices, dims=2), n_points, r_r)
    end
    return vec(state)
end
```

### `_build_fiber_indices` allocation

The inner `vcat(left_idx, [i], right_idx)` (line 170) allocates a new vector per fiber. Replace with direct sliced assignment into the pre-allocated `indices` matrix:

```julia
# instead of: indices[idx, :] = vcat(left_idx, [i], right_idx)
n_left = j - 1
n_right = N - j
j > 1  && (indices[idx, 1:n_left]         = lsets[j][r_left, :])
           indices[idx, j]                 = i
j < N  && (indices[idx, (j+1):(j+n_right)] = rsets[j][r_right, :])
```

### `_svdtrunc` consolidation

The local `_svdtrunc` in `tt_cross_interpolation.jl` (lines 144–161) uses plain `svd`, while `tt_tools.jl` uses MatrixAlgebraKit's `svd_compact`. Verify whether MatrixAlgebraKit's `svd_compact` handles complex matrices correctly; if so, consolidate. If not, add an explicit comment explaining the intentional separation.

---

## Out of Scope

- No changes to public API (`tt_cross`, `tt_integrate` signatures unchanged).
- No refactoring of algorithm sweep logic.
- No new exports.
- No changes to `tt_integrate` or the Gauss-Legendre quadrature.

---

## Success Criteria

1. All new test cases pass at stated tolerances for all three algorithms.
2. Complex separable functions (rank-1) achieve `relerr < 1e-8` for Greedy and DMRG.
3. No regression in existing tests.
4. `_evaluate_tt` rewrite passes all existing and new tests.
