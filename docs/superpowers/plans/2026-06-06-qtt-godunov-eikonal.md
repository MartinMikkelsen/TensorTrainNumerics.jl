# QTT Godunov Eikonal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not commit; the repository owner makes commits.

**Goal:** Build an Eikonal-specific QTT solver for variable-speed problems `|grad u| = s(x)` using a monotone Godunov residual and active-set Newton-MALS refinement.

**Architecture:** Keep the existing centered viscous Eikonal solver as a baseline and add a separate `src/solvers/eikonal_godunov.jl`. The new solver has dense Godunov residual/fast-sweeping reference routines, QTT one-sided difference operators, QTT active masks, and a non-Hermitian MALS Newton-correction loop. Benchmarks compare dense fast sweeping, QTT residual, residual history, ranks, and error.

**Tech Stack:** Julia, TensorTrainNumerics TT/QTT operators, TensorOperations, Test, CairoMakie for examples.

---

## File Structure

- Create `src/solvers/eikonal_godunov.jl`
  - Dense 1D/2D Godunov residuals.
  - Dense fast-sweeping reference for small 1D/2D variable-speed grids.
  - QTT one-sided derivative operators.
  - QTT active-set Newton-MALS routines.
- Modify `src/TensorTrainNumerics.jl`
  - Export and include the new Eikonal-specific solver APIs.
- Modify `examples/nonlinear_benchmark_utils.jl`
  - Add benchmark helpers for variable-speed Godunov Eikonal.
- Create `examples/eikonal_godunov.jl`
  - Plot dense reference, QTT solution, residual history, rank history, and error.
- Create `test/test_eikonal_godunov.jl`
  - TDD tests for dense reference, QTT APIs, residual convergence, variable speed support, and rank caps.
- Modify `test/runtests.jl`
  - Include `test_eikonal_godunov.jl`.

## Public API Target

```julia
dense_fast_sweeping_eikonal_2d(speed; h, boundary = :zero, max_sweeps = 10_000, tol = 1.0e-12)
godunov_residual_2d(u, speed; h)

eikonal_godunov_mals_1d(d; speed, boundary = :zero, init = :fast_sweeping,
    max_newton = 30, residual_tol = 1.0e-10, max_bond = 64, mals_tol = 0.0,
    damping = :backtracking, verbose = false)

eikonal_godunov_mals_2d(d; speed, boundary = :zero, init = :fast_sweeping,
    max_newton = 30, residual_tol = 1.0e-10, max_bond = 64, mals_tol = 0.0,
    damping = :backtracking, verbose = false)
```

`speed` accepts either a QTT-compatible function or a `TTvector`. Function inputs are sampled on the same interior grid as the current Eikonal examples.

## Task 1: Dense Godunov Residual And Fast-Sweeping Reference

**Files:**
- Create: `src/solvers/eikonal_godunov.jl`
- Test: `test/test_eikonal_godunov.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write failing dense residual tests**

```julia
using Test
using TensorTrainNumerics

@testset "dense Godunov Eikonal reference" begin
    N = 32
    h = 1.0 / (N + 1)
    x = [i * h for i in 1:N]
    u = min.(x, 1 .- x)
    speed = ones(N)

    r = TensorTrainNumerics.godunov_residual_1d(u, speed; h = h)

    @test maximum(abs.(r[2:end-1])) < 1.0e-12
end
```

- [ ] **Step 2: Run test and verify red**

Run: `/Users/pzb464/.juliaup/bin/julialauncher --project=. test/test_eikonal_godunov.jl`

Expected: `UndefVarError: godunov_residual_1d not defined`.

- [ ] **Step 3: Implement dense 1D Godunov residual**

```julia
function godunov_residual_1d(u::AbstractVector, speed::AbstractVector; h::Real)
    N = length(u)
    @assert length(speed) == N
    r = similar(float.(u))
    for i in 1:N
        uim = i == 1 ? 0.0 : u[i - 1]
        uip = i == N ? 0.0 : u[i + 1]
        dm = (u[i] - uim) / h
        dp = (uip - u[i]) / h
        gx = max(dm, -dp, 0.0)
        r[i] = gx^2 - speed[i]^2
    end
    return r
end
```

- [ ] **Step 4: Export/include and verify green**

Add `export godunov_residual_1d` and `include("solvers/eikonal_godunov.jl")` in `src/TensorTrainNumerics.jl`.

Run: `/Users/pzb464/.juliaup/bin/julialauncher --project=. test/test_eikonal_godunov.jl`

Expected: dense residual test passes.

- [ ] **Step 5: Add dense 2D residual and fast-sweeping tests**

```julia
@testset "dense 2D fast sweeping solves unit speed square" begin
    N = 32
    h = 1.0 / (N + 1)
    speed = ones(N, N)

    u, hist = TensorTrainNumerics.dense_fast_sweeping_eikonal_2d(speed; h = h, tol = 1.0e-12)
    r = TensorTrainNumerics.godunov_residual_2d(u, speed; h = h)

    @test hist[end] < 1.0e-12
    @test norm(vec(r), Inf) < 5.0e-10
    @test maximum(u) > 0.2
end
```

- [ ] **Step 6: Implement dense 2D sweeping update**

Use the standard local update for `|grad u| = s` with neighbor minima `a = min(u[i-1,j], u[i+1,j])`, `b = min(u[i,j-1], u[i,j+1])`:

```julia
function _godunov_update(a::Real, b::Real, s::Real, h::Real)
    sh = s * h
    lo, hi = minmax(a, b)
    if hi - lo >= sh
        return lo + sh
    end
    disc = 2sh^2 - (hi - lo)^2
    return (lo + hi + sqrt(max(disc, 0.0))) / 2
end
```

Sweep in the four standard orders until max update is below `tol`.

## Task 2: QTT Operators And Residual Evaluation

**Files:**
- Modify: `src/solvers/eikonal_godunov.jl`
- Test: `test/test_eikonal_godunov.jl`

- [ ] **Step 1: Write failing tests for QTT residual consistency**

```julia
@testset "QTT Godunov residual matches dense residual in 1D" begin
    d = 5
    N = 2^d
    h = 1.0 / (N + 1)
    x = [i * h for i in 1:N]
    uvals = min.(x, 1 .- x)
    speed_vals = ones(N)
    u = function_to_qtt(t -> min(t, 1 - t), d)
    speed = ones_tt(2, d)

    r_qtt = TensorTrainNumerics.godunov_residual_qtt_1d(u, speed; h = h, max_bond = 32)
    r_dense = TensorTrainNumerics.godunov_residual_1d(uvals, speed_vals; h = h)

    @test norm(real.(qtt_to_function(r_qtt)) .- r_dense, Inf) < 1.0e-10
end
```

- [ ] **Step 2: Verify red**

Run: `/Users/pzb464/.juliaup/bin/julialauncher --project=. test/test_eikonal_godunov.jl`

Expected: `UndefVarError: godunov_residual_qtt_1d not defined`.

- [ ] **Step 3: Implement one-sided QTT difference operators and QTT residual**

Build forward/backward differences from existing `shift` or explicit one-sided stencils. For this first vertical slice, construct masks by evaluating the current QTT on the grid and recompressing the binary mask; TT-cross mask construction is outside this plan.

```julia
Dminus = (I - Sminus) / h
Dplus = (Splus - I) / h
```

For 2D serial QTT, use Kronecker products:

```julia
Dxminus = Dminus ⊗ id_tto(d)
Dxplus = Dplus ⊗ id_tto(d)
Dyminus = id_tto(d) ⊗ Dminus
Dyplus = id_tto(d) ⊗ Dplus
```

## Task 3: Active Masks And Newton-MALS Linearization

**Files:**
- Modify: `src/solvers/eikonal_godunov.jl`
- Test: `test/test_eikonal_godunov.jl`

- [ ] **Step 1: Write failing active-mask tests**

```julia
@testset "active-set masks choose upwind derivative directions" begin
    d = 5
    N = 2^d
    h = 1.0 / (N + 1)
    u = function_to_qtt(x -> min(x, 1 - x), d)

    masks = TensorTrainNumerics.godunov_active_masks_1d(u; h = h)

    @test masks.minus isa TTvector
    @test masks.plus isa TTvector
    @test maximum(real.(qtt_to_function(masks.minus)) .+ real.(qtt_to_function(masks.plus))) <= 1.0 + 1.0e-12
end
```

- [ ] **Step 2: Verify red**

Run: `/Users/pzb464/.juliaup/bin/julialauncher --project=. test/test_eikonal_godunov.jl`

Expected: `UndefVarError: godunov_active_masks_1d not defined`.

- [ ] **Step 3: Implement active masks**

Evaluate the candidate one-sided derivatives, select the active derivative with larger positive contribution, and compress the binary masks:

```julia
mminus[i] = dm[i] >= max(-dp[i], 0.0) ? 1.0 : 0.0
mplus[i] = -dp[i] > max(dm[i], 0.0) ? 1.0 : 0.0
```

- [ ] **Step 4: Implement frozen active-set Jacobian**

For 1D, with active derivative `g = Mminus * Dminus*u + Mplus * (-Dplus*u)`, use

```julia
J = 2 * diag(g) * (diag(mminus) * Dminus - diag(mplus) * Dplus)
rhs = -R(u)
```

For 2D, sum the corresponding x/y terms.

## Task 4: Public QTT Newton-MALS Solvers

**Files:**
- Modify: `src/solvers/eikonal_godunov.jl`
- Modify: `src/TensorTrainNumerics.jl`
- Test: `test/test_eikonal_godunov.jl`

- [ ] **Step 1: Write failing solver tests**

```julia
@testset "QTT active-set Newton-MALS solves variable-speed 1D Eikonal" begin
    d = 6
    speed_fun = x -> 1.0 + 0.25 * sin(2π * x)

    u, info = TensorTrainNumerics.eikonal_godunov_mals_1d(d;
        speed = speed_fun,
        max_newton = 30,
        residual_tol = 1.0e-8,
        max_bond = 32,
        verbose = false
    )

    @test info.residual_history[end] < 1.0e-8
    @test maximum(u.ttv_rks) <= 32
end
```

- [ ] **Step 2: Verify red**

Run: `/Users/pzb464/.juliaup/bin/julialauncher --project=. test/test_eikonal_godunov.jl`

Expected: `UndefVarError: eikonal_godunov_mals_1d not defined`.

- [ ] **Step 3: Implement 1D solver loop**

Algorithm per iteration:

```julia
r = godunov_residual_qtt_1d(u, speed_qtt; h = h, max_bond = max_bond)
resid = norm(r) / max(norm(speed_qtt), eps())
resid < residual_tol && return u, info
J = godunov_jacobian_qtt_1d(u; h = h, max_bond = max_bond)
δ = _nonsymmetric_mals_linsolve(J, -r, zeros_tt(2, d); max_bond = max_bond, tol = mals_tol)
u = damped_update(u, δ, speed_qtt; h = h, max_bond = max_bond)
```

- [ ] **Step 4: Implement damping**

Backtracking accepts the first `α` from `1.0, 0.5, 0.25, ...` that reduces the Godunov residual norm. If none reduce it, use the smallest `α` and continue unless residual is `NaN`, in which case throw `DomainError`.

- [ ] **Step 5: Add 2D solver test and implementation**

Use `d = 4`, `speed = (x, y) -> 1 + 0.2sin(2πx)sin(2πy)`, `residual_tol = 1e-7`, `max_bond = 48`.

## Task 5: Benchmark Helper And Example

**Files:**
- Modify: `examples/nonlinear_benchmark_utils.jl`
- Create: `examples/eikonal_godunov.jl`
- Test: `test/test_nonlinear_examples.jl`

- [ ] **Step 1: Add benchmark test**

```julia
@testset "Godunov Eikonal benchmark reports QTT residual and ranks" begin
    bench = eikonal_godunov_2d_benchmark(;
        d = 4,
        speed = (x, y) -> 1.0 + 0.2 * sin(2π * x) * sin(2π * y),
        residual_tol = 1.0e-7,
        max_bond = 48,
        verbose = false
    )

    @test bench.metrics.final_residual < 1.0e-7
    @test bench.metrics.max_rank <= 48
    @test bench.method_label == "Godunov active-set Newton-MALS"
end
```

- [ ] **Step 2: Implement benchmark helper**

Return fields:

```julia
(
    equation = "Variable-speed Eikonal 2D",
    method = :godunov_active_set_mals,
    method_label = "Godunov active-set Newton-MALS",
    d = d,
    N = 2^d,
    solution = u,
    grid = grid,
    dense_reference = ref,
    residual_history = info.residual_history,
    rank_history = info.rank_history,
    metrics = (
        final_residual = last(info.residual_history),
        max_rank = maximum(info.rank_history),
        reference_error = norm(grid - ref) / max(norm(ref), eps()),
    ),
)
```

- [ ] **Step 3: Add example plot**

`examples/eikonal_godunov.jl` should save `examples/output/eikonal_godunov.png` with panels for speed, dense reference, QTT solution, absolute error, residual history, and rank history.

## Task 6: Verification

**Files:**
- All modified files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
/Users/pzb464/.juliaup/bin/julialauncher --project=. test/test_eikonal_godunov.jl
/Users/pzb464/.juliaup/bin/julialauncher --project=. test/test_nonlinear_examples.jl
```

Expected: all tests pass.

- [ ] **Step 2: Run example**

Run:

```bash
/Users/pzb464/.juliaup/bin/julialauncher --project=examples examples/eikonal_godunov.jl
```

Expected: figure saved to `examples/output/eikonal_godunov.png`, final residual printed below requested tolerance.

- [ ] **Step 3: Run package tests**

Run:

```bash
/Users/pzb464/.juliaup/bin/julialauncher --project=. -e 'using Pkg; Pkg.test()'
```

Expected: `Testing TensorTrainNumerics tests passed`.

## Self-Review

- Spec coverage: variable speed, Godunov residual, fast-sweeping reference, QTT active-set Newton-MALS, examples, tests, and full verification are covered.
- Placeholder scan: no task uses open-ended placeholder steps.
- Type consistency: public API names use `eikonal_godunov_mals_1d`, `eikonal_godunov_mals_2d`, `godunov_residual_1d`, `godunov_residual_2d`, and benchmark helper names consistently.
- Scope check: this is one vertical slice; higher-order ENO/WENO, obstacles, arbitrary boundary data, TT-cross mask optimization, and fast marching in compressed form are intentionally deferred.
