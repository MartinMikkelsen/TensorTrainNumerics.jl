# Solver API Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `linear_solve`/`eigen_solve` algorithm-object front doors, rename cross interpolation's `DMRG` algorithm to `DMRGcross`, and move time-stepping linear-solver dispatch out of the time-evolution file.

**Architecture:** Keep algorithm-specific numerical kernels in `als.jl`, `mals.jl`, and `dmrg.jl`. Add thin API/orchestration files `linear_solver.jl` and `eigen_solver.jl`, then make legacy solver names delegate through the new front doors. Rename `euler.jl` to `time_evolution.jl` and make implicit steppers call `linear_solve`.

**Tech Stack:** Julia 1.12, TensorTrainNumerics TTvector/TToperator APIs, KrylovKit, IterativeSolvers, LinearMaps, TensorOperations, Test stdlib.

## Global Constraints

- Do not make git commits.
- Do not change the mathematical algorithms in this refactor.
- Keep existing linear and eigen solver function names as wrappers around the new front doors.
- The cross-interpolation rename from `DMRG` to `DMRGcross` is intentionally breaking.
- `eigen_solve` must not call `linear_solve`; they share algorithm objects and kernels, not local solve semantics.
- Prefer mechanical moves/renames over rewrites.

---

## File Structure

- Create `src/solvers/linear_solver.jl`: algorithm structs, `linear_solve` front door, legacy linear wrappers, string/legacy solver normalization, Krylov TT linear solve.
- Create `src/solvers/eigen_solver.jl`: `eigen_solve` front door and legacy eigen wrappers.
- Rename `src/solvers/euler.jl` to `src/solvers/time_evolution.jl`: keep time steppers, remove private solver-type dispatch, call `linear_solve`.
- Modify `src/solvers/als.jl`: replace public linear/eigen function bodies with `linear_solve`/`eigen_solve` methods for `ALS`; keep local kernels.
- Modify `src/solvers/mals.jl`: replace public linear/eigen function bodies with `linear_solve`/`eigen_solve` methods for `MALS`; keep local kernels.
- Modify `src/solvers/dmrg.jl`: replace public linear/eigen function bodies with `linear_solve`/`eigen_solve` methods for `DMRG`; keep local kernels.
- Modify `src/tt_cross_interpolation.jl`: rename cross algorithm `DMRG` to `DMRGcross`.
- Modify `src/TensorTrainNumerics.jl`: update exports and include order.
- Add `test/test_solver_api.jl`: direct new API coverage and compatibility checks.
- Modify `test/test_tt_cross_interpolation.jl`: update `DMRG` cross calls to `DMRGcross`.
- Modify `test/test_euler.jl`: cover object-based `tt_solver = ALS(...)` selection.
- Modify docs examples only where `DMRG(...)` is used for cross interpolation.

---

### Task 1: Rename Cross-Interpolation `DMRG` To `DMRGcross`

**Files:**
- Modify: `src/tt_cross_interpolation.jl`
- Modify: `src/TensorTrainNumerics.jl`
- Modify: `test/test_tt_cross_interpolation.jl`
- Modify: docs files found by `rg -n "DMRG\\(" docs/src src test`

**Interfaces:**
- Consumes: existing `CrossAlgorithm`, `PivotAlgorithm`, and `tt_cross` methods.
- Produces: `DMRGcross(; maxiter, tol, rmax, kickrank, verbose, pivot) <: CrossAlgorithm`.

- [ ] **Step 1: Write the failing rename test**

In `test/test_tt_cross_interpolation.jl`, replace any cross-interpolation construction using `DMRG(...)` with `DMRGcross(...)`. If there is no direct constructor test, add one near the other algorithm-constructor tests:

```julia
@testset "DMRGcross constructor" begin
    alg = DMRGcross(maxiter = 3, tol = 1.0e-8, rmax = 7, kickrank = 2, verbose = false)
    @test alg isa TensorTrainNumerics.CrossAlgorithm
    @test alg.maxiter == 3
    @test alg.tol == 1.0e-8
    @test alg.rmax == 7
    @test alg.kickrank == 2
    @test alg.verbose == false
end
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run:

```bash
julia --project=. -e 'using TensorTrainNumerics; include("test/test_tt_cross_interpolation.jl")'
```

Expected: failure with `UndefVarError: DMRGcross not defined` or an export/import failure.

- [ ] **Step 3: Rename the cross algorithm type and constructor**

In `src/tt_cross_interpolation.jl`, change:

```julia
struct DMRG{T <: Real, P <: PivotAlgorithm} <: CrossAlgorithm
```

to:

```julia
struct DMRGcross{T <: Real, P <: PivotAlgorithm} <: CrossAlgorithm
```

Change the constructor:

```julia
function DMRG(;
        maxiter::Int = CROSS_MAXITER[],
        tol::Real = CROSS_TOL[],
        rmax::Int = CROSS_RMAX[],
        kickrank::Union{Nothing, Int} = CROSS_KICKRANK[],
        verbose::Bool = true,
        pivot::PivotAlgorithm = MaxVolPivot()
    )
    return DMRG(maxiter, tol, rmax, kickrank, verbose, pivot)
end
```

to:

```julia
function DMRGcross(;
        maxiter::Int = CROSS_MAXITER[],
        tol::Real = CROSS_TOL[],
        rmax::Int = CROSS_RMAX[],
        kickrank::Union{Nothing, Int} = CROSS_KICKRANK[],
        verbose::Bool = true,
        pivot::PivotAlgorithm = MaxVolPivot()
    )
    return DMRGcross(maxiter, tol, rmax, kickrank, verbose, pivot)
end
```

Change the cross dispatch signature:

```julia
function tt_cross(
        f::Function,
        domain::Vector{<:AbstractVector{T}},
        alg::DMRG;
        ranks::Union{Int, Vector{Int}} = 2,
        val_size::Int = 1000
    ) where {T <: Number}
```

to:

```julia
function tt_cross(
        f::Function,
        domain::Vector{<:AbstractVector{T}},
        alg::DMRGcross;
        ranks::Union{Int, Vector{Int}} = 2,
        val_size::Int = 1000
    ) where {T <: Number}
```

In the Greedy fallback, change:

```julia
dmrg_alg = DMRG(maxiter = alg.maxiter, tol = alg.tol, rmax = alg.rmax, kickrank = nothing, verbose = alg.verbose)
```

to:

```julia
dmrg_alg = DMRGcross(maxiter = alg.maxiter, tol = alg.tol, rmax = alg.rmax, kickrank = nothing, verbose = alg.verbose)
```

- [ ] **Step 4: Update exports**

In `src/TensorTrainNumerics.jl`, change:

```julia
export tt_cross, tt_integrate, MaxVol, DMRG, Greedy
```

to:

```julia
export tt_cross, tt_integrate, MaxVol, DMRGcross, Greedy
```

- [ ] **Step 5: Update remaining cross references**

Run:

```bash
rg -n "DMRG\\(" src test docs/src
```

Every remaining cross-interpolation use should become `DMRGcross(...)`. Do not change `dmrg_linsolve`, `dmrg_eigsolve`, or text that refers to the DMRG solver algorithm concept outside cross interpolation.

- [ ] **Step 6: Verify the rename**

Run:

```bash
julia --project=. -e 'using TensorTrainNumerics; include("test/test_tt_cross_interpolation.jl")'
```

Expected: all cross interpolation tests pass.

- [ ] **Step 7: No-commit checkpoint**

Run:

```bash
git status --short
```

Expected: modified source/tests/docs only. Do not commit.

---

### Task 2: Add Solver Algorithm Objects And `linear_solve` Front Door

**Files:**
- Create: `src/solvers/linear_solver.jl`
- Modify: `src/TensorTrainNumerics.jl`
- Add: `test/test_solver_api.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: `abstract type LinearSolverAlgorithm end`
- Produces: `abstract type EigenSolverAlgorithm <: LinearSolverAlgorithm end`
- Produces: `ALS`, `MALS`, `DMRG`, `Krylov`
- Produces: `linear_solve(A, b, guess; alg = MALS())`
- Produces: `linear_solve(A, b, guess, alg::LinearSolverAlgorithm)`
- Produces: legacy aliases `TTLinearSolver`, `ALSSolver`, `MALSSolver`, `DMRGSolver`, `KrylovSolver`

- [ ] **Step 1: Write failing constructor and front-door tests**

Add `test/test_solver_api.jl`:

```julia
using Test
using LinearAlgebra
using TensorTrainNumerics

@testset "solver algorithm constructors" begin
    @test ALS(sweep_count = 3).sweep_count == 3
    @test MALS(tol = 1.0e-9, rmax = 4).tol == 1.0e-9
    @test MALS(tol = 1.0e-9, rmax = 4).rmax == 4
    @test DMRG(N = 2, tol = 1.0e-8, rmax_schedule = [4]).N == 2
    @test DMRG(N = 2, tol = 1.0e-8, rmax_schedule = [4]).rmax_schedule == [4]
    @test Krylov(max_bond = 3, krylov_solver = :gmres).max_bond == 3
    @test Krylov(max_bond = 3, krylov_solver = :gmres).krylov_solver == :gmres
    @test ALSSolver === ALS
    @test MALSSolver === MALS
    @test DMRGSolver === DMRG
    @test KrylovSolver === Krylov
end

@testset "linear_solve front door with ALS" begin
    dims = (2, 2, 2)
    A = id_tto(3)
    b = rand_tt(dims, [1, 2, 2, 1])
    guess = rand_tt(dims, [1, 2, 2, 1])

    x = linear_solve(A, b, guess, ALS(sweep_count = 2))
    @test x isa TensorTrainNumerics.AbstractTTvector
    @test norm(A * x - b) / max(norm(b), eps()) < 1.0e-8
end
```

In `test/runtests.jl`, add after `include("test_euler.jl")` or before it:

```julia
include("test_solver_api.jl")
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
julia --project=. -e 'using TensorTrainNumerics; include("test/test_solver_api.jl")'
```

Expected: failure with `UndefVarError: ALS not defined` or `linear_solve not defined`.

- [ ] **Step 3: Create `linear_solver.jl` with algorithm types**

Create `src/solvers/linear_solver.jl` with:

```julia
using KrylovKit

const KRYLOV_ROUND_RANK = Ref{Int}(0)

abstract type LinearSolverAlgorithm end
abstract type EigenSolverAlgorithm <: LinearSolverAlgorithm end

struct ALS <: EigenSolverAlgorithm
    sweep_count::Int
    it_solver::Bool
    r_itsolver::Int
    return_info::Bool
    sweep_schedule::Union{Nothing, Vector{Int}}
    rmax_schedule::Union{Nothing, Vector{Int}}
    noise_schedule::Union{Nothing, Vector{Float64}}
    itslv_thresh::Int
    maxiter::Int
    linsolv_tol::Float64
end

function ALS(;
        sweep_count::Int = 2,
        it_solver::Bool = false,
        r_itsolver::Int = 5000,
        return_info::Bool = false,
        sweep_schedule::Union{Nothing, Vector{Int}} = nothing,
        rmax_schedule::Union{Nothing, Vector{Int}} = nothing,
        noise_schedule::Union{Nothing, Vector{Float64}} = nothing,
        itslv_thresh::Int = 1024,
        maxiter::Int = 200,
        linsolv_tol::Float64 = 1.0e-8
    )
    return ALS(
        sweep_count, it_solver, r_itsolver, return_info,
        sweep_schedule, rmax_schedule, noise_schedule,
        itslv_thresh, maxiter, linsolv_tol
    )
end

struct MALS <: EigenSolverAlgorithm
    tol::Float64
    rmax::Union{Nothing, Int}
    return_info::Bool
    sweep_schedule::Union{Nothing, Vector{Int}}
    rmax_schedule::Union{Nothing, Vector{Int}}
    it_solver::Bool
    linsolv_maxiter::Int
    linsolv_tol::Float64
    itslv_thresh::Int
end

function MALS(;
        tol::Float64 = 1.0e-12,
        rmax::Union{Nothing, Int} = nothing,
        return_info::Bool = false,
        sweep_schedule::Union{Nothing, Vector{Int}} = nothing,
        rmax_schedule::Union{Nothing, Vector{Int}} = nothing,
        it_solver::Bool = false,
        linsolv_maxiter::Int = 200,
        linsolv_tol::Float64 = 1.0e-6,
        itslv_thresh::Int = 256
    )
    return MALS(tol, rmax, return_info, sweep_schedule, rmax_schedule, it_solver, linsolv_maxiter, linsolv_tol, itslv_thresh)
end

struct DMRG <: EigenSolverAlgorithm
    sweep_count::Int
    N::Int
    tol::Float64
    sweep_schedule::Union{Nothing, Vector{Int}}
    rmax_schedule::Union{Nothing, Vector{Int}}
    it_solver::Bool
    linsolv_maxiter::Int
    linsolv_tol::Float64
    itslv_thresh::Int
    return_info::Bool
    verbose::Bool
end

function DMRG(;
        sweep_count::Int = 2,
        N::Int = 2,
        tol::Float64 = 1.0e-12,
        sweep_schedule::Union{Nothing, Vector{Int}} = nothing,
        rmax_schedule::Union{Nothing, Vector{Int}} = nothing,
        it_solver::Bool = false,
        linsolv_maxiter::Int = 200,
        linsolv_tol::Float64 = 1.0e-6,
        itslv_thresh::Int = 256,
        return_info::Bool = false,
        verbose::Bool = false
    )
    return DMRG(sweep_count, N, tol, sweep_schedule, rmax_schedule, it_solver, linsolv_maxiter, linsolv_tol, itslv_thresh, return_info, verbose)
end

struct Krylov <: LinearSolverAlgorithm
    max_bond::Int
    krylov_solver::Symbol
    krylovdim::Int
    maxiter::Int
    rtol::Float64
    atol::Float64
    tol::Union{Nothing, Float64}
    orth::Any
    issymmetric::Bool
    ishermitian::Bool
    isposdef::Bool
    verbosity::Int
end

function Krylov(;
        max_bond::Int = 0,
        krylov_solver::Symbol = :auto,
        krylovdim::Int = 8,
        maxiter::Int = 20,
        rtol::Float64 = 1.0e-8,
        atol::Float64 = 1.0e-12,
        tol::Union{Nothing, Real} = nothing,
        orth = KrylovKit.KrylovDefaults.orth,
        issymmetric::Bool = false,
        ishermitian::Union{Nothing, Bool} = nothing,
        isposdef::Bool = false,
        verbosity::Int = 0
    )
    tol_value = isnothing(tol) ? nothing : Float64(tol)
    hermitian_value = isnothing(ishermitian) ? issymmetric : ishermitian
    return Krylov(max_bond, krylov_solver, krylovdim, maxiter, Float64(rtol), Float64(atol), tol_value, orth, issymmetric, hermitian_value, isposdef, verbosity)
end

const TTLinearSolver = LinearSolverAlgorithm
const ALSSolver = ALS
const MALSSolver = MALS
const DMRGSolver = DMRG
const KrylovSolver = Krylov

linear_solve(A, b, guess; alg::LinearSolverAlgorithm = MALS()) = linear_solve(A, b, guess, alg)
```

- [ ] **Step 4: Add temporary linear dispatch through legacy implementations**

Append to `src/solvers/linear_solver.jl`:

```julia
function linear_solve(A, b, guess, alg::ALS)
    return als_linsolve(
        A, b, guess;
        sweep_count = alg.sweep_count,
        it_solver = alg.it_solver,
        r_itsolver = alg.r_itsolver,
        return_info = alg.return_info
    )
end

function linear_solve(A, b, guess, alg::MALS)
    rmax = isnothing(alg.rmax) ? round(Int, sqrt(prod(guess.ttv_dims)::Int)) : alg.rmax
    return mals_linsolve(A, b, guess; tol = alg.tol, rmax = rmax, return_info = alg.return_info)
end

function linear_solve(A, b, guess, alg::DMRG)
    sweep_schedule = isnothing(alg.sweep_schedule) ? [2] : alg.sweep_schedule
    rmax_schedule = isnothing(alg.rmax_schedule) ? [isqrt(prod(guess.ttv_dims)::Int)] : alg.rmax_schedule
    return dmrg_linsolve(
        A, b, guess;
        sweep_count = alg.sweep_count,
        N = alg.N,
        tol = alg.tol,
        sweep_schedule = sweep_schedule,
        rmax_schedule = rmax_schedule,
        it_solver = alg.it_solver,
        linsolv_maxiter = alg.linsolv_maxiter,
        linsolv_tol = alg.linsolv_tol,
        itslv_thresh = alg.itslv_thresh,
        return_info = alg.return_info,
        verbose = alg.verbose
    )
end
```

Krylov dispatch will be moved from `euler.jl` in Task 5.

- [ ] **Step 5: Update module exports and include order**

In `src/TensorTrainNumerics.jl`, replace the old solver exports/includes:

```julia
export als_linsolve, als_eigsolve, als_gen_eigsolv
include("solvers/als.jl")

export mals_eigsolve, mals_linsolve
include("solvers/mals.jl")

export dmrg_linsolve, dmrg_eigsolve
include("solvers/dmrg.jl")
```

with:

```julia
export LinearSolverAlgorithm, EigenSolverAlgorithm
export ALS, MALS, DMRG, Krylov
export TTLinearSolver, ALSSolver, MALSSolver, DMRGSolver, KrylovSolver
export linear_solve
include("solvers/linear_solver.jl")

export als_linsolve, als_eigsolve, als_gen_eigsolv
include("solvers/als.jl")

export mals_eigsolve, mals_linsolve
include("solvers/mals.jl")

export dmrg_linsolve, dmrg_eigsolve
include("solvers/dmrg.jl")
```

- [ ] **Step 6: Remove duplicate solver-tag definitions from `euler.jl`**

In `src/solvers/euler.jl`, delete:

```julia
const KRYLOV_ROUND_RANK = Ref{Int}(0)
abstract type TTLinearSolver end
struct ALSSolver <: TTLinearSolver end
struct MALSSolver <: TTLinearSolver end
struct DMRGSolver <: TTLinearSolver end
struct KrylovSolver <: TTLinearSolver end
_tt_linsolver(...)
_tt_linsolve(...)
```

Keep `krylov_linsolve` in `euler.jl` until Task 5 moves it.

- [ ] **Step 7: Verify the new constructors and ALS front door**

Run:

```bash
julia --project=. -e 'using TensorTrainNumerics; include("test/test_solver_api.jl")'
```

Expected: constructor tests and `linear_solve(..., ALS(...))` test pass.

- [ ] **Step 8: No-commit checkpoint**

Run:

```bash
git status --short
```

Expected: modified source/tests. Do not commit.

---

### Task 3: Make Legacy Linear Wrappers Delegate To `linear_solve`

**Files:**
- Modify: `src/solvers/linear_solver.jl`
- Modify: `src/solvers/als.jl`
- Modify: `src/solvers/mals.jl`
- Modify: `src/solvers/dmrg.jl`
- Modify: `test/test_solver_api.jl`

**Interfaces:**
- Consumes: `ALS`, `MALS`, `DMRG`, `linear_solve`.
- Produces: `als_linsolve`, `mals_linsolve`, and `dmrg_linsolve` wrappers that call `linear_solve`.

- [ ] **Step 1: Add legacy wrapper equivalence tests**

Append to `test/test_solver_api.jl`:

```julia
@testset "legacy linear wrappers delegate to linear_solve" begin
    dims = (2, 2, 2)
    A = id_tto(3)
    b = rand_tt(dims, [1, 2, 2, 1])
    guess = rand_tt(dims, [1, 2, 2, 1])

    x_new = linear_solve(A, b, guess, MALS(tol = 1.0e-10, rmax = 4))
    x_old = mals_linsolve(A, b, guess; tol = 1.0e-10, rmax = 4)
    @test norm(A * x_new - b) / max(norm(b), eps()) < 1.0e-8
    @test norm(A * x_old - b) / max(norm(b), eps()) < 1.0e-8

    x_dmrg = linear_solve(A, b, guess, DMRG(N = 2, sweep_schedule = [2], rmax_schedule = [4]))
    @test norm(A * x_dmrg - b) / max(norm(b), eps()) < 1.0e-8
end
```

- [ ] **Step 2: Run the targeted test before refactor**

Run:

```bash
julia --project=. -e 'using TensorTrainNumerics; include("test/test_solver_api.jl")'
```

Expected: may pass before wrapper inversion because Task 2 temporary dispatch calls old functions. Continue to Step 3 to remove the temporary direction.

- [ ] **Step 3: Move ALS linear body behind `linear_solve`**

In `src/solvers/als.jl`, change the public function header:

```julia
function als_linsolve(A::AbstractTToperator, b::AbstractTTvector, tt_start::AbstractTTvector; sweep_count = 2, it_solver = false, r_itsolver = 5000, return_info = false)
```

to:

```julia
function _als_linsolve_impl(A::AbstractTToperator, b::AbstractTTvector, tt_start::AbstractTTvector; sweep_count = 2, it_solver = false, r_itsolver = 5000, return_info = false)
```

Do not change the function body.

In `src/solvers/linear_solver.jl`, replace the Task 2 temporary `linear_solve(..., alg::ALS)` method with:

```julia
function linear_solve(A, b, guess, alg::ALS)
    return _als_linsolve_impl(
        A, b, guess;
        sweep_count = alg.sweep_count,
        it_solver = alg.it_solver,
        r_itsolver = alg.r_itsolver,
        return_info = alg.return_info
    )
end

function als_linsolve(A, b, guess; kwargs...)
    return linear_solve(A, b, guess, ALS(; kwargs...))
end
```

- [ ] **Step 4: Move MALS linear body behind `linear_solve`**

In `src/solvers/mals.jl`, change:

```julia
function mals_linsolve(
        A::AbstractTToperator, b::AbstractTTvector,
        tt_start::AbstractTTvector;
        tol::Float64 = 1.0e-12,
        rmax::Int = round(Int, sqrt(prod(tt_start.ttv_dims)::Int)),
        return_info::Bool = false
    )
```

to:

```julia
function _mals_linsolve_impl(
        A::AbstractTToperator, b::AbstractTTvector,
        tt_start::AbstractTTvector;
        tol::Float64 = 1.0e-12,
        rmax::Int = round(Int, sqrt(prod(tt_start.ttv_dims)::Int)),
        return_info::Bool = false
    )
```

Do not change the function body.

In `src/solvers/linear_solver.jl`, replace the Task 2 temporary `linear_solve(..., alg::MALS)` method with:

```julia
function linear_solve(A, b, guess, alg::MALS)
    rmax = isnothing(alg.rmax) ? round(Int, sqrt(prod(guess.ttv_dims)::Int)) : alg.rmax
    return _mals_linsolve_impl(A, b, guess; tol = alg.tol, rmax = rmax, return_info = alg.return_info)
end

function mals_linsolve(A, b, guess; kwargs...)
    return linear_solve(A, b, guess, MALS(; kwargs...))
end
```

- [ ] **Step 5: Move DMRG linear body behind `linear_solve`**

In `src/solvers/dmrg.jl`, change:

```julia
function dmrg_linsolve(
        A::AbstractTToperator, b::AbstractTTvector, tt_start::AbstractTTvector; sweep_count = 2, N = 2, tol = 1.0e-12::Float64,
```

to:

```julia
function _dmrg_linsolve_impl(
        A::AbstractTToperator, b::AbstractTTvector, tt_start::AbstractTTvector; sweep_count = 2, N = 2, tol = 1.0e-12::Float64,
```

Do not change the function body.

In `src/solvers/linear_solver.jl`, replace the Task 2 temporary `linear_solve(..., alg::DMRG)` method with:

```julia
function linear_solve(A, b, guess, alg::DMRG)
    sweep_schedule = isnothing(alg.sweep_schedule) ? [2] : alg.sweep_schedule
    rmax_schedule = isnothing(alg.rmax_schedule) ? [isqrt(prod(guess.ttv_dims)::Int)] : alg.rmax_schedule
    return _dmrg_linsolve_impl(
        A, b, guess;
        sweep_count = alg.sweep_count,
        N = alg.N,
        tol = alg.tol,
        sweep_schedule = sweep_schedule,
        rmax_schedule = rmax_schedule,
        it_solver = alg.it_solver,
        linsolv_maxiter = alg.linsolv_maxiter,
        linsolv_tol = alg.linsolv_tol,
        itslv_thresh = alg.itslv_thresh,
        return_info = alg.return_info,
        verbose = alg.verbose
    )
end

function dmrg_linsolve(A, b, guess; kwargs...)
    return linear_solve(A, b, guess, DMRG(; kwargs...))
end
```

- [ ] **Step 6: Verify linear API and legacy tests**

Run:

```bash
julia --project=. -e 'using TensorTrainNumerics; include("test/test_solver_api.jl")'
julia --project=. -e 'using TensorTrainNumerics; include("test/test_als.jl"); include("test/test_mals.jl"); include("test/test_dmrg.jl")'
```

Expected: all listed tests pass.

- [ ] **Step 7: No-commit checkpoint**

Run:

```bash
git status --short
```

Expected: modified source/tests. Do not commit.

---

### Task 4: Add `eigen_solve` Front Door And Legacy Eigen Wrappers

**Files:**
- Create: `src/solvers/eigen_solver.jl`
- Modify: `src/TensorTrainNumerics.jl`
- Modify: `src/solvers/als.jl`
- Modify: `src/solvers/mals.jl`
- Modify: `src/solvers/dmrg.jl`
- Modify: `test/test_solver_api.jl`

**Interfaces:**
- Consumes: `ALS`, `MALS`, `DMRG`, and existing eigen implementations.
- Produces: `eigen_solve(A, guess; alg = MALS())`
- Produces: `eigen_solve(A, guess, alg::EigenSolverAlgorithm)`
- Produces: legacy `als_eigsolve`, `mals_eigsolve`, `dmrg_eigsolve` wrappers.

- [ ] **Step 1: Add failing `eigen_solve` tests**

Append to `test/test_solver_api.jl`:

```julia
@testset "eigen_solve front door" begin
    dims = (2, 2, 2)
    A = id_tto(3)
    guess = rand_tt(dims, [1, 2, 2, 1])

    E_als, x_als = eigen_solve(A, guess, ALS(sweep_schedule = [2], rmax_schedule = [2], noise_schedule = [0.0]))
    @test !isempty(E_als)
    @test x_als isa TensorTrainNumerics.AbstractTTvector
    @test abs(last(E_als) - 1.0) < 1.0e-8

    E_mals, x_mals, r_hist_mals = eigen_solve(A, guess, MALS(sweep_schedule = [2], rmax_schedule = [4]))
    @test !isempty(E_mals)
    @test !isempty(r_hist_mals)
    @test x_mals isa TensorTrainNumerics.AbstractTTvector

    E_dmrg, x_dmrg, r_hist_dmrg = eigen_solve(A, guess, DMRG(N = 2, sweep_schedule = [2], rmax_schedule = [4]))
    @test !isempty(E_dmrg)
    @test !isempty(r_hist_dmrg)
    @test x_dmrg isa TensorTrainNumerics.AbstractTTvector
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
julia --project=. -e 'using TensorTrainNumerics; include("test/test_solver_api.jl")'
```

Expected: failure with `UndefVarError: eigen_solve not defined`.

- [ ] **Step 3: Create `eigen_solver.jl` front door and wrappers**

Create `src/solvers/eigen_solver.jl`:

```julia
eigen_solve(A, guess; alg::EigenSolverAlgorithm = MALS()) = eigen_solve(A, guess, alg)

function als_eigsolve(A, guess; kwargs...)
    return eigen_solve(A, guess, ALS(; kwargs...))
end

function mals_eigsolve(A, guess; kwargs...)
    return eigen_solve(A, guess, MALS(; kwargs...))
end

function dmrg_eigsolve(A, guess; kwargs...)
    return eigen_solve(A, guess, DMRG(; kwargs...))
end
```

- [ ] **Step 4: Move ALS eigen body behind `eigen_solve`**

In `src/solvers/als.jl`, change:

```julia
function als_eigsolve(
        A::AbstractTToperator,
        tt_start::AbstractTTvector;
```

to:

```julia
function _als_eigsolve_impl(
        A::AbstractTToperator,
        tt_start::AbstractTTvector;
```

Do not change the function body.

Add this method near the implementation:

```julia
function eigen_solve(A::AbstractTToperator, guess::AbstractTTvector, alg::ALS)
    sweep_schedule = isnothing(alg.sweep_schedule) ? [2] : alg.sweep_schedule
    rmax_schedule = isnothing(alg.rmax_schedule) ? [maximum(guess.ttv_rks)] : alg.rmax_schedule
    noise_schedule = isnothing(alg.noise_schedule) ? zeros(length(rmax_schedule)) : alg.noise_schedule
    return _als_eigsolve_impl(
        A, guess;
        sweep_schedule = sweep_schedule,
        rmax_schedule = rmax_schedule,
        noise_schedule = noise_schedule,
        it_solver = alg.it_solver,
        itslv_thresh = alg.itslv_thresh,
        maxiter = alg.maxiter,
        linsolv_tol = alg.linsolv_tol
    )
end
```

- [ ] **Step 5: Move MALS eigen body behind `eigen_solve`**

In `src/solvers/mals.jl`, change:

```julia
function mals_eigsolve(
        A::AbstractTToperator,
        tt_start::AbstractTTvector;
```

to:

```julia
function _mals_eigsolve_impl(
        A::AbstractTToperator,
        tt_start::AbstractTTvector;
```

Do not change the function body.

Add:

```julia
function eigen_solve(A::AbstractTToperator, guess::AbstractTTvector, alg::MALS)
    sweep_schedule = isnothing(alg.sweep_schedule) ? [2] : alg.sweep_schedule
    rmax_schedule = isnothing(alg.rmax_schedule) ? [round(Int, sqrt(prod(guess.ttv_dims)::Int))] : alg.rmax_schedule
    return _mals_eigsolve_impl(
        A, guess;
        tol = alg.tol,
        sweep_schedule = sweep_schedule,
        rmax_schedule = rmax_schedule,
        it_solver = alg.it_solver,
        linsolv_maxiter = alg.linsolv_maxiter,
        linsolv_tol = alg.linsolv_tol,
        itslv_thresh = alg.itslv_thresh
    )
end
```

- [ ] **Step 6: Move DMRG eigen body behind `eigen_solve`**

In `src/solvers/dmrg.jl`, change:

```julia
function dmrg_eigsolve(
        A::AbstractTToperator,
        tt_start::AbstractTTvector;
```

to:

```julia
function _dmrg_eigsolve_impl(
        A::AbstractTToperator,
        tt_start::AbstractTTvector;
```

Do not change the function body.

Add:

```julia
function eigen_solve(A::AbstractTToperator, guess::AbstractTTvector, alg::DMRG)
    sweep_schedule = isnothing(alg.sweep_schedule) ? [2] : alg.sweep_schedule
    rmax_schedule = isnothing(alg.rmax_schedule) ? [isqrt(prod(guess.ttv_dims)::Int)] : alg.rmax_schedule
    return _dmrg_eigsolve_impl(
        A, guess;
        N = alg.N,
        tol = alg.tol,
        sweep_schedule = sweep_schedule,
        rmax_schedule = rmax_schedule,
        it_solver = alg.it_solver,
        linsolv_maxiter = alg.linsolv_maxiter,
        linsolv_tol = alg.linsolv_tol,
        itslv_thresh = alg.itslv_thresh,
        verbose = alg.verbose
    )
end
```

- [ ] **Step 7: Update module exports and include**

In `src/TensorTrainNumerics.jl`, add:

```julia
export eigen_solve
include("solvers/eigen_solver.jl")
```

after `include("solvers/dmrg.jl")` so the algorithm-specific `eigen_solve` methods are loaded before the wrappers are used.

- [ ] **Step 8: Verify eigen API and legacy eigen tests**

Run:

```bash
julia --project=. -e 'using TensorTrainNumerics; include("test/test_solver_api.jl")'
julia --project=. -e 'using TensorTrainNumerics; include("test/test_als.jl"); include("test/test_mals.jl"); include("test/test_dmrg.jl")'
```

Expected: all listed tests pass.

- [ ] **Step 9: No-commit checkpoint**

Run:

```bash
git status --short
```

Expected: modified source/tests. Do not commit.

---

### Task 5: Move Time Evolution Dispatch To `linear_solve`

**Files:**
- Rename: `src/solvers/euler.jl` to `src/solvers/time_evolution.jl`
- Modify: `src/solvers/linear_solver.jl`
- Modify: `src/solvers/time_evolution.jl`
- Modify: `src/TensorTrainNumerics.jl`
- Modify: `test/test_euler.jl`

**Interfaces:**
- Consumes: `linear_solve` and algorithm objects.
- Produces: implicit time steppers that accept `tt_solver::Union{AbstractString, LinearSolverAlgorithm}` and call `linear_solve`.

- [ ] **Step 1: Add time-stepper algorithm-object test**

In `test/test_euler.jl`, extend the existing solver dispatch test with:

```julia
sol_short = implicit_euler_method(A, u₀, guess, steps; tt_solver = ALS(sweep_count = 4), normalize = false)
@test norm(sol_short - sol_str) / max(norm(sol_str), eps()) < 1.0e-10
```

where `sol_str` is the existing string-based result in the same testset. If no reusable `sol_str` exists, create both in the same testset:

```julia
sol_str = implicit_euler_method(A, u₀, guess, steps; tt_solver = "als", normalize = false, sweep_count = 4)
sol_short = implicit_euler_method(A, u₀, guess, steps; tt_solver = ALS(sweep_count = 4), normalize = false)
@test norm(sol_short - sol_str) / max(norm(sol_str), eps()) < 1.0e-10
```

- [ ] **Step 2: Run the time-stepper test before implementation**

Run:

```bash
julia --project=. -e 'using TensorTrainNumerics; include("test/test_euler.jl")'
```

Expected: failure if the current stepper does not accept `ALS(...)` directly.

- [ ] **Step 3: Move Krylov linear solve into `linear_solver.jl`**

Move these definitions from `src/solvers/euler.jl` to `src/solvers/linear_solver.jl`:

```julia
_krylov_algorithm(...)
krylov_linsolve(...)
```

Then add:

```julia
function linear_solve(A, b, guess, alg::Krylov)
    return krylov_linsolve(
        A, b, guess;
        max_bond = alg.max_bond,
        krylov_solver = alg.krylov_solver,
        krylovdim = alg.krylovdim,
        maxiter = alg.maxiter,
        rtol = alg.rtol,
        atol = alg.atol,
        tol = alg.tol,
        orth = alg.orth,
        issymmetric = alg.issymmetric,
        ishermitian = alg.ishermitian,
        isposdef = alg.isposdef,
        verbosity = alg.verbosity
    )
end
```

- [ ] **Step 4: Add string solver normalization**

In `src/solvers/linear_solver.jl`, add:

```julia
_linear_solver_alg(alg::LinearSolverAlgorithm; kwargs...) = alg

function _linear_solver_alg(name::AbstractString; kwargs...)
    name == "als" && return ALS(; kwargs...)
    name == "mals" && return MALS(; kwargs...)
    name == "dmrg" && return DMRG(; kwargs...)
    name == "krylov" && return Krylov(; kwargs...)
    throw(ArgumentError("Unknown TT solver: $name. Use \"als\", \"mals\", \"dmrg\", \"krylov\", or a LinearSolverAlgorithm instance."))
end
```

Replace `_linear_solver_alg(alg::LinearSolverAlgorithm; kwargs...) = alg` with merger methods so legacy calls such as `tt_solver = ALSSolver(), sweep_count = 4` still work:

```julia
_linear_solver_alg(alg::LinearSolverAlgorithm; kwargs...) = isempty(kwargs) ? alg : _merge_solver_kwargs(alg; kwargs...)

function _merge_solver_kwargs(alg::ALS; kwargs...)
    opts = merge((
        sweep_count = alg.sweep_count,
        it_solver = alg.it_solver,
        r_itsolver = alg.r_itsolver,
        return_info = alg.return_info,
        sweep_schedule = alg.sweep_schedule,
        rmax_schedule = alg.rmax_schedule,
        noise_schedule = alg.noise_schedule,
        itslv_thresh = alg.itslv_thresh,
        maxiter = alg.maxiter,
        linsolv_tol = alg.linsolv_tol,
    ), (; kwargs...))
    return ALS(; opts...)
end

function _merge_solver_kwargs(alg::MALS; kwargs...)
    opts = merge((
        tol = alg.tol,
        rmax = alg.rmax,
        return_info = alg.return_info,
        sweep_schedule = alg.sweep_schedule,
        rmax_schedule = alg.rmax_schedule,
        it_solver = alg.it_solver,
        linsolv_maxiter = alg.linsolv_maxiter,
        linsolv_tol = alg.linsolv_tol,
        itslv_thresh = alg.itslv_thresh,
    ), (; kwargs...))
    return MALS(; opts...)
end

function _merge_solver_kwargs(alg::DMRG; kwargs...)
    opts = merge((
        sweep_count = alg.sweep_count,
        N = alg.N,
        tol = alg.tol,
        sweep_schedule = alg.sweep_schedule,
        rmax_schedule = alg.rmax_schedule,
        it_solver = alg.it_solver,
        linsolv_maxiter = alg.linsolv_maxiter,
        linsolv_tol = alg.linsolv_tol,
        itslv_thresh = alg.itslv_thresh,
        return_info = alg.return_info,
        verbose = alg.verbose,
    ), (; kwargs...))
    return DMRG(; opts...)
end

function _merge_solver_kwargs(alg::Krylov; kwargs...)
    opts = merge((
        max_bond = alg.max_bond,
        krylov_solver = alg.krylov_solver,
        krylovdim = alg.krylovdim,
        maxiter = alg.maxiter,
        rtol = alg.rtol,
        atol = alg.atol,
        tol = alg.tol,
        orth = alg.orth,
        issymmetric = alg.issymmetric,
        ishermitian = alg.ishermitian,
        isposdef = alg.isposdef,
        verbosity = alg.verbosity,
    ), (; kwargs...))
    return Krylov(; opts...)
end
```

- [ ] **Step 5: Rename the file and update include**

Rename `src/solvers/euler.jl` to `src/solvers/time_evolution.jl`.

In `src/TensorTrainNumerics.jl`, change:

```julia
include("solvers/euler.jl")
```

to:

```julia
include("solvers/time_evolution.jl")
```

- [ ] **Step 6: Update implicit time steppers**

In `src/solvers/time_evolution.jl`, change the `tt_solver` keyword type in `implicit_euler_method` and `crank_nicholson_method` from:

```julia
tt_solver::Union{AbstractString, TTLinearSolver} = MALSSolver(),
```

to:

```julia
tt_solver::Union{AbstractString, LinearSolverAlgorithm} = MALS(),
```

In both methods, replace:

```julia
solver = _tt_linsolver(tt_solver)
```

with:

```julia
solver = _linear_solver_alg(tt_solver; kwargs...)
```

Replace:

```julia
next = _tt_linsolve(solver, M, solution, guess; max_bond = max_bond, kwargs...)::AbstractTTvector
```

with:

```julia
next = linear_solve(M, solution, guess, _with_time_stepper_max_bond(solver, max_bond))::AbstractTTvector
```

Add this helper in `linear_solver.jl`:

```julia
_with_time_stepper_max_bond(alg::LinearSolverAlgorithm, max_bond::Int) = alg

function _with_time_stepper_max_bond(alg::Krylov, max_bond::Int)
    max_bond <= 0 && return alg
    return Krylov(
        max_bond = max_bond,
        krylov_solver = alg.krylov_solver,
        krylovdim = alg.krylovdim,
        maxiter = alg.maxiter,
        rtol = alg.rtol,
        atol = alg.atol,
        tol = alg.tol,
        orth = alg.orth,
        issymmetric = alg.issymmetric,
        ishermitian = alg.ishermitian,
        isposdef = alg.isposdef,
        verbosity = alg.verbosity
    )
end
```

For `crank_nicholson_method`, use the same replacement with `LHS` and `RHS`:

```julia
next = linear_solve(LHS, RHS, guess, _with_time_stepper_max_bond(solver, max_bond))::AbstractTTvector
```

- [ ] **Step 7: Remove old private dispatch**

Ensure `src/solvers/time_evolution.jl` no longer defines:

```julia
_tt_linsolver
_tt_linsolve
TTLinearSolver
ALSSolver
MALSSolver
DMRGSolver
KrylovSolver
KRYLOV_ROUND_RANK
_krylov_algorithm
krylov_linsolve
```

Only time-evolution methods should remain.

- [ ] **Step 8: Verify time evolution tests**

Run:

```bash
julia --project=. -e 'using TensorTrainNumerics; include("test/test_euler.jl")'
```

Expected: all Euler/time-stepper tests pass.

- [ ] **Step 9: No-commit checkpoint**

Run:

```bash
git status --short
```

Expected: file rename, modified source/tests. Do not commit.

---

### Task 6: Export Cleanup, Docs Update, And Full Verification

**Files:**
- Modify: `src/TensorTrainNumerics.jl`
- Modify: docs files found by `rg -n "DMRG\\(|ALSSolver|MALSSolver|DMRGSolver|KrylovSolver|als_linsolve|mals_linsolve|dmrg_linsolve" docs/src`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: all new API files.
- Produces: final public exports and full-suite verification.

- [ ] **Step 1: Confirm exports are coherent**

In `src/TensorTrainNumerics.jl`, ensure the solver exports include:

```julia
export LinearSolverAlgorithm, EigenSolverAlgorithm
export ALS, MALS, DMRG, Krylov
export TTLinearSolver, ALSSolver, MALSSolver, DMRGSolver, KrylovSolver
export linear_solve, eigen_solve
export als_linsolve, als_eigsolve, als_gen_eigsolv
export mals_linsolve, mals_eigsolve
export dmrg_linsolve, dmrg_eigsolve
export euler_method, implicit_euler_method, crank_nicholson_method, rk4_method
export tt_cross, tt_integrate, MaxVol, DMRGcross, Greedy
```

- [ ] **Step 2: Confirm test runner includes new tests**

In `test/runtests.jl`, ensure:

```julia
include("test_solver_api.jl")
```

is present after the core solver files are available and before tests that depend on the new API.

- [ ] **Step 3: Update docs examples**

Run:

```bash
rg -n "DMRG\\(|ALSSolver|MALSSolver|DMRGSolver|KrylovSolver|tt_solver = \"|als_linsolve|mals_linsolve|dmrg_linsolve" docs/src
```

Apply these rules:

- Cross interpolation examples: `DMRG(...)` becomes `DMRGcross(...)`.
- New or edited time-stepper examples should prefer `tt_solver = MALS(...)` over `tt_solver = "mals"`.
- Existing legacy solver-function docs may mention `als_linsolve`, `mals_linsolve`, and `dmrg_linsolve` as compatibility wrappers.

- [ ] **Step 4: Run focused tests**

Run:

```bash
julia --project=. -e 'using TensorTrainNumerics; include("test/test_solver_api.jl")'
julia --project=. -e 'using TensorTrainNumerics; include("test/test_tt_cross_interpolation.jl")'
julia --project=. -e 'using TensorTrainNumerics; include("test/test_euler.jl")'
julia --project=. -e 'using TensorTrainNumerics; include("test/test_als.jl"); include("test/test_mals.jl"); include("test/test_dmrg.jl")'
```

Expected: all focused tests pass.

- [ ] **Step 5: Run full package tests**

Run:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: `Testing TensorTrainNumerics tests passed`.

- [ ] **Step 6: Inspect final diff**

Run:

```bash
git status --short
git diff --stat
```

Expected: source, tests, docs, and plan/spec changes only. Do not commit.
