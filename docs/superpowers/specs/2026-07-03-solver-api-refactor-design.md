# Solver API Refactor Design

## Purpose

Refactor the solver API so linear, eigenvalue, and future nonlinear TT solvers use
algorithm/configuration objects in the same style as `tt_cross`. The new API should
make solver choice explicit, keep solver options bundled with the chosen algorithm,
and remove the private linear-solver dispatch currently embedded in the time
evolution code.

## Goals

- Add public `linear_solve` and `eigen_solve` front doors.
- Use short algorithm names for solver configuration objects: `ALS`, `MALS`,
  `DMRG`, and `Krylov`.
- Store solver-specific options on algorithm objects, e.g. `MALS(tol = 1.0e-12)`.
- Keep existing solver functions such as `als_linsolve`, `mals_linsolve`,
  `dmrg_linsolve`, `als_eigsolve`, `mals_eigsolve`, and `dmrg_eigsolve` as wrappers
  around the new front doors.
- Avoid duplicated sweep and local-kernel code between linear and eigenvalue
  solvers.
- Prepare the layout for a future nonlinear solver API without implementing it now.

## Non-Goals

- Do not change the mathematical algorithms in this refactor.
- Do not rewrite ALS, MALS, DMRG, TDVP, or time-stepping internals beyond what is
  needed to route through the new public dispatch layer.
- Do not make `eigen_solve` call `linear_solve`; eigen solvers should share algorithm
  objects and common kernels, but their local microsteps solve eigenproblems rather
  than linear systems.

## Public API

The new linear API should be:

```julia
linear_solve(A, b, guess; alg = MALS())
linear_solve(A, b, guess, alg::LinearSolverAlgorithm)
```

The new eigenvalue API should be:

```julia
eigen_solve(A, guess; alg = MALS())
eigen_solve(A, guess, alg::EigenSolverAlgorithm)
```

Solver algorithm objects should be constructed with keyword options:

```julia
ALS(sweep_count = 2)
MALS(tol = 1.0e-12, rmax = nothing)
DMRG(N = 2, tol = 1.0e-12, sweep_schedule = [2], rmax_schedule = nothing)
Krylov(max_bond = 0, krylov_solver = :auto)
```

The exact field lists should mirror the existing keyword arguments of the legacy
functions. Values that currently depend on the problem, such as default `rmax`
schedules derived from `tt_start.ttv_dims`, should be represented as `nothing` on
the algorithm object and resolved inside `linear_solve` or `eigen_solve`.

Legacy wrappers should delegate into the new API:

```julia
als_linsolve(A, b, guess; kwargs...) = linear_solve(A, b, guess, ALS(; kwargs...))
mals_linsolve(A, b, guess; kwargs...) = linear_solve(A, b, guess, MALS(; kwargs...))
dmrg_linsolve(A, b, guess; kwargs...) = linear_solve(A, b, guess, DMRG(; kwargs...))

als_eigsolve(A, guess; kwargs...) = eigen_solve(A, guess, ALS(; kwargs...))
mals_eigsolve(A, guess; kwargs...) = eigen_solve(A, guess, MALS(; kwargs...))
dmrg_eigsolve(A, guess; kwargs...) = eigen_solve(A, guess, DMRG(; kwargs...))
```

## Cross-Interpolation Rename

The existing cross-interpolation algorithm named `DMRG` conflicts with the desired
short solver name. This refactor intentionally makes a breaking rename:

```julia
DMRGcross(...)
tt_cross(f, domain, DMRGcross(...))
```

`DMRG` becomes the solver algorithm object used by `linear_solve` and `eigen_solve`.
Tests and documentation that currently call `tt_cross(..., DMRG(...))` should be
updated to `DMRGcross(...)`.

## File Layout

Use the following solver layout:

```text
src/solvers/
  linear_solver.jl
  eigen_solver.jl
  time_evolution.jl
  tdvp.jl
  als.jl
  mals.jl
  dmrg.jl
```

`linear_solver.jl` owns:

- `LinearSolverAlgorithm`
- solver algorithm constructors used for linear solves
- `linear_solve`
- legacy `*_linsolve` wrappers
- the Krylov TT linear solver currently located in `euler.jl`

`eigen_solver.jl` owns:

- `EigenSolverAlgorithm`
- `eigen_solve`
- legacy `*_eigsolve` wrappers
- shared eigen-solver API routing

`time_evolution.jl` owns:

- `euler_method`
- `implicit_euler_method`
- `crank_nicholson_method`
- `rk4_method`

Time evolution should call `linear_solve` with an algorithm object instead of using
private `_tt_linsolve` dispatch.

`tdvp.jl` can remain separate for now because it is already a substantial, specialized
time-evolution implementation.

`als.jl`, `mals.jl`, and `dmrg.jl` should keep the algorithm-specific local kernels,
environment updates, core moves, and sweep implementation details. API files should
route to these internals rather than duplicate them.

## Reuse Boundary

Linear and eigen solvers should share:

- algorithm objects and option resolution
- environment construction helpers where the existing code permits it
- rank schedule handling
- core movement and truncation helpers

They should not share the final local microstep when the mathematical operation
differs:

- linear solvers solve local systems such as `K*x = Pb`
- eigen solvers solve local eigenproblems such as `K*x = lambda*x`

This keeps the public API coherent while avoiding a false abstraction that would
obscure different numerical operations.

## Compatibility

The refactor keeps legacy linear and eigen solver function names as wrappers.
The breaking API change is limited to the cross-interpolation algorithm rename from
`DMRG` to `DMRGcross`.

String-based time-stepper solver selection can remain temporarily for compatibility:

```julia
tt_solver = "als"
tt_solver = "mals"
tt_solver = "dmrg"
tt_solver = "krylov"
```

Internally, strings should be converted to algorithm objects. New examples should use
algorithm objects directly.

## Testing Strategy

- Add direct tests for `linear_solve(A, b, guess, ALS(...))`,
  `linear_solve(A, b, guess, MALS(...))`, and `linear_solve(A, b, guess, DMRG(...))`.
- Add direct tests for `eigen_solve(A, guess, ALS(...))`,
  `eigen_solve(A, guess, MALS(...))`, and `eigen_solve(A, guess, DMRG(...))`.
- Keep existing legacy wrapper tests and verify they match the new front-door results
  on small systems.
- Update time-evolution tests to cover algorithm-object solver selection.
- Update cross-interpolation tests and docs from `DMRG(...)` to `DMRGcross(...)`.
- Run the full package test suite after the refactor.
