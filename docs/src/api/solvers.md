# Solvers and algorithms

```@meta
CurrentModule = TensorTrainNumerics
```

Linear, eigenvalue, and nonlinear solvers, time integrators, and cross interpolation.

## Linear and eigenvalue solvers

```@docs
linear_solve
eigen_solve
LinearSolverAlgorithm
EigenSolverAlgorithm
ALS
MALS
DMRG
Krylov
als_linsolve
als_eigsolve
als_gen_eigsolv
mals_linsolve
mals_eigsolve
dmrg_linsolve
dmrg_eigsolve
TTLinearSolver
ALSSolver
MALSSolver
DMRGSolver
KrylovSolver
```

## Nonlinear solver

```@docs
non_linear_solve
NonLinearSolverAlgorithm
PenaltyALS
MGR
gpe_energy
```

## Time evolution

```@docs
euler_method
implicit_euler_method
crank_nicholson_method
rk4_method
tdvp
tdvp2
```

## Cross interpolation

```@docs
tt_cross
tt_integrate
MaxVol
DMRGcross
Greedy
```
