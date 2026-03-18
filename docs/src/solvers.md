# Solver Guide

TensorTrainNumerics.jl provides three families of iterative solvers for linear systems
`Ax = b` and eigenvalue problems `Ax = λx` in tensor-train format: **ALS**, **MALS**,
and **DMRG**. All three sweep back and forth over the TT cores, updating one (or more)
cores at a time by solving a small dense local problem. They differ in how they handle
bond dimensions and local subproblems.

---

## Overview

| Property | ALS | MALS | DMRG |
|---|---|---|---|
| Bond dimensions | Fixed | Adaptive (SVD) | Adaptive (SVD) |
| Local problem size | `r²·n` | `r²·n²` | `r²·n^N` |
| Memory per sweep | Low | Moderate | Moderate–high |
| Convergence speed | Moderate | Often faster | Often fastest |
| Recommended for | Fixed-rank refinement | General use | Large problems, eigsolve |
| Functions | `als_linsolve`, `als_eigsolve`, `als_gen_eigsolv` | `mals_linsolve`, `mals_eigsolve` | `dmrg_linsolve`, `dmrg_eigsolve` |
