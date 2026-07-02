# Global Space-Time QTT Crank-Nicholson Solver Design

## Purpose

Add a global block Crank-Nicholson solver that computes the full trajectory in one compressed space-time TT/QTT object instead of advancing one time step at a time. The first implementation targets the homogeneous autonomous equation already used by the package:

```julia
u_t = A * u
```

The returned object represents the unknown time slices `[u1, u2, ..., uNt]`, so users can extract any intermediate QTT state without forming a dense space-time array.

## Sign Convention

The literature notes use

```math
y_t = -A y + f(t)
```

with `A_+ = I + tau/2 A` and `A_- = I - tau/2 A`. The existing package API uses the operator directly in

```julia
u_t = A * u
```

and the existing `crank_nicholson_method` applies

```math
u_{k+1} =
\left(I-\frac{\tau}{2}A\right)^{-1}
\left(I+\frac{\tau}{2}A\right)u_k .
```

The global solver will follow the package convention. For a constant step `tau`, define

```math
L = I - \frac{\tau}{2}A,
\qquad
R = I + \frac{\tau}{2}A .
```

The global block system is

```math
\mathcal{A} U = b,
\qquad
\mathcal{A} = I_t \otimes L - S_t \otimes R,
\qquad
b = e_1 \otimes (R u_0),
```

where `S_t` is the strict lower shift on the unknown time slices. Block row `k` enforces `L*u_k - R*u_{k-1} = 0`, with `u0` moved to the first right-hand side block.

## Public API

Add:

```julia
U = global_crank_nicholson_method(A, u0, guess, steps; kwargs...)
u_k = space_time_slice(U, k)
```

The first implementation targets QTT inputs: `A::QTToperator`, `u0::QTTvector`, and a spatial `guess::QTTvector`. Internally the solver may strip metadata and use the existing TT linear solvers, but the public global method returns a space-time QTT wrapper with explicit time metadata.

The first version requires:

- `steps` has at least two entries so the time index has at least one QTT bit.
- all entries of `steps` are equal to a common `tau`;
- `length(steps)` is a power of two;
- no forcing term is supplied;
- no per-step normalization is applied inside the block solve.

Keyword behavior should mirror the existing time steppers where practical:

- `tt_solver = "mals"` by default;
- `"als"`, `"dmrg"`, and `"krylov"` route to the existing linear solvers;
- `max_bond` is passed to the Krylov rank control path and may be used for optional post-solve compression;
- `return_error = false` returns only `U`;
- `return_error = true` returns `(U, residual)` with `residual = norm(Aglobal * U - b) / norm(b)`.

## Result Type

Add a lightweight wrapper:

```julia
struct SpaceTimeQTTvector{T,M} <: AbstractTTvector
    N::Int64
    ttv_vec::Vector{Array{T,3}}
    ttv_dims::NTuple{M,Int64}
    ttv_rks::Vector{Int64}
    ttv_ot::Vector{Int64}
    time_bits::Int
    space_n_dims::Int
    space_bits_per_dim::Int
    space_ordering::Symbol
end
```

This wrapper preserves the normal TT fields so existing norm, arithmetic, and solver internals can work through `TTvector(U)`. It records enough metadata for exact time slicing back to spatial QTT states. Time cores are placed first, followed by spatial cores.

Plain TT inputs are outside the first implementation because slicing a plain space-time TT would need separate metadata for the number of time cores and the spatial shape. That can be added later with a separate `SpaceTimeTTvector` or a more general result wrapper.

## Time Operators

Add constructors for the time direction:

```julia
qtt_lower_shift(time_bits)
qtt_time_identity(time_bits)
```

`qtt_lower_shift(time_bits)` represents the strict lower shift matrix of size `2^time_bits` in QTT/MPO form. It should be built using the existing Toeplitz QTT constructor where possible:

```julia
S_t = toeplitz_to_qtto(0, 0, 1, time_bits)
```

The exact orientation must be verified against dense matrix tests. The chosen convention must satisfy `S_t[k, k-1] = 1` for `k = 2, ..., Nt`.

## Space-Time RHS And Guess

The RHS for the homogeneous problem is rank-separable:

```julia
b = qtt_basis_vector(time_bits, 1) ⊗ (R * u0)
```

The solver accepts either:

- a spatial `QTTvector` guess, which is expanded as a rank-one time vector of ones over all unknown time slices tensor-producted with the spatial guess; or
- a full `SpaceTimeQTTvector` guess with `time_bits + A.N` cores.

For the first implementation, the spatial guess path is the main supported path because it matches the requested syntax:

```julia
global_crank_nicholson_method(A, u0, guess, steps; ...)
```

## Slicing

`space_time_slice(U, k)` extracts the `k`-th unknown time slice, where `k = 1` corresponds to the state after the first Crank-Nicholson step. It contracts the time cores with the QTT basis vector for `k`, leaving only spatial cores.

The return value is a `QTTvector` with the original spatial metadata.

Index validation:

- `1 <= k <= 2^time_bits`;
- extracted states do not include `u0`.

## Testing

Add tests to `test/test_euler.jl` or a new focused test file included from `test/runtests.jl`.

Required tests:

- dense orientation test for `qtt_lower_shift`;
- global Crank-Nicholson slices match sequential `crank_nicholson_method` on a small 1D QTT operator;
- final slice matches the final sequential result;
- `return_error = true` gives a finite small global residual;
- `SpaceTimeQTTvector` preserves `time_bits`, spatial dimension metadata, and spatial ordering;
- plain TT inputs throw a clear `MethodError` or `ArgumentError` in the first implementation;
- validation rejects nonconstant steps and non-power-of-two step counts.

## Future Extensions

The design leaves room for:

- forcing terms through a space-time RHS `F(t,x)`;
- terminal-condition/backward-time variants for Kolmogorov and Feynman-Kac equations;
- helper conversion of small space-time results to dense arrays for diagnostics;
- interleaved time-space ordering if it proves rank-beneficial.

These are intentionally outside the first implementation so the homogeneous global block path can be verified against the existing sequential solver.
