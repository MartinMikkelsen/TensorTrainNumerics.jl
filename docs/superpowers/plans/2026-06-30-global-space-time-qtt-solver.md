# Global Space-Time QTT Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a homogeneous global Crank-Nicholson solver that returns the full QTT space-time trajectory `[u1, ..., uNt]` and supports extracting individual time slices.

**Architecture:** The solver constructs the block MPO `I_t ⊗ L - S_t ⊗ R`, with time QTT cores first and spatial QTT cores second. It solves the global TT linear system with the existing ALS/MALS/DMRG/Krylov solvers, wraps the result in `SpaceTimeQTTvector`, and slices by contracting the time cores with a QTT basis vector.

**Tech Stack:** Julia 1.10, TensorTrainNumerics TT/QTT structs, existing `toeplitz_to_qtto`, `id_tto`, `⊗`, `als_linsolve`, `mals_linsolve`, `dmrg_linsolve`, `krylov_linsolve`, and `Test`.

## Global Constraints

- Do not make git commits; the repository owner commits changes.
- First implementation targets `A::QTToperator`, `u0::QTTvector`, and `guess::QTTvector` or `guess::SpaceTimeQTTvector`.
- `steps` must have at least two equal entries and have power-of-two length so the time index has at least one QTT bit.
- The package sign convention is `u_t = A*u`; use `L = I - tau/2*A` and `R = I + tau/2*A`.
- The returned space-time unknown stores `[u1, ..., uNt]`; `u0` is not included.
- Time cores are first, followed by spatial QTT cores.
- No forcing term and no per-step normalization in the first implementation.

---

## File Structure

- `src/qtt_tools.jl`: define `SpaceTimeQTTvector`, conversion to `TTvector`, display, copy, and `space_time_slice`.
- `src/tt_operators.jl`: define `qtt_time_identity` and `qtt_lower_shift`.
- `src/solvers/euler.jl`: define validation helpers, guess expansion, and `global_crank_nicholson_method`.
- `src/TensorTrainNumerics.jl`: export new public API.
- `test/test_global_crank_nicholson.jl`: targeted tests for time operators, slicing, validation, and global CN accuracy.
- `test/runtests.jl`: include the new test file.

---

### Task 1: Add Failing Tests For The Global Solver Surface

**Files:**
- Create: `test/test_global_crank_nicholson.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Consumes: intended public functions `qtt_lower_shift`, `qtt_time_identity`, `global_crank_nicholson_method`, `space_time_slice`, and type `SpaceTimeQTTvector`.
- Produces: executable tests that fail before implementation and pass after Tasks 2-4.

- [ ] **Step 1: Create the test file**

Add `test/test_global_crank_nicholson.jl`:

```julia
using Test
using TensorTrainNumerics
using LinearAlgebra

@testset "QTT time direction operators" begin
    d = 3
    S = qtt_lower_shift(d)
    Iₜ = qtt_time_identity(d)

    S_dense = qtto_to_matrix(S)
    I_dense = qtto_to_matrix(Iₜ)
    expected = zeros(2^d, 2^d)
    for k in 2:(2^d)
        expected[k, k - 1] = 1.0
    end

    @test S_dense == expected
    @test I_dense == Matrix{Float64}(I, 2^d, 2^d)
end

@testset "SpaceTimeQTTvector slicing" begin
    time_bits = 2
    d = 3
    u1 = qtt_sin(d)
    u2 = 2.0 * qtt_sin(d)
    u3 = 3.0 * qtt_sin(d)
    u4 = 4.0 * qtt_sin(d)

    tensor = zeros(ntuple(_ -> 2, time_bits + d))
    slices = [qtt_to_vector(u1), qtt_to_vector(u2), qtt_to_vector(u3), qtt_to_vector(u4)]
    for k in 1:4
        bits = reverse(digits(k - 1, base = 2, pad = time_bits)) .+ 1
        for x in CartesianIndices(ntuple(_ -> 2, d))
            spatial_bits = Tuple(x)
            spatial_index = TensorTrainNumerics.tuple_to_index(spatial_bits)
            tensor[CartesianIndex((bits..., spatial_bits...))] = slices[k][spatial_index]
        end
    end

    st = SpaceTimeQTTvector(ttv_decomp(tensor), time_bits, 1, d, :serial)

    @test st.time_bits == time_bits
    @test st.space_n_dims == 1
    @test st.space_bits_per_dim == d
    @test st.space_ordering == :serial

    for k in 1:4
        uk = space_time_slice(st, k)
        @test uk isa QTTvector
        @test uk.n_dims == 1
        @test uk.bits_per_dim == d
        @test uk.ordering == :serial
        @test norm(qtt_to_vector(uk) - slices[k]) / norm(slices[k]) < 1.0e-10
    end

    @test_throws BoundsError space_time_slice(st, 0)
    @test_throws BoundsError space_time_slice(st, 5)
end

@testset "global_crank_nicholson_method matches sequential CN" begin
    d = 3
    A_raw = -0.1 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    A = QTToperator(A_raw, 1, d, :serial)

    u0_raw = qtt_sin(d)
    u0 = QTTvector(u0_raw, 1, d, :serial)
    guess = u0
    steps = fill(0.02, 4)

    U, residual = global_crank_nicholson_method(
        A, u0, guess, steps;
        normalize = false,
        tt_solver = "krylov",
        tol = 1.0e-12,
        return_error = true
    )

    @test U isa SpaceTimeQTTvector
    @test U.time_bits == 2
    @test U.space_n_dims == 1
    @test U.space_bits_per_dim == d
    @test U.space_ordering == :serial
    @test isfinite(residual)
    @test residual < 1.0e-8

    sequential = u0
    for k in eachindex(steps)
        sequential = crank_nicholson_method(
            A, sequential, sequential, [steps[k]];
            normalize = false,
            tt_solver = "krylov",
            tol = 1.0e-12
        )
        global_slice = space_time_slice(U, k)
        rel_error = norm(qtt_to_vector(global_slice) - qtt_to_vector(sequential)) / norm(qtt_to_vector(sequential))
        @test rel_error < 1.0e-8
    end
end

@testset "global_crank_nicholson_method validation" begin
    d = 3
    A = QTToperator(-0.1 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d), 1, d, :serial)
    u0 = QTTvector(qtt_sin(d), 1, d, :serial)

    @test_throws ArgumentError global_crank_nicholson_method(A, u0, u0, Float64[]; normalize = false)
    @test_throws ArgumentError global_crank_nicholson_method(A, u0, u0, [0.1, 0.2]; normalize = false)
    @test_throws ArgumentError global_crank_nicholson_method(A, u0, u0, fill(0.1, 3); normalize = false)
end
```

- [ ] **Step 2: Include the new tests**

Modify `test/runtests.jl` by adding the include immediately after `include("test_euler.jl")`:

```julia
include("test_global_crank_nicholson.jl")
```

- [ ] **Step 3: Run the focused tests and verify they fail for missing API**

Run:

```bash
/Users/pzb464/.juliaup/bin/julialauncher --project=. test/test_global_crank_nicholson.jl
```

Expected: FAIL with `UndefVarError` for `qtt_lower_shift`, `SpaceTimeQTTvector`, or `global_crank_nicholson_method`.

---

### Task 2: Add SpaceTimeQTTvector And Time Slicing

**Files:**
- Modify: `src/qtt_tools.jl`
- Modify: `src/TensorTrainNumerics.jl`
- Test: `test/test_global_crank_nicholson.jl`

**Interfaces:**
- Consumes: existing `TTvector`, `QTTvector`, `qtt_basis_vector`, `ttv_decomp`, and `tuple_to_index`.
- Produces:
  - `SpaceTimeQTTvector(ttv::TTvector, time_bits::Int, space_n_dims::Int, space_bits_per_dim::Int, space_ordering::Symbol)`
  - `TTvector(st::SpaceTimeQTTvector)`
  - `space_time_slice(st::SpaceTimeQTTvector, k::Int)::QTTvector`

- [ ] **Step 1: Add the result wrapper and conversion helpers**

In `src/qtt_tools.jl`, add this block after the `QTToperator` definition and before `Base.eltype(::QTTvector...)`:

```julia
"""
A QTT vector over time and space with time cores first.

The represented unknown is `[u1, ..., uNt]`; the known initial state `u0` is not
stored in this tensor.
"""
struct SpaceTimeQTTvector{T <: Number, M} <: AbstractTTvector
    N::Int64
    ttv_vec::Vector{Array{T, 3}}
    ttv_dims::NTuple{M, Int64}
    ttv_rks::Vector{Int64}
    ttv_ot::Vector{Int64}
    time_bits::Int
    space_n_dims::Int
    space_bits_per_dim::Int
    space_ordering::Symbol
end
```

Add these methods near the existing QTT `Base.eltype` and conversion helpers:

```julia
Base.eltype(::SpaceTimeQTTvector{T, M}) where {T, M} = T

function SpaceTimeQTTvector(
        ttv::TTvector{T, M},
        time_bits::Int,
        space_n_dims::Int,
        space_bits_per_dim::Int,
        space_ordering::Symbol
    ) where {T, M}
    @assert time_bits ≥ 1 "time_bits must be at least 1"
    @assert space_n_dims ≥ 1 "space_n_dims must be at least 1"
    @assert space_bits_per_dim ≥ 1 "space_bits_per_dim must be at least 1"
    @assert space_ordering ∈ (:interleaved, :serial) "space_ordering must be :interleaved or :serial"
    @assert time_bits + space_n_dims * space_bits_per_dim == ttv.N "time_bits + space_n_dims * space_bits_per_dim must equal the number of TT cores"
    @assert all(==(2), ttv.ttv_dims) "All physical dimensions must be 2 for a space-time QTT vector"
    return SpaceTimeQTTvector{T, M}(
        ttv.N,
        ttv.ttv_vec,
        ttv.ttv_dims,
        ttv.ttv_rks,
        ttv.ttv_ot,
        time_bits,
        space_n_dims,
        space_bits_per_dim,
        space_ordering
    )
end

TTvector(st::SpaceTimeQTTvector{T, M}) where {T, M} =
    TTvector{T, M}(st.N, st.ttv_vec, st.ttv_dims, st.ttv_rks, st.ttv_ot)

function Base.show(io::IO, st::SpaceTimeQTTvector{T, M}) where {T, M}
    return print(io, "SpaceTimeQTT{$T}($(st.time_bits) time bits, $(st.space_n_dims)d×$(st.space_bits_per_dim) space bits, $(st.space_ordering))")
end

function Base.show(io::IO, ::MIME"text/plain", st::SpaceTimeQTTvector{T, M}) where {T, M}
    println(io, "SpaceTimeQTT{$T} with $(st.N) sites")
    println(io, "  Time bits     : $(st.time_bits)")
    println(io, "  Space         : $(st.space_n_dims)d × $(st.space_bits_per_dim) bits/dim")
    println(io, "  Ordering      : $(st.space_ordering)")
    println(io, "  Physical dims : $(st.ttv_dims)")
    println(io, "  Bond dims     : $(st.ttv_rks)")
    return print(io, "  Orthogonality : $(_ot_description(st.ttv_ot))")
end
```

- [ ] **Step 2: Add `space_time_slice`**

In `src/qtt_tools.jl`, after `TTvector(q::QTTvector...)`, add:

```julia
function _contract_time_prefix(st::SpaceTimeQTTvector{T}, k::Int) where {T}
    Nt = 2^st.time_bits
    1 ≤ k ≤ Nt || throw(BoundsError(st, k))
    bits = reverse(digits(k - 1, base = 2, pad = st.time_bits))

    left = ones(T, 1, 1)
    @inbounds for site in 1:st.time_bits
        selected = @view st.ttv_vec[site][bits[site] + 1, :, :]
        left = left * selected
    end
    return left
end

"""
    space_time_slice(st, k)

Extract the `k`-th unknown time slice from a `SpaceTimeQTTvector`.
`k = 1` corresponds to the first state after the initial condition.
"""
function space_time_slice(st::SpaceTimeQTTvector{T}, k::Int) where {T}
    left = _contract_time_prefix(st, k)
    first_space_site = st.time_bits + 1
    space_cores = deepcopy(st.ttv_vec[first_space_site:end])
    first_core = space_cores[1]
    new_first = zeros(T, size(first_core, 1), 1, size(first_core, 3))

    @inbounds for s in 1:size(first_core, 1), a in 1:size(first_core, 2), r in 1:size(first_core, 3)
        new_first[s, 1, r] += left[1, a] * first_core[s, a, r]
    end
    space_cores[1] = new_first

    space_dims = ntuple(_ -> 2, length(space_cores))
    space_rks = ones(Int64, length(space_cores) + 1)
    @inbounds for site in 1:length(space_cores)
        space_rks[site + 1] = size(space_cores[site], 3)
    end
    space_ot = zeros(Int64, length(space_cores))
    ttv = TTvector{T, length(space_cores)}(
        length(space_cores),
        space_cores,
        space_dims,
        space_rks,
        space_ot
    )
    return QTTvector(ttv, st.space_n_dims, st.space_bits_per_dim, st.space_ordering)
end
```

- [ ] **Step 3: Export the wrapper and slicer**

In `src/TensorTrainNumerics.jl`, extend the QTT export line:

```julia
export gauss_chebyshev_lobatto
export index_to_point, tuple_to_index, function_to_tensor, tensor_to_grid, function_to_qtt, qtt_to_function, qtt_to_vector, function_to_qtt_uniform, qtt_polynom, qtt_cos, qtt_sin, qtt_exp, qtto_to_matrix, qtt_basis_vector, qtt_chebyshev, qtt_trapezoidal, to_qtt, to_ttv, QTTvector, QTToperator, SpaceTimeQTTvector, space_time_slice, check_compat, function_to_qttv, qttv_to_array, reorder
```

- [ ] **Step 4: Run the slicing testset**

Run:

```bash
/Users/pzb464/.juliaup/bin/julialauncher --project=. -e 'using Test; include("test/test_global_crank_nicholson.jl")'
```

Expected: the `SpaceTimeQTTvector slicing` testset passes after time operators from Task 3 are added; before Task 3 the file still fails earlier on `qtt_lower_shift`.

---

### Task 3: Add QTT Time Operators

**Files:**
- Modify: `src/tt_operators.jl`
- Modify: `src/TensorTrainNumerics.jl`
- Test: `test/test_global_crank_nicholson.jl`

**Interfaces:**
- Consumes: existing `id_tto`, `toeplitz_to_qtto`, and `TToperator`.
- Produces:
  - `qtt_time_identity(time_bits::Int)::TToperator`
  - `qtt_time_identity(::Type{T}, time_bits::Int)::TToperator{T}`
  - `qtt_lower_shift(time_bits::Int)::TToperator`
  - `qtt_lower_shift(::Type{T}, time_bits::Int)::TToperator{T}`

- [ ] **Step 1: Add conversion helper and time operators**

In `src/tt_operators.jl`, after `shift(d::Int)`, add:

```julia
function _convert_tto_eltype(::Type{T}, A::TToperator{S, N}) where {T <: Number, S <: Number, N}
    return TToperator{T, N}(
        A.N,
        [convert(Array{T, 4}, core) for core in A.tto_vec],
        A.tto_dims,
        copy(A.tto_rks),
        copy(A.tto_ot)
    )
end

"""
    qtt_time_identity(time_bits)

Identity operator on the quantized time index with `2^time_bits` entries.
"""
qtt_time_identity(time_bits::Int) = qtt_time_identity(Float64, time_bits)

function qtt_time_identity(::Type{T}, time_bits::Int) where {T <: Number}
    time_bits ≥ 1 || throw(ArgumentError("time_bits must be at least 1"))
    return id_tto(T, time_bits)
end

"""
    qtt_lower_shift(time_bits)

Strict lower shift on the quantized time index, with dense entries
`S[k, k - 1] = 1` for `k = 2, ..., 2^time_bits`.
"""
qtt_lower_shift(time_bits::Int) = qtt_lower_shift(Float64, time_bits)

function qtt_lower_shift(::Type{T}, time_bits::Int) where {T <: Number}
    time_bits ≥ 1 || throw(ArgumentError("time_bits must be at least 1"))
    return _convert_tto_eltype(T, toeplitz_to_qtto(0.0, 0.0, 1.0, time_bits))
end
```

- [ ] **Step 2: Export the time operators**

In `src/TensorTrainNumerics.jl`, extend the operator export line:

```julia
export toeplitz_to_qtto, qtto_prolongation, qtto_constant_prolongation, qtto_linear_prolongation, ∇, Δ_DN, Δ_ND, Δ_NN, Δ_P, Δ, Δ⁻¹_DN, shift, qtt_time_identity, qtt_lower_shift, pauli_matrix, pauli_sum_tto, pauli_pair_sum_tto, H_μ, H_μν, heisenberg_xyz_tto, ising_tto, xxz_tto, xxx_tto, xy_tto, zeros_tt, zeros_tto, rand_tt, id_tto, rand_tto, qtt_laplacian
```

- [ ] **Step 3: Run the operator and slicing tests**

Run:

```bash
/Users/pzb464/.juliaup/bin/julialauncher --project=. -e 'using Test; include("test/test_global_crank_nicholson.jl")'
```

Expected: time operator and slicing tests pass; global solver tests fail with `UndefVarError: global_crank_nicholson_method not defined`.

---

### Task 4: Implement The Global Crank-Nicholson Solver

**Files:**
- Modify: `src/solvers/euler.jl`
- Modify: `src/TensorTrainNumerics.jl`
- Test: `test/test_global_crank_nicholson.jl`

**Interfaces:**
- Consumes: `SpaceTimeQTTvector`, `TTvector(::QTTvector)`, `TToperator(::QTToperator)`, `qtt_time_identity`, `qtt_lower_shift`, `qtt_basis_vector`, and existing linear solvers.
- Produces:
  - `global_crank_nicholson_method(A::QTToperator, u0::QTTvector, guess::QTTvector, steps::Vector{Float64}; kwargs...)`
  - `global_crank_nicholson_method(A::QTToperator, u0::QTTvector, guess::SpaceTimeQTTvector, steps::Vector{Float64}; kwargs...)`

- [ ] **Step 1: Add validation and conversion helpers**

In `src/solvers/euler.jl`, after `crank_nicholson_method`, add:

```julia
function _global_cn_time_bits(steps::Vector{Float64})
    isempty(steps) && throw(ArgumentError("steps must be nonempty"))
    τ = steps[1]
    all(step -> step == τ, steps) || throw(ArgumentError("global_crank_nicholson_method requires constant time steps"))
    Nt = length(steps)
    Nt < 2 && throw(ArgumentError("global_crank_nicholson_method requires at least two time steps so the time index has at least one QTT bit"))
    ispow2(Nt) || throw(ArgumentError("global_crank_nicholson_method requires length(steps) to be a power of two"))
    return τ, round(Int, log2(Nt))
end

function _convert_ttv_eltype(::Type{T}, v::TTvector{S, N}) where {T <: Number, S <: Number, N}
    return TTvector{T, N}(
        v.N,
        [convert(Array{T, 3}, core) for core in v.ttv_vec],
        v.ttv_dims,
        copy(v.ttv_rks),
        copy(v.ttv_ot)
    )
end

function _space_time_guess(
        ::Type{T},
        guess::QTTvector,
        time_bits::Int,
        space_n_dims::Int,
        space_bits_per_dim::Int,
        space_ordering::Symbol
    ) where {T <: Number}
    @assert guess.n_dims == space_n_dims "guess n_dims must match u0"
    @assert guess.bits_per_dim == space_bits_per_dim "guess bits_per_dim must match u0"
    @assert guess.ordering == space_ordering "guess ordering must match u0"
    time_guess = ones_tt(T, ntuple(_ -> 2, time_bits))
    return time_guess ⊗ _convert_ttv_eltype(T, TTvector(guess))
end

function _space_time_guess(
        ::Type{T},
        guess::SpaceTimeQTTvector,
        time_bits::Int,
        space_n_dims::Int,
        space_bits_per_dim::Int,
        space_ordering::Symbol
    ) where {T <: Number}
    @assert guess.time_bits == time_bits "space-time guess time_bits must match length(steps)"
    @assert guess.space_n_dims == space_n_dims "space-time guess space_n_dims must match u0"
    @assert guess.space_bits_per_dim == space_bits_per_dim "space-time guess space_bits_per_dim must match u0"
    @assert guess.space_ordering == space_ordering "space-time guess ordering must match u0"
    return _convert_ttv_eltype(T, TTvector(guess))
end

function _global_tt_linsolve(
        A::AbstractTToperator,
        b::AbstractTTvector,
        guess::AbstractTTvector,
        tt_solver::String,
        max_bond::Int;
        kwargs...
    )
    return (
        tt_solver == "mals" ? mals_linsolve(A, b, guess; kwargs...) :
            tt_solver == "als" ? als_linsolve(A, b, guess; kwargs...) :
            tt_solver == "dmrg" ? dmrg_linsolve(A, b, guess; kwargs...) :
            tt_solver == "krylov" ? krylov_linsolve(A, b, guess; max_bond = max_bond, kwargs...) :
            error("Unknown TT solver: $tt_solver")
    )::AbstractTTvector
end
```

- [ ] **Step 2: Add the public solver**

In `src/solvers/euler.jl`, immediately after the helpers from Step 1, add:

```julia
function global_crank_nicholson_method(
        A::QTToperator,
        u0::QTTvector,
        guess::Union{QTTvector, SpaceTimeQTTvector},
        steps::Vector{Float64};
        normalize::Bool = false,
        return_error::Bool = false,
        tt_solver::String = "mals",
        max_bond::Int = 0,
        kwargs...
    )
    normalize && throw(ArgumentError("global_crank_nicholson_method does not support per-step normalization"))
    check_compat(A, u0)

    τ, time_bits = _global_cn_time_bits(steps)
    T = promote_type(eltype(A), eltype(u0), typeof(τ))
    τT = convert(T, τ)

    A_tt = _convert_tto_eltype(T, TToperator(A))
    u0_tt = _convert_ttv_eltype(T, TTvector(u0))

    I_space = id_tto(T, A.N)
    L = I_space - (τT / 2) * A_tt
    R = I_space + (τT / 2) * A_tt

    I_time = qtt_time_identity(T, time_bits)
    S_time = qtt_lower_shift(T, time_bits)
    A_global = (I_time ⊗ L) - (S_time ⊗ R)

    first_rhs = R * u0_tt
    time_rhs = _convert_ttv_eltype(T, qtt_basis_vector(time_bits, 1))
    rhs = time_rhs ⊗ first_rhs

    guess_tt = _space_time_guess(
        T,
        guess,
        time_bits,
        u0.n_dims,
        u0.bits_per_dim,
        u0.ordering
    )

    solution_tt = _global_tt_linsolve(
        A_global,
        rhs,
        guess_tt,
        tt_solver,
        max_bond;
        kwargs...
    )
    if max_bond > 0
        solution_tt = tt_compress!(solution_tt, max_bond)
    else
        solution_tt = orthogonalize(solution_tt)
    end

    U = SpaceTimeQTTvector(
        solution_tt,
        time_bits,
        u0.n_dims,
        u0.bits_per_dim,
        u0.ordering
    )

    if return_error
        residual = norm(A_global * solution_tt - rhs) / max(norm(rhs), eps(real(T)))
        return U, residual
    end
    return U
end
```

- [ ] **Step 3: Export the solver**

In `src/TensorTrainNumerics.jl`, extend the Euler solver export line:

```julia
export euler_method, implicit_euler_method, crank_nicholson_method, global_crank_nicholson_method, rk4_method
```

- [ ] **Step 4: Run the focused global tests**

Run:

```bash
/Users/pzb464/.juliaup/bin/julialauncher --project=. test/test_global_crank_nicholson.jl
```

Expected: PASS.

---

### Task 5: Run Regression Tests And Inspect The Patch

**Files:**
- No new files.
- Verify: modified source and test files from Tasks 1-4.

**Interfaces:**
- Consumes: completed implementation.
- Produces: verified local patch with focused tests passing.

- [ ] **Step 1: Run Euler and global solver tests together**

Run:

```bash
/Users/pzb464/.juliaup/bin/julialauncher --project=. -e 'using Test; include("test/test_euler.jl"); include("test/test_global_crank_nicholson.jl")'
```

Expected: PASS.

- [ ] **Step 2: Run QTT multidimensional tests**

Run:

```bash
/Users/pzb464/.juliaup/bin/julialauncher --project=. test/test_qtt_multidim.jl
```

Expected: PASS.

- [ ] **Step 3: Run the full package test suite**

Run:

```bash
/Users/pzb464/.juliaup/bin/julialauncher --project=. -e 'using Pkg; Pkg.test()'
```

Expected: PASS. If the command is interrupted by the environment, report the targeted test results and the interruption point.

- [ ] **Step 4: Inspect changed files**

Run:

```bash
git diff -- src/qtt_tools.jl src/tt_operators.jl src/solvers/euler.jl src/TensorTrainNumerics.jl test/runtests.jl test/test_global_crank_nicholson.jl docs/superpowers/specs/2026-06-30-global-space-time-qtt-solver-design.md docs/superpowers/plans/2026-06-30-global-space-time-qtt-solver.md
```

Expected: diff only contains the global space-time QTT solver implementation, tests, and planning/spec files.

---

## Self-Review Notes

- Spec coverage: the tasks cover QTT-only inputs, constant power-of-two steps, package sign convention, time-first space-time output, slicing, residual reporting, and validation.
- Placeholder scan: no task contains open-ended placeholders; forcing terms and terminal-condition variants from the spec are excluded from this plan.
- Type consistency: `SpaceTimeQTTvector`, `space_time_slice`, `qtt_lower_shift`, `qtt_time_identity`, and `global_crank_nicholson_method` names are consistent across tests, exports, and implementation tasks.
