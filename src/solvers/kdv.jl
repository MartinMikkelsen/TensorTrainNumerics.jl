# KdV (Korteweg–de Vries) solver via SCF-ALS / SCF-MALS
#
# PDE:  u_t + 6u·u_x + u_xxx = 0
#
# Two time-stepping schemes, both with Picard (SCF) linearisation:
#
# Implicit Euler (kdv_*):
#   [(1/dt)·I + 6·diag(u^k)·D_x + D_xxx] · u^{n+1} = u^n/dt
#
# Crank-Nicolson (kdv_cn_*):
#   [(1/dt)·I + 3·diag(u^k)·D_x + (1/2)·D_xxx] · u^{n+1}
#     = (1/dt)·u^n − 3·(u^k ⊙ D_x·u^n) − (1/2)·D_xxx·u^n
#
# CN treats both the nonlinear and dispersive terms symmetrically in time
# (2nd-order accurate), eliminating the O(dt) phase error that shifts the
# soliton peak in the implicit-Euler scheme.
#
# 1-soliton IC (c = 4k², amplitude = c/2, width k = sqrt(c)/2):
#   u₀(x) = (c/2) * sech(sqrt(c)/2 * (x - x₀))²   ← correct (NOT sqrt(c/2))
#
# Recommended periodic operators (use ∇_c_P and ∇3_P from tt_operators.jl):
#   D_x   = (1/(2dx))  * ∇_c_P(d)   — centered first derivative
#   D_xxx = (1/(2dx³)) * ∇3_P(d)    — centered third derivative
#
# All time-steppers return Vector{TTvector} (one snapshot per step,
# including the initial condition) for space-time visualisation.

using LinearAlgebra

"""
    kdv_als_step(u_prev, D_x, D_xxx, dt; ...)

One implicit-Euler time step for the KdV equation

    u_t + 6u·u_x + u_xxx = 0

via Picard linearization: freeze the advection coefficient at the current
iterate and solve the linear system with 1-site ALS.

    [(1/dt)·I + 6·diag(u^k)·D_x + D_xxx] · u^{k+1} = u^{prev}/dt

# Arguments
- `u_prev  :: TTvector{T}`   — solution at the previous time step
- `D_x     :: TToperator{T}` — first-derivative operator (scaled by 1/(2dx))
- `D_xxx   :: TToperator{T}` — third-derivative operator (scaled by 1/(2dx³))
- `dt      :: Real`          — time step

# Keyword arguments
- `max_scf   :: Int  = 10`
- `scf_tol   :: Real = 1e-8`
- `max_bond  :: Int  = 20`
- `als_sweeps:: Int  = 5`
- `verbose   :: Bool = false`
"""
function kdv_als_step(
        u_prev    :: TTvector{T},
        D_x       :: TToperator{T},
        D_xxx     :: TToperator{T},
        dt        :: Real;
        max_scf   :: Int  = 10,
        scf_tol   :: Real = 1e-8,
        max_bond  :: Int  = 20,
        als_sweeps:: Int  = 5,
        verbose   :: Bool = false
    ) where {T <: Number}

    d     = u_prev.N
    I_tto = id_tto(T, d; n_dim = 2)
    invdt = convert(T, inv(dt))
    rhs   = invdt * u_prev

    u = u_prev
    for iter in 1:max_scf
        u_old = u
        A_adv = ttv_to_diag_tto(u) * D_x
        A_eff = invdt * I_tto + convert(T, 6) * A_adv + D_xxx
        u     = als_linsolve(A_eff, rhs, u_old; sweep_count = als_sweeps)
        u     = tt_compress!(u, max_bond)
        rel_diff = norm(u - u_old) / (norm(u) + eps(real(T)))
        verbose && println("    Picard $iter  rel_diff = $(round(rel_diff, sigdigits=4))")
        rel_diff < scf_tol && break
    end
    return u
end

"""
    kdv_als(u₀, D_x, D_xxx, dt, n_steps; ...)

Time-integrate the KdV equation for `n_steps` implicit-Euler steps using
`kdv_als_step` (1-site ALS, fixed rank).

# Returns
`Vector{TTvector{T}}` of length `n_steps + 1`: `sol[1] == u₀`,
`sol[k+1]` is the solution at time `k*dt`.
"""
function kdv_als(
        u₀           :: TTvector{T},
        D_x          :: TToperator{T},
        D_xxx        :: TToperator{T},
        dt           :: Real,
        n_steps      :: Int;
        max_scf      :: Int  = 10,
        scf_tol      :: Real = 1e-8,
        max_bond     :: Int  = 20,
        als_sweeps   :: Int  = 5,
        verbose      :: Bool = false,
        verbose_steps:: Bool = false
    ) where {T <: Number}

    u         = u₀
    snapshots = TTvector{T}[u₀]
    for step in 1:n_steps
        u = kdv_als_step(u, D_x, D_xxx, dt;
                max_scf = max_scf, scf_tol = scf_tol,
                max_bond = max_bond, als_sweeps = als_sweeps,
                verbose = verbose)
        push!(snapshots, u)
        verbose_steps &&
            println("  step $step / $n_steps  max_rank = $(maximum(u.ttv_rks))")
    end
    return snapshots
end

"""
    kdv_mals_step(u_prev, D_x, D_xxx, dt; ...)

One implicit-Euler time step for the KdV equation via Picard linearization
with a rank-adaptive two-site ALS sweep.

Identical to `kdv_als_step` but uses a non-symmetric 2-site sweep (same
structure as `burgers_scf_mals_step`) that grows bond dimension up to
`max_bond`.  No `Hermitian()` wrapping is applied to the local matrix because
the advection term makes the operator non-symmetric.
"""
function kdv_mals_step(
        u_prev   :: TTvector{T},
        D_x      :: TToperator{T},
        D_xxx    :: TToperator{T},
        dt       :: Real;
        max_scf  :: Int  = 10,
        scf_tol  :: Real = 1e-8,
        max_bond :: Int  = 20,
        verbose  :: Bool = false
    ) where {T <: Number}

    d     = u_prev.N
    dims  = u_prev.ttv_dims
    I_tto = id_tto(T, d; n_dim = 2)
    invdt = convert(T, inv(dt))
    rhs   = invdt * u_prev

    u = orthogonalize(u_prev)

    for iter in 1:max_scf
        u_old = u

        A_adv = ttv_to_diag_tto(u) * D_x
        A_eff = invdt * I_tto + convert(T, 6) * A_adv + D_xxx
        A_rks = A_eff.tto_rks
        b_rks = rhs.ttv_rks

        G   = Array{Array{T, 5}}(undef, d)
        G_b = Array{Array{T, 3}}(undef, d)
        for i in 1:d
            rmax_i = min(max_bond, prod(dims[1:(i - 1)]), prod(dims[i:end]))
            G[i]   = zeros(T, dims[i], rmax_i, dims[i], rmax_i, A_rks[i + 1])
            G_b[i] = zeros(T, dims[i], rmax_i, b_rks[i + 1])
        end
        G[1][:, 1:1, :, 1:1, :] = reshape(A_eff.tto_vec[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)
        G_b[1] = reshape(rhs.ttv_vec[1], dims[1], 1, :)

        H   = init_H_mals(u, A_eff, max_bond)
        H_b = init_Hb_mals(u, rhs, max_bond)

        for i in 1:(d - 1)
            rks  = u.ttv_rks
            Gi   = @view G[i][:, 1:rks[i],     :, 1:rks[i],     :]
            Hi   = @view H[i][:, :, 1:rks[i+2], :, 1:rks[i+2]]
            G_bi = @view G_b[i][:, 1:rks[i],   :]
            H_bi = @view H_b[i][:, :, 1:rks[i+2]]

            K_dims = (dims[i], rks[i], dims[i+1], rks[i+2])
            K  = zeros(T, prod(K_dims), prod(K_dims))
            Kr = reshape(K, (K_dims..., K_dims...))
            @tensor Kr[a, b, c, q, e, f, g, h] = Gi[a, b, e, f, z] * Hi[z, c, q, g, h]
            Pb = zeros(T, K_dims)
            @tensor Pb[a, b, c, q] = G_bi[a, b, z] * H_bi[z, c, q]

            V = reshape(K \ Pb[:], K_dims)
            u = right_core_move_mals(u, i, V, 0.0, max_bond)

            rks   = u.ttv_rks
            Gip   = @view G[i+1][:, 1:rks[i+1], :, 1:rks[i+1], :]
            G_bip = @view G_b[i+1][:, 1:rks[i+1], :]
            update_G!(u.ttv_vec[i], A_eff.tto_vec[i+1], Gi, Gip)
            update_Gb!(u.ttv_vec[i], rhs.ttv_vec[i+1], G_bi, G_bip)
        end

        for i in (d - 1):-1:1
            rks  = u.ttv_rks
            Gi   = @view G[i][:, 1:rks[i],     :, 1:rks[i],     :]
            Hi   = @view H[i][:, :, 1:rks[i+2], :, 1:rks[i+2]]
            G_bi = @view G_b[i][:, 1:rks[i],   :]
            H_bi = @view H_b[i][:, :, 1:rks[i+2]]

            K_dims = (dims[i], rks[i], dims[i+1], rks[i+2])
            K  = zeros(T, prod(K_dims), prod(K_dims))
            Kr = reshape(K, (K_dims..., K_dims...))
            @tensor Kr[a, b, c, q, e, f, g, h] = Gi[a, b, e, f, z] * Hi[z, c, q, g, h]
            Pb = zeros(T, K_dims)
            @tensor Pb[a, b, c, q] = G_bi[a, b, z] * H_bi[z, c, q]

            V = reshape(K \ Pb[:], K_dims)
            u = left_core_move_mals(u, i, V, 0.0, max_bond)

            if i > 1
                rks   = u.ttv_rks
                Him   = @view H[i-1][:, :, 1:rks[i+1], :, 1:rks[i+1]]
                H_bim = @view H_b[i-1][:, :, 1:rks[i+1]]
                updateH_mals!(u.ttv_vec[i+1], A_eff.tto_vec[i], Hi, Him)
                updateHb_mals!(u.ttv_vec[i+1], rhs.ttv_vec[i], H_bi, H_bim)
            end
        end

        rel_diff = norm(u - u_old) / (norm(u) + eps(real(T)))
        verbose && println("    Picard $iter  rel_diff = $(round(rel_diff, sigdigits=4))")
        rel_diff < scf_tol && break
    end
    return u
end

"""
    kdv_mals(u₀, D_x, D_xxx, dt, n_steps; ...)

Time-integrate the KdV equation for `n_steps` implicit-Euler steps using
`kdv_mals_step` (rank-adaptive two-site ALS).

# Returns
`Vector{TTvector{T}}` of length `n_steps + 1`: `sol[1] == u₀`,
`sol[k+1]` is the solution at time `k*dt`.
"""
function kdv_mals(
        u₀           :: TTvector{T},
        D_x          :: TToperator{T},
        D_xxx        :: TToperator{T},
        dt           :: Real,
        n_steps      :: Int;
        max_scf      :: Int  = 10,
        scf_tol      :: Real = 1e-8,
        max_bond     :: Int  = 20,
        verbose      :: Bool = false,
        verbose_steps:: Bool = false
    ) where {T <: Number}

    u         = u₀
    snapshots = TTvector{T}[u₀]
    for step in 1:n_steps
        u = kdv_mals_step(u, D_x, D_xxx, dt;
                max_scf = max_scf, scf_tol = scf_tol,
                max_bond = max_bond, verbose = verbose)
        push!(snapshots, u)
        verbose_steps &&
            println("  step $step / $n_steps  max_rank = $(maximum(u.ttv_rks))")
    end
    return snapshots
end

"""
    kdv_cn_mals_step(u_prev, D_x, D_xxx, dt; ...)

One Crank-Nicolson time step for the KdV equation via Picard linearization.

The nonlinear and dispersive terms are both treated as time-averages:

    [(1/dt)·I + 3·diag(u^k)·D_x + (1/2)·D_xxx] · u^{n+1}
    = (1/dt)·u^n − 3·(u^k ⊙ D_x·u^n) − (1/2)·D_xxx·u^n

This is 2nd-order accurate in time, eliminating the O(dt) phase error that
shifts the soliton peak in the implicit-Euler scheme.
"""
function kdv_cn_mals_step(
        u_prev   :: TTvector{T},
        D_x      :: TToperator{T},
        D_xxx    :: TToperator{T},
        dt       :: Real;
        max_scf  :: Int  = 10,
        scf_tol  :: Real = 1e-8,
        max_bond :: Int  = 20,
        verbose  :: Bool = false
    ) where {T <: Number}

    d     = u_prev.N
    dims  = u_prev.ttv_dims
    I_tto = id_tto(T, d; n_dim = 2)
    invdt = convert(T, inv(dt))
    half  = convert(T, 0.5)
    three = convert(T, 3)

    # Precompute u^n-dependent terms (constant across Picard iterations)
    Du_prev  = tt_compress!(D_x * u_prev,   max_bond)
    D3u_prev = tt_compress!(D_xxx * u_prev, max_bond)

    u = orthogonalize(u_prev)

    for iter in 1:max_scf
        u_old = u

        # u^{n+1/2} ≈ (u^k + u^n)/2 gives the symmetric (midpoint) treatment
        # that makes the nonlinear term 2nd-order accurate in time
        u_mid = tt_compress!(half * (u + u_prev), max_bond)
        A_adv = ttv_to_diag_tto(u_mid) * D_x
        A_eff = invdt * I_tto + three * A_adv + half * D_xxx

        rhs = tt_compress!(invdt * u_prev - three * hadamard(u_mid, Du_prev) - half * D3u_prev, max_bond)

        A_rks = A_eff.tto_rks
        b_rks = rhs.ttv_rks

        G   = Array{Array{T, 5}}(undef, d)
        G_b = Array{Array{T, 3}}(undef, d)
        for i in 1:d
            rmax_i = min(max_bond, prod(dims[1:(i - 1)]), prod(dims[i:end]))
            G[i]   = zeros(T, dims[i], rmax_i, dims[i], rmax_i, A_rks[i + 1])
            G_b[i] = zeros(T, dims[i], rmax_i, b_rks[i + 1])
        end
        G[1][:, 1:1, :, 1:1, :] = reshape(A_eff.tto_vec[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)
        G_b[1] = reshape(rhs.ttv_vec[1], dims[1], 1, :)

        H   = init_H_mals(u, A_eff, max_bond)
        H_b = init_Hb_mals(u, rhs, max_bond)

        for i in 1:(d - 1)
            rks  = u.ttv_rks
            Gi   = @view G[i][:, 1:rks[i],     :, 1:rks[i],     :]
            Hi   = @view H[i][:, :, 1:rks[i+2], :, 1:rks[i+2]]
            G_bi = @view G_b[i][:, 1:rks[i],   :]
            H_bi = @view H_b[i][:, :, 1:rks[i+2]]

            K_dims = (dims[i], rks[i], dims[i+1], rks[i+2])
            K  = zeros(T, prod(K_dims), prod(K_dims))
            Kr = reshape(K, (K_dims..., K_dims...))
            @tensor Kr[a, b, c, q, e, f, g, h] = Gi[a, b, e, f, z] * Hi[z, c, q, g, h]
            Pb = zeros(T, K_dims)
            @tensor Pb[a, b, c, q] = G_bi[a, b, z] * H_bi[z, c, q]

            V = reshape(K \ Pb[:], K_dims)
            u = right_core_move_mals(u, i, V, 0.0, max_bond)

            rks   = u.ttv_rks
            Gip   = @view G[i+1][:, 1:rks[i+1], :, 1:rks[i+1], :]
            G_bip = @view G_b[i+1][:, 1:rks[i+1], :]
            update_G!(u.ttv_vec[i], A_eff.tto_vec[i+1], Gi, Gip)
            update_Gb!(u.ttv_vec[i], rhs.ttv_vec[i+1], G_bi, G_bip)
        end

        for i in (d - 1):-1:1
            rks  = u.ttv_rks
            Gi   = @view G[i][:, 1:rks[i],     :, 1:rks[i],     :]
            Hi   = @view H[i][:, :, 1:rks[i+2], :, 1:rks[i+2]]
            G_bi = @view G_b[i][:, 1:rks[i],   :]
            H_bi = @view H_b[i][:, :, 1:rks[i+2]]

            K_dims = (dims[i], rks[i], dims[i+1], rks[i+2])
            K  = zeros(T, prod(K_dims), prod(K_dims))
            Kr = reshape(K, (K_dims..., K_dims...))
            @tensor Kr[a, b, c, q, e, f, g, h] = Gi[a, b, e, f, z] * Hi[z, c, q, g, h]
            Pb = zeros(T, K_dims)
            @tensor Pb[a, b, c, q] = G_bi[a, b, z] * H_bi[z, c, q]

            V = reshape(K \ Pb[:], K_dims)
            u = left_core_move_mals(u, i, V, 0.0, max_bond)

            if i > 1
                rks   = u.ttv_rks
                Him   = @view H[i-1][:, :, 1:rks[i+1], :, 1:rks[i+1]]
                H_bim = @view H_b[i-1][:, :, 1:rks[i+1]]
                updateH_mals!(u.ttv_vec[i+1], A_eff.tto_vec[i], Hi, Him)
                updateHb_mals!(u.ttv_vec[i+1], rhs.ttv_vec[i], H_bi, H_bim)
            end
        end

        rel_diff = norm(u - u_old) / (norm(u) + eps(real(T)))
        verbose && println("    Picard $iter  rel_diff = $(round(rel_diff, sigdigits=4))")
        rel_diff < scf_tol && break
    end
    return u
end

"""
    kdv_cn_mals(u₀, D_x, D_xxx, dt, n_steps; ...)

Time-integrate the KdV equation for `n_steps` Crank-Nicolson steps using
`kdv_cn_mals_step` (rank-adaptive two-site ALS, 2nd-order in time).

# Returns
`Vector{TTvector{T}}` of length `n_steps + 1`: `sol[1] == u₀`,
`sol[k+1]` is the solution at time `k*dt`.
"""
function kdv_cn_mals(
        u₀           :: TTvector{T},
        D_x          :: TToperator{T},
        D_xxx        :: TToperator{T},
        dt           :: Real,
        n_steps      :: Int;
        max_scf      :: Int  = 10,
        scf_tol      :: Real = 1e-8,
        max_bond     :: Int  = 20,
        verbose      :: Bool = false,
        verbose_steps:: Bool = false
    ) where {T <: Number}

    u         = u₀
    snapshots = TTvector{T}[u₀]
    for step in 1:n_steps
        u = kdv_cn_mals_step(u, D_x, D_xxx, dt;
                max_scf = max_scf, scf_tol = scf_tol,
                max_bond = max_bond, verbose = verbose)
        push!(snapshots, u)
        verbose_steps &&
            println("  step $step / $n_steps  max_rank = $(maximum(u.ttv_rks))")
    end
    return snapshots
end
