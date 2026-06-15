# Viscous Eikonal equation on (0,1)² with T = 0 on the boundary:
#
#     -ε·ΔT + |∇T|² = s(x,y)²,        ε → 0
#
# where s is the slowness field. As ε → 0 the solution converges to the
# viscosity solution of the Eikonal equation |∇T| = s.
#
# Method: ε-continuation with Picard (frozen-gradient) linearisation,
#
#     [ε·L + diag(∂x T_k)·D_x + diag(∂y T_k)·D_y] T_{k+1} = s² ,
#
# solved per iteration with fixed-rank ALS (the local solves handle the
# non-symmetric advection part directly, like the Burgers solver; MALS is
# avoided because `mals_linsolve` symmetrises the local matrix).
# The frozen gradient coefficients are linear in T (KdV-like), so they are
# formed by direct operator application plus gauged compression; no
# interpolative projection is needed inside the loop.

"""
    eikonal_viscous_solve(d; slowness, ε_schedule, ...)

Solve the viscous Eikonal equation `-ε·ΔT + |∇T|² = s²` on `(0,1)²` with
`T = 0` on the boundary, continued through decreasing viscosities
`ε_schedule`, in serial 2D QTT format with `d` bits per dimension.

The grid is the interior lattice `x_i = i·h`, `h = 1/(2^d + 1)`.

# Arguments
- `slowness`: function `(x, y) -> s` or a `TTvector` with `2d` binary sites
  holding `s²` on the interior grid (serial ordering, x sites first).

# Keyword arguments
- `ε_schedule::Vector{<:Real} = [0.2, 0.1, 0.05, 0.025]` — decreasing viscosities,
  each warm-started from the previous solution.
- `max_scf::Int = 30` — maximum Picard iterations per viscosity level.
- `scf_tol::Real = 1.0e-10` — relative Picard convergence tolerance.
- `max_bond::Int = 32` — TT bond dimension (the iterate is padded to this rank).
- `coeff_bond::Int = max_bond` — bond cap for the frozen gradient coefficients.
- `als_sweeps::Int = 5` — ALS sweeps per Picard iteration.
- `verbose::Bool = false`

# Returns
`(T, info)` where `info` carries `Lap`, `Dx`, `Dy`, `s2` (the operators and
data actually used, for external referencing), `h`, per-level
`eikonal_residual` (mean |­|∇T|−s| on the grid), `picard_history`, and
`rank_history`.
"""
function eikonal_viscous_solve(
        d::Int;
        slowness,
        ε_schedule::Vector{<:Real} = [0.2, 0.1, 0.05, 0.025],
        max_scf::Int = 30,
        scf_tol::Real = 1.0e-10,
        max_bond::Int = 32,
        coeff_bond::Int = max_bond,
        als_sweeps::Int = 5,
        verbose::Bool = false
    )
    isempty(ε_schedule) && throw(ArgumentError("ε_schedule must not be empty"))
    all(>(0), ε_schedule) || throw(ArgumentError("all viscosities must be positive"))

    N = 2^d
    h = 1.0 / (N + 1)

    I_d = id_tto(d)
    L1 = (1.0 / h^2) * Δ(d)
    C1 = (1.0 / (2h)) * ∇_c(d)
    Lap = L1 ⊗ I_d + I_d ⊗ L1
    Dx = C1 ⊗ I_d
    Dy = I_d ⊗ C1

    s2 = if slowness isa TTvector
        slowness.N == 2d ||
            throw(ArgumentError("slowness TTvector must have $(2d) sites, got $(slowness.N)"))
        slowness
    else
        # interior grid: a + i·(b-a)/(N-1) = (i+1)·h for a = h, b = N·h
        TTvector(function_to_qttv(
            c -> Float64(slowness(c[1], c[2]))^2,
            2, d; ordering = :serial, a = h, b = N * h,
        ))
    end

    # Initial guess: screened-Poisson solve ε₀·L T = s² at the largest viscosity
    # (zero advection), from a random full-rank start.
    T = rand_tt(fill(2, 2d), max_bond; normalise = true)
    T = als_linsolve(Float64(ε_schedule[1]) * Lap, s2, T; sweep_count = als_sweeps)

    eikonal_residual = Float64[]
    picard_history = Vector{Float64}[]
    rank_history = Int[]

    for ε in Float64.(ε_schedule)
        level_history = Float64[]
        for iter in 1:max_scf
            T_old = T
            px = tt_compress!(orthogonalize(Dx * T), coeff_bond)
            py = tt_compress!(orthogonalize(Dy * T), coeff_bond)
            A = ε * Lap + ttv_to_diag_tto(px) * Dx + ttv_to_diag_tto(py) * Dy
            T = als_linsolve(A, s2, T_old; sweep_count = als_sweeps)
            rel_diff = norm(T - T_old) / (norm(T) + eps(Float64))
            push!(level_history, rel_diff)
            verbose && println("    ε=$(round(ε, sigdigits = 3))  Picard $iter  rel_diff = $(round(rel_diff, sigdigits = 4))")
            rel_diff < scf_tol && break
        end
        push!(picard_history, level_history)
        T_eff = tt_compress!(orthogonalize(1.0 * T), typemax(Int); truncerr = 1e-10)
        push!(rank_history, maximum(T_eff.ttv_rks))

        px = real.(qtt_to_function(Dx * T))
        py = real.(qtt_to_function(Dy * T))
        s_vals = sqrt.(max.(real.(qtt_to_function(s2)), 0.0))
        res = abs.(sqrt.(px .^ 2 .+ py .^ 2) .- s_vals)
        push!(eikonal_residual, sum(res) / length(res))
        verbose && println("  ε=$(round(ε, sigdigits = 3))  mean |∇T|-s residual = $(round(eikonal_residual[end], sigdigits = 4))  max_rank = $(maximum(T.ttv_rks))")
    end

    info = (
        Lap = Lap,
        Dx = Dx,
        Dy = Dy,
        s2 = s2,
        h = h,
        ε_schedule = Float64.(ε_schedule),
        eikonal_residual = eikonal_residual,
        picard_history = picard_history,
        rank_history = rank_history,
    )
    return T, info
end
