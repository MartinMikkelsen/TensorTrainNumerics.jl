using TensorTrainNumerics
using CairoMakie

const GLMAKIE_LOAD_ERROR = Ref{Union{Nothing, String}}(nothing)

try
    @eval import GLMakie
catch err
    GLMAKIE_LOAD_ERROR[] = sprint(showerror, err)
end

function load_makie_backend()
    if isdefined(@__MODULE__, :GLMakie)
        return GLMakie, "GLMakie"
    end
    if GLMAKIE_LOAD_ERROR[] !== nothing
        println("GLMakie unavailable ($(GLMAKIE_LOAD_ERROR[])); using CairoMakie instead.")
    end
    return CairoMakie, "CairoMakie"
end

coordinate_from_bits(bits, row, cols, a, b) = a + (b - a) * index_to_point(@view bits[row, cols])

# Keep the affine map explicit here so the physical coordinates are correct.
coordinate_qtt(d, a, b) = function_to_qtt(t -> a + (b - a) * t, d)

function build_initial_density(
        d,
        bounds,
        center,
        spreads,
        max_bond;
        tci_tol = 1.0e-12
    )
    xmin, xmax = bounds.x
    ymin, ymax = bounds.y
    zmin, zmax = bounds.z
    x₀, y₀, z₀ = center
    σ_x, σ_y, σ_z = spreads
    domain_ic = [collect(1:2) for _ in 1:(3d)]

    function f_gauss(bits)
        out = zeros(Float64, size(bits, 1))
        @inbounds for s in axes(bits, 1)
            x = coordinate_from_bits(bits, s, 1:d, xmin, xmax)
            y = coordinate_from_bits(bits, s, (d + 1):(2d), ymin, ymax)
            z = coordinate_from_bits(bits, s, (2d + 1):(3d), zmin, zmax)
            out[s] = exp(-0.5 * (((x - x₀) / σ_x)^2 + ((y - y₀) / σ_y)^2 + ((z - z₀) / σ_z)^2))
        end
        return out
    end

    u₀ = tt_cross(f_gauss, domain_ic, MaxVol(tol = tci_tol, verbose = false))
    return tt_compress!(u₀, max_bond)
end

function build_lorenz_operator(d, bounds, spacings, params)
    xmin, xmax = bounds.x
    ymin, ymax = bounds.y
    zmin, zmax = bounds.z
    h_x, h_y, h_z = spacings
    σ_L, ρ_L, β_L = params

    ones_d = ones_tt(ntuple(_ -> 2, d))
    qtt_x = coordinate_qtt(d, xmin, xmax)
    qtt_y = coordinate_qtt(d, ymin, ymax)
    qtt_z = coordinate_qtt(d, zmin, zmax)

    qtt_x3 = qtt_x ⊗ ones_d ⊗ ones_d
    qtt_y3 = ones_d ⊗ qtt_y ⊗ ones_d
    qtt_z3 = ones_d ⊗ ones_d ⊗ qtt_z
    qtt_xz = qtt_x ⊗ ones_d ⊗ qtt_z
    qtt_xy = qtt_x ⊗ qtt_y ⊗ ones_d

    C1 = ttv_to_diag_tto(σ_L * qtt_y3 - σ_L * qtt_x3)
    C2 = ttv_to_diag_tto(ρ_L * qtt_x3 - qtt_y3 - qtt_xz)
    C3 = ttv_to_diag_tto(qtt_xy - β_L * qtt_z3)

    I_d = id_tto(d)
    D_x = (1 / h_x) * (∇(d) ⊗ I_d ⊗ I_d)
    D_y = (1 / h_y) * (I_d ⊗ ∇(d) ⊗ I_d)
    D_z = (1 / h_z) * (I_d ⊗ I_d ⊗ ∇(d))

    L = (-1.0) * (C1 * D_x) - ((C2 * D_y) + (C3 * D_z)) + (σ_L + 1 + β_L) * id_tto(3d)
    return L, (C1 = C1, C2 = C2, C3 = C3)
end

function approx_mass(u, cell_volume)
    weights = ones_tt(ntuple(_ -> 2, u.N))
    return cell_volume * dot(weights, u)
end

function reconstruct_density(u, N)
    ρ_raw = reshape(qtt_to_vector(u), N, N, N)
    return permutedims(ρ_raw, (3, 2, 1))
end

function density_for_display(ρ)
    ρ_vis = max.(ρ, 0.0)
    peak = maximum(ρ_vis)
    if peak <= eps(Float64)
        println("Density is non-positive after clipping; using absolute values for visualization.")
        ρ_vis = abs.(ρ)
        peak = maximum(ρ_vis)
    end
    peak = max(peak, eps(Float64))
    return ρ_vis ./ peak
end

normalize_curve(v) = v ./ max(maximum(v), eps(Float64))

function density_diagnostics(ρ_vis)
    ρ_xy = dropdims(sum(ρ_vis, dims = 3), dims = 3)
    ρ_xz = dropdims(sum(ρ_vis, dims = 2), dims = 2)
    ρ_yz = dropdims(sum(ρ_vis, dims = 1), dims = 1)

    marginal_x = vec(sum(ρ_vis, dims = (2, 3)))
    marginal_y = vec(sum(ρ_vis, dims = (1, 3)))
    marginal_z = vec(sum(ρ_vis, dims = (1, 2)))

    return (
        ρ_xy = ρ_xy,
        ρ_xz = ρ_xz,
        ρ_yz = ρ_yz,
        marginal_x = marginal_x,
        marginal_y = marginal_y,
        marginal_z = marginal_z,
    )
end

function render_density(
        makie,
        ρ,
        bounds,
        N,
        d,
        total_time;
        ρ_initial = nothing,
        isovalue = 0.3,
        output_path = joinpath(@__DIR__, "lorenz_density.png"),
        show_figure = true
    )
    xmin, xmax = bounds.x
    ymin, ymax = bounds.y
    zmin, zmax = bounds.z
    ρ_vis = density_for_display(ρ)
    diagnostics = density_diagnostics(ρ_vis)
    x_grid = collect(range(xmin, xmax, length = N))
    y_grid = collect(range(ymin, ymax, length = N))
    z_grid = collect(range(zmin, zmax, length = N))
    diagnostics_initial = nothing
    ρ_xy_change = nothing
    if ρ_initial !== nothing
        diagnostics_initial = density_diagnostics(density_for_display(ρ_initial))
        ρ_xy_change = diagnostics.ρ_xy - diagnostics_initial.ρ_xy
    end

    makie.set_theme!(makie.theme_black())
    fig = makie.Figure(size = (1200, 800))
    ax = makie.Axis3(fig[1:2, 1];
        xlabel = "x",
        ylabel = "y",
        zlabel = "z",
        title = "Lorenz Phase-Space Density - QTT (d=$(d), T=$(round(total_time, digits = 2)))",
        protrusions = (20, 20, 20, 20),
        viewmode = :fit,
        limits = (xmin, xmax, ymin, ymax, zmin, zmax))

    if nameof(makie) == :GLMakie
        makie.volume!(ax, (xmin, xmax), (ymin, ymax), (zmin, zmax), ρ_vis;
            algorithm = :iso,
            isorange = 0.05,
            isovalue = isovalue,
            colormap = :inferno,
            transparency = true)
    else
        println("CairoMakie does not support volume rendering here; using a 3D point-cloud fallback.")
        points = findall(>=(isovalue), ρ_vis)

        if isempty(points)
            fallback_level = max(0.05, 0.5 * maximum(ρ_vis))
            println("No points above isovalue=$(isovalue); retrying fallback at $(fallback_level).")
            points = findall(>=(fallback_level), ρ_vis)
        end

        max_points = 4000
        if length(points) > max_points
            stride = ceil(Int, length(points) / max_points)
            points = points[1:stride:end]
        end

        xs = Float64[]
        ys = Float64[]
        zs = Float64[]
        cs = Float64[]
        sizehint!(xs, length(points))
        sizehint!(ys, length(points))
        sizehint!(zs, length(points))
        sizehint!(cs, length(points))

        for idx in points
            ix, iy, iz = Tuple(idx)
            push!(xs, x_grid[ix])
            push!(ys, y_grid[iy])
            push!(zs, z_grid[iz])
            push!(cs, ρ_vis[ix, iy, iz])
        end

        makie.scatter!(ax, xs, ys, zs;
            color = cs,
            colormap = :inferno,
            markersize = 7,
            transparency = true)
    end

    ax_xy = makie.Axis(fig[1, 2],
        xlabel = "x",
        ylabel = "y",
        title = "XY projection")
    hm_xy = makie.heatmap!(ax_xy, x_grid, y_grid, diagnostics.ρ_xy;
        colormap = :inferno)
    makie.Colorbar(fig[1, 3], hm_xy, label = "projected density")

    if diagnostics_initial === nothing
        ax_xz = makie.Axis(fig[1, 4],
            xlabel = "x",
            ylabel = "z",
            title = "XZ projection")
        makie.heatmap!(ax_xz, x_grid, z_grid, diagnostics.ρ_xz;
            colormap = :inferno)
    else
        ax_change = makie.Axis(fig[1, 4],
            xlabel = "x",
            ylabel = "y",
            title = "XY change from initial")
        change_scale = max(maximum(abs, ρ_xy_change), eps(Float64))
        makie.heatmap!(ax_change, x_grid, y_grid, ρ_xy_change;
            colormap = :balance,
            colorrange = (-change_scale, change_scale))
    end

    ax_marginals = makie.Axis(fig[2, 2:4],
        xlabel = "coordinate value",
        ylabel = "normalised marginal",
        title = diagnostics_initial === nothing ? "1D marginals" : "1D marginals (solid = final, dashed = initial)")
    makie.lines!(ax_marginals, x_grid, normalize_curve(diagnostics.marginal_x),
        color = :gold, linewidth = 3, label = "x")
    makie.lines!(ax_marginals, y_grid, normalize_curve(diagnostics.marginal_y),
        color = :deepskyblue, linewidth = 3, label = "y")
    makie.lines!(ax_marginals, z_grid, normalize_curve(diagnostics.marginal_z),
        color = :tomato, linewidth = 3, label = "z")
    if diagnostics_initial !== nothing
        makie.lines!(ax_marginals, x_grid, normalize_curve(diagnostics_initial.marginal_x),
            color = :gold, linewidth = 2, linestyle = :dash)
        makie.lines!(ax_marginals, y_grid, normalize_curve(diagnostics_initial.marginal_y),
            color = :deepskyblue, linewidth = 2, linestyle = :dash)
        makie.lines!(ax_marginals, z_grid, normalize_curve(diagnostics_initial.marginal_z),
            color = :tomato, linewidth = 2, linestyle = :dash)
    end
    makie.axislegend(ax_marginals, position = :rb)

    fig[2, 1] = makie.Label(fig,
        "Normalised density view\nnegative values clipped only for plotting",
        tellwidth = false)

    show_figure && display(fig)
    if output_path !== nothing
        mkpath(dirname(output_path))
        makie.save(output_path, fig)
        println("Saved $(abspath(output_path))")
    end
    return fig, ρ_vis
end

d = 4
σ_L = 10.0
ρ_L = 28.0
β_L = 8 / 3
bounds = (x = (-25.0, 25.0), y = (-30.0, 30.0), z = (0.0, 60.0))
center = (1.0, 1.0, 1.0)
spreads = (2.0, 2.0, 2.0)
dt = 0.01
nsteps = 5
max_bond = 10
tci_tol = 1.0e-12
isovalue = 0.15
output_path = joinpath(@__DIR__, "lorenz_density.png")
show_figure = true

N = 2^d
xmin, xmax = bounds.x
ymin, ymax = bounds.y
zmin, zmax = bounds.z
h_x = (xmax - xmin) / (N - 1)
h_y = (ymax - ymin) / (N - 1)
h_z = (zmax - zmin) / (N - 1)
cell_volume = h_x * h_y * h_z

println("Grid: $(N)^3 = $(N^3) points  |  h_x=$(round(h_x, digits = 3))  h_y=$(round(h_y, digits = 3))  h_z=$(round(h_z, digits = 3))")
println("Demo defaults favour a stable runnable example; try d=5, nsteps=100, max_bond=12 for a denser plot.")
println("Set show_figure=true when calling main(...) if you want an interactive Makie window.")
println("Building initial condition via TCI...")
u₀ = build_initial_density(d, bounds, center, spreads, max_bond; tci_tol = tci_tol)
mass₀ = approx_mass(u₀, cell_volume)
ρ_initial = reconstruct_density(u₀, N)
println("u₀: N=$(u₀.N) cores, max rank=$(maximum(u₀.ttv_rks)), L2=$(norm(u₀)), mass≈$(mass₀)")

println("Assembling Liouville operator...")
L, coeffs = build_lorenz_operator(d, bounds, (h_x, h_y, h_z), (σ_L, ρ_L, β_L))
println("  C1 max rank: $(maximum(coeffs.C1.tto_rks))")
println("  C2 max rank: $(maximum(coeffs.C2.tto_rks))")
println("  C3 max rank: $(maximum(coeffs.C3.tto_rks))")
println("  L max rank: $(maximum(L.tto_rks))")

Lu₀ = L * u₀
println("L*u₀ L2 = $(norm(Lu₀))")

steps = fill(dt, nsteps)
total_time = nsteps * dt
println("Running RK4 for $(nsteps) steps (T = $(total_time))...")
solution = rk4_method(L, u₀, steps, max_bond; normalize = false)

mass₁ = approx_mass(solution, cell_volume)
mass_drift = abs(mass₁ - mass₀) / max(abs(mass₀), eps(Float64))
println("solution max rank = $(maximum(solution.ttv_rks))")
println("solution L2 = $(norm(solution))")
println("solution mass≈$(mass₁)  |  relative drift≈$(mass_drift)")

ρ = reconstruct_density(solution, N)
println("density range before clipping = ($(minimum(ρ)), $(maximum(ρ)))")

fig = nothing
backend_name = nothing
if show_figure || output_path !== nothing
makie, backend_name = load_makie_backend()
println("Rendering density with $(backend_name)...")
fig, _ = render_density(
    makie,
    ρ,
    bounds,
    N,
    d,
    total_time;
    ρ_initial = ρ_initial,
    isovalue = isovalue,
    output_path = output_path,
    show_figure = show_figure,
)
end
