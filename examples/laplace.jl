using CairoMakie
using TensorTrainNumerics

function right_hand_side(cores::Int64)::Array{Float64,2}
    points = 2^cores
    a, b, d, e = 0, 1, 0, 1
    x = range(a, stop=b, length=points)
    y = range(d, stop=e, length=points)
    f = ((b - a) / points) * ((e - d) / points) .* source_term(x, y, points)
    bound = boundary_term(x, y, points)
    return reshape(f .- bound, points, points)
end

function source_term(x, y, points)
    X = repeat(x', points, 1)
    Y = repeat(y, 1, points)
    return zeros(points, points)
end

function boundary_term(x, y, points::Int)::Array{Float64,2}

    bc_left(y) = sin.(y)
    bc_right(y) = 0.1
    bc_bottom(x) = 0.1
    bc_top(x) = 0.1

    b_bottom_top = zeros(points, points)
    b_left_right = zeros(points, points)

    # Apply boundary conditions using broadcasting
    b_bottom_top[1, :] .= bc_bottom.(x)         # Bottom boundary
    b_bottom_top[end, :] .= bc_top.(x)          # Top boundary

    b_left_right[:, 1] .= bc_left.(y)           # Left boundary
    b_left_right[:, end] .= bc_right.(y)        # Right boundary

    return b_left_right .+ b_bottom_top
end

function solve_Laplace(cores::Int)::Array{Float64,2}
    points = 2^cores
    reshape_dims = ntuple(_ -> 2, 2 * cores)
    
    # Create QTT gradient operator
    A = Δ_tto(points, 2, Δ_DD) 

    qtt_levels = Int(log2(points)) 

    row_dims = [fill(2, qtt_levels) for _ in 1:2]  
    col_dims = [fill(2, qtt_levels) for _ in 1:2]

    L = tt2qtt(A,row_dims,col_dims)

    # Create right-hand side in QTT format
    b = reshape(right_hand_side(cores), reshape_dims...)
    b_tt = ttv_decomp(b, index=1, tol=1e-7)

    # Ensure consistent initialization for ALS solver
    x_tt = rand_tt(Float64, b_tt.ttv_dims, b_tt.ttv_rks)

    # Solve using ALS
    x_tt = als_linsolv(L, b_tt, x_tt)

    # Convert back to tensor
    y = ttv_to_tensor(x_tt)
    reshape(y, points, points)
end

K = solve_Laplace(8)
fig = Figure()
ax = Axis(fig[1, 1], title="Laplace Solution", xlabel="x", ylabel="y", limits=(0, 1, 0, 1))
heatmap!(ax, LinRange(0, 1, size(K, 1)), LinRange(0, 1, size(K, 2)), K)
fig