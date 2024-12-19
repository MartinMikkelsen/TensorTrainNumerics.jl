using CairoMakie
using TensorTrainNumerics

function compute_boundary_conditions(c,N)
    ket_0 = χ(c,1.0,0.0)
    ket_1 = χ(c,0.0,1.0)

    boundary_left = interpolating_qtt(bc_left,c,N)
    boundary_right = interpolating_qtt(bc_right,c,N)
    boundary_bottom = interpolating_qtt(bc_bottom,c,N)
    boundary_top = interpolating_qtt(bc_top,c,N)

    qtt_bottom = concatenate(ket_0,boundary_bottom)
    qtt_top = concatenate(ket_1,boundary_top)
    qtt_left = concatenate(boundary_left,ket_0)
    qtt_right = concatenate(boundary_right,ket_1)

    boundary_x = concatenate(qtt_left,qtt_right)
    boundary_y = concatenate(qtt_bottom,qtt_top)
    boundary = boundary_x + boundary_y
    return  (qtt_bottom + qtt_top + qtt_left + qtt_right)

end

function solve_Laplace(cores::Int)::Array{Float64,2}
    points = 2^cores
    
    # Create QTT gradient operator
    A = Δ_tto(points, 2, Δ_DD) 

    qtt_levels = Int(log2(points)) 

    row_dims = [fill(2, qtt_levels) for _ in 1:2]  
    col_dims = [fill(2, qtt_levels) for _ in 1:2]

    L = tt2qtt(A,row_dims,col_dims)

    # Create right-hand side in QTT format
    b_tt = compute_boundary_conditions(cores,25)

    # Ensure consistent initialization for ALS solver
    q_tt = rand_tt(Float64, b_tt.ttv_dims, b_tt.ttv_rks)

    matricize(L)

    # Solve using ALS
    x_tt = als_linsolv(L, b_tt, b_tt)

    # Convert back to tensor
    y = ttv_to_tensor(x_tt)
    reshape(y, points, points)
end

bc_left(y) = sin(π * y) 
bc_right(y) = 0.0
bc_bottom(x) = 0.0
bc_top(x) = 0.0

K = solve_Laplace(8)
fig = Figure()
ax = Axis(fig[1, 1], title="Laplace Solution", xlabel="x", ylabel="y", limits=(0, 1, 0, 1))
heatmap!(ax, LinRange(0, 1, size(K, 1)), LinRange(0, 1, size(K, 2)), K)
fig