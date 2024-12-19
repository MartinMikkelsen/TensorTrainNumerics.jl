using CairoMakie
using TensorTrainNumerics

function compute_boundary_conditions(c,N)
    ket_0 = χ(c,1.0,0.0)
    ket_1 = χ(c,0.0,1.0)

    boundary_left = lagrange_rank_revealing(bc_left,c,N)
    boundary_right = lagrange_rank_revealing(bc_right,c,N)
    boundary_bottom = lagrange_rank_revealing(bc_bottom,c,N)
    boundary_top = lagrange_rank_revealing(bc_top,c,N)

    qtt_bottom = concatenate(ket_0,boundary_bottom)
    qtt_top = concatenate(ket_1,boundary_top)
    qtt_left = concatenate(boundary_left,ket_0)
    qtt_right = concatenate(boundary_right,ket_1)

    boundary_x = concatenate(qtt_left,qtt_right)
    boundary_y = concatenate(qtt_bottom,qtt_top)
    boundary = boundary_x + boundary_y

    return boundary
end

function kron(A::TToperator{T,NA}, B::TToperator{T,NB}) where {T<:Number,NA,NB}
    N = A.N + B.N
    new_dims = (A.tto_dims..., B.tto_dims...)
    new_cores = vcat(A.tto_vec, B.tto_vec)
    new_rks = vcat(A.tto_rks[1:end-1], B.tto_rks)
    new_ot = vcat(A.tto_ot, B.tto_ot)
    return TToperator{T, N}(N, new_cores, new_dims, new_rks, new_ot)
end

function build_1d_diff_elements_DD(c::Int)
    # Define I, J, JT
    I = [1.0 0.0; 0.0 1.0]
    J = [0.0 1.0; 0.0 0.0]
    JT = transpose(J)

    # f_core: shape (1,2,2,3)
    f_core = zeros(Float64, 1, 2, 2, 3)
    f_core[1, :, :, 1] = I
    f_core[1, :, :, 2] = JT
    f_core[1, :, :, 3] = J

    # m_core: shape (3,2,2,3)
    m_core = zeros(Float64, 3, 2, 2, 3)
    # Python: m_core[0,:,:,0] = I => Julia: m_core[1,:,:,1]
    m_core[1, :, :, 1] = I
    # m_core[0,:,:,1] = JT => m_core[1,:,:,2]
    m_core[1, :, :, 2] = JT
    # m_core[0,:,:,2] = J => m_core[1,:,:,3]
    m_core[1, :, :, 3] = J
    # m_core[1,:,:,1] = J => m_core[2,:,:,2]
    m_core[2, :, :, 2] = J
    # m_core[2,:,:,2] = JT => m_core[3,:,:,3]
    m_core[3, :, :, 3] = JT

    # Compute h, h2, alpha, beta, gamma
    alpha = -2.0 
    beta = 1.0
    gamma = 1.0

    l_core = zeros(Float64, 3, 2, 2, 1)
    l_core[1, :, :, 1] = alpha * I + beta * J + gamma * JT
    l_core[2, :, :, 1] = gamma * J
    l_core[3, :, :, 1] = beta * JT

    all_cores = Vector{Array{Float64,4}}()
    push!(all_cores, f_core)
    for i in 1:(c-2)
        push!(all_cores, copy(m_core))
    end
    push!(all_cores, l_core)

    tto_dims = ntuple(_->2, c)

    tto_rks = [1, 3]
    if c > 2
        append!(tto_rks, fill(3, c-2))
    end
    push!(tto_rks, 1)

    tto_ot = zeros(Int, c)

    return TToperator{Float64,c}(c, all_cores, tto_dims, tto_rks, tto_ot)
end

function solve_Laplace(cores::Int)
    points = 2^cores

    # Create the 1D differential operator and identity operator in TT format
    tt_operator = build_1d_diff_elements_DD(cores)  # This gives ranks like [1,3,3,1]
    # Extract the ranks
    rks = tt_operator.tto_rks

    # Build identity operator with the same ranks
    I_TT = id(Float64, rks, 2)

    # Now I_TT and tt_operator have compatible ranks and shapes
    A = kron(tt_operator, I_TT)
    B = kron(I_TT, tt_operator)
    C = A + B  # Should now work without dimension mismatch

    # Create right-hand side in QTT format
    b_tt = compute_boundary_conditions(cores, 25)

    # Ensure consistent initialization for ALS solver
    q_tt = rand_tt(b_tt.ttv_dims, b_tt.ttv_rks)
    
    # Solve the system using ALS solver
    x_tt = als_linsolv(C, b_tt, q_tt, it_solver=true)

    # Convert the solution back to tensor format
    y = ttv_to_tensor(x_tt)

    visualize(x_tt)
    return y
end

bc_left(y) = sin(π * y) 
bc_right(y) = sin(π * y) 
bc_bottom(x) = sin(π * x) 
bc_top(x) = sin(π * x) 

K = solve_Laplace(8)
