"""
1d-discrete Laplacian
"""
function Δ_DD(n)
  return Matrix(SymTridiagonal(2ones(n),-ones(n-1)))
end 

"""
1D Discrete Laplacian for Neumann-Neumann (NN) Boundary Conditions
"""
function Δ_NN(n)
    T = SymTridiagonal(2ones(n), -ones(n-1))
    T[1, 1] = 1  
    T[n, n] = 1  
    return Matrix(T)
end

"""
1D Discrete Laplacian for Dirichlet-Neumann (DN) Boundary Conditions
"""
function Δ_DN(n)
    T = SymTridiagonal(2ones(n), -ones(n-1))
    T[n, n] = 1 
    return Matrix(T)
end

"""
1D Discrete Laplacian for Neumann-Dirichlet (ND) Boundary Conditions
"""
function Δ_ND(n)
    T = SymTridiagonal(2ones(n), -ones(n-1))
    T[1, 1] = 1  
    return Matrix(T)
end

"""
1D Discrete Laplacian for Periodic Boundary Conditions
"""
function Δ_Periodic(n)
    T = zeros(n, n)
    for i in 1:n
        T[i, i] = 2
    end

    for i in 1:(n-1)
        T[i, i+1] = -1
        T[i+1, i] = -1
    end

    T[1, n] = -1
    T[n, 1] = -1

    return T
end

"""
n^d Discrete Laplacian in TTO format or 1D Laplacian if d=1.
"""
function Δ_tto(n, d, Δ_func)
    if d == 1
        return Δ_func(n)
    end
    h = [Δ_func(n) for _ in 1:d] 
    H_vec = Vector{Array{Float64,4}}(undef, d)
    rks = vcat(1, 2ones(Int64, d-1), 1)

    # First TTO core
    H_vec[1] = zeros(n, n, 1, 2)
    H_vec[1][:, :, 1, 1] = h[1]
    H_vec[1][:, :, 1, 2] = Matrix(I, n, n)

    # Middle TTO cores
    for i in 2:(d-1)
        H_vec[i] = zeros(n, n, 2, 2)
        H_vec[i][:, :, 1, 1] = Matrix(I, n, n)
        H_vec[i][:, :, 2, 1] = h[i]
        H_vec[i][:, :, 2, 2] = Matrix(I, n, n)
    end

    # Last TTO core
    H_vec[d] = zeros(n, n, 2, 1)
    H_vec[d][:, :, 1, 1] = Matrix(I, n, n)
    H_vec[d][:, :, 2, 1] = h[d]

    return TToperator{Float64, d}(d, H_vec, Tuple(n * ones(Int64, d)), rks, zeros(Int64, d))
end

"""
Constructs the QTT representation of a tridiagonal Toeplitz matrix
with parameters α (main diagonal), β (upper diagonal), γ (lower diagonal)
for size 2^l x 2^l.
"""
function QTT_Tridiagonal_Toeplitz(α, β, γ, l)
    if l < 2
        throw(ArgumentError("QTT representation requires l >= 2."))
    end

    I = [1 0; 0 1]  
    J = [0 1; 0 0]  

     first_core = zeros(Float64, 2, 2, 1, 3)
     first_core[:, :, 1, 1] = I
     first_core[:, :, 1, 2] = J'
     first_core[:, :, 1, 3] = J
 
     middle_core = zeros(Float64, 2, 2, 3, 3)
     middle_core[:, :, 1, 1] = I
     middle_core[:, :, 1, 2] = J'
     middle_core[:, :, 1, 3] = J
     middle_core[:, :, 2, 2] = J
     middle_core[:, :, 3, 3] = J'
 
     last_core = zeros(Float64, 2, 2, 3, 1)
     last_core[:, :, 1, 1] = α * I + β * J + γ * J'
     last_core[:, :, 2, 1] = γ * J
     last_core[:, :, 3, 1] = β * J'
 
     cores = [first_core]
     for _ in 2:(l-1)
         push!(cores, middle_core)
     end
     push!(cores, last_core)
 
     # Define dimensions and ranks
     dims = Tuple([2 for _ in 1:l])
     ranks = [1; fill(3, l-1); 1]
 
     return TToperator{Float64, l}(l, cores, dims, ranks, zeros(Int, l))
 end

"""
1D discrete shift matrix
"""
function shift_matrix(n)
  S = zeros(Float64, n, n)
  for i in 1:n-1
    S[i, i+1] = 1.0
  end
  return S
end

"""
n^d discrete shift in TTO format with rank 2 of
S = s ⊗ id ⊗ … ⊗ id + ⋯ + id ⊗ … ⊗ id ⊗ s
"""
function shift_tto(n, d; s=[shift_matrix(n) for i in 1:d])
  S_vec = Vector{Array{Float64,4}}(undef, d)
  rks = vcat(1, 2ones(Int64, d-1), 1)
  
  # first TTO core
  S_vec[1] = zeros(n, n, 1, 2)
  S_vec[1][:,:,1,1] = s[1]
  S_vec[1][:,:,1,2] = Matrix{Float64}(I, n, n)
  
  for i in 2:d-1
    S_vec[i] = zeros(n, n, 2, 2)
    S_vec[i][:,:,1,1] = Matrix{Float64}(I, n, n)
    S_vec[i][:,:,2,1] = s[i]
    S_vec[i][:,:,2,2] = Matrix{Float64}(I, n, n)
  end
  
  S_vec[d] = zeros(n, n, 2, 1)
  S_vec[d][:,:,1,1] = Matrix{Float64}(I, n, n)
  S_vec[d][:,:,2,1] = s[d]
  
  return TToperator{Float64, d}(d, S_vec, Tuple(n*ones(Int64, d)), rks, zeros(Int64, d))
end

"""
1D Discrete Gradient Operator for Dirichlet-Dirichlet (DD) Boundary Conditions
"""
function ∇_DD(n)
    G = zeros(Float64, n, n)
    for i in 1:(n-1)
        G[i, i] = 1   
        G[i, i+1] = -1  
    end
    return G
end

"""
n^d Discrete Gradient in TTO format or 1D Gradient if d=1.
"""
function ∇_tto(n, d, ∇_func)
    if d == 1
        # For 1D, return the full gradient matrix
        return ∇_func(n)
    end

    # Generate the gradient operators for each dimension
    h = [∇_func(n) for _ in 1:d]
    H_vec = Vector{Array{Float64,4}}(undef, d)
    rks = vcat(1, 2ones(Int64, d-1), 1)

    # First TTO core
    H_vec[1] = zeros(n, n, 1, 2)
    H_vec[1][:, :, 1, 1] = h[1]               # Gradient operator for the first dimension
    H_vec[1][:, :, 1, 2] = Matrix(I, n, n)    # Identity matrix

    # Middle TTO cores
    for i in 2:(d-1)
        H_vec[i] = zeros(n, n, 2, 2)
        H_vec[i][:, :, 1, 1] = Matrix(I, n, n)  # Identity matrix
        H_vec[i][:, :, 2, 1] = h[i]             # Gradient operator
        H_vec[i][:, :, 2, 2] = Matrix(I, n, n)  # Identity matrix
    end

    # Last TTO core
    H_vec[d] = zeros(n, n, 2, 1)
    H_vec[d][:, :, 1, 1] = Matrix(I, n, n)    # Identity matrix
    H_vec[d][:, :, 2, 1] = h[d]               # Gradient operator for the last dimension

    # Create the TToperator
    return TToperator{Float64, d}(d, H_vec, Tuple(n * ones(Int64, d)), rks, zeros(Int64, d))
end

function Jacobian_tto(n, d, ∇_func)
    # Generate the gradient operators for each dimension
    gradient_ops = [∇_tto(n, d, ∇_func) for _ in 1:d]

    # Adjust rank structure to accommodate all gradients
    H_vec = Vector{Array{Float64, 4}}(undef, d)
    rks = [1; fill(2 * d, d - 1); 1]  # Combined ranks for d gradients

    # First core
    H_vec[1] = zeros(n, n, rks[1], rks[2])
    for i in 1:d
        H_vec[1][:, :, 1, i] = gradient_ops[i].tto_vec[1][:, :, 1, 1]
    end

    # Middle cores
    for k in 2:(d - 1)
        H_vec[k] = zeros(n, n, rks[k], rks[k + 1])
        for i in 1:d
            H_vec[k][:, :, 1:2, (2 * i - 1):(2 * i)] .= gradient_ops[i].tto_vec[k]
        end
    end

    # Last core
    H_vec[d] = zeros(n, n, rks[d], rks[d + 1])
    for i in 1:d
        H_vec[d][:, :, (2 * i - 1):(2 * i), 1] .= gradient_ops[i].tto_vec[d]
    end

    dims = Tuple(n * ones(Int64, d))
    return TToperator{Float64, d}(d, H_vec, dims, rks, zeros(Int64, d))
end

function matricize(tt::TToperator{T, M}) where {T, M}
    first_core = tt.tto_vec[1]
    if prod(size(first_core)) != tt.tto_dims[1] * tt.tto_dims[1] * tt.tto_rks[2]
        error("First core size mismatch: expected $(tt.tto_dims[1] * tt.tto_dims[1] * tt.tto_rks[2]), got $(prod(size(first_core)))")
    end
    tt_mat = reshape(first_core, (tt.tto_dims[1], tt.tto_dims[1], tt.tto_rks[2]))

    for i in 2:tt.N
        next_core = tt.tto_vec[i]
        @tensor temp[a, b, c, d, r2] := tt_mat[a, b, r1] * next_core[c, d, r1, r2]
        tt_mat = permutedims(temp, (1, 3, 2, 4, 5))
        tt_mat = reshape(tt_mat, prod(tt.tto_dims[1:i]), prod(tt.tto_dims[1:i]), tt.tto_rks[i+1])
    end

    m = prod(tt.tto_dims)
    n = prod(tt.tto_dims)
    return reshape(tt_mat, m, n)
end

function matricize(tt::TTvector{T, M}) where {T, M}
    first_core = tt.ttv_vec[1]
    if prod(size(first_core)) != tt.ttv_dims[1] * tt.ttv_rks[2]
        error("First core size mismatch: expected $(tt.ttv_dims[1] * tt.ttv_rks[2]), got $(prod(size(first_core)))")
    end
    tt_mat = reshape(first_core, (tt.ttv_dims[1], tt.ttv_rks[2]))

    for i in 2:tt.N
        next_core = tt.ttv_vec[i]
        @tensor temp[a, b, r2] := tt_mat[a, r1] * next_core[b, r1, r2]
        tt_mat = permutedims(temp, (1, 3, 2))
        tt_mat = reshape(tt_mat, prod(tt.ttv_dims[1:i]), tt.ttv_rks[i+1])
    end

    m = prod(tt.ttv_dims)
    return reshape(tt_mat, m * tt.ttv_rks[end])
end

function tt2qtt(tt_tensor::TToperator{T,N}, row_dims::Vector{Vector{Int}}, col_dims::Vector{Vector{Int}}, threshold::Float64=0.0) where {T<:Number,N}
    
    qtt_cores = Array{Array{T,4}}(undef, 0)
    tto_rks = [1]
    tto_dims = Int64[]

    # For each core in tt_tensor
    for i in 1:tt_tensor.N

        # Get core, rank_prev, rank_next, row_dim, col_dim
        core = permutedims(tt_tensor.tto_vec[i], (3,1,2,4))  # Now core is (r_{k-1}, n_k_row, n_k_col, r_k)
        rank_prev = tto_rks[end]
        rank_next = tt_tensor.tto_rks[i+1]
        row_dim = tt_tensor.tto_dims[i]
        col_dim = tt_tensor.tto_dims[i]  # Assuming square dimensions

        # Begin splitting
        for j in 1:(length(row_dims[i]) - 1)

            # Update row_dim and col_dim
            row_dim = div(row_dim, row_dims[i][j])
            col_dim = div(col_dim, col_dims[i][j])

            # Reshape and permute core
            core = reshape(core, (rank_prev, row_dims[i][j], row_dim, col_dims[i][j], col_dim, rank_next))
            core = permutedims(core, (1,2,4,3,5,6))  # Now core is (r_{k-1}, row_dims[i][j], col_dims[i][j], row_dim, col_dim, r_k)

            # Reshape core into 2D matrix for SVD
            core_reshaped = reshape(core, (rank_prev * row_dims[i][j] * col_dims[i][j], row_dim * col_dim * rank_next))

            # Compute SVD
            F = svd(core_reshaped; full=false)
            U = F.U
            S = F.S
            Vt = F.Vt

            # Rank reduction
            if threshold != 0.0
                indices = findall(S ./ S[1] .> threshold)
                U = U[:, indices]
                S = S[indices]
                Vt = Vt[indices, :]
            end

            # Update rank
            new_rank = length(S)

            # Reshape U into core and append to qtt_cores
            U_reshaped = reshape(U, (rank_prev, row_dims[i][j], col_dims[i][j], new_rank))
            # Permute back to (n_k_row, n_k_col, r_{k-1}, r_k)
            core_to_append = permutedims(U_reshaped, (2,3,1,4))
            push!(qtt_cores, core_to_append)

            # Update tto_rks
            push!(tto_rks, new_rank)

            # Update tto_dims
            push!(tto_dims, row_dims[i][j])
            push!(tto_dims, col_dims[i][j])

            # Update core for next iteration
            core = Diagonal(S) * Vt
            rank_prev = new_rank
        end

        # For the last QTT core
        core = reshape(core, (rank_prev, row_dim, col_dim, rank_next))
        core_to_append = permutedims(core, (2,3,1,4))
        push!(qtt_cores, core_to_append)

        # Update tto_dims
        push!(tto_dims, row_dim)
        push!(tto_dims, col_dim)

        # Update tto_rks
        push!(tto_rks, rank_next)
    end

    N_qtt = length(qtt_cores)
    M = length(tto_dims)
    tto_ot = zeros(Int64, N_qtt)

    qtt_tensor = TToperator{T,M}(N_qtt, qtt_cores, Tuple(tto_dims), tto_rks, tto_ot)

    return qtt_tensor
end

function tt2qtt(tt_tensor::TTvector{T,N}, dims::Vector{Vector{Int}}, threshold::Float64=0.0) where {T<:Number,N}
    
    qtt_cores = Array{Array{T,3}}(undef, 0)
    ttv_rks = [1]
    ttv_dims = Int64[]

    # For each core in tt_tensor
    for i in 1:tt_tensor.N

        # Get core, rank_prev, rank_next, dim
        core = permutedims(tt_tensor.ttv_vec[i], (2,1,3))  # Now core is (r_{k-1}, n_k, r_k)
        rank_prev = ttv_rks[end]
        rank_next = tt_tensor.ttv_rks[i+1]
        dim = tt_tensor.ttv_dims[i]

        # Begin splitting
        for j in 1:(length(dims[i]) - 1)

            # Update dim
            dim = div(dim, dims[i][j])

            # Reshape and permute core
            core = reshape(core, (rank_prev, dims[i][j], dim, rank_next))
            core = permutedims(core, (1,2,3,4))  # Now core is (r_{k-1}, dims[i][j], dim, r_k)

            # Reshape core into 2D matrix for SVD
            core_reshaped = reshape(core, (rank_prev * dims[i][j], dim * rank_next))

            # Compute SVD
            F = svd(core_reshaped; full=false)
            U = F.U
            S = F.S
            Vt = F.Vt

            # Rank reduction
            if threshold != 0.0
                indices = findall(S ./ S[1] .> threshold)
                U = U[:, indices]
                S = S[indices]
                Vt = Vt[indices, :]
            end

            # Update rank
            new_rank = length(S)

            # Reshape U into core and append to qtt_cores
            U_reshaped = reshape(U, (rank_prev, dims[i][j], new_rank))
            # Permute back to (n_k, r_{k-1}, r_k)
            core_to_append = permutedims(U_reshaped, (2,1,3))
            push!(qtt_cores, core_to_append)

            # Update ttv_rks
            push!(ttv_rks, new_rank)

            # Update ttv_dims
            push!(ttv_dims, dims[i][j])

            # Update core for next iteration
            core = Diagonal(S) * Vt
            rank_prev = new_rank
        end

        # For the last QTT core
        core = reshape(core, (rank_prev, dim, rank_next))
        core_to_append = permutedims(core, (2,1,3))
        push!(qtt_cores, core_to_append)

        # Update ttv_dims
        push!(ttv_dims, dim)

        # Update ttv_rks
        push!(ttv_rks, rank_next)
    end

    N_qtt = length(qtt_cores)
    M = length(ttv_dims)
    ttv_ot = zeros(Int64, N_qtt)

    qtt_tensor = TTvector{T,M}(N_qtt, qtt_cores, Tuple(ttv_dims), ttv_rks, ttv_ot)

    return qtt_tensor
end

using LinearAlgebra

# Define the TTvector
dims = (4, 8, 16)
rks = [1, 2, 2, 1]
tt_vec = [randn(Float64, dims[i], rks[i], rks[i+1]) for i in 1:length(dims)]
tt = TTvector{Float64, 3}(3, tt_vec, dims, rks, zeros(Int64, 3))

# Define the dimensions for QTT
qtt_dims = [[2, 2], [2, 2, 2], [2, 2, 2, 2]]

# Convert TTvector to QTTvector
qtt = tt2qtt(tt, qtt_dims)

