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

