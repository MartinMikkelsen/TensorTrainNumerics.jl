using LinearAlgebra
using FastGaussQuadrature

"""
    chebyshev_lobatto_nodes(N::Int) -> Vector{Float64}

Generate Chebyshev-Lobatto nodes.

# Arguments
- `N::Int`: The number of intervals, which results in `N+1` nodes.

# Returns
- `Vector{Float64}`: A vector containing the Chebyshev-Lobatto nodes scaled to the interval [0, 1].
"""
function chebyshev_lobatto_nodes(N::Int)
    nodes = (cos.(π * (0:N) / N) .+ 1) ./ 2
    return nodes
end

function gauss_chebyshev_lobatto(n; shifted = true)
    x = [cos(π * j / (n - 1)) for j in 0:(n - 1)]
    w = π / (n - 1) * ones(n)
    w[1] /= 2
    w[end] /= 2

    if shifted
        x .= (x .+ 1) ./ 2
        w .= w ./ 2
    end

    return x, w
end

"""
    equally_spaced_nodes(N::Int) -> Vector{Float64}

Generate a vector of `N + 1` equally spaced nodes in the interval [0, 1].

# Arguments
- `N::Int`: The number of intervals between the nodes.

# Returns
- `Vector{Float64}`: A vector containing `N + 1` equally spaced nodes in the interval [0, 1].
"""
function equally_spaced_nodes(N::Int)
    return collect(range(0, 1, length = N + 1))
end

"""
    legendre_nodes(N::Int) -> Vector{Float64}

Compute the Legendre nodes for Gauss-Legendre quadrature.

# Arguments
- `N::Int`: The number of Legendre nodes to compute.

# Returns
- `Vector{Float64}`: A vector containing the Legendre nodes scaled to the interval [0, 1].
"""
function legendre_nodes(N::Int)
    nodes, _ = gausslegendre(N)
    nodes = (nodes .+ 1) ./ 2
    return nodes
end

"""
    get_nodes(N::Int, node_type::String="chebyshev")

Generate interpolation nodes based on the specified type.

# Arguments
- `N::Int`: The number of nodes to generate.
- `node_type::String`: The type of nodes to generate. Supported values are:
  - `"chebyshev"`: Chebyshev-Lobatto nodes.
  - `"equally_spaced"`: Equally spaced nodes.
  - `"legendre"`: Legendre nodes.

# Returns
- An array of nodes corresponding to the specified type.

# Throws
- `ArgumentError`: If an unknown `node_type` is provided.
"""
function get_nodes(N::Int, node_type::String = "chebyshev")
    if node_type == "chebyshev"
        return chebyshev_lobatto_nodes(N)
    elseif node_type == "equally_spaced"
        return equally_spaced_nodes(N)
    elseif node_type == "legendre"
        return legendre_nodes(N)
    else
        error("Unknown node type: $node_type")
    end
end

"""
    lagrange_basis(nodes::Vector{Float64}, x::Float64, j::Int) -> Float64

Compute the Lagrange basis polynomial `L_j(x)` at a given point `x` for a set of interpolation nodes.

# Arguments
- `nodes::Vector{Float64}`: A vector containing the interpolation nodes.
- `x::Float64`: The point at which to evaluate the Lagrange basis polynomial.
- `j::Int`: The index of the Lagrange basis polynomial to compute.

# Returns
- `Float64`: The value of the Lagrange basis polynomial `L_j(x)` at the point `x`.
"""
function lagrange_basis(nodes::Vector{Float64}, x::Float64, j::Int)
    N = length(nodes) - 1
    result = 1.0
    for k in 0:N
        if k != j
            denominator = nodes[j + 1] - nodes[k + 1]
            if denominator != 0
                result *= (x - nodes[k + 1]) / denominator
            end
        end
    end
    return result
end

"""
    lagrange_basis(nodes::Vector{Float64}, x::Vector{Float64}, j::Int) -> Vector{Float64}

Compute the Lagrange basis polynomial `L_j(x)` at the points `x` for the given set of interpolation `nodes`.

# Arguments
- `nodes::Vector{Float64}`: A vector of interpolation nodes.
- `x::Vector{Float64}`: A vector of points at which to evaluate the Lagrange basis polynomial.
- `j::Int`: The index of the Lagrange basis polynomial to compute.

# Returns
- `Vector{Float64}`: A vector containing the values of the Lagrange basis polynomial `L_j(x)` evaluated at the points `x`.
"""
function lagrange_basis(nodes::Vector{Float64}, x::Vector{Float64}, j::Int)
    N = length(nodes) - 1
    result = ones(Float64, length(x))
    for k in 0:N
        if k != j
            denominator = nodes[j + 1] - nodes[k + 1]
            if denominator != 0
                result .= result .* ((x .- nodes[k + 1]) ./ denominator)
            end
        end
    end
    return result
end

function A_L(func::Function, nodes::Vector{Float64}, start::Float64 = 0.0, stop::Float64 = 1.0)
    D = length(nodes)
    A = zeros(Float64, 1, 2, 1, D)
    for σ in 0:1
        A[1, σ + 1, 1, :] = func.(0.5 .* (σ .+ nodes) .* (stop - start) .+ start)
    end
    return A
end

function A_C(nodes::Vector{Float64})
    D = length(nodes)
    A = zeros(Float64, D, 2, 1, D)
    for σ in 0:1
        x = 0.5 * (σ .+ nodes)
        for α in 1:D
            A[α, σ + 1, 1, :] = lagrange_basis(nodes, x, α - 1)
        end
    end
    return A
end

function A_R(nodes::Vector{Float64})
    D = length(nodes)
    A = zeros(Float64, D, 2, 1, 1)
    for σ in 0:1
        x = 0.5 * σ
        for α in 1:D
            A[α, σ + 1, 1, 1] = lagrange_basis(nodes, x, α - 1)
        end
    end
    return A
end

function A_L_x(func::Function, nodes::Vector{Float64}, start::Float64 = 0.0, stop::Float64 = 1.0)
    D = length(nodes)
    A = zeros(Float64, 1, 2, 1, D^2)

    for σ in 0:1
        x_val = (0.5 .* (σ .+ nodes)) .* (stop - start) .+ start
        y_val = nodes .* (stop - start) .+ start

        values = Float64[]
        for x in x_val
            for y in y_val
                push!(values, func(y, x))
            end
        end

        A[1, σ + 1, 1, :] = reshape(values, D^2)
    end

    return A
end

function A_C_y(nodes::Vector{Float64})
    D = length(nodes)
    A = zeros(Float64, D^2, 2, 1, D^2)
    A_c = A_C(nodes)
    for σ in 0:1
        kron_res = kron(Matrix{Float64}(I, D, D), A_c[:, σ + 1, 1, :])
        A[:, σ + 1, 1, :] = kron_res
    end
    return A
end

function A_C_x(nodes::Vector{Float64})
    D = length(nodes)
    A = zeros(Float64, D^2, 2, 1, D^2)
    A_c = A_C(nodes)
    for σ in 0:1
        kron_res = kron(A_c[:, σ + 1, 1, :], Matrix{Float64}(I, D, D))
        A[:, σ + 1, 1, :] = kron_res
    end
    return A
end

function A_R_x(nodes::Vector{Float64})
    D = length(nodes)
    A = zeros(Float64, D^2, 2, 1, D)
    A_r = A_R(nodes)
    for σ in 0:1
        kron_res = kron(A_r[:, σ + 1, 1, :], Matrix{Float64}(I, D, D))
        A[:, σ + 1, 1, :] = kron_res
    end
    return A
end

function A_R_y(nodes::Vector{Float64})
    return A_R(nodes)
end

"""
    interpolating_qtt(func::Function, core::Int, N::Int; node_type::String="chebyshev", start::Float64=0.0, stop::Float64=1.0)

Constructs an interpolating Quantized Tensor Train (QTT) for a given function.

# Arguments
- `func::Function`: The function to be interpolated.
- `core::Int`: The number of cores in the QTT.
- `N::Int`: The number of nodes for interpolation.
- `node_type::String`: The type of nodes to use for interpolation. Default is `"chebyshev"`.
- `start::Float64`: The start of the interval for interpolation. Default is `0.0`.
- `stop::Float64`: The end of the interval for interpolation. Default is `1.0`.

# Returns
- `tn`: A `TTvector` representing the interpolated QTT.

# Description
This function constructs an interpolating QTT by first generating the interpolation nodes and constructing the corresponding tensors. The tensors are then reshaped and permuted to fit the TTvector format. The resulting TTvector is returned.
"""
function interpolating_qtt(
        func::Function, core::Int, N::Int; node_type::String = "chebyshev",
        start::Float64 = 0.0, stop::Float64 = 1.0
    )
    nodes = get_nodes(N, node_type)
    Al = A_L(func, nodes, start, stop)
    Ac = A_C(nodes)
    Ar = A_R(nodes)

    tensors = [Al]
    for _ in 1:(core - 2)
        push!(tensors, Ac)
    end
    push!(tensors, Ar)

    # Convert tensors to TTvector format
    N_ = length(tensors)
    ttv_vec = Vector{Array{Float64, 3}}()
    ttv_rks = Int[]
    ttv_rks = [1]
    for i in 1:N_
        # Reshape tensors to (n_i, r_{i-1}, r_i)
        T = tensors[i]
        n_i = size(T, 2)
        r_prev = size(T, 1) * size(T, 3)
        r_next = size(T, 4)
        # Permute dimensions to (μ_i, α_{i-1}, α_i)
        T = permutedims(T, (2, 1, 3, 4))
        T = reshape(T, n_i, r_prev, r_next)
        push!(ttv_vec, T)
        push!(ttv_rks, r_next)
    end
    ttv_dims = ntuple(i -> size(ttv_vec[i], 1), N_)
    ttv_ot = zeros(Int, N_)
    tn = TTvector{Float64, N_}(N_, ttv_vec, ttv_dims, ttv_rks, ttv_ot)
    return tn
end

"""
    lagrange_rank_revealing(func::Function, core::Int, N::Int)

Perform Lagrange rank-revealing tensor train (TT) decomposition.

# Arguments
- `func::Function`: The function to be approximated.
- `core::Int`: The number of cores in the TT decomposition.
- `N::Int`: The number of Chebyshev-Lobatto nodes.

# Returns
- `tn::TTvector{Float64, N_}`: The resulting tensor train vector.

# Description
This function performs a rank-revealing TT decomposition using Lagrange interpolation. It constructs the TT cores by performing QR decomposition and singular value decomposition (SVD) on intermediate matrices. The numerical rank is determined based on a threshold, and the resulting TT cores are truncated accordingly.

# Details
1. Compute Chebyshev-Lobatto nodes.
2. Construct the first core using QR decomposition.
3. For intermediate cores, perform SVD and truncate based on numerical rank.
4. Construct the last core.
5. Convert the list of TT cores to a `TTvector` format.
"""
function lagrange_rank_revealing(func::Function, core::Int, N::Int)
    nodes = chebyshev_lobatto_nodes(N)
    tensors = []

    # First core
    AL = A_L(func, nodes)
    AL_mat = reshape(AL, 2, N + 1)  # AL is of size (1, 2, 1, N+1), reshaped to (2, N+1)

    # Perform QR decomposition and extract Q and R
    qr_result = qr(AL_mat)
    U = Matrix(qr_result.Q)  # Convert Q to a full matrix
    R = qr_result.R

    # Reshape U to match expected dimensions
    U = reshape(U, 1, size(U, 1), 1, size(U, 2))  # U is (1, 2, 1, N_r)
    push!(tensors, U)

    count_zero = false
    # Intermediate cores
    for d in 2:(core - 1)
        Ak = A_C(nodes)
        Ak_mat = reshape(Ak, N + 1, :)
        B = R * Ak_mat  # B is of size (size(R,1), size(Ak_mat,2))

        # Reshape B to (r_prev, 2, N+1) for SVD
        B = reshape(B, size(B, 1), 2, N + 1)
        B_mat = reshape(B, size(B, 1) * 2, N + 1)  # Flatten for SVD

        # Perform SVD
        U_svd, S, V = svd(B_mat, full = false)
        D = sum(S .> 1.0e-10)  # Numerical rank based on threshold

        # Truncate U_svd, S, V based on rank D
        U_svd = U_svd[:, 1:D]
        S = S[1:D]
        V = V[:, 1:D]

        # Update R for next iteration
        R = Diagonal(S) * V'

        if D > 1
            # Reshape U_svd to (r_prev, 2, 1, D)
            U = reshape(U_svd, size(B, 1), 2, 1, D)
            push!(tensors, U)
        else
            push!(tensors, zeros(Float64, size(B, 1), 2, 1, D))
            count_zero = true
        end
    end

    # Last core
    Ar = A_R(nodes)
    Ar_mat = reshape(Ar, N + 1, 2)
    UR = R * Ar_mat
    if !count_zero
        UR = reshape(UR, size(UR, 1), 2, 1, 1)
        push!(tensors, UR)
    else
        push!(tensors, zeros(Float64, size(UR, 1), 2, 1, 1))
    end

    # Convert tensors to TTvector format
    N_ = length(tensors)
    ttv_vec = Vector{Array{Float64, 3}}()
    ttv_rks = [1]
    for i in 1:N_
        T = tensors[i]
        n_i = size(T, 2)
        r_prev = size(T, 1) * size(T, 3)
        r_next = size(T, 4)
        # Permute dimensions to (μ_i, α_{i-1}, α_i)
        T = permutedims(T, (2, 1, 3, 4))
        T = reshape(T, n_i, r_prev, r_next)
        push!(ttv_vec, T)
        push!(ttv_rks, r_next)
    end
    ttv_dims = ntuple(i -> size(ttv_vec[i], 1), N_)
    ttv_ot = zeros(Int, N_)
    tn = TTvector{Float64, N_}(N_, ttv_vec, ttv_dims, ttv_rks, ttv_ot)
    return tn
end
