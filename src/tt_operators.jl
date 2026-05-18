"""
Constructs a tensor train operator (TTO) representation of a Toeplitz matrix parameterized by `α`, `β`, and `γ` over `d` dimensions.
"""
function toeplitz_to_qtto(α, β, γ, d)
    out = zeros_tto(2, d, 3)
    id = Matrix{Float64}(I, 2, 2)
    J = zeros(2, 2)
    J[1, 2] = 1
    for i in 1:2
        for j in 1:2
            out.tto_vec[1][i, j, 1, :] = [id[i, j];J[j, i];J[i, j]]
            for k in 2:(d - 1)
                out.tto_vec[k][i, j, :, :] = [id[i, j] J[j, i] J[i, j]; 0 J[i, j] 0 ; 0 0 J[j, i]]
            end
            out.tto_vec[d][i, j, :, 1] = [α * id[i, j] + β * J[i, j] + γ * J[j, i]; γ * J[i, j] ; β * J[j, i]]
        end
    end
    return out
end

"""
Constructs a tensor train operator (TTO) representation of the shift matrix
"""
function shift(d::Int)
    return toeplitz_to_qtto(0, 1, 0, d)
end

"""
Constructs a centered first-difference TTO representing S₊₁ − S₋₁.
Multiply by `1/(2dx)` to obtain the centered first-derivative operator.
"""
function ∇_c(d::Int)
    return toeplitz_to_qtto(0, 1, -1, d)
end

"""
Constructs a centered third-difference TTO representing S₊₂ − 2S₊₁ + 2S₋₁ − S₋₂.
Multiply by `1/(2dx³)` to obtain the centered third-derivative operator.
Computed as `∇_c(d) * (S₊₁ + S₋₁ − 2I)`.
"""
function ∇3(d::Int)
    return ∇_c(d) * toeplitz_to_qtto(-2, 1, 1, d)
end

"""
Constructs a tensor train operator (TTO) representation of the gradient matrix
"""
function ∇(d::Int)
    return toeplitz_to_qtto(1, 0, -1, d)
end

# Rank-1 TTO whose matrix is ⊗_k M_k (same 2×2 matrix at every site).
function _periodic_wrap(d::Int, M::Matrix{Float64})
    out = zeros_tto(2, d, 1)
    for k in 1:d
        out.tto_vec[k][:, :, 1, 1] = M
    end
    return out
end

"""
Periodic centered first-difference: S₊₁ − S₋₁ with wrap-around.
Equivalent to ∇_c(d) plus the two corner corrections that make the shift operators circular.
Multiply by `1/(2dx)` to obtain the centered first-derivative operator.
"""
function ∇_c_P(d::Int)
    Jt = [0.0 0.0; 1.0 0.0]   # |eₙ⟩⟨e₁|: adds (row N, col 1) = S₊₁ wrap
    J  = [0.0 1.0; 0.0 0.0]   # |e₁⟩⟨eₙ|: adds (row 1, col N) = S₋₁ wrap
    return ∇_c(d) + _periodic_wrap(d, Jt) - _periodic_wrap(d, J)
end

"""
Periodic centered third-difference: S₊₂ − 2S₊₁ + 2S₋₁ − S₋₂ with wrap-around.
Computed as `∇_c_P(d) * (S₊₁_P + S₋₁_P − 2I)`.
Multiply by `1/(2dx³)` to obtain the centered third-derivative operator.
"""
function ∇3_P(d::Int)
    Jt = [0.0 0.0; 1.0 0.0]
    J  = [0.0 1.0; 0.0 0.0]
    T_P = toeplitz_to_qtto(-2, 1, 1, d) + _periodic_wrap(d, Jt) + _periodic_wrap(d, J)
    return ∇_c_P(d) * T_P
end

"""
Constructs a tensor train operator (TTO) representation of the Laplacian with Dirichlet-Dirichlet boundary conditions
"""
function Δ(d::Int)
    return toeplitz_to_qtto(2, -1, -1, d)
end

"""
Constructs a tensor train operator (TTO) representation of the Laplacian with Dirichlet-Neumann boundary conditions
"""
function Δ_DN(d::Int)
    @assert d ≥ 4 "Dimension must be at least 4"
    out = zeros_tto(2, d, 4)
    id = [1 0; 0 1]
    J = [0 1; 0 0]
    I₂ = [0 0; 0 1]
    for i in 1:2
        for j in 1:2
            out.tto_vec[1][i, j, 1, :] = [id[i, j]; J[j, i]; J[i, j]; I₂[i, j]]
            for k in 2:(d - 1)
                out.tto_vec[k][i, j, :, :] = [id[i, j] J[j, i] J[i, j] 0; 0 J[i, j] 0 0; 0 0 J[j, i] 0; 0 0 0 I₂[i, j]]
            end
            out.tto_vec[d][i, j, :, 1] = [2 * id[i, j] - J[i, j] - J[j, i]; -J[i, j]; -J[j, i]; -I₂[i, j]]
        end
    end
    return out
end

"""
Constructs a tensor train operator (TTO) representation of the Laplacian with Neumann-Dirichlet boundary conditions
"""
function Δ_ND(d::Int)
    @assert d ≥ 4 "Dimension must be at least 4"
    out = zeros_tto(2, d, 4)
    id = [1 0; 0 1]
    J = [0 1; 0 0]
    I₁ = [1 0; 0 0]
    for i in 1:2
        for j in 1:2
            out.tto_vec[1][i, j, 1, :] = [id[i, j]; J[j, i]; J[i, j]; I₁[i, j]]
            for k in 2:(d - 1)
                out.tto_vec[k][i, j, :, :] = [id[i, j] J[j, i] J[i, j] 0; 0 J[i, j] 0 0; 0 0 J[j, i] 0; 0 0 0 I₁[i, j]]
            end
            out.tto_vec[d][i, j, :, 1] = [2 * id[i, j] - J[i, j] - J[j, i]; -J[i, j]; -J[j, i]; -I₁[i, j]]
        end
    end
    return out
end

"""
Constructs a tensor train operator (TTO) representation of the Laplacian with Neumann-Neumann boundary conditions
"""
function Δ_NN(d)
    @assert d ≥ 4 "Dimension must be at least 4"
    out = zeros_tto(ntuple(_ -> 2, d), [1; fill(5, d - 1); 1])
    id = [1 0; 0 1]
    J = [0 1; 0 0]
    I₁ = [1 0; 0 0]
    I₂ = [0 0; 0 1]
    for i in 1:2
        for j in 1:2
            out.tto_vec[1][i, j, 1, :] = [id[i, j]; J[j, i]; J[i, j]; I₂[i, j]; I₁[i, j]]
            for k in 2:(d - 1)
                out.tto_vec[k][i, j, :, :] = [id[i, j] J[j, i] J[i, j] 0 0; 0 J[i, j] 0 0 0; 0 0 J[j, i] 0 0; 0 0 0 I₂[i, j] 0; 0 0 0 0 -I₁[i, j]]
            end
            out.tto_vec[d][i, j, :, 1] = [2 * id[i, j] - J[i, j] - J[j, i]; -J[i, j]; -J[j, i]; -I₂[i, j]; -I₁[i, j]]
        end
    end
    return out
end

"""
Constructs a tensor train operator (TTO) representation of the Laplacian with periodic boundary conditions
"""
function Δ_P(d)
    @assert d ≥ 4 "Dimension must be at least 4"
    out = zeros_tto(ntuple(_ -> 2, d), [1; fill(5, d - 1); 1])
    id = [1 0; 0 1]
    J = [0 1; 0 0]
    for i in 1:2
        for j in 1:2
            out.tto_vec[1][i, j, 1, :] = [id[i, j], J[j, i], J[i, j], J[i, j], J[j, i]]
            for k in 2:(d - 1)
                out.tto_vec[k][i, j, :, :] = [
                    id[i, j] J[j, i] J[i, j] 0 0;
                    0 J[i, j] 0 0 0;
                    0 0 J[j, i] 0 0;
                    0 0 0 J[i, j] 0;
                    0 0 0 0 J[j, i]
                ]
            end
            out.tto_vec[d][i, j, :, 1] = [
                2 * id[i, j] - J[i, j] - J[j, i];
                -J[i, j];
                -J[j, i];
                -J[i, j];
                -J[j, i]
            ]
        end
    end
    return out
end

"""
Constructs a tensor train operator (TTO) representation of the inverse Laplacian with Dirichlet-Neumann boundary conditions
"""
function Δ⁻¹_DN(d::Int)
    @assert d ≥ 2 "Dimension must be at least 2"
    out = zeros_tto(2, d, 4)
    id = [1 0; 0 1]
    E = [1 1; 1 1]
    I₂ = [0 0; 0 1]
    J = [0 1; 0 0]
    for i in 1:2
        for j in 1:2
            out.tto_vec[1][i, j, 1, :] = [id[i, j]; I₂[i, j]; J[i, j]; J[j, i]]
            for k in 2:(d - 1)
                out.tto_vec[k][i, j, :, :] = [
                    id[i, j] I₂[i, j] J[i, j] J[j, i];
                    0 2 * E[i, j] 0 0;
                    0 I₂[i, j] + J[j, i] E[i, j] 0;
                    0 I₂[i, j] + J[i, j] 0 E[i, j];
                ]
            end
            out.tto_vec[d][i, j, :, 1] = [
                E[i, j] + I₂[i, j];
                2 * E[i, j];
                E[i, j] + I₂[i, j] + J[j, i];
                E[i, j] + I₂[i, j] + J[i, j]
            ]
        end
    end
    return out
end

"""
Constructs a tensor train operator (TTO) representation of the prolongation operator for multigrid methods
"""
function qtto_prolongation(d::Int)
    @assert d ≥ 2 "Dimension must be at least 2"
    out = zeros_tto(2, d, 2)
    id = [1.0 0.0; 0.0 1.0]
    J = [0.0 1.0; 0.0 0.0]
    for i in 1:2
        for j in 1:2
            out.tto_vec[1][i, j, 1, :] = 0.5 * [id[i, j]; J[j, i]]
            for k in 2:(d - 1)
                out.tto_vec[k][i, j, :, :] = [id[i, j] J[j, i]; 0 J[i, j]]
            end
        end
    end
    out.tto_vec[d][1, 1, 1, 1] = 1.0
    out.tto_vec[d][2, 1, 1, 1] = 2.0
    out.tto_vec[d][1, 2, 1, 1] = 1.0
    out.tto_vec[d][2, 2, 1, 1] = 0.0
    return out
end


"""
    id_tto(d; n_dim=2)

Create an identity tensor train operator (TTO) of dimension `d` with optional keyword argument `n_dim` specifying the number of dimensions (default is 2).

# Arguments
- `d::Int`: The dimension of the identity tensor train operator.
- `n_dim::Int`: The number of dimensions of the identity tensor train operator (default is 2).

# Returns
- An identity tensor train operator of the specified dimension and number of dimensions.
"""
function id_tto(d; n_dim = 2)
    return id_tto(Float64, d; n_dim = n_dim)
end


function id_tto(::Type{T}, d; n_dim = 2) where {T}
    dims = Tuple(n_dim * ones(Int64, d))
    A = Array{Array{T, 4}, 1}(undef, d)
    for j in 1:d
        A[j] = zeros(T, 2, 2, 1, 1)
        A[j][:, :, 1, 1] = Matrix{T}(I, 2, 2)
    end
    return TToperator{T, d}(d, A, dims, ones(Int64, d + 1), zeros(d))
end

function rand_tto(dims, rmax::Int; T = Float64)
    d = length(dims)
    tt_vec = Vector{Array{T, 4}}(undef, d)
    rks = ones(Int, d + 1)
    for i in eachindex(tt_vec)
        ri = min(prod(dims[1:(i - 1)]), prod(dims[i:d]), rmax)
        rip = min(prod(dims[1:i]), prod(dims[(i + 1):d]), rmax)
        rks[i + 1] = rip
        tt_vec[i] = randn(T, dims[i], dims[i], ri, rip)
    end
    return TToperator{T, d}(d, tt_vec, dims, rks, zeros(Int, d))
end


function zeros_tt(dims, rks; ot = zeros(Int64, length(dims)))
    return zeros_tt(Float64, dims, rks; ot = ot)
end

function zeros_tt(::Type{T}, dims::NTuple{N, Int64}, rks; ot = zeros(Int64, length(dims))) where {T, N}
    @assert length(dims) + 1 == length(rks) "Dimensions and ranks are not compatible"
    tt_vec = [zeros(T, dims[i], rks[i], rks[i + 1]) for i in eachindex(dims)]
    return TTvector{T, N}(N, tt_vec, dims, deepcopy(rks), deepcopy(ot))
end

function zeros_tt(n::Integer, d::Integer, r; ot = zeros(Int64, d), r_and_d = true)
    dims = ntuple(x -> n, d)
    if r_and_d
        rks = r_and_d_to_rks(r * ones(Int64, d + 1), dims)
    else
        rks = r * ones(Int64, d + 1)
        rks[1], rks[end] = 1, 1
    end
    return zeros_tt(Float64, dims, rks; ot = ot)
end

function zeros_tt(::Type{T}, dims::Vector{Int}, rks::Vector{Int}; ot = zeros(Int64, length(dims))) where {T}
    return zeros_tt(T, Tuple(dims), rks; ot = ot)
end

function zeros_tt(::Type{T}, dims::NTuple{N, Int64}, rks::NTuple{M, Int64}; ot = zeros(Int64, length(dims))) where {T, N, M}
    return zeros_tt(T, collect(Int64, dims), collect(Int64, rks); ot = ot)
end

function zeros_tt!(A::TTvector)
    @assert isa(A.ttv_vec, Vector)
    for core in A.ttv_vec
        fill!(core, zero(eltype(core)))
    end
    return A
end

function ones_tt(dims)
    return ones_tt(Float64, dims)
end

function ones_tt(::Type{T}, dims) where {T}
    tdims = Tuple(dims)
    N = length(tdims)
    vec = [ones(T, n, 1, 1) for n in tdims]
    rks = ones(Int64, N + 1)
    ot = zeros(Int64, N)
    return TTvector{T, N}(N, vec, tdims, rks, ot)
end

function ones_tt(n::Integer, d::Integer)
    return ones_tt(ntuple(_ -> Int64(n), d))
end


function zeros_tto(dims, rks)
    return zeros_tto(Float64, dims, rks)
end

function zeros_tto(::Type{T}, dims::NTuple{N, Int64}, rks) where {T, N}
    @assert length(dims) + 1 == length(rks) "Dimensions and ranks are not compatible"
    vec = [zeros(T, dims[i], dims[i], rks[i], rks[i + 1]) for i in eachindex(dims)]
    return TToperator{T, N}(N, vec, dims, rks, zeros(Int64, N))
end

function zeros_tto(n, d, r)
    dims = ntuple(x -> n, d)
    rks = r * ones(Int64, d + 1)
    rks = r_and_d_to_rks(rks, dims .^ 2; rmax = r)
    return zeros_tto(Float64, dims, rks)
end
