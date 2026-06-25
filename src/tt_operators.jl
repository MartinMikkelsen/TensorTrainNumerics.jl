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
Constructs a tensor train operator (TTO) representation of the gradient matrix
"""
function ∇(d::Int)
    return toeplitz_to_qtto(1, 0, -1, d)
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
    out = zeros_tto(ntuple(_ -> 2, d), [4; fill(5, d - 1); 4])
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
    out = zeros_tto(ntuple(_ -> 2, d), fill(5, d + 1))
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
Constructs a constant QTT prolongation operator from `d` to `d + 1` binary sites.
"""
function qtto_constant_prolongation(d::Int)
    @assert d ≥ 1 "Dimension must be at least 1"

    identity_branch = id_tto(d)
    out = Vector{Array{Float64, 4}}(undef, d + 1)
    @inbounds for k in 1:d
        out[k] = copy(identity_branch.tto_vec[k])
    end
    out[d + 1] = ones(Float64, 2, 1, 1, 1)

    return TToperator{Float64, d + 1}(
        d + 1,
        out,
        ntuple(_ -> 2, d + 1),
        ones(Int64, d + 2),
        zeros(Int64, d + 1)
    )
end

"""
Constructs a linear QTT prolongation operator from `d` to `d + 1` binary sites.
"""
function qtto_linear_prolongation(d::Int)
    @assert d ≥ 1 "Dimension must be at least 1"

    identity_branch = id_tto(d)
    if d == 1
        average_core = zeros(Float64, 2, 2, 1, 1)
        average_core[:, :, 1, 1] .= 0.5 .* [1.0 1.0; 0.0 1.0]
        average_branch = TToperator{Float64, 1}(1, [average_core], (2,), [1, 1], [0])
    else
        average_branch = 0.5 * (id_tto(d) + shift(d))
    end
    out_rks = Vector{Int64}(undef, d + 2)
    out_rks[1] = 1
    @inbounds for k in 2:(d + 1)
        out_rks[k] = identity_branch.tto_rks[k] + average_branch.tto_rks[k]
    end
    out_rks[d + 2] = 1

    out = Vector{Array{Float64, 4}}(undef, d + 1)
    out[1] = zeros(Float64, 2, 2, 1, out_rks[2])
    r₀ = identity_branch.tto_rks[2]
    out[1][:, :, 1:1, 1:r₀] .= identity_branch.tto_vec[1]
    out[1][:, :, 1:1, (r₀ + 1):out_rks[2]] .= average_branch.tto_vec[1]

    @inbounds for k in 2:d
        l₀ = identity_branch.tto_rks[k]
        r₀ = identity_branch.tto_rks[k + 1]
        l₁ = average_branch.tto_rks[k]
        r₁ = average_branch.tto_rks[k + 1]
        out[k] = zeros(Float64, 2, 2, out_rks[k], out_rks[k + 1])
        out[k][:, :, 1:l₀, 1:r₀] .= identity_branch.tto_vec[k]
        out[k][:, :, (l₀ + 1):(l₀ + l₁), (r₀ + 1):(r₀ + r₁)] .= average_branch.tto_vec[k]
    end

    l₀ = identity_branch.tto_rks[d + 1]
    l₁ = average_branch.tto_rks[d + 1]
    out[d + 1] = zeros(Float64, 2, 1, out_rks[d + 1], 1)
    out[d + 1][1, 1, 1:l₀, 1] .= 1.0
    out[d + 1][2, 1, (l₀ + 1):(l₀ + l₁), 1] .= 1.0

    return TToperator{Float64, d + 1}(d + 1, out, ntuple(_ -> 2, d + 1), out_rks, zeros(Int64, d + 1))
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
    rks_vec = collect(Int64, rks)
    ot_vec = collect(Int64, ot)
    return TTvector{T, N}(N, tt_vec, dims, rks_vec, ot_vec)
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
    return zeros_tt(T, Tuple(dims), Tuple(rks); ot = ot)
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
    N = length(dims)
    vec = [ones(T, n, 1, 1) for n in dims]
    rks = ones(Int64, N + 1)
    ot = zeros(Int64, N)
    return TTvector{T, N}(N, vec, dims, rks, ot)
end

function ones_tt(n::Integer, d::Integer)
    dims = n * ones(Int64, d)
    return ones_tt(dims)
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

"""
    qtt_laplacian(n_dims, bits_per_dim; ordering=:interleaved, a=0.0, b=1.0, bc=:DN)

Build the `n_dims`-dimensional Laplacian operator in QTT format as the Kronecker sum
of 1D second-derivative operators:

    Δ_nd = Δ₁⊗I⊗…⊗I + I⊗Δ₂⊗I⊗…⊗I + … + I⊗…⊗I⊗Δₙ

Each 1D operator acts on `bits_per_dim` sites (a uniform grid of `2^bits_per_dim`
points over `[a, b]`). The finite-difference scaling `1/h²` is included.

# Arguments
- `n_dims::Int`: Number of spatial dimensions (≥ 1).
- `bits_per_dim::Int`: Number of QTT bits per dimension.

# Keyword arguments
- `ordering::Symbol`: `:serial` (sites grouped by dimension) or `:interleaved`
  (sites interleaved across dimensions). Default: `:interleaved`.
- `a::Real`, `b::Real`: Interval endpoints. Default: `[0, 1]`.
- `bc::Symbol`: Boundary conditions — `:DD` (Dirichlet–Dirichlet), `:DN`
  (Dirichlet–Neumann), `:ND` (Neumann–Dirichlet), `:NN` (Neumann–Neumann).
  Default: `:DN`.

# Returns
A `QTToperator` with `N = n_dims * bits_per_dim` sites.
"""
function qtt_laplacian(
        n_dims::Int, bits_per_dim::Int;
        ordering::Symbol = :interleaved, a::Real = 0.0, b::Real = 1.0,
        bc::Symbol = :DN
    )
    @assert ordering ∈ (:interleaved, :serial) "ordering must be :interleaved or :serial"
    @assert n_dims ≥ 1 "n_dims must be at least 1"
    @assert bc ∈ (:DD, :DN, :ND, :NN) "bc must be :DD, :DN, :ND, or :NN"
    @assert !(bc == :NN && n_dims > 1) "bc=:NN is only supported for n_dims=1 (the Δ_NN MPO has non-unit boundary ranks, which are incompatible with the TToperator Kronecker sum)"

    d = bits_per_dim
    h = (b - a) / (2^d - 1)
    scale = 1.0 / h^2

    # Select 1D Laplacian with correct boundary conditions
    lap_1d = if bc == :DD
        Δ(d)
    elseif bc == :DN
        Δ_DN(d)
    elseif bc == :ND
        Δ_ND(d)
    else  # :NN
        Δ_NN(d)
    end

    id_1d = id_tto(d)

    if n_dims == 1
        # Single dimension: just scale the 1D Laplacian
        scaled = scale * lap_1d
        return QTToperator(scaled, 1, d, ordering)
    end

    # For n_dims ≥ 2: build Kronecker sum in serial ordering.
    # Term k: I ⊗ … ⊗ Δ_k ⊗ … ⊗ I
    # kron(A, B) concatenates TT cores (= Kronecker product of operators on disjoint sites)
    function build_term(k::Int)
        # Sites for dim 1..k-1: identity; sites for dim k: Δ; sites for dim k+1..n: identity
        ops = [dim == k ? lap_1d : id_1d for dim in 1:n_dims]
        term = ops[1]
        for dim in 2:n_dims
            term = kron(term, ops[dim])
        end
        return term
    end

    # Sum all n_dims terms (with h² scaling on the first term to avoid repeated scaling)
    result = scale * build_term(1)
    for k in 2:n_dims
        result = result + (scale * build_term(k))
    end

    serial_qtto = QTToperator(result, n_dims, d, :serial)

    if ordering == :serial
        return serial_qtto
    else  # :interleaved — reorder from serial to interleaved
        return reorder(serial_qtto, :interleaved)
    end
end
