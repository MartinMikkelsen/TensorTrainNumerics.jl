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

function _pauli_axis(μ)
    axis = lowercase(string(μ))
    if axis == "x"
        return :x
    elseif axis == "y"
        return :y
    elseif axis == "z"
        return :z
    end
    throw(ArgumentError("Pauli axis must be :x, :y, or :z"))
end

"""
    pauli_matrix(μ)

Return the Pauli matrix for axis `μ`, where `μ` is `:x`, `:y`, or `:z`.
"""
function pauli_matrix(μ)
    axis = _pauli_axis(μ)
    if axis == :x
        return [0.0 1.0; 1.0 0.0]
    elseif axis == :y
        return ComplexF64[0.0 -im; im 0.0]
    else
        return [1.0 0.0; 0.0 -1.0]
    end
end

function _pauli_pair_factors(μ, ν)
    axisμ = _pauli_axis(μ)
    axisν = _pauli_axis(ν)
    if axisμ == :y && axisν == :y
        y_real = [0.0 -1.0; 1.0 0.0]
        return -y_real, y_real
    end
    return pauli_matrix(axisμ), pauli_matrix(axisν)
end

"""
    pauli_sum_tto(μ, d)

Construct the rank-2 TT operator

    H_μ = sum_i I ⊗ ... ⊗ P_μ ⊗ ... ⊗ I

on `d` spin-1/2 sites with open boundaries.
"""
function pauli_sum_tto(μ, d::Int)
    @assert d ≥ 1 "number of spin sites must be at least 1"

    P = pauli_matrix(μ)
    T = eltype(P)
    id = Matrix{T}(I, 2, 2)
    dims = ntuple(_ -> 2, d)

    if d == 1
        return TToperator{T, 1}(1, [reshape(P, 2, 2, 1, 1)], dims, [1, 1], zeros(Int64, 1))
    end

    rks = vcat(1, fill(2, d - 1), 1)
    cores = Array{Array{T, 4}, 1}(undef, d)

    cores[1] = zeros(T, 2, 2, 1, 2)
    cores[1][:, :, 1, 1] = P
    cores[1][:, :, 1, 2] = id

    @inbounds for k in 2:(d - 1)
        core = zeros(T, 2, 2, 2, 2)
        core[:, :, 1, 1] = id
        core[:, :, 2, 1] = P
        core[:, :, 2, 2] = id
        cores[k] = core
    end

    cores[d] = zeros(T, 2, 2, 2, 1)
    cores[d][:, :, 1, 1] = id
    cores[d][:, :, 2, 1] = P

    return TToperator{T, d}(d, cores, dims, rks, zeros(Int64, d))
end

"""
    pauli_pair_sum_tto(μ, ν, d)

Construct the rank-3 nearest-neighbor TT operator

    H_{μ,ν} = sum_i I ⊗ ... ⊗ P_μ ⊗ P_ν ⊗ ... ⊗ I

on `d` spin-1/2 sites with open boundaries.
"""
function pauli_pair_sum_tto(μ, ν, d::Int)
    @assert d ≥ 2 "nearest-neighbor Pauli pair sum needs at least 2 spin sites"

    Pμ_raw, Pν_raw = _pauli_pair_factors(μ, ν)
    T = promote_type(eltype(Pμ_raw), eltype(Pν_raw))
    Pμ = convert.(T, Pμ_raw)
    Pν = convert.(T, Pν_raw)
    id = Matrix{T}(I, 2, 2)
    dims = ntuple(_ -> 2, d)
    rks = vcat(1, fill(3, d - 1), 1)
    cores = Array{Array{T, 4}, 1}(undef, d)

    cores[1] = zeros(T, 2, 2, 1, 3)
    cores[1][:, :, 1, 2] = Pμ
    cores[1][:, :, 1, 3] = id

    @inbounds for k in 2:(d - 1)
        core = zeros(T, 2, 2, 3, 3)
        core[:, :, 1, 1] = id
        core[:, :, 2, 1] = Pν
        core[:, :, 3, 2] = Pμ
        core[:, :, 3, 3] = id
        cores[k] = core
    end

    cores[d] = zeros(T, 2, 2, 3, 1)
    cores[d][:, :, 1, 1] = id
    cores[d][:, :, 2, 1] = Pν

    return TToperator{T, d}(d, cores, dims, rks, zeros(Int64, d))
end

H_μ(μ, d::Int) = pauli_sum_tto(μ, d)
H_μν(μ, ν, d::Int) = pauli_pair_sum_tto(μ, ν, d)

"""
    heisenberg_xyz_tto(d; jx=1.0, jy=1.0, jz=1.0, λ=0.0, field=:x)

Construct the open-boundary Heisenberg XYZ Hamiltonian

    H = jx H_{x,x} + jy H_{y,y} + jz H_{z,z} + λ H_field

as a direct low-rank TT operator on `d` spin-1/2 sites.
"""
function heisenberg_xyz_tto(d::Int; jx = 1.0, jy = 1.0, jz = 1.0, λ = 0.0, field = :x)
    @assert d ≥ 2 "Heisenberg XYZ chain needs at least 2 spin sites"

    Px1_raw, Px2_raw = _pauli_pair_factors(:x, :x)
    Py1_raw, Py2_raw = _pauli_pair_factors(:y, :y)
    Pz1_raw, Pz2_raw = _pauli_pair_factors(:z, :z)
    Pf_raw = pauli_matrix(field)

    T = promote_type(
        typeof(jx), typeof(jy), typeof(jz), typeof(λ),
        eltype(Px1_raw), eltype(Px2_raw),
        eltype(Py1_raw), eltype(Py2_raw),
        eltype(Pz1_raw), eltype(Pz2_raw),
        iszero(λ) ? Float64 : eltype(Pf_raw),
    )

    jxT, jyT, jzT, λT = convert(T, jx), convert(T, jy), convert(T, jz), convert(T, λ)
    Px1, Px2 = convert.(T, Px1_raw), convert.(T, Px2_raw)
    Py1, Py2 = convert.(T, Py1_raw), convert.(T, Py2_raw)
    Pz1, Pz2 = convert.(T, Pz1_raw), convert.(T, Pz2_raw)
    Pf = convert.(T, Pf_raw)
    id = Matrix{T}(I, 2, 2)

    dims = ntuple(_ -> 2, d)
    rks = vcat(1, fill(5, d - 1), 1)
    cores = Array{Array{T, 4}, 1}(undef, d)

    cores[1] = zeros(T, 2, 2, 1, 5)
    cores[1][:, :, 1, 1] = λT * Pf
    cores[1][:, :, 1, 2] = jxT * Px1
    cores[1][:, :, 1, 3] = jyT * Py1
    cores[1][:, :, 1, 4] = jzT * Pz1
    cores[1][:, :, 1, 5] = id

    @inbounds for k in 2:(d - 1)
        core = zeros(T, 2, 2, 5, 5)
        core[:, :, 1, 1] = id
        core[:, :, 2, 1] = Px2
        core[:, :, 3, 1] = Py2
        core[:, :, 4, 1] = Pz2
        core[:, :, 5, 1] = λT * Pf
        core[:, :, 5, 2] = jxT * Px1
        core[:, :, 5, 3] = jyT * Py1
        core[:, :, 5, 4] = jzT * Pz1
        core[:, :, 5, 5] = id
        cores[k] = core
    end

    cores[d] = zeros(T, 2, 2, 5, 1)
    cores[d][:, :, 1, 1] = id
    cores[d][:, :, 2, 1] = Px2
    cores[d][:, :, 3, 1] = Py2
    cores[d][:, :, 4, 1] = Pz2
    cores[d][:, :, 5, 1] = λT * Pf

    return TToperator{T, d}(d, cores, dims, rks, zeros(Int64, d))
end

"""
    ising_tto(d; J=1.0, h=0.0, interaction=:z, field=:x)

Construct the open-boundary Ising Hamiltonian

    H = J H_{interaction,interaction} + h H_field

on `d` spin-1/2 sites. Coefficients are used with their given sign.
"""
function ising_tto(d::Int; J = 1.0, h = 0.0, interaction = :z, field = :x)
    axis = _pauli_axis(interaction)
    if axis == :x
        return heisenberg_xyz_tto(d; jx = J, jy = zero(J), jz = zero(J), λ = h, field = field)
    elseif axis == :y
        return heisenberg_xyz_tto(d; jx = zero(J), jy = J, jz = zero(J), λ = h, field = field)
    else
        return heisenberg_xyz_tto(d; jx = zero(J), jy = zero(J), jz = J, λ = h, field = field)
    end
end

"""
    xxz_tto(d; J=1.0, Δ=1.0, h=0.0, field=:z)

Construct the open-boundary XXZ Hamiltonian

    H = J(H_{x,x} + H_{y,y}) + JΔ H_{z,z} + h H_field.
"""
function xxz_tto(d::Int; J = 1.0, Δ = 1.0, h = 0.0, field = :z)
    return heisenberg_xyz_tto(d; jx = J, jy = J, jz = J * Δ, λ = h, field = field)
end

"""
    xxx_tto(d; J=1.0, h=0.0, field=:z)

Construct the open-boundary isotropic Heisenberg XXX Hamiltonian

    H = J(H_{x,x} + H_{y,y} + H_{z,z}) + h H_field.
"""
function xxx_tto(d::Int; J = 1.0, h = 0.0, field = :z)
    return heisenberg_xyz_tto(d; jx = J, jy = J, jz = J, λ = h, field = field)
end

"""
    xy_tto(d; jx=1.0, jy=1.0, h=0.0, field=:z)

Construct the open-boundary XY Hamiltonian

    H = jx H_{x,x} + jy H_{y,y} + h H_field.
"""
function xy_tto(d::Int; jx = 1.0, jy = 1.0, h = 0.0, field = :z)
    return heisenberg_xyz_tto(d; jx = jx, jy = jy, jz = zero(jx + jy), λ = h, field = field)
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
function qtt_laplacian(n_dims::Int, bits_per_dim::Int;
        ordering::Symbol = :interleaved, a::Real = 0.0, b::Real = 1.0,
        bc::Symbol = :DN)
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
