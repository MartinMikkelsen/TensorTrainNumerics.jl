"""
Constructs a tensor train operator (TTO) representation of a Toeplitz matrix parameterized by `α`, `β`, and `γ` over `d` dimensions.
"""
function toeplitz_to_qtto(α, β, γ, d)
    d == 1 && return _single_site_qtto(float.([α β; γ α]))
    out = zeros_tto(2, d, 3)
    id = Matrix{Float64}(I, 2, 2)
    J = zeros(2, 2)
    J[1, 2] = 1
    for i in 1:2
        for j in 1:2
            out.cores[1][i, j, 1, :] = [id[i, j];J[j, i];J[i, j]]
            for k in 2:(d - 1)
                out.cores[k][i, j, :, :] = [id[i, j] J[j, i] J[i, j]; 0 J[i, j] 0 ; 0 0 J[j, i]]
            end
            out.cores[d][i, j, :, 1] = [α * id[i, j] + β * J[i, j] + γ * J[j, i]; γ * J[i, j] ; β * J[j, i]]
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
        return TTOperator{T, 1}([reshape(P, 2, 2, 1, 1)], dims, [1, 1])
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

    return TTOperator{T, d}(cores, dims, rks)
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

    return TTOperator{T, d}(cores, dims, rks)
end

"""
    H_μ(μ, d)

Alias for [`pauli_sum_tto`](@ref)`(μ, d)`.
"""
H_μ(μ, d::Int) = pauli_sum_tto(μ, d)

"""
    H_μν(μ, ν, d)

Alias for [`pauli_pair_sum_tto`](@ref)`(μ, ν, d)`.
"""
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

    return TTOperator{T, d}(cores, dims, rks)
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
    Δ(d; bc=:DD) -> TTOperator

Second-difference (negative Laplacian) operator on a grid of `2^d` points in
QTT format, without the `1/h²` scaling.

`bc` selects the boundary conditions at the left and right end of the grid:
`:DD` (Dirichlet–Dirichlet), `:DN` (Dirichlet–Neumann), `:ND`
(Neumann–Dirichlet), `:NN` (Neumann–Neumann), or `:periodic`. All except `:DD`
require `d ≥ 4`.
"""
function Δ(d::Int; bc::Symbol = :DD)
    bc === :DD && return toeplitz_to_qtto(2, -1, -1, d)
    bc === :DN && return _Δ_DN(d)
    bc === :ND && return _Δ_ND(d)
    bc === :NN && return _Δ_NN(d)
    bc === :periodic && return _Δ_periodic(d)
    throw(ArgumentError("`bc` must be :DD, :DN, :ND, :NN, or :periodic; got :$bc"))
end

function _Δ_DN(d::Int)
    @assert d ≥ 4 "Dimension must be at least 4"
    out = zeros_tto(2, d, 4)
    id = [1 0; 0 1]
    J = [0 1; 0 0]
    I₂ = [0 0; 0 1]
    for i in 1:2
        for j in 1:2
            out.cores[1][i, j, 1, :] = [id[i, j]; J[j, i]; J[i, j]; I₂[i, j]]
            for k in 2:(d - 1)
                out.cores[k][i, j, :, :] = [id[i, j] J[j, i] J[i, j] 0; 0 J[i, j] 0 0; 0 0 J[j, i] 0; 0 0 0 I₂[i, j]]
            end
            out.cores[d][i, j, :, 1] = [2 * id[i, j] - J[i, j] - J[j, i]; -J[i, j]; -J[j, i]; -I₂[i, j]]
        end
    end
    return out
end

function _Δ_ND(d::Int)
    @assert d ≥ 4 "Dimension must be at least 4"
    out = zeros_tto(2, d, 4)
    id = [1 0; 0 1]
    J = [0 1; 0 0]
    I₁ = [1 0; 0 0]
    for i in 1:2
        for j in 1:2
            out.cores[1][i, j, 1, :] = [id[i, j]; J[j, i]; J[i, j]; I₁[i, j]]
            for k in 2:(d - 1)
                out.cores[k][i, j, :, :] = [id[i, j] J[j, i] J[i, j] 0; 0 J[i, j] 0 0; 0 0 J[j, i] 0; 0 0 0 I₁[i, j]]
            end
            out.cores[d][i, j, :, 1] = [2 * id[i, j] - J[i, j] - J[j, i]; -J[i, j]; -J[j, i]; -I₁[i, j]]
        end
    end
    return out
end

function _Δ_NN(d::Int)
    @assert d ≥ 4 "Dimension must be at least 4"
    out = zeros_tto(ntuple(_ -> 2, d), [1; fill(5, d - 1); 1])
    id = [1 0; 0 1]
    J = [0 1; 0 0]
    I₁ = [1 0; 0 0]
    I₂ = [0 0; 0 1]
    for i in 1:2
        for j in 1:2
            out.cores[1][i, j, 1, :] = [id[i, j]; J[j, i]; J[i, j]; I₂[i, j]; I₁[i, j]]
            for k in 2:(d - 1)
                out.cores[k][i, j, :, :] = [id[i, j] J[j, i] J[i, j] 0 0; 0 J[i, j] 0 0 0; 0 0 J[j, i] 0 0; 0 0 0 I₂[i, j] 0; 0 0 0 0 I₁[i, j]]
            end
            out.cores[d][i, j, :, 1] = [2 * id[i, j] - J[i, j] - J[j, i]; -J[i, j]; -J[j, i]; -I₂[i, j]; -I₁[i, j]]
        end
    end
    return out
end

function _Δ_periodic(d::Int)
    @assert d ≥ 4 "Dimension must be at least 4"
    out = zeros_tto(ntuple(_ -> 2, d), [1; fill(5, d - 1); 1])
    id = [1 0; 0 1]
    J = [0 1; 0 0]
    for i in 1:2
        for j in 1:2
            out.cores[1][i, j, 1, :] = [id[i, j], J[j, i], J[i, j], J[i, j], J[j, i]]
            for k in 2:(d - 1)
                out.cores[k][i, j, :, :] = [
                    id[i, j] J[j, i] J[i, j] 0 0;
                    0 J[i, j] 0 0 0;
                    0 0 J[j, i] 0 0;
                    0 0 0 J[i, j] 0;
                    0 0 0 0 J[j, i]
                ]
            end
            out.cores[d][i, j, :, 1] = [
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
    Δ⁻¹(d; bc=:DN) -> TTOperator

Inverse of [`Δ`](@ref) in QTT format. Only `bc = :DN` (Dirichlet–Neumann) is
available.
"""
function Δ⁻¹(d::Int; bc::Symbol = :DN)
    bc === :DN || throw(ArgumentError("`Δ⁻¹` is only available for `bc = :DN`; got :$bc"))
    @assert d ≥ 2 "Dimension must be at least 2"
    out = zeros_tto(2, d, 4)
    id = [1 0; 0 1]
    E = [1 1; 1 1]
    I₂ = [0 0; 0 1]
    J = [0 1; 0 0]
    for i in 1:2
        for j in 1:2
            out.cores[1][i, j, 1, :] = [id[i, j]; I₂[i, j]; J[i, j]; J[j, i]]
            for k in 2:(d - 1)
                out.cores[k][i, j, :, :] = [
                    id[i, j] I₂[i, j] J[i, j] J[j, i];
                    0 2 * E[i, j] 0 0;
                    0 I₂[i, j] + J[j, i] E[i, j] 0;
                    0 I₂[i, j] + J[i, j] 0 E[i, j];
                ]
            end
            out.cores[d][i, j, :, 1] = [
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
            out.cores[1][i, j, 1, :] = 0.5 * [id[i, j]; J[j, i]]
            for k in 2:(d - 1)
                out.cores[k][i, j, :, :] = [id[i, j] J[j, i]; 0 J[i, j]]
            end
        end
    end
    out.cores[d][1, 1, 1, 1] = 1.0
    out.cores[d][2, 1, 1, 1] = 2.0
    out.cores[d][1, 2, 1, 1] = 1.0
    out.cores[d][2, 2, 1, 1] = 0.0
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
        out[k] = copy(identity_branch.cores[k])
    end
    out[d + 1] = ones(Float64, 2, 1, 1, 1)

    return TTOperator{Float64, d + 1}(
        out,
        ntuple(_ -> 2, d + 1),
        ones(Int64, d + 2)
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
        average_branch = TTOperator{Float64, 1}([average_core], (2,), [1, 1])
    else
        average_branch = 0.5 * (id_tto(d) + shift(d))
    end
    out_rks = Vector{Int64}(undef, d + 2)
    out_rks[1] = 1
    @inbounds for k in 2:(d + 1)
        out_rks[k] = identity_branch.ranks[k] + average_branch.ranks[k]
    end
    out_rks[d + 2] = 1

    out = Vector{Array{Float64, 4}}(undef, d + 1)
    out[1] = zeros(Float64, 2, 2, 1, out_rks[2])
    r₀ = identity_branch.ranks[2]
    out[1][:, :, 1:1, 1:r₀] .= identity_branch.cores[1]
    out[1][:, :, 1:1, (r₀ + 1):out_rks[2]] .= average_branch.cores[1]

    @inbounds for k in 2:d
        l₀ = identity_branch.ranks[k]
        r₀ = identity_branch.ranks[k + 1]
        l₁ = average_branch.ranks[k]
        r₁ = average_branch.ranks[k + 1]
        out[k] = zeros(Float64, 2, 2, out_rks[k], out_rks[k + 1])
        out[k][:, :, 1:l₀, 1:r₀] .= identity_branch.cores[k]
        out[k][:, :, (l₀ + 1):(l₀ + l₁), (r₀ + 1):(r₀ + r₁)] .= average_branch.cores[k]
    end

    l₀ = identity_branch.ranks[d + 1]
    l₁ = average_branch.ranks[d + 1]
    out[d + 1] = zeros(Float64, 2, 1, out_rks[d + 1], 1)
    out[d + 1][1, 1, 1:l₀, 1] .= 1.0
    out[d + 1][2, 1, (l₀ + 1):(l₀ + l₁), 1] .= 1.0

    return TTOperator{Float64, d + 1}(out, ntuple(_ -> 2, d + 1), out_rks)
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
function id_tto(d; n_dim::Int = 2)
    return id_tto(Float64, d; n_dim = n_dim)
end


function id_tto(::Type{T}, d; n_dim::Int = 2) where {T}
    dims = ntuple(_ -> n_dim, d)
    A = Array{Array{T, 4}, 1}(undef, d)
    for j in 1:d
        A[j] = zeros(T, n_dim, n_dim, 1, 1)
        A[j][:, :, 1, 1] = Matrix{T}(I, n_dim, n_dim)
    end
    return TTOperator{T, d}(A, dims, ones(Int64, d + 1))
end

# Identity operator with the element type and physical dimensions of `A`.
function _identity_like(A::AbstractTTOperator)
    T = eltype(A)
    dims = _square_dims(A)
    d = length(dims)
    cores = [reshape(Matrix{T}(I, n, n), n, n, 1, 1) for n in dims]
    return TTOperator{T, d}(cores, dims, ones(Int, d + 1))
end

"""
    rand_tto(dims, max_bond::Int; T=Float64) -> TTOperator

Return a random [`TTOperator`](@ref) with physical dimensions `dims` and entries
drawn from `randn`. Every interior rank is `max_bond`, reduced where the dimensions
force a smaller rank.
"""
function rand_tto(dims, max_bond::Int; T = Float64)
    d = length(dims)
    tt_vec = Vector{Array{T, 4}}(undef, d)
    rks = ones(Int, d + 1)
    for i in eachindex(tt_vec)
        ri = min(prod(dims[1:(i - 1)]), prod(dims[i:d]), max_bond)
        rip = min(prod(dims[1:i]), prod(dims[(i + 1):d]), max_bond)
        rks[i + 1] = rip
        tt_vec[i] = randn(T, dims[i], dims[i], ri, rip)
    end
    return TTOperator{T, d}(tt_vec, dims, rks)
end

"""
    zeros_tt([T=Float64,] dims, ranks; orthogonality=(1, length(dims))) -> TTVector
    zeros_tt(n::Integer, d::Integer, r; orthogonality=(1, d), admissible=true) -> TTVector

Return a [`TTVector`](@ref) with element type `T`, physical dimensions `dims`,
TT ranks `ranks` (length `length(dims) + 1`), and all cores zero. `orthogonality`
sets the orthogonality interval `(left, right)` recorded on the result.

The second form uses `d` sites of dimension `n` and interior ranks `r`. With
`admissible = true`, ranks are reduced where the dimensions force a smaller rank
(see [`admissible_ranks`](@ref)); otherwise every interior rank is `r`.
"""
function zeros_tt(dims, ranks; kwargs...)
    return zeros_tt(Float64, dims, ranks; kwargs...)
end

function zeros_tt(::Type{T}, dims::NTuple{N, Int64}, ranks; orthogonality = (1, N)) where {T, N}
    @assert length(dims) + 1 == length(ranks) "Dimensions and ranks are not compatible"
    tt_vec = [zeros(T, dims[i], ranks[i], ranks[i + 1]) for i in eachindex(dims)]
    ranks_vec = collect(Int64, ranks)
    return TTVector{T, N}(tt_vec, dims, ranks_vec; orthogonality)
end

function zeros_tt(n::Integer, d::Integer, r; admissible = true, kwargs...)
    dims = ntuple(x -> n, d)
    if admissible
        ranks = admissible_ranks(r * ones(Int64, d + 1), dims)
    else
        ranks = r * ones(Int64, d + 1)
        ranks[1], ranks[end] = 1, 1
    end
    return zeros_tt(Float64, dims, ranks; kwargs...)
end

function zeros_tt(::Type{T}, dims::Vector{Int}, ranks::Vector{Int}; kwargs...) where {T}
    return zeros_tt(T, Tuple(dims), Tuple(ranks); kwargs...)
end

function zeros_tt!(A::AbstractTTVector)
    @assert isa(A.cores, Vector)
    for core in A.cores
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
    return TTVector{T, N}(vec, Tuple(dims), rks)
end

function ones_tt(n::Integer, d::Integer)
    dims = n * ones(Int64, d)
    return ones_tt(dims)
end

"""
    zeros_tto([T=Float64,] dims, ranks) -> TTOperator
    zeros_tto(T, row_dims, col_dims, ranks) -> TTOperator
    zeros_tto(n, d, r) -> TTOperator

Return a [`TTOperator`](@ref) with element type `T`, TT ranks `ranks`, and all
cores zero. The first form is square with physical dimensions `dims`; the
second takes row and column dimensions separately. The third form uses `d`
sites of dimension `n` and interior ranks `r`, reduced where the dimensions
force a smaller rank.
"""
function zeros_tto(dims, ranks)
    return zeros_tto(Float64, dims, ranks)
end

zeros_tto(::Type{T}, dims::NTuple{N, Int64}, ranks) where {T, N} = zeros_tto(T, dims, dims, ranks)

function zeros_tto(::Type{T}, row_dims::NTuple{N, Int64}, col_dims::NTuple{N, Int64}, ranks) where {T, N}
    @assert N + 1 == length(ranks) "Dimensions and ranks are not compatible"
    vec = [zeros(T, row_dims[i], col_dims[i], ranks[i], ranks[i + 1]) for i in 1:N]
    return TTOperator{T, N}(vec, row_dims, col_dims, ranks)
end

function zeros_tto(n, d, r)
    dims = ntuple(x -> n, d)
    ranks = r * ones(Int64, d + 1)
    ranks = admissible_ranks(ranks, dims .^ 2; max_bond = r)
    return zeros_tto(Float64, dims, ranks)
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
- `bc::Symbol`: Boundary conditions of each 1D operator, as in [`Δ`](@ref):
  `:DD`, `:DN`, `:ND`, `:NN`, or `:periodic`. Default: `:DN`.

# Returns
A `QTTOperator` with `N = n_dims * bits_per_dim` sites.
"""
function qtt_laplacian(
        n_dims::Int, bits_per_dim::Int;
        ordering::Symbol = :interleaved, a::Real = 0.0, b::Real = 1.0,
        bc::Symbol = :DN
    )
    @assert ordering ∈ (:interleaved, :serial) "ordering must be :interleaved or :serial"
    @assert n_dims ≥ 1 "n_dims must be at least 1"

    d = bits_per_dim
    h = (b - a) / (2^d - 1)
    scale = 1.0 / h^2

    lap_1d = Δ(d; bc)

    id_1d = id_tto(d)

    if n_dims == 1
        # Single dimension: just scale the 1D Laplacian
        scaled = scale * lap_1d
        return QTTOperator(scaled, 1, d, ordering)
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

    serial_qtto = QTTOperator(result, n_dims, d, :serial)

    if ordering == :serial
        return serial_qtto
    else  # :interleaved — reorder from serial to interleaved
        return reorder(serial_qtto, :interleaved)
    end
end
