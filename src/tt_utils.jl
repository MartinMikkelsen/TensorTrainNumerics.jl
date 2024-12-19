using Random
using LinearAlgebra
using TensorTrainNumerics

"""
    χ(c::Int, b1::Float64, b2::Float64) -> TTvector{Float64}

Constructs a tensor train vector (TTvector) with specified parameters.

# Arguments
- `c::Int`: The number of cores in the tensor train. If `c == 2`, a specific structure is created. For `c > 2`, a different structure is created.
- `b1::Float64`: The value to be assigned to the (1, 1, 2) element of the first core.
- `b2::Float64`: The value to be assigned to the (1, 2, 1) element of the first core.

# Returns
- `TTvector{Float64}`: A tensor train vector with the specified structure and values.
"""
function χ(c::Int, b1::Float64, b2::Float64)
    f_core = zeros(Float64, 1, 2, 2)
    f_core[1, 1, 2] = b1
    f_core[1, 2, 1] = b2

    if c == 2
        l_core = zeros(Float64, 2, 2, 1)
        l_core[2, 1, 1] = 1.0
        l_core[1, 2, 1] = 1.0

        ttv_vec = [f_core, l_core]
        ttv_dims = (2, 2)
        ttv_rks = [1, 2, 1]
        ttv_ot = zeros(Int, 2)  
        return TTvector{Float64,2}(c, ttv_vec, ttv_dims, ttv_rks, ttv_ot)
    end

    m_core_1 = zeros(Float64, 2, 2, 2)
    m_core_1[1, 2, 1] = -1.0
    m_core_1[2, 1, 2] = 1.0

    m_core_2 = zeros(Float64, 2, 2, 2)
    m_core_2[1, 2, 1] = 1.0
    m_core_2[2, 1, 2] = 1.0

    l_core = zeros(Float64, 2, 2, 1)
    l_core[2, 1, 1] = 1.0
    l_core[1, 2, 1] = -1.0

    all_cores = Vector{Array{Float64,3}}()
    push!(all_cores, f_core)
    push!(all_cores, m_core_1)

    for i in 1:(c-3)
        push!(all_cores, copy(m_core_2))
    end

    push!(all_cores, l_core)

    ttv_dims = ntuple(i->2, c) 
    ttv_rks = [1; fill(2, c-1); 1] 
    ttv_ot = zeros(Int, c)

    return TTvector{Float64,c}(c, all_cores, ttv_dims, ttv_rks, ttv_ot)
end

function build_1d_diff_elements_DD(c::Int; P=1.0, S=0.0, V=0.0, start=0.0, stop=1.0)
    I = [1.0 0.0; 0.0 1.0]
    J = [0.0 1.0; 0.0 0.0]
    JT = J'  

    f_core = zeros(Float64, 1, 2, 2, 3)
    f_core[1, :, :, 1] = I
    f_core[1, :, :, 2] = JT
    f_core[1, :, :, 3] = J

    m_core = zeros(Float64, 3, 2, 2, 3)
    m_core[1, :, :, 1] = I
    m_core[1, :, :, 2] = JT
    m_core[1, :, :, 3] = J
    m_core[2, :, :, 2] = J
    m_core[3, :, :, 3] = JT

    h = (stop - start) / c^2
    h2 = h^2
    alpha = h2 * V - 2 * P
    beta = P + h * S / 2
    gamma = P - h * S / 2

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

    tto_dims = ntuple(i->2, c) 
    tto_rks = [1; fill(3, c-1); 1]
    tto_ot = zeros(Int, c)  

    return TToperator{Float64,c}(c, all_cores, tto_dims, tto_rks, tto_ot)
end

cores = 4
A  = Δ_tto(2^cores, 2, Δ_DD)
matricize(A)