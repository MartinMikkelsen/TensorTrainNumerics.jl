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
