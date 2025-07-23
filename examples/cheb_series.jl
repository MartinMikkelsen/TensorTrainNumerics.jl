using CairoMakie
using TensorTrainNumerics

"""
    gauss_chebyshev_lobatto(n; shifted=true)

Compute Chebyshev-Lobatto nodes and weights for integration.

Returns nodes in [0,1] if `shifted=true`.

# Arguments
- `n`: number of nodes.
- `shifted`: whether to shift nodes to [0, 1].

# Returns
- Tuple `(x, w)` of nodes and weights.
"""
function gauss_chebyshev_lobatto(n; shifted=true)
    x = [cos(π * j / (n - 1)) for j in 0:n-1]
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
    chebyshev_polynomial(k::Int, x::Float64)

Evaluate Chebyshev polynomial T_k(x).
"""
function chebyshev_polynomial(k::Int, x::Float64)
    if k == 0
        return 1.0
    elseif k == 1
        return x
    else
        Tkm1 = x
        Tkm2 = 1.0
        for _ in 2:k
            Tk = 2x * Tkm1 - Tkm2
            Tkm2, Tkm1 = Tkm1, Tk
        end
        return Tkm1
    end
end

chebyshev_polynomial(k::Int, x::Vector{Float64}) = [chebyshev_polynomial(k, xi) for xi in x]


function A_L(func::Function, nodes::Vector{Float64}, start::Float64=0.0, stop::Float64=1.0)
    D = length(nodes)
    A = zeros(Float64, 1, 2, 1, D)
    for σ in 0:1
        A[1, σ + 1, 1, :] = func.(0.5 .* (σ .+ nodes) .* (stop - start) .+ start)
    end
    return A
end

function A_C_chebyshev(nodes::Vector{Float64})
    D = length(nodes)
    A = zeros(Float64, D, 2, 1, D)
    for σ in 0:1
        x = 0.5 .* (σ .+ nodes)
        for α in 1:D
            A[α, σ + 1, 1, :] = chebyshev_polynomial(α - 1, x)
        end
    end
    return A
end

function A_R_chebyshev(nodes::Vector{Float64}, weights::Vector{Float64})
    D = length(nodes)
    A = zeros(Float64, D, 2, 1, 1)
    for σ in 0:1
        x = 0.5 * σ
        for α in 1:D
            A[α, σ + 1, 1, 1] = chebyshev_polynomial(α - 1, x) * weights[α]
        end
    end
    return A
end

# ---- 4. QTT Constructor ----

"""
    interpolating_qtt(func::Function, core::Int, N::Int)

Construct QTT Chebyshev representation of `func` using `core` cores and `N` Chebyshev-Lobatto nodes.
"""
function interpolating_qtt(
    func::Function, core::Int, N::Int; start::Float64=0.0, stop::Float64=1.0
)
    nodes, weights = gauss_chebyshev_lobatto(N)
    Al = A_L(func, nodes, start, stop)
    Ac = A_C_chebyshev(nodes)
    Ar = A_R_chebyshev(nodes, weights)

    tensors = [Al]
    for _ in 1:(core - 2)
        push!(tensors, Ac)
    end
    push!(tensors, Ar)

    N_ = length(tensors)
    ttv_vec = Vector{Array{Float64,3}}()
    ttv_rks = [1]

    for i in 1:N_
        T = tensors[i]
        n_i = size(T, 2)
        r_prev = size(T, 1) * size(T, 3)
        r_next = size(T, 4)
        T = permutedims(T, (2, 1, 3, 4))
        T = reshape(T, n_i, r_prev, r_next)
        push!(ttv_vec, T)
        push!(ttv_rks, r_next)
    end

    ttv_dims = ntuple(i -> size(ttv_vec[i], 1), N_)
    ttv_ot = zeros(Int, N_)
    return TTvector{Float64, N_}(N_, ttv_vec, ttv_dims, ttv_rks, ttv_ot)
end

d = 6

f(x) = exp(x)
tn_cheb = interpolating_qtt(f, d, 2^d)

let 
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="x", ylabel="f(x)", title="Chebyshev QTT Reconstruction")
    xes, _ = gauss_chebyshev_lobatto(2^d; shifted=true)  # Chebyshev nodes

    f_exact = f.(xes)                        # Ground truth
    f_qtt    = qtt_to_function(tn_cheb)                           

    lines!(ax, xes, f_qtt, label="QTT Reconstruction")
    lines!(ax, xes, f_exact, color=:red, linestyle=:dash, label="Exact f(x)")
    
    fig
end
