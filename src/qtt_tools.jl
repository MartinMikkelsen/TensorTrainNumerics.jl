using LinearAlgebra

function index_to_point(t; L = 1.0)
    d = length(t)
    return sum(2.0^(d - i) * (t[i] - 1) / (2^d - 1) for i in 1:d)
end

function tuple_to_index(t)
    d = length(t)
    return sum(2^(d - i) * (t[i] - 1) for i in 1:d) + 1
end

function function_to_tensor(f, d; a = 0.0, b = 1.0)
    out = zeros(ntuple(x -> 2, d))
    for t in CartesianIndices(out)
        out[t] = f(index_to_point(Tuple(t); L = b - a))
    end
    return out
end

function tensor_to_grid(tensor)
    T = eltype(tensor)
    out = Vector{T}(undef, length(tensor))
    @inbounds for t in CartesianIndices(tensor)
        out[tuple_to_index(Tuple(t))] = tensor[t]
    end
    return out
end

"""
Converts a univariate function `f` into its Quantized Tensor Train (QTT) representation.
"""
function function_to_qtt(f, d; a = 0.0, b = 1.0)
    tensor = function_to_tensor(f, d; a = a, b = b)
    return ttv_decomp(tensor)
end

"""
Converts a quantized tensor train (QTT) vector `qtt` into a function representation.
"""
function qtt_to_function(qtt::TTvector{T, d}) where {T <: Number, d}
    tensor = ttv_to_tensor(qtt)
    out = tensor_to_grid(tensor)
    return out
end

function function_to_qtt_uniform(f, d::Int)
    N = 2^d
    y = [f(n / N) for n in 0:(N - 1)]
    A = zeros(eltype(y), ntuple(_ -> 2, d)...)
    @inbounds for n in 0:(N - 1)
        bits = (digits(n, base = 2, pad = d)) .+ 1
        A[CartesianIndex(Tuple(bits))] = y[n + 1]
    end
    return ttv_decomp(A)
end

"""
Constructs a Quantized Tensor Train (QTT) representation a polynomial with given coefficients
over a uniform grid in the interval `[a, b]` with `2^d` points.
"""
function qtt_polynom(coef, d; a = 0.0, b = 1.0)
    p = length(coef)
    h = (b - a) / (2^d - 1)
    out = zeros_tt(2, d, p; r_and_d = false)
    φ(x, s) = sum(coef[k + 1] * x^(k - s) * binomial(k, s) for k in s:(p - 1))
    t₁ = a
    out.ttv_vec[1][1, 1, :] = [φ(t₁, k) for k in 0:(p - 1)]
    t₁ = a + h * 2^(d - 1) #convention : coarsest first
    out.ttv_vec[1][2, 1, :] = [φ(t₁, k) for k in 0:(p - 1)]
    for k in 2:(d - 1)
        for j in 0:(p - 1)
            out.ttv_vec[k][1, j + 1, j + 1] = 1.0
            for i in 0:(p - 1)
                tₖ = h * 2^(d - k)
                out.ttv_vec[k][2, i + 1, j + 1] = binomial(i, i - j) * tₖ^(i - j)
            end
        end
    end
    out.ttv_vec[d][1, 1, 1] = 1.0
    td = h
    out.ttv_vec[d][2, :, 1] = [td^k for k in 0:(p - 1)]
    return out
end

"""
Constructs a Quantized Tensor Train (QTT) representation of the cosine function
over a uniform grid in the interval `[a, b]` with `2^d` points.
"""
function qtt_cos(d; a = 0.0, b = 1.0, λ = 1.0)
    out = zeros_tt(2, d, 2)
    h = (b - a) / (2^d - 1)
    t₁ = a
    out.ttv_vec[1][1, 1, :] = [cos(λ * π * t₁); -sin(λ * π * t₁)]
    t₁ = a + h * 2^(d - 1) #convention : coarsest first
    out.ttv_vec[1][2, 1, :] = [cos(λ * π * t₁); -sin(λ * π * t₁)]
    for k in 2:(d - 1)
        out.ttv_vec[k][1, :, :] = [1 0;0 1]
        tₖ = h * 2^(d - k)
        out.ttv_vec[k][2, :, :] = [cos(λ * π * tₖ) -sin(λ * π * tₖ); sin(λ * π * tₖ) cos(λ * π * tₖ)]
    end
    out.ttv_vec[d][1, 1, 1] = 1.0
    td = h
    out.ttv_vec[d][2, :, 1] = [cos(λ * π * td); sin(λ * π * td)]
    return out
end

"""
Constructs a Quantized Tensor Train (QTT) representation of the sine function
over a uniform grid in the interval `[a, b]` with `2^d` points.
"""
function qtt_sin(d; a = 0.0, b = 1.0, λ = 1.0)
    out = zeros_tt(2, d, 2)
    h = (b - a) / (2^d - 1)
    t₁ = a
    out.ttv_vec[1][1, 1, :] = [sin(λ * π * t₁); cos(λ * π * t₁)]
    t₁ = a + h * 2^(d - 1) #convention : coarsest first
    out.ttv_vec[1][2, 1, :] = [sin(λ * π * t₁); cos(λ * π * t₁)]
    for k in 2:(d - 1)
        out.ttv_vec[k][1, :, :] = [1 0;0 1]
        tₖ = h * 2^(d - k)
        out.ttv_vec[k][2, :, :] = [cos(λ * π * tₖ) -sin(λ * π * tₖ); sin(λ * π * tₖ) cos(λ * π * tₖ)]
    end
    out.ttv_vec[d][1, 1, 1] = 1.0
    td = h
    out.ttv_vec[d][2, :, 1] = [cos(λ * π * td); sin(λ * π * td)]
    return out
end

"""
Constructs a Quantized Tensor Train (QTT) representation of the exponential function
over a uniform grid in the interval `[a, b]` with `2^d` points.
"""
function qtt_exp(d; a = 0.0, b = 1.0, α = 1.0, β = 0.0)
    out = zeros_tt(2, d, 1)
    h = (b - a) / (2^d - 1)
    t₁ = a
    out.ttv_vec[1][1, 1, 1] = exp(α * t₁ + β)
    t₁ = a + h * 2^(d - 1)
    out.ttv_vec[1][2, 1, 1] = exp(α * t₁ + β)
    for k in 2:(d - 1)
        tₖ = h * 2^(d - k)
        out.ttv_vec[k][1, 1, 1] = 1.0
        out.ttv_vec[k][2, 1, 1] = exp(α * tₖ)
    end
    out.ttv_vec[d][1, 1, 1] = 1.0
    td = h
    out.ttv_vec[d][2, 1, 1] = exp(α * td)
    return out
end

"""
Converts a quantum tensor train operator (`TToperator`) into its full matrix representation.
"""
function qtto_to_matrix(Aqtto::TToperator{T, d}) where {T, d}
    A = zeros(2^d, 2^d)
    A_tensor = tto_to_tensor(Aqtto)
    for t in CartesianIndices(A_tensor)
        A[tuple_to_index(Tuple(t)[1:d]), tuple_to_index(Tuple(t)[(d + 1):end])] = A_tensor[t]
    end
    return A
end

function qtt_basis_vector(d, pos::Int, val::Number = 1.0)
    out = zeros_tt(2, d, 1)
    bits = reverse(digits(pos - 1, base = 2, pad = d))
    for k in 1:d
        out.ttv_vec[k][:, 1, 1] .= 0.0
        out.ttv_vec[k][bits[k] + 1, 1, 1] = val
        val = 1.0
    end
    return out
end

"""
Constructs a Quantized Tensor Train (QTT) representation of the Chebyshev polynomial of degree `n` over `2^d` Chebyshev-Lobatto nodes.

# Details
- The function uses the Gauss-Chebyshev-Lobatto nodes, shifted to the interval [0, 1].
"""
function qtt_chebyshev(n, d)
    out = zeros_tt(2, d, 2)
    N = 2^d
    x_nodes, _ = gauss_chebyshev_lobatto(N; shifted = true)
    θ = acos.(clamp.(2 .* x_nodes .- 1, -1.0, 1.0))
    out.ttv_vec[1][1, 1, :] = [cos(n * θ[1]); -sin(n * θ[1])]
    out.ttv_vec[1][2, 1, :] = [cos(n * θ[2^(d - 1) + 1]); -sin(n * θ[2^(d - 1) + 1])]
    for k in 2:(d - 1)
        out.ttv_vec[k][1, :, :] .= [1.0 0.0; 0.0 1.0]
        idx = 2^(d - k) + 1
        out.ttv_vec[k][2, :, :] .= [cos(n * θ[idx]) -sin(n * θ[idx]);sin(n * θ[idx])  cos(n * θ[idx])]
    end
    out.ttv_vec[d][1, :, 1] .= [1.0, 0.0]
    out.ttv_vec[d][2, :, 1] .= [cos(n * θ[2]), sin(n * θ[2])]

    return out
end

function qtt_trapezoidal(d; a = 0.0, b = 1.0)
    out = zeros_tt(2, d, 1)
    h = (b - a) / (2^d - 1)

    out.ttv_vec[1][1, 1, 1] = 1.0
    out.ttv_vec[1][2, 1, 1] = 1.0

    for k in 2:(d - 1)
        out.ttv_vec[k][1, 1, 1] = 1.0
        out.ttv_vec[k][2, 1, 1] = 1.0
    end

    out.ttv_vec[d][1, 1, 1] = 1.0
    out.ttv_vec[d][2, 1, 1] = 1.0

    return h * out
end

function qtt_simpson(d; a = 0.0, b = 1.0)
    N = 2^d
    h = (b - a) / (N - 1)
    tensors = [zeros_tt(2, d, 1) for _ in 1:N]
    for i in 0:(N - 1)
        weight = (i == 0 || i == N - 1) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        bits = reverse(digits(i, base = 2, pad = d)) .+ 1

        for k in 1:d
            fill!(tensors[i + 1].ttv_vec[k], 0.0)
            tensors[i + 1].ttv_vec[k][bits[k], 1, 1] = 1.0
        end

        tensors[i + 1] = weight * tensors[i + 1]
    end

    simpson_tt = tensors[1]
    for j in 2:N
        simpson_tt += tensors[j]
    end

    return (h / 3) * simpson_tt
end
