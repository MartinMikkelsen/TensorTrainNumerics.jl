using LinearAlgebra

function index_to_point(t;L=1.0)
  d = length(t)
  return sum(2.0^(d-i)*(t[i]-1)/(2^d-1) for i in 1:d)
end

function tuple_to_index(t)
  d = length(t)
  return sum(2^(d-i)*(t[i]-1) for i in 1:d)+1
end

function function_to_tensor(f,d;a=0.0,b=1.0)
  out = zeros(ntuple(x->2,d))
  for t in CartesianIndices(out)
    out[t] = f(index_to_point(Tuple(t);L=b-a))
  end
  return out
end

function tensor_to_grid(tensor)
  out = zeros(prod(size(tensor)))
  for t in CartesianIndices(tensor)
    out[tuple_to_index(t)] = tensor[t]
  end
  return out
end

function function_to_qtt(f,d;a=0.0,b=1.0)
  tensor = function_to_tensor(f,d;a=a,b=b)
  return ttv_decomp(tensor)
end

function qtt_to_function(ftt::TTvector{T,d}) where {T<:Number,d}
  tensor = ttv_to_tensor(ftt)
  out = tensor_to_grid(tensor)
  return out
end

"""
QTT of a polynom p(x) = ∑ₖ₌₀ⁿ cₖ xᵏ
"""
function qtt_polynom(coef,d;a=0.0,b=1.0)
  p = length(coef)
  h = (b-a)/(2^d-1)
  out = zeros_tt(2,d,p;r_and_d=false)
  φ(x,s) = sum(coef[k+1]*x^(k-s)*binomial(k,s) for k in s:(p-1))
  t₁ = a
  out.ttv_vec[1][1,1,:] = [φ(t₁,k) for k in 0:p-1] 
  t₁ = a+h*2^(d-1) #convention : coarsest first
  out.ttv_vec[1][2,1,:] = [φ(t₁,k) for k in 0:p-1] 
  for k in 2:d-1
    for j in 0:p-1
      out.ttv_vec[k][1,j+1,j+1] = 1.0
      for i in 0:p-1 
        tₖ = h*2^(d-k)
        out.ttv_vec[k][2,i+1,j+1] = binomial(i,i-j)*tₖ^(i-j)
      end
    end
  end
  out.ttv_vec[d][1,1,1] = 1.0
  td = h
  out.ttv_vec[d][2,:,1] = [td^k for k in 0:p-1]
  return out
end

"""
QTT of cos(λ*π*x)
"""
function qtt_cos(d;a=0.0,b=1.0,λ=1.0)
  out = zeros_tt(2,d,2)
  h = (b-a)/(2^d-1)
  t₁ = a
  out.ttv_vec[1][1,1,:] = [cos(λ*π*t₁); -sin(λ*π*t₁)] 
  t₁ = a+h*2^(d-1) #convention : coarsest first
  out.ttv_vec[1][2,1,:] = [cos(λ*π*t₁); -sin(λ*π*t₁)] 
  for k in 2:d-1
    out.ttv_vec[k][1,:,:] = [1 0;0 1]
    tₖ = h*2^(d-k)
    out.ttv_vec[k][2,:,:] = [cos(λ*π*tₖ) -sin(λ*π*tₖ); sin(λ*π*tₖ) cos(λ*π*tₖ)]
  end
  out.ttv_vec[d][1,1,1] = 1.0
  td = h
  out.ttv_vec[d][2,:,1] = [cos(λ*π*td); sin(λ*π*td)]
  return out
end

"""
QTT of sin(λ*π*x)
"""
function qtt_sin(d;a=0.0,b=1.0,λ=1.0)
  out = zeros_tt(2,d,2)
  h = (b-a)/(2^d-1)
  t₁ = a
  out.ttv_vec[1][1,1,:] = [sin(λ*π*t₁); cos(λ*π*t₁)] 
  t₁ = a+h*2^(d-1) #convention : coarsest first
  out.ttv_vec[1][2,1,:] = [sin(λ*π*t₁); cos(λ*π*t₁)] 
  for k in 2:d-1
    out.ttv_vec[k][1,:,:] = [1 0;0 1]
    tₖ = h*2^(d-k)
    out.ttv_vec[k][2,:,:] = [cos(λ*π*tₖ) -sin(λ*π*tₖ); sin(λ*π*tₖ) cos(λ*π*tₖ)]
  end
  out.ttv_vec[d][1,1,1] = 1.0
  td = h
  out.ttv_vec[d][2,:,1] = [cos(λ*π*td); sin(λ*π*td)]
  return out
end

"""
  qtt_exp(d; a=0.0, b=1.0, α=1.0, β=0.0)

Constructs a Quantized Tensor Train (QTT) representation of the exponential function
over a uniform grid in the interval `[a, b]` with `2^d` points.
"""
function qtt_exp(d; a=0.0, b=1.0, α=1.0, β=0.0)
  out = zeros_tt(2, d, 1)
  h = (b - a) / (2^d - 1)
  t₁ = a
  out.ttv_vec[1][1, 1, 1] = exp(α * t₁ + β)
  t₁ = a + h * 2^(d-1)
  out.ttv_vec[1][2, 1, 1] = exp(α * t₁ + β)
  for k in 2:d-1
    tₖ = h * 2^(d - k)
    out.ttv_vec[k][1, 1, 1] = 1.0
    out.ttv_vec[k][2, 1, 1] = exp(α * tₖ)
  end
  out.ttv_vec[d][1, 1, 1] = 1.0
  td = h
  out.ttv_vec[d][2, 1, 1] = exp(α * td)
  return out
end

function qtto_to_matrix(Aqtto::TToperator{T,d}) where {T,d}
  A = zeros(2^d,2^d)
  A_tensor = tto_to_tensor(Aqtto)
  for t in CartesianIndices(A_tensor)
    A[tuple_to_index(Tuple(t)[1:d]),tuple_to_index(Tuple(t)[d+1:end])] = A_tensor[t]
  end
  return A 
end

function qtt_basis_vector(d, pos::Int, val::Number=1.0)
    out = zeros_tt(2, d, 1)
    bits = reverse(digits(pos - 1, base=2, pad=d))
    for k in 1:d
        out.ttv_vec[k][:,1,1] .= 0.0
        out.ttv_vec[k][bits[k] + 1, 1, 1] = val
        val = 1.0
    end
    return out
end

function qtt_chebyshev(n, d)
    out = zeros_tt(2, d, 2)
    N = 2^d

    x_nodes, _ = gauss_chebyshev_lobatto(N; shifted=true)
    θ = acos.(clamp.(2 .* x_nodes .- 1, -1.0, 1.0))
    out.ttv_vec[1][1,1,:] = [cos(n * θ[1]); -sin(n * θ[1])]
    out.ttv_vec[1][2,1,:] = [cos(n * θ[2^(d-1)+1]); -sin(n * θ[2^(d-1)+1])]

    for k in 2:d-1
        out.ttv_vec[k][1,:,:] .= [1.0 0.0; 0.0 1.0]
        idx = 2^(d-k) + 1
        out.ttv_vec[k][2,:,:] .= [cos(n * θ[idx]) -sin(n * θ[idx]);
                                  sin(n * θ[idx])  cos(n * θ[idx])]
    end

    out.ttv_vec[d][1,:,1] .= [1.0, 0.0]
    out.ttv_vec[d][2,:,1] .= [cos(n * θ[2]), sin(n * θ[2])]

    return out
end

"""
    qtt_fft1(T::Type, d::Int; inverse::Bool=false)

Construct the 1D FFT operator in QTT (quantized tensor train) format for vectors of length 2^d.
"""
function qtt_fft1(T::Type, d::Int; inverse::Bool=false)
    sign = inverse ? 1 : -1
    W = [exp(sign * 2π * im / 2^k) for k in 1:d]


    rks = [1; fill(2, d-1); 1]
    dims = fill(2, d)
    cores = Vector{Array{T,4}}(undef, d)

    for k in 1:d
        rk = rks[k]
        rkp = rks[k+1]
        core = zeros(T, 2, 2, rk, rkp)
        for α in 1:rk
            for β in 1:rkp
                # Butterfly block
                if rk == 1 && rkp == 2
                    # First core
                    core[:,:,α,β] = (1/sqrt(2)) * [1 1; 1 -1]
                elseif rk == 2 && rkp == 2
                    for i in 1:2, j in 1:2
                        twiddle = (j == 2) ? W[k]^(α-1) : 1
                        core[i,j,α,β] = (1/sqrt(2)) * (i == j ? 1 : (j == 2 ? twiddle : 0))
                    end
                elseif rk == 2 && rkp == 1
                    # Last core
                    core[:,:,α,β] = (1/sqrt(2)) * [1 1; 1 -1]
                end
            end
        end
        cores[k] = core
    end

    return TToperator{T, d}(d, cores, Tuple(dims), rks, zeros(Int, d))
end
