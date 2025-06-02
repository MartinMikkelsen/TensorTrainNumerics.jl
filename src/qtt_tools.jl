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
QTT of sin(λ*π*x/(b-a))
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

# Arguments
- `d::Int`: The number of quantization levels (the resulting grid has `2^d` points).
- `a::Float64=0.0`: The left endpoint of the interval.
- `b::Float64=1.0`: The right endpoint of the interval.
- `α::Float64=1.0`: The coefficient multiplying the grid points in the exponent.
- `β::Float64=0.0`: The constant term in the exponent.

# Returns
- A QTT tensor (in TT format) representing the function `exp(α * t + β)` evaluated at the grid points.

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