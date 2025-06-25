using CairoMakie
using TensorTrainNumerics

f = x -> sin(π * x)
num_cores = 10  
N = 150 

qtt = integrating_qtt(num_cores, N)

function chebyshev_coefficients(f::Function, N::Int)
    c = zeros(Float64, N+1)
    # α_k factors
    α = ones(Float64, N+1)
    α[1] = 2.0
    α[end] = 2.0

    # loop over k = 0:N
    for k in 0:N
        acc = 0.0
        # sum j=0..2N-1
        for j in 0:(2N-1)
            x = cos(j * π / N)
            acc += f(x) * cos(k * j * π / N)
        end
        c[k+1] = acc / (α[k+1] * N)
    end

    return c
end


function qtt_chebyshev(k::Int,d;a=0.0,b=1.0)
  p = length(coef)
  xⱼ = chebyshev_lobatto_nodes(k)
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

T_k = qtt_chebyshev(1,d,a=-1.0,b=1.0)

let
    fig = Figure()
    xes = collect(range(-1.0, 1.0, 2^d))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "Comparison of Time-Stepping Methods")
    lines!(ax, xes, qtt_to_function(T_k), label = "Explicit Euler", linestyle = :solid, linewidth=3)
    display(fig)
end

h(x;ν=2.0) = 2*(x/2)^ν/(sqrt(π)*gamma(ν+0.5))

d = 8

Q = function_to_qtt(h, d; a=-1, b=1)
P = function_to_qtt(g,d; a=-1, b=1)
  
dot(Q,P)

function integrant(x; ν=2.0, d=8)
  # 1) f-TT on [0,1]
  fqtt = function_to_qtt(t -> (1 - t^2)^(ν - 0.5) * sin(x*t), d; a=0, b=1)

  # 2) weight-TT via trapezoid
  h = 1/(2^d - 1)
  wqtt = function_to_qtt(t -> (isapprox(t,0) || isapprox(t,1) ? h/2 : h), d; a=0, b=1)

  # 3) QTT inner product ≃ ∫₀¹ f(t) dt
  return dot(wqtt, fqtt)
end

Q_test = [h(xs).*integrant(xs) for xs in collect(range(-5, 5, 2^8))]

using Struve
let
    fig = Figure()
    xes = collect(range(-5, 5, 2^8))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "Comparison of Time-Stepping Methods")
    lines!(ax, xes, Q_test, label = "Explicit Euler", linestyle = :solid, linewidth=3)
    lines!(ax, xes, struveh.(2,xes), label = "Explicit Euler", linestyle = :dash, linewidth=3)
    display(fig)
end