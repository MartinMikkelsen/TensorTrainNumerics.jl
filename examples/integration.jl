using CairoMakie
using TensorTrainNumerics

f = x -> sin(π * x)
num_cores = 10  
N = 150 

qtt = integrating_qtt(num_cores, N)

fqtt = interpolating_qtt(f, num_cores, N)

dot(qtt, fqtt)


# 1) pick your QTT dimension:
d = 10              # 2^10 points on [0,1]
λ = 1.0             # for sin(π x)

# 2) build f(x)=sin(π x) exactly in QTT
fqtt = qtt_sin(d; a=0.0, b=1.0, λ=λ)

# 3) build the composite‐trapezoid weights as a tiny function and turn into QTT
h = 1/(2^d - 1)
weightfun = x -> (isapprox(x,0.0) || isapprox(x,1.0) ? h/2 : h)
wqtt = function_to_qtt(weightfun, d; a=0.0, b=1.0)

# 4) take the TT‐inner‐product to get the integral
I = dot(wqtt, fqtt)

println("∫₀¹ sin(π x) dx ≃ ", I)
# → 0.6366197723675814  (i.e. 2/π)