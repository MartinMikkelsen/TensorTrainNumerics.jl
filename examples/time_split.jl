using TensorTrainNumerics
using FFTW
using LinearAlgebra
using CairoMakie

d = 8
N = 2^d

x = collect(range(0.0,1.0,2^d))

Δx = x[2] - x[1]

k² = ((2*π*fftfreq(N))./Δx).^2
Δt = 0.01
Dₓₓ = [exp(1im*Δt*i) for i in k²]

Dₓₓ_qtt = function_to_qtt_uniform(x -> exp(1im * Δt * x), d)

Dₓₓ_mpo = TTdiag(Dₓₓ_qtt)

λ(ψ_var) = exp(1im*Δt*(2*TensorTrainNumerics.dot(ψ_var,ψ_var)))^(1/n)

steps = 5
n = 2

a=0.01
k=0.1/a
HH(x) = sqrt(2)*x*k^2*a
usol(x) = a*(-1+4/(1+4*HH(x)^2))
ψ₀ = Complex{Float64}[usol(i) for i in x]

ψₜ = complex(function_to_qtt(x -> a*(-1+4/(1+4*HH(x)^2)), d))

for _ in 1:steps
    ψₜ = ψₜ * λ(ψₜ)
    ℱψₜ = fourier_qtto(d) * ψₜ
    k²ℱψₜ = orthogonalize(Dₓₓ_mpo * ℱψₜ)
    ψₜ = orthogonalize(fourier_qtto(d; sign=1.0) * k²ℱψₜ)
end

lines(x,abs.(qtt_to_function(ψₜ)))

