using TensorTrainNumerics
using ForwardDiff
using LinearAlgebra
import TensorTrainNumerics: dot

dims = (2, 2, 2)
rks = [1, 3, 3, 1]
ψ = rand_tt(dims, rks)

grad, val = tt_gradient(x -> TensorTrainNumerics.dot(x, x), ψ)
println("Squared norm: ", val)
println("Gradient norm: ", norm(grad))

loss = NormLoss()
grad, val = tt_gradient(loss, ψ)

target = rand_tt(dims, rks)
loss = DistanceLoss(target)
grad, val = tt_gradient(loss, ψ)
println("Distance squared: ", val)

d = 6
H = Δ(d)
ψ = qtt_sin(d, λ = π)
loss = ExpectationLoss(H)
grad, val = tt_gradient(loss, ψ)
println("⟨ψ|H|ψ⟩ = ", val)

loss = RayleighLoss(H)
grad, val = tt_gradient(loss, ψ)
println("Rayleigh quotient: ", val)

loss = CustomLoss(x -> dot(x, H * x) + 0.1 * dot(x, x))
grad, val = tt_gradient(loss, ψ)

ψ₀ = rand_tt(target.ttv_dims, target.ttv_rks)
loss = DistanceLoss(target)
ψ_opt, history = tt_gradient_descent(
    loss, ψ₀;
    lr = 0.01,
    maxiter = 200,
    verbose = true
)
println("Initial loss: ", history[1])
println("Final loss: ", history[end])

d = 8
H = Δ(d)
ψ₀ = rand_tt(ntuple(_ -> 2, d), [1; fill(4, d-1); 1])
loss = RayleighLoss(H)
ψ_opt, history = tt_gradient_descent(
    loss, ψ₀;
    lr = 0.001,
    maxiter = 500,
    normalize = true,
    verbose = true
)

# Gradient w.r.t. operator
d = 4
H = rand_tto(ntuple(_ -> 2, d), 3)
ψ = qtt_sin(d, λ = π)
grad_H, val = tto_gradient(A -> dot(ψ, A * ψ), H)

# Joint gradient w.r.t. operator and state
grad_H, grad_ψ, val = tt_joint_gradient((A, x) -> dot(x, A * x), H, ψ)

