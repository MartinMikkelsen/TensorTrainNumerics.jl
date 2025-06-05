using LinearMaps
using KrylovKit
using LinearAlgebra

"""
    tdvp_step(A::TToperator, x::TTvector, dt; tol=1e-12)

Propagate `x` for a time step `dt` under the generator `A` using
`KrylovKit.exponentiate`. The vector is reconstructed in tensor-train
format after each step.
"""
function tdvp_step(A::TToperator{T}, x::TTvector{T}, dt::Real; tol::Real=1e-12) where {T<:Number}
    Amat = reshape(tto_to_tensor(A), prod(A.tto_dims), prod(A.tto_dims))
    linop = LinearMap(z -> Amat * z, size(Amat))
    xvec = vec(ttv_to_tensor(x))
    yvec = KrylovKit.exponentiate(linop, xvec, dt)
    ytensor = reshape(yvec, A.tto_dims)
    return ttv_decomp(ytensor; index=1, tol=tol)
end

"""
    tdvp_solve(A::TToperator, x::TTvector, t_final; dt=0.1, tol=1e-12)

Evolve `x` from `t=0` to `t_final` using uniform time steps of size `dt`.
Returns the propagated tensor-train vector.
"""
function tdvp_solve(A::TToperator{T}, x::TTvector{T}, t_final::Real; dt::Real=0.1, tol::Real=1e-12) where {T<:Number}
    t = 0.0
    y = x
    while t < t_final - 1e-15
        step = min(dt, t_final - t)
        y = tdvp_step(A, y, step; tol=tol)
        t += step
    end
    return y
end
