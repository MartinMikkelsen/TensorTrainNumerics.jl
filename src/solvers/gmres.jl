using LinearAlgebra
using TensorTrainNumerics

"""
    tt_gmres(A, u0, b; eps=1e-6, maxit=100, m=20, callback=nothing, verbose=0)

Flexible TT-GMRES for TToperator/TTvector.
- `A`: function (TTvector; eps=...) -> TTvector (matvec)
- `u0`: initial TTvector
- `b`: right-hand side TTvector
- `eps`: required accuracy
- `maxit`: max number of iterations
- `m`: restart parameter
- `callback`: optional callback
- `verbose`: print debug info

Returns: (solution::TTvector, relres::Float64)
"""
function tt_gmres(A, b::TTvector, u0::TTvector; eps=1e-6, maxit=100, m=20, callback=nothing, verbose=0, _iteration=0)
    # Helper: norm for TTvector
    tt_norm(x) = sqrt(dot(x, x))

    # Matvec: allow A to be a TToperator or a function
    matvec = (A isa TToperator) ? (v -> A * v) : A

    maxitexceeded = false
    converged = false

    if verbose > 0
        println("GMRES(m=$m, _iteration=$_iteration, maxit=$maxit)")
    end

    v = Vector{typeof(u0)}(undef, m+1)
    R = fill(NaN, m, m)
    g = zeros(m)
    s = fill(NaN, m)
    c = fill(NaN, m)

    v[1] = b - matvec(u0)
    v[1] = orthogonalize(v[1])
    resnorm = tt_norm(v[1])
    curr_beta = resnorm
    bnorm = tt_norm(b)
    wlen = resnorm
    q = m

    for j in 1:m
        _iteration += 1
        delta = eps / (curr_beta / resnorm)
        if verbose > 0
            println("it = $_iteration, delta = $delta")
        end

        v[j] = (1.0 / wlen) * v[j]
        v[j+1] = matvec(v[j])
        for i in 1:j
            R[i, j] = dot(v[j+1], v[i])
            v[j+1] = v[j+1] - R[i, j] * v[i]
        end
        v[j+1] = orthogonalize(v[j+1])
        wlen = tt_norm(v[j+1])

        for i in 1:j-1
            r1 = R[i, j]
            r2 = R[i+1, j]
            R[i, j] = c[i] * r1 - s[i] * r2
            R[i+1, j] = c[i] * r2 + s[i] * r1
        end
        denom = hypot(wlen, R[j, j])
        s[j] = wlen / denom
        c[j] = -R[j, j] / denom
        R[j, j] = -denom

        g[j] = c[j] * curr_beta
        curr_beta *= s[j]

        if verbose > 0
            println("it = $_iteration, ||r|| = $(curr_beta / bnorm)")
        end

        converged = (curr_beta / bnorm) < eps || (curr_beta / resnorm) < eps
        maxitexceeded = _iteration >= maxit
        if converged || maxitexceeded
            q = j
            break
        end
    end

    y = R[1:q, 1:q] \ g[1:q]
    for idx in 1:q
        u0 = u0 + v[idx] * y[idx]
    end
    u0 = orthogonalize(u0)

    if callback !== nothing
        callback(u0)
    end

    if converged || maxitexceeded
        return u0, resnorm / bnorm
    end
    return tt_gmres(A, b, u0; eps=eps, maxit=maxit, m=m, callback=callback, verbose=verbose, _iteration=_iteration)
end