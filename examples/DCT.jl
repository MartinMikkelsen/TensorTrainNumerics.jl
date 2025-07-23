using CairoMakie
using TensorTrainNumerics
using SpecialFunctions
using LinearAlgebra

function gauss_chebyshev_lobatto(n; shifted=true)
    x = [cos(π * j / (n - 1)) for j=0:n-1]
    w = π / (n-1) * ones(n)
    w[1] /= 2
    w[end] /= 2

    if shifted
        x .= (x .+ 1) ./ 2
        w .= w ./ 2
    end

    return x, w
end

function qtt_chebyshev(n, d)
    out = zeros_tt(2, d, 2)
    N = 2^d

    x_nodes, _ = gauss_chebyshev_lobatto(N; shifted=true)
    θ = acos.(clamp.(2 .* x_nodes .- 1, -1.0, 1.0))
    # First core
    out.ttv_vec[1][1,1,:] = [cos(n * θ[1]); -sin(n * θ[1])]
    out.ttv_vec[1][2,1,:] = [cos(n * θ[2^(d-1)+1]); -sin(n * θ[2^(d-1)+1])]

    # Middle cores
    for k in 2:d-1
        out.ttv_vec[k][1,:,:] .= [1.0 0.0; 0.0 1.0]
        idx = 2^(d-k) + 1
        out.ttv_vec[k][2,:,:] .= [cos(n * θ[idx]) -sin(n * θ[idx]);
                                  sin(n * θ[idx])  cos(n * θ[idx])]
    end

    # Final core
    out.ttv_vec[d][1,:,1] .= [1.0, 0.0]
    out.ttv_vec[d][2,:,1] .= [cos(n * θ[2]), sin(n * θ[2])]

    return out
end

n = 7
d = 8
A = qtt_chebyshev(n, d)

f_vals = matricize(A, d)
x_points, w = gauss_chebyshev_lobatto(2^d; shifted=true)
g(x) = cos(n*acos(2*x-1))

let 
    fig = Figure()
    ax = Axis(fig[1,1], title="T₅(x) on shifted Chebyshev nodes", xlabel="x", ylabel="T₅(x)")
    lines!(ax, x_points, f_vals)
    lines!(ax, x_points, g.(x_points), color=:red, label="cos(5*acos(2x-1))", linestyle=:dash)
    fig
end

max = 15

C_chebs = [qtt_chebyshev(n, d) for n in 0:max] 
C_cos = [qtt_chebyshev(n, d) for n in 0:max]
C_fun = [qtt_function(test_function, d) for n in 0:max]

function chebyshev_series_qtt(f, d; max=10)
    C_chebs = [qtt_chebyshev(n, d) for n in 0:max]
    f_qtt = qtt_function(f, d)

    c = zeros(Float64, max+1)
    Q = c[1] * C_chebs[1] 

    for n in 1:max
        c[n+1] = TensorTrainNumerics.dot(f_qtt, C_chebs[n+1])  # indexing from 1
        Q = Q + c[n+1] * C_chebs[n+1]
    end

    return Q, c
end

Q, coeffs = chebyshev_series_qtt(test_function, d)

let 
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="x", ylabel="f(x)", title="Chebyshev QTT Reconstruction")
    xes, _ = gauss_chebyshev_lobatto(2^d; shifted=true)  # Chebyshev nodes

    f_exact = test_function.(xes)                        # Ground truth
    f_qtt    = matricize(Q, d)                           # Reconstructed f(x)

    lines!(ax, xes, f_qtt, label="QTT Reconstruction")
    lines!(ax, xes, f_exact, color=:red, linestyle=:dash, label="Exact f(x)")
    
    axislegend(ax)
    fig
end



