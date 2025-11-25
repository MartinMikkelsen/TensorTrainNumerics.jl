using CairoMakie
using TensorTrainNumerics
using TensorOperations
using LinearAlgebra

function laplace_qtto(d::Int; λ::Float64 = 1.0, K::Int = 15)
    @assert d ≥ 1
    P = cheb_lobatto_grid(K)
    r = K + 1
    r2 = r * r
    
    @inline double_to_linear(α, β) = α * r + β + 1
    @inline linear_to_double(idx) = ((idx - 1) ÷ r, (idx - 1) % r)
    
    Δt = 2.0^(-d)
    
    # A_L: shape (1, 2, 2, r²) - evaluate exponential at first level
    AL_tensor = zeros(Float64, 1, 2, 2, r2)
    for σ in 0:1, τ in 0:1
        for idx_out in 1:r2
            (γ, δ) = linear_to_double(idx_out)
            cγ = P.grid[γ + 1]
            cδ = P.grid[δ + 1]
            
            # First level contribution
            s = 0.5 * (σ + cγ)
            t = 0.5 * (τ + cδ)
            
            AL_tensor[1, σ+1, τ+1, idx_out] = exp(-λ * s * t) * Δt
        end
    end
    
    # A_C: shape (r², 2, 2, r²) - interpolation cores (depth-dependent!)
    AC_tensors = []
    for m in 2:(d-1)
        AC = zeros(Float64, r2, 2, 2, r2)
        
        for σ in 0:1, τ in 0:1
            for idx_in in 1:r2, idx_out in 1:r2
                (α, β) = linear_to_double(idx_in)
                (γ, δ) = linear_to_double(idx_out)
                
                cα = P.grid[α + 1]
                cβ = P.grid[β + 1]
                cγ = P.grid[γ + 1]
                cδ = P.grid[δ + 1]
                
                # Interpolation
                x = 0.5 * (σ + cγ)
                y = 0.5 * (τ + cδ)
                Pα = lagrange_eval(P, α, x)
                Pβ = lagrange_eval(P, β, y)
                
                # Cross terms at level m
                s_prev = cα * 2.0^(-(m-1))
                t_prev = cβ * 2.0^(-(m-1))
                s_new = σ * 2.0^(-m)
                t_new = τ * 2.0^(-m)
                
                exponent = -λ * (s_prev * t_new + s_new * t_prev + s_new * t_new)
                
                AC[idx_in, σ+1, τ+1, idx_out] = Pα * Pβ * exp(exponent)
            end
        end
        
        push!(AC_tensors, AC)
    end
    
    # A_R: shape (r², 2, 2, 1) - final interpolation
    AR_tensor = zeros(Float64, r2, 2, 2, 1)
    for σ in 0:1, τ in 0:1
        for idx_in in 1:r2
            (α, β) = linear_to_double(idx_in)
            
            cα = P.grid[α + 1]
            cβ = P.grid[β + 1]
            
            # Final interpolation
            Pα = lagrange_eval(P, α, 0.5 * σ)
            Pβ = lagrange_eval(P, β, 0.5 * τ)
            
            # Final cross terms
            s_prev = cα * 2.0^(-(d-1))
            t_prev = cβ * 2.0^(-(d-1))
            s_new = σ * 2.0^(-d)
            t_new = τ * 2.0^(-d)
            
            exponent = -λ * (s_prev * t_new + s_new * t_prev + s_new * t_new)
            
            AR_tensor[idx_in, σ+1, τ+1, 1] = Pα * Pβ * exp(exponent)
        end
    end
    
    # Now reshape and convert to TToperator format (matching interpolating_qtt)
    tensors = [AL_tensor]
    for AC in AC_tensors
        push!(tensors, AC)
    end
    push!(tensors, AR_tensor)
    
    # Convert to TToperator format
    N_ = length(tensors)
    cores = Vector{Array{Float64, 4}}(undef, N_)
    
    for i in 1:N_
        T = tensors[i]
        # T has shape (r_prev, n_σ, n_τ, r_next)
        # Need to convert to (n_σ, n_τ, r_prev, r_next)
        T = permutedims(T, (2, 3, 1, 4))
        cores[i] = T
    end
    
    dims = ntuple(_ -> 2, d)
    rks = vcat(1, fill(r2, d - 1), 1)
    ot = zeros(Int, d)
    
    return TToperator{Float64, d}(d, cores, dims, rks, ot)
end

function discrete_laplace_sine(ω, λ, s_vals, Δt, N)
    # Discrete sum approximation
    result = zeros(length(s_vals))
    for (i, s) in enumerate(s_vals)
        sum_val = 0.0
        for k in 0:(N-1)
            t = k * Δt
            sum_val += sin(ω * t) * exp(-λ * s * t)
        end
        result[i] = sum_val * Δt  # Include discretization step
    end
    return result
end

# Parameters
ω = 5.0
d = 8
N = 2^d

Δt = 1.0 / N  # Discretization step in [0, 1)
t_vals = collect(0:(N-1)) .* Δt  # Input points in [0, 1)
s_vals = collect(0:(N-1)) .* Δt  # Output points in [0, 1)

f = x -> sin(ω*x)
qtt = interpolating_qtt(f, d, 10)
L = laplace_qtto(d; K=15)
F_qtt = L * (qtt)
approx = (qtt_to_function(F_qtt))

function discrete_laplace_sine(ω, λ, s_vals, t_vals, Δt)
    result = zeros(length(s_vals))
    for (i, s) in enumerate(s_vals)
        sum_val = 0.0
        for t in t_vals
            sum_val += sin(ω * t) * exp(-λ * s * t)
        end
        result[i] = sum_val * Δt
    end
    return result
end

analytical_discrete = discrete_laplace_sine(ω, 1.0, s_vals, t_vals, Δt)

let
    fig = Figure()
    ax = Axis(fig[1,1], 
              title="Laplace Transform Approximation",
              xlabel="s", 
              ylabel="F(s)")
    lines!(ax, s_vals, analytical_discrete, label="Analytical (discrete)", linewidth=2)
    lines!(ax, s_vals, approx, linestyle=:dash, color=:red, label="QTT", linewidth=2)
    axislegend(ax)
    fig
end


function laplace_matrix(d::Int; λ::Float64 = 1.0)
    N = 2^d
    L = zeros(N, N)
    Δt = 1.0 / N
    
    for i in 0:(N-1), j in 0:(N-1)
        s = i * Δt
        t = j * Δt
        L[i+1, j+1] = exp(-λ * s * t) * Δt
    end
    return L
end

# Test with dense matrix
L_dense = laplace_matrix(d; λ=1.0)
f_vec = [f(t) for t in t_vals]
approx_dense = L_dense * f_vec

