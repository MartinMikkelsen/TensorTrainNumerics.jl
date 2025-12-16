using ForwardDiff
using LinearAlgebra

function tt_to_vec(ψ::TTvector{T, N}) where {T <: Number, N}
    return vcat([vec(core) for core in ψ.ttv_vec]...)
end

function vec_to_tt(θ::AbstractVector{T}, dims::NTuple{N, Int64}, rks::Vector{Int64}) where {T <: Number, N}
    d = length(dims)
    ttv_vec = Vector{Array{T, 3}}(undef, d)
    idx = 1
    for k in 1:d
        n = dims[k] * rks[k] * rks[k + 1]
        ttv_vec[k] = reshape(θ[idx:(idx + n - 1)], dims[k], rks[k], rks[k + 1])
        idx += n
    end
    return TTvector{T, N}(d, ttv_vec, dims, copy(rks), zeros(Int64, d))
end

function tto_to_vec(A::TToperator{T, N}) where {T <: Number, N}
    return vcat([vec(core) for core in A.tto_vec]...)
end

function vec_to_tto(θ::AbstractVector{T}, dims::NTuple{N, Int64}, rks::Vector{Int64}) where {T <: Number, N}
    d = length(dims)
    tto_vec = Vector{Array{T, 4}}(undef, d)
    idx = 1
    for k in 1:d
        n = dims[k] * dims[k] * rks[k] * rks[k + 1]
        tto_vec[k] = reshape(θ[idx:(idx + n - 1)], dims[k], dims[k], rks[k], rks[k + 1])
        idx += n
    end
    return TToperator{T, N}(d, tto_vec, dims, copy(rks), zeros(Int64, d))
end

function tt_add(x::TTvector{T, N}, y::TTvector{S, N}) where {T <: Number, S <: Number, N}
    @assert x.ttv_dims == y.ttv_dims
    R = promote_type(T, S)
    d = x.N
    ttv_vec = Vector{Array{R, 3}}(undef, d)
    rks = x.ttv_rks + y.ttv_rks
    rks[1] = 1
    rks[d + 1] = 1
    for k in 1:d
        ttv_vec[k] = zeros(R, x.ttv_dims[k], rks[k], rks[k + 1])
    end
    ttv_vec[1][:, :, 1:x.ttv_rks[2]] = x.ttv_vec[1]
    ttv_vec[1][:, :, (x.ttv_rks[2] + 1):rks[2]] = y.ttv_vec[1]
    for k in 2:(d - 1)
        ttv_vec[k][:, 1:x.ttv_rks[k], 1:x.ttv_rks[k + 1]] = x.ttv_vec[k]
        ttv_vec[k][:, (x.ttv_rks[k] + 1):rks[k], (x.ttv_rks[k + 1] + 1):rks[k + 1]] = y.ttv_vec[k]
    end
    ttv_vec[d][:, 1:x.ttv_rks[d], 1] = x.ttv_vec[d]
    ttv_vec[d][:, (x.ttv_rks[d] + 1):rks[d], 1] = y.ttv_vec[d]
    return TTvector{R, N}(d, ttv_vec, x.ttv_dims, rks, zeros(Int64, d))
end

function tt_scale(a::S, x::TTvector{T, N}) where {S <: Number, T <: Number, N}
    R = promote_type(S, T)
    d = x.N
    ttv_vec = Vector{Array{R, 3}}(undef, d)
    for k in 1:d
        ttv_vec[k] = R.(x.ttv_vec[k])
    end
    i = findfirst(==(0), x.ttv_ot)
    i === nothing && (i = 1)
    ttv_vec[i] = a .* ttv_vec[i]
    return TTvector{R, N}(d, ttv_vec, x.ttv_dims, copy(x.ttv_rks), copy(x.ttv_ot))
end

function tt_sub(x::TTvector{T, N}, y::TTvector{S, N}) where {T <: Number, S <: Number, N}
    return tt_add(x, tt_scale(-one(S), y))
end

function tt_dot(x::TTvector{T, N}, y::TTvector{S, N}) where {T <: Number, S <: Number, N}
    @assert x.ttv_dims == y.ttv_dims
    R = promote_type(T, S)
    x_rks = x.ttv_rks
    y_rks = y.ttv_rks
    out = zeros(R, maximum(x_rks), maximum(y_rks))
    out[1, 1] = one(R)
    for k in eachindex(x.ttv_dims)
        M_new = zeros(R, x_rks[k + 1], y_rks[k + 1])
        for α in 1:x_rks[k + 1], β in 1:y_rks[k + 1]
            s = zero(R)
            for z in 1:x.ttv_dims[k], α′ in 1:x_rks[k], β′ in 1:y_rks[k]
                s += x.ttv_vec[k][z, α′, α] * y.ttv_vec[k][z, β′, β] * out[α′, β′]
            end
            M_new[α, β] = s
        end
        out[1:x_rks[k + 1], 1:y_rks[k + 1]] .= M_new
    end
    return out[1, 1]
end

function tto_mul(A::TToperator{T, M}, x::TTvector{S, M}) where {T <: Number, S <: Number, M}
    @assert A.tto_dims == x.ttv_dims
    R = promote_type(T, S)
    d = A.N
    A_rks = A.tto_rks
    x_rks = x.ttv_rks
    rks = A_rks .* x_rks
    ttv_vec = Vector{Array{R, 3}}(undef, d)
    for k in 1:d
        n = A.tto_dims[k]
        core = zeros(R, n, rks[k], rks[k + 1])
        for i in 1:n, j in 1:n, α_A in 1:A_rks[k], α_x in 1:x_rks[k], β_A in 1:A_rks[k + 1], β_x in 1:x_rks[k + 1]
            α = (α_A - 1) * x_rks[k] + α_x
            β = (β_A - 1) * x_rks[k + 1] + β_x
            core[i, α, β] += A.tto_vec[k][i, j, α_A, β_A] * x.ttv_vec[k][j, α_x, β_x]
        end
        ttv_vec[k] = core
    end
    return TTvector{R, M}(d, ttv_vec, A.tto_dims, rks, zeros(Int64, d))
end

function tt_gradient(f::Function, ψ::TTvector{T, N}) where {T <: Real, N}
    θ = tt_to_vec(ψ)
    loss_fn = θ_inner -> f(vec_to_tt(θ_inner, ψ.ttv_dims, ψ.ttv_rks))
    grad_vec = ForwardDiff.gradient(loss_fn, θ)
    return vec_to_tt(grad_vec, ψ.ttv_dims, ψ.ttv_rks), loss_fn(θ)
end

function tto_gradient(f::Function, A::TToperator{T, N}) where {T <: Real, N}
    θ = tto_to_vec(A)
    loss_fn = θ_inner -> f(vec_to_tto(θ_inner, A.tto_dims, A.tto_rks))
    grad_vec = ForwardDiff.gradient(loss_fn, θ)
    return vec_to_tto(grad_vec, A.tto_dims, A.tto_rks), loss_fn(θ)
end

function tt_joint_gradient(f::Function, A::TToperator{T, N}, ψ::TTvector{T, N}) where {T <: Real, N}
    θ_A = tto_to_vec(A)
    θ_ψ = tt_to_vec(ψ)
    n_A = length(θ_A)
    θ = vcat(θ_A, θ_ψ)
    function loss_fn(θ_inner)
        A_inner = vec_to_tto(θ_inner[1:n_A], A.tto_dims, A.tto_rks)
        ψ_inner = vec_to_tt(θ_inner[(n_A + 1):end], ψ.ttv_dims, ψ.ttv_rks)
        return f(A_inner, ψ_inner)
    end
    grad_vec = ForwardDiff.gradient(loss_fn, θ)
    grad_A = vec_to_tto(grad_vec[1:n_A], A.tto_dims, A.tto_rks)
    grad_ψ = vec_to_tt(grad_vec[(n_A + 1):end], ψ.ttv_dims, ψ.ttv_rks)
    return grad_A, grad_ψ, loss_fn(θ)
end

abstract type TTLossFunction end

struct NormLoss <: TTLossFunction end

struct DistanceLoss{T, N} <: TTLossFunction
    target::TTvector{T, N}
end

struct ExpectationLoss{T, N} <: TTLossFunction
    H::TToperator{T, N}
end

struct RayleighLoss{T, N} <: TTLossFunction
    H::TToperator{T, N}
end

struct CustomLoss{F} <: TTLossFunction
    f::F
end

function evaluate_loss(::NormLoss, ψ::TTvector)
    return tt_dot(ψ, ψ)
end

function evaluate_loss(loss::DistanceLoss, ψ::TTvector)
    diff = tt_sub(ψ, loss.target)
    return tt_dot(diff, diff)
end

function evaluate_loss(loss::ExpectationLoss, ψ::TTvector)
    return tt_dot(ψ, tto_mul(loss.H, ψ))
end

function evaluate_loss(loss::RayleighLoss, ψ::TTvector)
    return tt_dot(ψ, tto_mul(loss.H, ψ)) / tt_dot(ψ, ψ)
end

function evaluate_loss(loss::CustomLoss, ψ::TTvector)
    return loss.f(ψ)
end

function tt_gradient(loss::TTLossFunction, ψ::TTvector{T, N}) where {T <: Real, N}
    return tt_gradient(ψ_inner -> evaluate_loss(loss, ψ_inner), ψ)
end

function tto_gradient(loss::TTLossFunction, A::TToperator{T, N}) where {T <: Real, N}
    return tto_gradient(A_inner -> evaluate_loss(loss, A_inner), A)
end

function tt_gradient_descent(
        loss::TTLossFunction,
        ψ₀::TTvector{T, N};
        lr::T = T(0.01),
        maxiter::Int = 100,
        tol::T = T(1e-8),
        normalize::Bool = false,
        verbose::Bool = false
    ) where {T <: Real, N}
    ψ = deepcopy(ψ₀)
    history = T[]
    for iter in 1:maxiter
        grad, loss_val = tt_gradient(loss, ψ)
        push!(history, loss_val)
        if verbose && (iter % 10 == 0 || iter == 1)
            @info "Iteration $iter" loss = loss_val grad_norm = norm(grad)
        end
        grad_norm = norm(grad)
        if grad_norm < tol
            verbose && @info "Converged at iteration $iter"
            break
        end
        θ = tt_to_vec(ψ) - lr * tt_to_vec(grad)
        ψ = vec_to_tt(θ, ψ.ttv_dims, ψ.ttv_rks)
        if normalize
            ψ = (1 / norm(ψ)) * ψ
        end
    end
    return ψ, history
end