using TensorTrainNumerics
using ProgressMeter
using Plots

function euler_method(A::TToperator, u₀::TTvector, steps::Vector{Float64}; normalize::Bool=true, return_error::Bool=false)
    solution = copy(u₀)
    I = id_tto(A.N)

    @showprogress for h in steps
        update = A * solution
        solution = orthogonalize(solution + h * update)
        if normalize
            norm² = dot(solution, solution)
            solution = (1 / sqrt(norm²)) * solution
        end
    end

    if return_error
        h = steps[end]
        residual = solution - (I + h * A) * solution
        rel_error = norm(residual) / norm(solution)
        return solution, rel_error
    end

    return solution
end

function implicit_euler_method(
    A::TToperator,
    u₀::TTvector,
    guess::TTvector,
    steps::Vector{Float64};
    normalize::Bool=true,
    return_error::Bool=false,
    tt_solver::String="mals",
    kwargs...
)
    solution = copy(u₀)
    u_prev = copy(u₀)
    I = id_tto(A.N)

    @showprogress for h in steps
        M = I - h * A

        next = tt_solver == "mals" ? mals_linsolve(M, solution, guess; kwargs...) :
               tt_solver == "als"  ? als_linsolve(M, solution, guess; kwargs...) :
               tt_solver == "dmrg" ? dmrg_linsolve(M, solution, guess; kwargs...) :
               error("Unknown TT solver: $tt_solver")

        if normalize
            next = next / norm(next)
        end

        u_prev = solution
        solution = orthogonalize(next)
        guess = solution
    end

    if return_error
        h = steps[end]
        M = I - h * A
        residual = M * solution - u_prev
        rel_error = norm(residual) / norm(solution)
        return solution, rel_error
    end

    return solution
end

function crank_nicholson_method(
    A::TToperator,
    u₀::TTvector,
    guess::TTvector,
    steps::Vector{Float64};
    normalize::Bool=true,
    return_error::Bool=false,
    tt_solver::String="mals",
    kwargs...
)
    solution = copy(u₀)
    u_prev = copy(u₀)
    I = id_tto(A.N)

    @showprogress for h in steps
        LHS = I - (h/2) * A
        RHS = (I + (h/2) * A) * solution

        next = tt_solver == "mals" ? mals_linsolve(LHS, RHS, guess; kwargs...) :
               tt_solver == "als"  ? als_linsolve(LHS, RHS, guess; kwargs...) :
               tt_solver == "dmrg" ? dmrg_linsolve(LHS, RHS, guess; kwargs...) :
               error("Unknown TT solver: $tt_solver")

        if normalize
            next = next / norm(next)
        end

        u_prev = solution
        solution = orthogonalize(next)
        guess = solution
    end

    if return_error
        h = steps[end]
        LHS = I - (h/2) * A
        RHS = (I + (h/2) * A) * u_prev
        residual = LHS * solution - RHS
        rel_error = norm(residual) / norm(solution)
        return solution, rel_error
    end

    return solution
end

function build_block_operator(A::TToperator, steps::Vector{Float64})
  d      = A.N
  N_t    = length(steps)
  τ      = steps[1]
  d_time = Int(round(log2(N_t+1)))

  Isp = id_tto(d)
  M   = Isp + (τ/2)*A
  B   = -1.0*(Isp) + (τ/2)*A

  It = id_tto(d_time)
  # subdiagonal shift in time:
  J  = zeros(2,2); J[1,2] = 1
  Smat = J'
  St = shift_tto(2, d_time; s = [Smat for _ in 1:d_time])

  return kron(It, M) + kron(St, B)
end

function build_rhs_block(
    A::TToperator,            # spatial operator
    y0::TTvector,             # initial condition y⁰
    steps::Vector{Float64}    # time steps [τ, τ, …, τ]
)
    d      = A.N
    N_t    = length(steps)
    τ      = steps[1]                   # assume uniform
    d_time = Int(round(log2(N_t + 1)))

    # 1) Spatial initial contribution
    Isp = id_tto(d)
    M0  = (Isp - (τ/2)*A) * y0                 # TTvector of length Nₓ

    rhs = kron(M0, qtt_basis_vector(d_time, 1))

    return rhs
end
function build_rhs_source_block(
    f::Vector{<:TTvector},  # any Vector of TTvectors
    steps::Vector{Float64}
)
    N_t     = length(steps)
    τ       = steps[1]
    d_time  = Int(round(log2(N_t + 1)))
    d_space = f[1].N                     # spatial TT-order from the first element
    total_d = d_space + d_time
    # zero out a full space–time TTvector
    rhs = zeros_tt(Float64,
                   ntuple(_ -> 2, total_d),
                   ones(Int, total_d + 1))

    for k in 0:(N_t-1)
        w    = (τ/2) * (f[k+1] + f[k+2])      # spatial TTvector
        tvec = qtt_basis_vector(d_time, k+2) # temporal TTvector
        rhs += kron(w, tvec)
    end

    return rhs
end
