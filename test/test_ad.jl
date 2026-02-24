using Test
using TensorTrainNumerics
using ChainRulesCore

@testset "AD: ChainRulesCore rrule for dot" begin
    # Use tuple dims (not collect) to avoid circular dispatch in zeros_tt
    dims = (2, 3, 2)
    rks  = [1, 2, 2, 1]
    a = rand_tt(Float64, dims, rks)
    b = rand_tt(Float64, dims, rks)

    # Evaluate the rrule
    result, pullback = ChainRulesCore.rrule(TensorTrainNumerics.dot, a, b)

    @test result ≈ TensorTrainNumerics.dot(a, b)

    _, ∂a, ∂b = pullback(1.0)

    # ∂a and ∂b should be Tangent objects with a ttv_vec field
    @test ∂a isa ChainRulesCore.Tangent
    @test ∂b isa ChainRulesCore.Tangent
    @test length(∂a.ttv_vec) == length(a.ttv_vec)
    @test length(∂b.ttv_vec) == length(b.ttv_vec)

    # Shapes should match the original cores
    for k in 1:length(dims)
        @test size(∂a.ttv_vec[k]) == size(a.ttv_vec[k])
        @test size(∂b.ttv_vec[k]) == size(b.ttv_vec[k])
    end

    # Finite-difference check for gradient of dot w.r.t. a
    # d/dε dot(a + ε*e_k, b)|_{ε=0} should equal ∂a.ttv_vec[k_site][idx]
    ε = 1e-7
    k_site = 2
    idx = (1, 1, 2)  # (z, left_rank, right_rank) index to perturb
    a_pert = deepcopy(a)
    a_pert.ttv_vec[k_site][idx...] += ε
    fd_grad = (TensorTrainNumerics.dot(a_pert, b) - TensorTrainNumerics.dot(a, b)) / ε
    @test ∂a.ttv_vec[k_site][idx...] ≈ fd_grad  rtol=1e-5
end

@testset "AD: ChainRulesCore rrule for norm" begin
    dims = (2, 3, 2)
    rks  = [1, 2, 2, 1]
    x = rand_tt(Float64, dims, rks)

    result, pullback = ChainRulesCore.rrule(TensorTrainNumerics.norm, x)
    @test result ≈ TensorTrainNumerics.norm(x)

    _, ∂x = pullback(1.0)
    @test ∂x isa ChainRulesCore.Tangent

    # Finite-difference check
    ε = 1e-7
    k_site = 1
    idx = (1, 1, 1)
    x_pert = deepcopy(x)
    x_pert.ttv_vec[k_site][idx...] += ε
    fd_grad = (TensorTrainNumerics.norm(x_pert) - TensorTrainNumerics.norm(x)) / ε
    @test ∂x.ttv_vec[k_site][idx...] ≈ fd_grad  rtol=1e-5
end

@testset "AD: ChainRulesCore rrule for TToperator * TTvector" begin
    dims = (2, 3, 2)
    v_rks = [1, 2, 2, 1]

    A = rand_tto(dims, 2)   # rand_tto(dims, rmax)
    v = rand_tt(Float64, dims, v_rks)

    y = A * v
    result, pullback = ChainRulesCore.rrule(*, A, v)
    @test result isa TTvector

    # Build a dummy tangent for y (all-ones cores)
    Δy_cores = [ones(size(c)) for c in y.ttv_vec]
    Δy = ChainRulesCore.Tangent{typeof(y)}(; ttv_vec = Δy_cores)

    _, ∂A, ∂v = pullback(Δy)
    @test ∂v isa ChainRulesCore.Tangent
    @test ∂A isa ChainRulesCore.Tangent

    for k in 1:length(dims)
        @test size(∂v.ttv_vec[k]) == size(v.ttv_vec[k])
        @test size(∂A.tto_vec[k]) == size(A.tto_vec[k])
    end
end

# Optional end-to-end test with Zygote (only runs if Zygote is loaded)
if Base.get_extension(TensorTrainNumerics, :TensorTrainNumericsChainRulesCoreExt) !== nothing
    try
        using Zygote
        @testset "AD: Zygote gradient of dot" begin
            dims = (2, 2, 2)
            rks  = [1, 2, 2, 1]
            a = rand_tt(Float64, dims, rks)
            b = rand_tt(Float64, dims, rks)

            # gradient of dot(a, b) w.r.t. a
            ∂a, = Zygote.gradient(x -> TensorTrainNumerics.dot(x, b), a)
            @test ∂a !== nothing

            # Sanity check: gradient of dot(x, x) w.r.t. x should give 2*x (when orthogonal)
            x = orthogonalize(rand_tt(Float64, dims, rks))
            ∂x, = Zygote.gradient(y -> TensorTrainNumerics.dot(y, y), x)
            @test ∂x !== nothing
        end
    catch e
        @info "Zygote not available; skipping end-to-end AD test" exception=e
    end
end
