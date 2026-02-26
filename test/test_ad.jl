using Test
using TensorTrainNumerics
using DifferentiationInterface
using ForwardDiff

# f(tt) = dot(tt, tt) = norm(tt)^2.
# Using the same TTvector for both args means the promoted TTvector{Dual} satisfies
# dot(::TTvector{T}, ::TTvector{T}) — no mixed-type issues with ForwardDiff.
normsq(tt) = real(TensorTrainNumerics.dot(tt, tt))

@testset "AD: tt_core_gradient" begin
    dims = (2, 3, 2)
    rks  = [1, 2, 2, 1]
    x = rand_tt(Float64, dims, rks)

    grad = tt_core_gradient(normsq, x, 2, AutoForwardDiff())
    @test size(grad) == size(x.ttv_vec[2])

    # finite-difference check at one index
    ε = 1e-6
    idx = (1, 1, 2)
    x_pert = deepcopy(x)
    x_pert.ttv_vec[2][idx...] += ε
    fd = (normsq(x_pert) - normsq(x)) / ε
    @test grad[idx...] ≈ fd  rtol=1e-4
end

@testset "AD: tt_gradient" begin
    dims = (2, 3, 2)
    rks  = [1, 2, 2, 1]
    x = rand_tt(Float64, dims, rks)

    grads = tt_gradient(normsq, x, AutoForwardDiff())

    @test length(grads) == x.N
    for k in 1:x.N
        @test size(grads[k]) == size(x.ttv_vec[k])
    end
end

@testset "AD: tt_core_hessian" begin
    dims = (2, 2, 2)
    rks  = [1, 2, 2, 1]
    x = rand_tt(Float64, dims, rks)

    H = tt_core_hessian(normsq, x, 2, AutoForwardDiff())

    n = prod(size(x.ttv_vec[2]))
    @test size(H) == (n, n)
    @test H ≈ H'  rtol=1e-8
end

@testset "AD: tt_core_jacobian" begin
    dims = (2, 2, 2)
    rks  = [1, 2, 2, 1]
    x = rand_tt(Float64, dims, rks)

    # Two-output function: [norm², 2·norm²]. Row 2 should be exactly 2× row 1.
    f_vec = tt -> [real(TensorTrainNumerics.dot(tt, tt)),
                   2*real(TensorTrainNumerics.dot(tt, tt))]
    J = tt_core_jacobian(f_vec, x, 2, AutoForwardDiff())

    n = prod(size(x.ttv_vec[2]))
    @test size(J) == (2, n)
    @test J[2, :] ≈ 2 * J[1, :]  rtol=1e-8
end

@testset "AD: tt_core_curl" begin
    dims = (2, 2, 2)
    rks  = [1, 2, 2, 1]
    x = rand_tt(Float64, dims, rks)

    n = prod(size(x.ttv_vec[2]))
    import Random; Random.seed!(99)
    A = randn(n, n)

    # f(tt) = A * vec(core_2): Jacobian is A, so curl = A - A'
    f_curl = tt -> A * vec(tt.ttv_vec[2])
    C = tt_core_curl(f_curl, x, 2, AutoForwardDiff())

    @test size(C) == (n, n)
    @test C ≈ A - A'   rtol=1e-5   # matches expected curl
    @test C ≈ -C'      rtol=1e-8   # antisymmetry
end
