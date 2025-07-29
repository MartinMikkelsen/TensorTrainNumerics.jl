using Test
using Random

import TensorTrainNumerics: update_H!

@testset "update_H!" begin
    n, r1, r2, r3 = 2, 1, 1, 1
    x_vec = randn(Float64, n, r1, r2)
    A_vec = randn(Float64, n, n, r3, r1)
    Hi = randn(Float64, r3, r2, r2)
    Him = zeros(Float64, r1, r2, r2)
    update_H!(x_vec, A_vec, Hi, Him)
    @test size(Him) == (r1, r2, r2)
    @test eltype(Him) <: Number
end


@testset "ALS Tests" begin
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]

    tt_start = rand_tt(dims, rks)

    A_dims = (2, 2, 2)
    A_rks = [1, 2, 2, 1]
    A = rand_tto(A_dims, 3)

    b = rand_tt(dims, rks)

    # Test ALS linear solver
    tt_opt = als_linsolve(A, b, tt_start)
    @test typeof(tt_opt) == TTvector{Float64, 3}
    @test tt_opt.N == 3

end