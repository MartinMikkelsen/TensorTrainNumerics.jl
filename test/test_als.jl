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

