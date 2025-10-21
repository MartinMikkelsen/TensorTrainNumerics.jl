using Test
using TensorTrainNumerics
import TensorTrainNumerics: tt_linsolve, power_method
using LinearAlgebra


@testset "tt_linsolve: dispatch and basic behavior" begin
    dims = (2, 2, 2)
    rks  = [1, 2, 2, 1]

    A  = rand_tto(dims, length(dims))  
    b  = rand_tt(dims, rks)
    x0 = rand_tt(dims, rks)

    x_als  = tt_linsolve("als",  A, b, x0)
    @test x_als isa TTvector{Float64, 3}
    @test x_als.N == 3

    x_dmrg = tt_linsolve("dmrg", A, b, x0)
    @test x_dmrg isa TTvector{Float64, 3}
    @test x_dmrg.N == 3

    x_mals = tt_linsolve("mals", A, b, x0)
    @test x_mals isa TTvector{Float64, 3}
    @test x_mals.N == 3

    # Unknown solver string should throw
    @test_throws ErrorException tt_linsolve("unknown", A, b, x0)
end

@testset "Inverse power iteration" begin
    
    dims = (2, 2, 2)
    rks = [1, 2, 2, 1]

    tt_start = rand_tt(dims, rks)

    A = toeplitz_to_qtto(1,2,3,3)

    eigenvalues, eigenvector = power_method(A,tt_start; repeats=8, tt_solver="mals")

    A_dense = qtto_to_matrix(A)
    eigenvalues_dense = real.(eigvals(A_dense))

    atol = abs.(eigenvalues_dense .- eigenvalues) ./ eigenvalues_dense 

end