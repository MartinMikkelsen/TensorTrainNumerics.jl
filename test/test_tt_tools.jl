using Test
using LinearAlgebra
using TensorTrains

@testset "tt_up_rks" begin
	x = randn(4,4,4,4)
	x_tt = ttv_decomp(x,tol=0.1)
	x = ttv_to_tensor(x_tt)
	y_tt = tt_up_rks(x_tt,20)
	@test isapprox(x,ttv_to_tensor(y_tt),atol=1e-10)
	z_tt = tt_up_rks(x_tt,20,ϵ_wn=1e-10)
	@test isapprox(x,ttv_to_tensor(z_tt),atol=1e-6)
end

@testset "TT operations" begin
    n = 3
    d = 3
    L = randn(n,n,n,n,n,n)
    S = randn(n,n,n,n,n,n)
    L_tto = tto_decomp(L)
    S_tto = tto_decomp(S)
    x = randn(n,n,n)
    y = randn(n,n,n)
    x_tt = ttv_decomp(x)
    y_tt = ttv_decomp(y)
    a = randn()
    #mult_a_tt
    @test isapprox(a.*x, ttv_to_tensor(a*x_tt))
    #tt_add
    z_tt = x_tt+y_tt
    @test(isapprox(ttv_to_tensor(z_tt),x+y)) 
    #tt_dot
    @test isapprox(x[:]'*(x+y)[:], dot(x_tt,z_tt))
    #tto_add
    A_tto = L_tto+S_tto
    @test(isapprox(tto_to_tensor(A_tto),L+S)) #tto_add test
    #tto*ttv multiplication
    z = reshape(L,n^d,:)*x[:]
    z_tt = L_tto*x_tt
    @test(isapprox(ttv_to_tensor(z_tt)[:],z))
    #tto*tto multiplication
    A = reshape(L,n^d,:)*reshape(S,n^d,:)
    A_tto = L_tto*S_tto
    @test(isapprox(reshape(tto_to_tensor(A_tto),n^d,:),A))
end

@testset "TT orthogonalize" begin
    dims = Tuple(2*ones(Int,8))
    rks = [1,2,4,8,8,8,4,2,1]
    x_tt = rand_tt(dims,rks)
    y_tt = orthogonalize(x_tt)
    i = rand(2:8)
    @test(isapprox(sum(y_tt.ttv_vec[i][k,:,:]*y_tt.ttv_vec[i][k,:,:]' for k in 1:dims[i]),Matrix(I,rks[i],rks[i])))
    @test(size(y_tt.ttv_vec[i])==(dims[i],rks[i],rks[i+1]))
    @test(isapprox(ttv_to_tensor(x_tt),ttv_to_tensor(y_tt)))
end

