using Test
using TensorTrainNumerics
using LinearAlgebra
import TensorTrainNumerics: shift_matrix, qtto_prolongation

@testset "Laplacian Tests" begin
    @testset "Δ_DD Tests" begin
        @testset "n=5" begin
            expected = [2.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 2.0]
            result = Δ_DD(5)
            @test result == expected
        end
    end

    @testset "Δ_NN Tests" begin
        @testset "n=5" begin
            expected = [1.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 1.0]
            result = Δ_NN(5)
            @test result == expected
        end
    end

    @testset "Δ_DN Tests" begin
        @testset "n=5" begin
            expected = [2.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 1.0]
            result = Δ_DN(5)
            @test result == expected
        end
    end

    @testset "Δ_ND Tests" begin
        @testset "n=5" begin
            expected = [1.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 2.0]
            result = Δ_ND(5)
            @test result == expected
        end
    end

    @testset "Δ_Periodic Tests" begin
        @testset "n=5" begin
            expected = [2.0 -1.0 0.0 0.0 -1.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        -1.0 0.0 0.0 -1.0 2.0]
            result = Δ_Periodic(5)
            @test result == expected
        end
    end

    @testset "Δ_tto Tests" begin
        @testset "n=2, d=2" begin
            dims = (2, 2)
            rks = [1, 2, 1]
            expected = Δ_tto(2, 2, Δ_DD)
            
            # Check the structure and content of the TToperator
            @test expected.tto_dims == dims
            @test expected.tto_rks == rks
        end
    end
end
@testset "QTT_Tridiagonal_Toeplitz" begin
    α, β, γ = 2.0, -1.0, -1.0
    l = 3

    qtt = QTT_Tridiagonal_Toeplitz(α, β, γ, l)

    # Check dimensions
    @test qtt.N == l
    @test qtt.tto_dims == (2, 2, 2)
    @test qtt.tto_rks == [1, 3, 3, 1]

    # Check first core
    first_core = qtt.tto_vec[1]
    @test size(first_core) == (2, 2, 1, 3)
    @test first_core[:, :, 1, 1] == [1 0; 0 1]
    @test first_core[:, :, 1, 2] == [0 0; 1 0]
    @test first_core[:, :, 1, 3] == [0 1; 0 0]

    # Check middle core
    middle_core = qtt.tto_vec[2]
    @test size(middle_core) == (2, 2, 3, 3)
    @test middle_core[:, :, 1, 1] == [1 0; 0 1]
    @test middle_core[:, :, 1, 2] == [0 0; 1 0]
    @test middle_core[:, :, 1, 3] == [0 1; 0 0]
    @test middle_core[:, :, 2, 2] == [0 1; 0 0]
    @test middle_core[:, :, 3, 3] == [0 0; 1 0]

    # Check last core
    last_core = qtt.tto_vec[3]
    @test size(last_core) == (2, 2, 3, 1)
    @test last_core[:, :, 1, 1] == α * [1 0; 0 1] + β * [0 1; 0 0] + γ * [0 0; 1 0]
    @test last_core[:, :, 2, 1] == γ * [0 1; 0 0]
    @test last_core[:, :, 3, 1] == β * [0 0; 1 0]
end


@testset "Laplacian Tests" begin
    @testset "Δ_DD Tests" begin
        @testset "n=5" begin
            expected = [2.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 2.0]
            result = Δ_DD(5)
            @test result == expected
        end
    end

    @testset "Δ_NN Tests" begin
        @testset "n=5" begin
            expected = [1.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 1.0]
            result = Δ_NN(5)
            @test result == expected
        end
    end

    @testset "Δ_DN Tests" begin
        @testset "n=5" begin
            expected = [2.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 1.0]
            result = Δ_DN(5)
            @test result == expected
        end
    end

    @testset "Δ_ND Tests" begin
        @testset "n=5" begin
            expected = [1.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 2.0]
            result = Δ_ND(5)
            @test result == expected
        end
    end

    @testset "Δ_Periodic Tests" begin
        @testset "n=5" begin
            expected = [2.0 -1.0 0.0 0.0 -1.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        -1.0 0.0 0.0 -1.0 2.0]
            result = Δ_Periodic(5)
            @test result == expected
        end
    end

    @testset "Δ_tto Tests" begin
        @testset "n=2, d=2" begin
            dims = (2, 2)
            rks = [1, 2, 1]
            expected = Δ_tto(2, 2, Δ_DD)
            
            # Check the structure and content of the TToperator
            @test expected.tto_dims == dims
            @test expected.tto_rks == rks
        end
    end
end

@testset "QTT_Tridiagonal_Toeplitz" begin
    α, β, γ = 2.0, -1.0, -1.0
    l = 3

    qtt = QTT_Tridiagonal_Toeplitz(α, β, γ, l)

    # Check dimensions
    @test qtt.N == l
    @test qtt.tto_dims == (2, 2, 2)
    @test qtt.tto_rks == [1, 3, 3, 1]

    # Check first core
    first_core = qtt.tto_vec[1]
    @test size(first_core) == (2, 2, 1, 3)
    @test first_core[:, :, 1, 1] == [1 0; 0 1]
    @test first_core[:, :, 1, 2] == [0 0; 1 0]
    @test first_core[:, :, 1, 3] == [0 1; 0 0]

    # Check middle core
    middle_core = qtt.tto_vec[2]
    @test size(middle_core) == (2, 2, 3, 3)
    @test middle_core[:, :, 1, 1] == [1 0; 0 1]
    @test middle_core[:, :, 1, 2] == [0 0; 1 0]
    @test middle_core[:, :, 1, 3] == [0 1; 0 0]
    @test middle_core[:, :, 2, 2] == [0 1; 0 0]
    @test middle_core[:, :, 3, 3] == [0 0; 1 0]

    # Check last core
    last_core = qtt.tto_vec[3]
    @test size(last_core) == (2, 2, 3, 1)
    @test last_core[:, :, 1, 1] == α * [1 0; 0 1] + β * [0 1; 0 0] + γ * [0 0; 1 0]
    @test last_core[:, :, 2, 1] == γ * [0 1; 0 0]
    @test last_core[:, :, 3, 1] == β * [0 0; 1 0]
end

@testset "Gradient Tests" begin
    @testset "∇_DD Tests" begin
        @testset "n=5" begin
            expected = [1.0 -1.0 0.0 0.0 0.0;
                        0.0 1.0 -1.0 0.0 0.0;
                        0.0 0.0 1.0 -1.0 0.0;
                        0.0 0.0 0.0 1.0 -1.0;
                        0.0 0.0 0.0 0.0 0.0]
            result = ∇_DD(5)
            @test result == expected
        end
    end

    @testset "∇_tto Tests" begin
        @testset "n=2, d=2" begin
            dims = (2, 2)
            rks = [1, 2, 1]
            expected = ∇_tto(2, 2, ∇_DD)
            
            # Check the structure and content of the TToperator
            @test expected.tto_dims == dims
            @test expected.tto_rks == rks
        end
    end
end


@testset "Laplacian Tests" begin
    @testset "Δ_DD Tests" begin
        @testset "n=5" begin
            expected = [2.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 2.0]
            result = Δ_DD(5)
            @test result == expected
        end
    end

    @testset "Δ_NN Tests" begin
        @testset "n=5" begin
            expected = [1.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 1.0]
            result = Δ_NN(5)
            @test result == expected
        end
    end

    @testset "Δ_DN Tests" begin
        @testset "n=5" begin
            expected = [2.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 1.0]
            result = Δ_DN(5)
            @test result == expected
        end
    end

    @testset "Δ_ND Tests" begin
        @testset "n=5" begin
            expected = [1.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 2.0]
            result = Δ_ND(5)
            @test result == expected
        end
    end

    @testset "Δ_Periodic Tests" begin
        @testset "n=5" begin
            expected = [2.0 -1.0 0.0 0.0 -1.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        -1.0 0.0 0.0 -1.0 2.0]
            result = Δ_Periodic(5)
            @test result == expected
        end
    end

    @testset "Δ_tto Tests" begin
        @testset "n=2, d=2" begin
            dims = (2, 2)
            rks = [1, 2, 1]
            expected = Δ_tto(2, 2, Δ_DD)
            
            # Check the structure and content of the TToperator
            @test expected.tto_dims == dims
            @test expected.tto_rks == rks
        end
    end
end

@testset "QTT_Tridiagonal_Toeplitz" begin
    α, β, γ = 2.0, -1.0, -1.0
    l = 3

    qtt = QTT_Tridiagonal_Toeplitz(α, β, γ, l)

    # Check dimensions
    @test qtt.N == l
    @test qtt.tto_dims == (2, 2, 2)
    @test qtt.tto_rks == [1, 3, 3, 1]

    # Check first core
    first_core = qtt.tto_vec[1]
    @test size(first_core) == (2, 2, 1, 3)
    @test first_core[:, :, 1, 1] == [1 0; 0 1]
    @test first_core[:, :, 1, 2] == [0 0; 1 0]
    @test first_core[:, :, 1, 3] == [0 1; 0 0]

    # Check middle core
    middle_core = qtt.tto_vec[2]
    @test size(middle_core) == (2, 2, 3, 3)
    @test middle_core[:, :, 1, 1] == [1 0; 0 1]
    @test middle_core[:, :, 1, 2] == [0 0; 1 0]
    @test middle_core[:, :, 1, 3] == [0 1; 0 0]
    @test middle_core[:, :, 2, 2] == [0 1; 0 0]
    @test middle_core[:, :, 3, 3] == [0 0; 1 0]

    # Check last core
    last_core = qtt.tto_vec[3]
    @test size(last_core) == (2, 2, 3, 1)
    @test last_core[:, :, 1, 1] == α * [1 0; 0 1] + β * [0 1; 0 0] + γ * [0 0; 1 0]
    @test last_core[:, :, 2, 1] == γ * [0 1; 0 0]
    @test last_core[:, :, 3, 1] == β * [0 0; 1 0]
end

@testset "Gradient Tests" begin
    @testset "∇_DD Tests" begin
        @testset "n=5" begin
            expected = [1.0 -1.0 0.0 0.0 0.0;
                        0.0 1.0 -1.0 0.0 0.0;
                        0.0 0.0 1.0 -1.0 0.0;
                        0.0 0.0 0.0 1.0 -1.0;
                        0.0 0.0 0.0 0.0 0.0]
            result = ∇_DD(5)
            @test result == expected
        end
    end

    @testset "∇_tto Tests" begin
        @testset "n=2, d=2" begin
            dims = (2, 2)
            rks = [1, 2, 1]
            expected = ∇_tto(2, 2, ∇_DD)
            
            # Check the structure and content of the TToperator
            @test expected.tto_dims == dims
            @test expected.tto_rks == rks
        end
    end
end

@testset "Jacobian_tto Tests" begin
    @testset "n=2, d=2" begin
        dims = (2, 2)
        rks = [1, 4, 1]
        expected = Jacobian_tto(2, 2, ∇_DD)
        
        # Check the structure and content of the TToperator
        @test expected.tto_dims == dims
        @test expected.tto_rks == rks
    end
end


@testset "Laplacian Tests" begin
    @testset "Δ_DD Tests" begin
        @testset "n=5" begin
            expected = [2.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 2.0]
            result = Δ_DD(5)
            @test result == expected
        end
    end

    @testset "Δ_NN Tests" begin
        @testset "n=5" begin
            expected = [1.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 1.0]
            result = Δ_NN(5)
            @test result == expected
        end
    end

    @testset "Δ_DN Tests" begin
        @testset "n=5" begin
            expected = [2.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 1.0]
            result = Δ_DN(5)
            @test result == expected
        end
    end

    @testset "Δ_ND Tests" begin
        @testset "n=5" begin
            expected = [1.0 -1.0 0.0 0.0 0.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        0.0 0.0 0.0 -1.0 2.0]
            result = Δ_ND(5)
            @test result == expected
        end
    end

    @testset "Δ_Periodic Tests" begin
        @testset "n=5" begin
            expected = [2.0 -1.0 0.0 0.0 -1.0;
                        -1.0 2.0 -1.0 0.0 0.0;
                        0.0 -1.0 2.0 -1.0 0.0;
                        0.0 0.0 -1.0 2.0 -1.0;
                        -1.0 0.0 0.0 -1.0 2.0]
            result = Δ_Periodic(5)
            @test result == expected
        end
    end
end

@testset "toeplitz_to_qtto Tests" begin
    α, β, γ = 2.0, -1.0, -1.0
    d = 3

    tt = toeplitz_to_qtto(α, β, γ, d)

    # Check TT dimensions and ranks
    @test tt.tto_dims == (2, 2, 2)
    @test tt.tto_rks == [1, 3, 3, 1]

    # Check first core
    first_core = tt.tto_vec[1]
    @test size(first_core) == (2, 2, 1, 3)
    id = [1.0 0.0; 0.0 1.0]
    J = [0.0 1.0; 0.0 0.0]
    @test first_core[:, :, 1, 1] == id
    @test first_core[:, :, 1, 2] == J'
    @test first_core[:, :, 1, 3] == J

    # Check middle core
    middle_core = tt.tto_vec[2]
    @test size(middle_core) == (2, 2, 3, 3)
    # Check a few entries for correctness
    @test middle_core[:, :, 1, 1] == id
    @test middle_core[:, :, 1, 2] == J'
    @test middle_core[:, :, 1, 3] == J
    @test middle_core[:, :, 2, 2] == J
    @test middle_core[:, :, 3, 3] == J'

    # Check last core
    last_core = tt.tto_vec[3]
    @test size(last_core) == (2, 2, 3, 1)
    @test last_core[:, :, 1, 1] == α * id + β * J + γ * J'
    @test last_core[:, :, 2, 1] == γ * J
    @test last_core[:, :, 3, 1] == β * J'
end

@testset "Δ_DD Tests" begin
    @testset "n=1" begin
        expected = [2.0;;]
        result = Δ_DD(1)
        @test result == expected
    end

    @testset "n=2" begin
        expected = [2.0 -1.0;
                    -1.0 2.0]
        result = Δ_DD(2)
        @test result == expected
    end

    @testset "n=3" begin
        expected = [2.0 -1.0 0.0;
                    -1.0 2.0 -1.0;
                    0.0 -1.0 2.0]
        result = Δ_DD(3)
        @test result == expected
    end

    @testset "n=5" begin
        expected = [2.0 -1.0 0.0 0.0 0.0;
                    -1.0 2.0 -1.0 0.0 0.0;
                    0.0 -1.0 2.0 -1.0 0.0;
                    0.0 0.0 -1.0 2.0 -1.0;
                    0.0 0.0 0.0 -1.0 2.0]
        result = Δ_DD(5)
        @test result == expected
    end

    @testset "Symmetry" begin
        n = 6
        result = Δ_DD(n)
        @test result == result'
    end

    @testset "Diagonal values" begin
        n = 10
        result = Δ_DD(n)
        @test all(result[i,i] == 2.0 for i in 1:n)
    end

    @testset "Off-diagonal values" begin
        n = 7
        result = Δ_DD(n)
        for i in 1:n-1
            @test result[i,i+1] == -1.0
            @test result[i+1,i] == -1.0
        end
        # All other off-diagonal elements should be zero
        for i in 1:n
            for j in 1:n
                if abs(i-j) > 1
                    @test result[i,j] == 0.0
                end
            end
        end
    end
end

@testset "Δ_tto Tests" begin
    @testset "d=1 returns matrix" begin
        n = 4
        result = Δ_tto(n, 1, Δ_DD)
        expected = Δ_DD(n)
        @test result == expected
    end

    @testset "d=2, n=2, structure and values" begin
        n, d = 2, 2
        tt = Δ_tto(n, d, Δ_DD)
        @test tt.N == d
        @test tt.tto_dims == (2, 2)
        @test tt.tto_rks == [1, 2, 1]
        @test length(tt.tto_vec) == d

        # Check first core
        first_core = tt.tto_vec[1]
        @test size(first_core) == (2, 2, 1, 2)
        @test first_core[:, :, 1, 1] == Δ_DD(2)
        @test first_core[:, :, 1, 2] == Matrix(I, 2, 2)

        # Check second (last) core
        last_core = tt.tto_vec[2]
        @test size(last_core) == (2, 2, 2, 1)
        @test last_core[:, :, 1, 1] == Matrix(I, 2, 2)
        @test last_core[:, :, 2, 1] == Δ_DD(2)
    end

    @testset "d=3, n=2, structure" begin
        n, d = 2, 3
        tt = Δ_tto(n, d, Δ_DD)
        @test tt.N == d
        @test tt.tto_dims == (2, 2, 2)
        @test tt.tto_rks == [1, 2, 2, 1]
        @test length(tt.tto_vec) == d

        # First core
        first_core = tt.tto_vec[1]
        @test size(first_core) == (2, 2, 1, 2)
        @test first_core[:, :, 1, 1] == Δ_DD(2)
        @test first_core[:, :, 1, 2] == Matrix(I, 2, 2)

        # Middle core
        middle_core = tt.tto_vec[2]
        @test size(middle_core) == (2, 2, 2, 2)
        @test middle_core[:, :, 1, 1] == Matrix(I, 2, 2)
        @test middle_core[:, :, 2, 1] == Δ_DD(2)
        @test middle_core[:, :, 2, 2] == Matrix(I, 2, 2)

        # Last core
        last_core = tt.tto_vec[3]
        @test size(last_core) == (2, 2, 2, 1)
        @test last_core[:, :, 1, 1] == Matrix(I, 2, 2)
        @test last_core[:, :, 2, 1] == Δ_DD(2)
    end

    @testset "d=2, n=3, values" begin
        n, d = 3, 2
        tt = Δ_tto(n, d, Δ_DD)
        @test tt.tto_dims == (3, 3)
        @test tt.tto_rks == [1, 2, 1]
        @test size(tt.tto_vec[1]) == (3, 3, 1, 2)
        @test size(tt.tto_vec[2]) == (3, 3, 2, 1)
        @test tt.tto_vec[1][:, :, 1, 1] == Δ_DD(3)
        @test tt.tto_vec[1][:, :, 1, 2] == Matrix(I, 3, 3)
        @test tt.tto_vec[2][:, :, 1, 1] == Matrix(I, 3, 3)
        @test tt.tto_vec[2][:, :, 2, 1] == Δ_DD(3)
    end
end

@testset "QTT_Tridiagonal_Toeplitz" begin
    α, β, γ = 2.0, -1.0, -1.0
    l = 3

    qtt = QTT_Tridiagonal_Toeplitz(α, β, γ, l)

    # Check dimensions
    @test qtt.N == l
    @test qtt.tto_dims == (2, 2, 2)
    @test qtt.tto_rks == [1, 3, 3, 1]

    # Check first core
    first_core = qtt.tto_vec[1]
    @test size(first_core) == (2, 2, 1, 3)
    @test first_core[:, :, 1, 1] == [1 0; 0 1]
    @test first_core[:, :, 1, 2] == [0 0; 1 0]
    @test first_core[:, :, 1, 3] == [0 1; 0 0]

    # Check middle core
    middle_core = qtt.tto_vec[2]
    @test size(middle_core) == (2, 2, 3, 3)
    @test middle_core[:, :, 1, 1] == [1 0; 0 1]
    @test middle_core[:, :, 1, 2] == [0 0; 1 0]
    @test middle_core[:, :, 1, 3] == [0 1; 0 0]
    @test middle_core[:, :, 2, 2] == [0 1; 0 0]
    @test middle_core[:, :, 3, 3] == [0 0; 1 0]

    # Check last core
    last_core = qtt.tto_vec[3]
    @test size(last_core) == (2, 2, 3, 1)
    @test last_core[:, :, 1, 1] == α * [1 0; 0 1] + β * [0 1; 0 0] + γ * [0 0; 1 0]
    @test last_core[:, :, 2, 1] == γ * [0 1; 0 0]
    @test last_core[:, :, 3, 1] == β * [0 0; 1 0]

    # Test for l < 2 throws error
    @test_throws ArgumentError QTT_Tridiagonal_Toeplitz(α, β, γ, 1)
end

@testset "shift_matrix Tests" begin
    @testset "n=1" begin
        expected = [0.0;;]
        result = shift_matrix(1)
        @test result == expected
    end

    @testset "n=2" begin
        expected = [0.0 1.0;
                    0.0 0.0]
        result = shift_matrix(2)
        @test result == expected
    end

    @testset "n=3" begin
        expected = [0.0 1.0 0.0;
                    0.0 0.0 1.0;
                    0.0 0.0 0.0]
        result = shift_matrix(3)
        @test result == expected
    end

    @testset "n=5" begin
        expected = [0.0 1.0 0.0 0.0 0.0;
                    0.0 0.0 1.0 0.0 0.0;
                    0.0 0.0 0.0 1.0 0.0;
                    0.0 0.0 0.0 0.0 1.0;
                    0.0 0.0 0.0 0.0 0.0]
        result = shift_matrix(5)
        @test result == expected
    end

    @testset "All elements except superdiagonal are zero" begin
        n = 6
        S = shift_matrix(n)
        for i in 1:n
            for j in 1:n
                if j == i+1
                    @test S[i, j] == 1.0
                else
                    @test S[i, j] == 0.0
                end
            end
        end
    end
end



@testset "toeplitz_to_qtto Tests" begin
    α, β, γ = 2.0, -1.0, -1.0
    d = 3

    tt = toeplitz_to_qtto(α, β, γ, d)

    # Check TT dimensions and ranks
    @test tt.tto_dims == (2, 2, 2)
    @test tt.tto_rks == [1, 3, 3, 1]

    # Check first core
    first_core = tt.tto_vec[1]
    @test size(first_core) == (2, 2, 1, 3)
    id = [1.0 0.0; 0.0 1.0]
    J = [0.0 1.0; 0.0 0.0]
    @test first_core[:, :, 1, 1] == id
    @test first_core[:, :, 1, 2] == J'
    @test first_core[:, :, 1, 3] == J

    # Check middle core
    middle_core = tt.tto_vec[2]
    @test size(middle_core) == (2, 2, 3, 3)
    # Check a few entries for correctness
    @test middle_core[:, :, 1, 1] == id
    @test middle_core[:, :, 1, 2] == J'
    @test middle_core[:, :, 1, 3] == J
    @test middle_core[:, :, 2, 2] == J
    @test middle_core[:, :, 3, 3] == J'

    # Check last core
    last_core = tt.tto_vec[3]
    @test size(last_core) == (2, 2, 3, 1)
    @test last_core[:, :, 1, 1] == α * id + β * J + γ * J'
    @test last_core[:, :, 2, 1] == γ * J
    @test last_core[:, :, 3, 1] == β * J'
end

@testset "Prolongation TToperator Tests" begin
    for d in 1:4
        tt = qtto_prolongation(d)
        @test tt.N == d
        @test tt.tto_dims == ntuple(_->2, d)
        @test tt.tto_rks == fill(2, d+1)
        @test length(tt.tto_vec) == d

        # Check the structure of each core
        for j in 1:d
            core = tt.tto_vec[j]
            @test size(core) == (2, 2, 2, 2)
            # Only the specified entries should be 1.0, all others 0.0
            for i1 in 1:2, i2 in 1:2, r1 in 1:2, r2 in 1:2
                val = core[i1, i2, r1, r2]
                if (i1 == 1 && i2 == 1 && r1 == 1 && r2 == 1) ||
                   (i1 == 1 && i2 == 1 && r1 == 2 && r2 == 2) ||
                   (i1 == 1 && i2 == 2 && r1 == 2 && r2 == 1) ||
                   (i1 == 2 && i2 == 2 && r1 == 1 && r2 == 2)
                    @test val == 1.0
                else
                    @test val == 0.0
                end
            end
        end
    end
end
