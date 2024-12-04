include("../src/tt_operators.jl")

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

@testset "Matricize Tests" begin
    @testset "n=2, d=2" begin
        tt = Δ_tto(2, 2, Δ_DD)
        expected = [2.0 -1.0 0.0 0.0;
                    -1.0 2.0 -1.0 0.0;
                    0.0 -1.0 2.0 -1.0;
                    0.0 0.0 -1.0 2.0]
        result = matricize(tt)
        @test result == expected
    end
end

@testset "TT to QTT Conversion Tests" begin
    @testset "tt2qtt Tests" begin
        @testset "n=2, d=2" begin
            tt = Δ_tto(2, 2, Δ_DD)
            row_dims = [[2, 1], [2, 1]]
            col_dims = [[2, 1], [2, 1]]
            threshold = 0.0

            qtt = tt2qtt(tt, row_dims, col_dims, threshold)

            # Check dimensions
            @test qtt.N == 4
            @test qtt.tto_dims == (2, 1, 2, 1)
            @test qtt.tto_rks == [1, 2, 2, 2, 1]

            # Check first core
            first_core = qtt.tto_vec[1]
            @test size(first_core) == (2, 1, 1, 2)
            @test first_core[:, :, 1, 1] == [2.0; -1.0]
            @test first_core[:, :, 1, 2] == [0.0; 0.0]

            # Check second core
            second_core = qtt.tto_vec[2]
            @test size(second_core) == (1, 2, 2, 2)
            @test second_core[:, :, 1, 1] == [1.0 0.0]
            @test second_core[:, :, 2, 2] == [0.0 1.0]

            # Check third core
            third_core = qtt.tto_vec[3]
            @test size(third_core) == (2, 1, 2, 2)
            @test third_core[:, :, 1, 1] == [1.0; 0.0]
            @test third_core[:, :, 2, 2] == [0.0; 1.0]

            # Check fourth core
            fourth_core = qtt.tto_vec[4]
            @test size(fourth_core) == (1, 2, 2, 1)
            @test fourth_core[:, :, 1, 1] == [2.0 -1.0]
            @test fourth_core[:, :, 2, 1] == [-1.0 2.0]
        end
    end
end
@testset "TT to QTT Conversion Tests" begin
    @testset "tt2qtt Tests" begin
        @testset "n=2, d=2" begin
            tt = Δ_tto(2, 2, Δ_DD)
            row_dims = [[2, 1], [2, 1]]
            col_dims = [[2, 1], [2, 1]]
            threshold = 0.0

            qtt = tt2qtt(tt, row_dims, col_dims, threshold)

            # Check dimensions
            @test qtt.N == 4
            @test qtt.tto_dims == (2, 1, 2, 1)
            @test qtt.tto_rks == [1, 2, 2, 2, 1]

            # Check first core
            first_core = qtt.tto_vec[1]
            @test size(first_core) == (2, 1, 1, 2)
            @test first_core[:, :, 1, 1] == [2.0; -1.0]
            @test first_core[:, :, 1, 2] == [0.0; 0.0]

            # Check second core
            second_core = qtt.tto_vec[2]
            @test size(second_core) == (1, 2, 2, 2)
            @test second_core[:, :, 1, 1] == [1.0 0.0]
            @test second_core[:, :, 2, 2] == [0.0 1.0]

            # Check third core
            third_core = qtt.tto_vec[3]
            @test size(third_core) == (2, 1, 2, 2)
            @test third_core[:, :, 1, 1] == [1.0; 0.0]
            @test third_core[:, :, 2, 2] == [0.0; 1.0]

            # Check fourth core
            fourth_core = qtt.tto_vec[4]
            @test size(fourth_core) == (1, 2, 2, 1)
            @test fourth_core[:, :, 1, 1] == [2.0 -1.0]
            @test fourth_core[:, :, 2, 1] == [-1.0 2.0]
        end

        @testset "TTvector to QTTvector Tests" begin
            dims = (4, 8, 16)
            rks = [1, 2, 2, 1]
            tt_vec = [randn(Float64, dims[i], rks[i], rks[i+1]) for i in 1:eachindex(dims)]
            tt = TTvector{Float64, 3}(3, tt_vec, dims, rks, zeros(Int64, 3))

            qtt_dims = [[2, 2], [2, 2, 2], [2, 2, 2, 2]]
            qtt = tt2qtt(tt, qtt_dims)

            # Check dimensions
            @test qtt.N == 9
            @test qtt.ttv_dims == (2, 2, 2, 2, 2, 2, 2, 2, 2)
            @test qtt.ttv_rks == [1, 2, 2, 2, 2, 2, 2, 2, 2, 1]

            # Check first core
            first_core = qtt.ttv_vec[1]
            @test size(first_core) == (2, 1, 2)

            # Check last core
            last_core = qtt.ttv_vec[9]
            @test size(last_core) == (2, 2, 1)
        end
    end
end
