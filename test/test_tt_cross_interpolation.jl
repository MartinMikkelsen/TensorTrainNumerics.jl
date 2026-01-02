using Test
import TensorTrainNumerics: MaxVolPivot, RandomPivot, MaxVol, Greedy, DMRG, _cap_ranks!

@testset "Cross Interpolation Algorithms" begin

    @testset "MaxVolPivot" begin
        pivot = MaxVolPivot()
        @test pivot.tol ≈ 1.05
        @test pivot.maxiter == 100

        pivot = MaxVolPivot(tol = 1.1, maxiter = 50)
        @test pivot.tol ≈ 1.1
        @test pivot.maxiter == 50
    end

    @testset "RandomPivot" begin
        pivot = RandomPivot()
        @test pivot.nsamples == 1000
        @test pivot.seed === nothing

        pivot = RandomPivot(nsamples = 500, seed = 42)
        @test pivot.nsamples == 500
        @test pivot.seed == 42
    end

    @testset "MaxVol" begin
        alg = MaxVol()
        @test alg.maxiter == 50
        @test alg.tol ≈ 1.0e-10
        @test alg.rmax == 500
        @test alg.kickrank == 5
        @test alg.verbose == true
        @test alg.pivot isa MaxVolPivot

        alg = MaxVol(maxiter = 50, tol = 1.0e-6, rmax = 100, kickrank = nothing, verbose = false)
        @test alg.maxiter == 50
        @test alg.tol ≈ 1.0e-6
        @test alg.rmax == 100
        @test alg.kickrank === nothing
        @test alg.verbose == false
    end

    @testset "Greedy" begin
        alg = Greedy()
        @test alg.maxiter == 50
        @test alg.tol ≈ 1.0e-10
        @test alg.rmax == 500
        @test alg.verbose == true
        @test alg.nsamples == 1000
        @test alg.pivot isa RandomPivot

        alg = Greedy(maxiter = 100, tol = 1.0e-8, nsamples = 500, verbose = false)
        @test alg.maxiter == 100
        @test alg.tol ≈ 1.0e-8
        @test alg.nsamples == 500
        @test alg.verbose == false
    end

    @testset "DMRG" begin
        alg = DMRG()
        @test alg.maxiter == 50
        @test alg.tol ≈ 1.0e-10
        @test alg.rmax == 500
        @test alg.kickrank == 5
        @test alg.verbose == true
        @test alg.pivot isa MaxVolPivot
        @test alg.nsweeps_inner == 2

        alg = DMRG(maxiter = 30, tol = 1.0e-12, nsweeps_inner = 3, kickrank = 5)
        @test alg.maxiter == 30
        @test alg.tol ≈ 1.0e-12
        @test alg.nsweeps_inner == 3
        @test alg.kickrank == 5
    end
    @testset "tt_cross" begin

        @testset "MaxVol algorithm" begin
            f(x) = sin.(sum(x, dims = 2))
            domain = [range(0, 1, length = 10) |> collect for _ in 1:4]

            tt = tt_cross(f, domain, MaxVol(verbose = false, tol = 1.0e-6))
            @test tt isa TTvector
            @test tt.N == 4
            @test all(tt.ttv_dims .== 10)
            @test tt.ttv_rks[1] == 1
            @test tt.ttv_rks[end] == 1
        end

        @testset "Greedy algorithm" begin
            f(x) = prod(x, dims = 2)
            domain = [range(0, 1, length = 8) |> collect for _ in 1:3]

            tt = tt_cross(f, domain, Greedy(verbose = false, tol = 1.0e-6, maxiter = 50))
            @test tt isa TTvector
            @test tt.N == 3
        end

        @testset "DMRG algorithm" begin
            f(x) = exp.(-sum(x .^ 2, dims = 2))
            domain = [range(-1, 1, length = 12) |> collect for _ in 1:4]

            tt = tt_cross(f, domain, DMRG(verbose = false, tol = 1.0e-6))
            @test tt isa TTvector
            @test tt.N == 4
        end

        @testset "Default algorithm" begin
            f(x) = sum(x, dims = 2)
            domain = [collect(1.0:5.0) for _ in 1:3]

            tt = tt_cross(f, domain; alg = MaxVol(verbose = false))
            @test tt isa TTvector
        end

        @testset "Tuple dimensions" begin
            f(x) = ones(size(x, 1))
            dims = (4, 5, 6)

            tt = tt_cross(f, dims; alg = MaxVol(verbose = false))
            @test tt isa TTvector
            @test tt.ttv_dims == dims
        end

        @testset "Vector dimensions" begin
            f(x) = sum(x, dims = 2)
            dims = [4, 5, 6, 7]

            tt = tt_cross(f, dims; alg = MaxVol(verbose = false))
            @test tt isa TTvector
            @test tt.N == 4
        end

    end


    @testset "Helper Functions" begin

        @testset "_cap_ranks!" begin
            Rs = [1, 10, 10, 10, 1]
            Is = [3, 4, 5, 6]
            TensorTrainNumerics._cap_ranks!(Rs, Is, 100)
            @test Rs[1] == 1
            @test Rs[end] == 1
            @test all(Rs[2:(end - 1)] .<= 100)

            Rs = [1, 100, 100, 1]
            Is = [2, 2, 2]
            TensorTrainNumerics._cap_ranks!(Rs, Is, 50)
            @test Rs[2] <= 2
            @test Rs[3] <= 4
        end

        @testset "_build_fiber_indices_left" begin
            Is = [3, 4, 5]
            Rs = [1, 2, 2, 1]
            N = 3
            lsets = [ones(Int, 1, 0), [1; 2;;], [1 1; 2 2]]
            rsets = [[1 1; 2 2], [1; 2;;], ones(Int, 1, 0)]

            indices = TensorTrainNumerics._build_fiber_indices_left(lsets, rsets, 1, Is, Rs, N)
            @test size(indices, 1) == Rs[1] * Is[1] * Rs[2]
            @test size(indices, 2) == N
        end

        @testset "_build_fiber_indices_right" begin
            Is = [3, 4, 5]
            Rs = [1, 2, 2, 1]
            N = 3
            lsets = [ones(Int, 1, 0), [1; 2;;], [1 1; 2 2]]
            rsets = [[1 1; 2 2], [1; 2;;], ones(Int, 1, 0)]

            indices = TensorTrainNumerics._build_fiber_indices_right(lsets, rsets, 2, Is, Rs, N)
            @test size(indices, 1) == Rs[2] * Is[2] * Rs[3]
            @test size(indices, 2) == N
        end

        @testset "_evaluate_on_domain" begin
            domain = [[1.0, 2.0, 3.0], [10.0, 20.0]]
            indices = [1 1; 2 2; 3 1]
            f(x) = sum(x, dims = 2)

            result = TensorTrainNumerics._evaluate_on_domain(f, domain, indices)
            @test result ≈ [11.0, 22.0, 13.0]
        end

        @testset "_evaluate_tt" begin
            cores = [
                reshape([1.0, 2.0], 2, 1, 1),
                reshape([1.0, 10.0, 100.0], 3, 1, 1),
            ]
            indices = [1 1; 1 2; 2 3]

            result = TensorTrainNumerics._evaluate_tt(cores, indices, 2)
            @test result ≈ [1.0, 10.0, 200.0]
        end
    end

    @testset "DMRG Helper Functions" begin

        @testset "_sample_superblock" begin
            domain = [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]
            f(x) = sum(x, dims = 2)
            I_l = [ones(Int, 1, 0), [1; 2;;], [1 1; 2 2]]
            I_g = [[1 1; 2 2], [1; 2;;], ones(Int, 1, 0)]
            Is = [2, 2, 2]
            N = 3

            superblock = TensorTrainNumerics._sample_superblock(f, domain, I_l, I_g, 1, Is, N)
            @test size(superblock) == (1, 2, 2, 2)
        end

        @testset "_combine_indices_left" begin
            I_l_k = [1 2; 3 4]
            s = 3

            result = TensorTrainNumerics._combine_indices_left(I_l_k, s)
            @test size(result) == (6, 3)
            @test result[1, :] == [1, 2, 1]
            @test result[3, :] == [1, 2, 2]
        end

        @testset "_combine_indices_right" begin
            I_g_k = [1 2; 3 4]
            s = 3

            result = TensorTrainNumerics._combine_indices_right(s, I_g_k)
            @test size(result) == (6, 3)
            @test result[1, :] == [1, 1, 2]
            @test result[4, :] == [1, 3, 4]
        end

        @testset "_dmrg_update!" begin
            domain = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
            f(x) = sum(x, dims = 2)
            Is = [3, 3, 3]
            Rs = [1, 2, 2, 1]
            N = 3

            I_l = [ones(Int, 1, 0), [1; 2;;], [1 1; 2 2]]
            I_g = [[1 1; 2 2], [1; 2;;], ones(Int, 1, 0)]

            cores = [zeros(3, 1, 2), zeros(3, 2, 2), zeros(3, 2, 1)]
            alg = DMRG(verbose = false)

            TensorTrainNumerics._dmrg_update!(cores, I_l, I_g, Rs, f, domain, Is, 1, N, true, alg)
            @test size(cores[1], 1) == 3
            @test size(cores[1], 2) == 1
        end

    end

    @testset "Greedy Helper Functions" begin

        @testset "_indexmerge" begin
            J1 = [1; 2;;]
            J2 = [10; 20; 30;;]

            result = TensorTrainNumerics._indexmerge(J1, J2)
            @test size(result) == (6, 2)
            @test result[1, :] == [1, 10]
            @test result[2, :] == [2, 10]
            @test result[3, :] == [1, 20]
        end

        @testset "_indexmerge with empty" begin
            J1 = ones(Int, 1, 0)
            J2 = [1; 2; 3;;]

            result = TensorTrainNumerics._indexmerge(J1, J2)
            @test size(result) == (3, 1)
            @test result[:, 1] == [1, 2, 3]
        end

        @testset "_form_tensor" begin
            N = 2
            Rs = [1, 2, 1]
            Is = [3, 4]

            y = [
                reshape(collect(1.0:6.0), 1, 3, 2),
                reshape(collect(1.0:8.0), 2, 4, 1),
            ]
            mid_inv_L = [ones(1, 1), Matrix(1.0I, 2, 2), ones(1, 1)]
            mid_inv_U = [ones(1, 1), Matrix(1.0I, 2, 2), ones(1, 1)]

            cores = TensorTrainNumerics._form_tensor(y, mid_inv_L, mid_inv_U, N, Rs, Is)
            @test length(cores) == 2
            @test size(cores[1]) == (3, 1, 2)
            @test size(cores[2]) == (4, 2, 1)
        end

    end

    @testset "Greedy Fiber Helper Functions" begin

        @testset "_sample_fiber" begin
            domain = [[1.0, 2.0], [10.0, 20.0, 30.0], [100.0, 200.0]]
            f(x) = sum(x, dims = 2)
            I_l = [ones(Int, 1, 0), [1; 2;;], [1 1; 2 2]]
            I_g = [[1 1; 2 2], [1; 2;;], ones(Int, 1, 0)]
            Is = [2, 3, 2]
            N = 3

            fiber = TensorTrainNumerics._sample_fiber(f, domain, I_l, I_g, 2, Is, N)
            @test size(fiber) == (2, 3, 2)
        end

        @testset "_fiber_to_Q3R" begin
            fiber = randn(2, 3, 4)

            Q3, R = TensorTrainNumerics._fiber_to_Q3R(fiber)
            @test size(Q3, 1) == 2
            @test size(Q3, 2) == 3
            @test size(R, 2) == 4
        end

        @testset "_Q3_to_core" begin
            Q3 = randn(2, 3, 4)
            row_indices = [1, 3, 5, 6]

            core = TensorTrainNumerics._Q3_to_core(Q3, row_indices)
            @test size(core, 1) == 2
            @test size(core, 2) == 3
            @test size(core, 3) == 4
        end

        @testset "_Q3_to_core empty indices" begin
            Q3 = randn(2, 3, 4)

            core = TensorTrainNumerics._Q3_to_core(Q3, Int[])
            @test core === Q3
        end

        @testset "_combine_indices_left_fiber" begin
            I_l_k = [1 2; 3 4]
            s = 3

            result = TensorTrainNumerics._combine_indices_left_fiber(I_l_k, s)
            @test size(result) == (6, 3)
            @test result[1, :] == [1, 2, 1]
            @test result[3, :] == [1, 2, 2]
        end

        @testset "_combine_indices_right_fiber" begin
            I_g_k = [1 2; 3 4]
            s = 3

            result = TensorTrainNumerics._combine_indices_right_fiber(s, I_g_k)
            @test size(result) == (6, 3)
            @test result[1, :] == [1, 1, 2]
            @test result[4, :] == [1, 3, 4]
        end

        @testset "_get_row_indices" begin
            rows = [1 2; 3 4]
            all_rows = [1 2; 5 6; 3 4; 7 8]

            indices = TensorTrainNumerics._get_row_indices(rows, all_rows)
            @test indices == [1, 3]
        end

        @testset "_get_row_indices empty" begin
            rows = zeros(Int, 0, 2)
            all_rows = [1 2; 3 4]

            indices = TensorTrainNumerics._get_row_indices(rows, all_rows)
            @test isempty(indices)
        end

    end


    @testset "Greedy Superblock Functions" begin

        @testset "_sample_superblock_greedy" begin
            domain = [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]
            f(x) = sum(x, dims = 2)
            I_l = [ones(Int, 1, 0), [1; 2;;], [1 1; 2 2]]
            I_g = [[1 1; 2 2], [1; 2;;], ones(Int, 1, 0)]
            Is = [2, 2, 2]
            N = 3
            j_l = [1, 2]
            j_g = [1, 2]

            result = TensorTrainNumerics._sample_superblock_greedy(f, domain, I_l, I_g, Is, 1, N, j_l, j_g)
            @test size(result) == (2, 2)
        end

        @testset "_sample_skeleton_greedy" begin
            cores = [randn(1, 3, 2), randn(2, 4, 1)]
            fibers = [randn(1, 3, 2), randn(2, 4, 1)]
            j_l = [1, 2, 3]
            j_g = [1, 2]

            result = TensorTrainNumerics._sample_skeleton_greedy(cores, fibers, 1, j_l, j_g)
            @test size(result) == (3, 2)
        end

        @testset "_sample_skeleton_greedy dimension mismatch" begin
            cores = [randn(1, 3, 2), randn(3, 4, 1)]
            fibers = [randn(1, 3, 2), randn(3, 4, 1)]
            j_l = [1, 2]
            j_g = [1, 2]

            result = TensorTrainNumerics._sample_skeleton_greedy(cores, fibers, 1, j_l, j_g)
            @test size(result) == (2, 2)
            @test all(result .== 0)
        end

    end

    @testset "Integration Functions" begin

        @testset "_gauss_legendre" begin
            nodes, weights = TensorTrainNumerics._gauss_legendre(5, 0.0, 1.0)
            @test length(nodes) == 5
            @test length(weights) == 5
            @test sum(weights) ≈ 1.0
            @test all(0 .< nodes .< 1)
        end

        @testset "_gauss_legendre custom bounds" begin
            nodes, weights = TensorTrainNumerics._gauss_legendre(3, -2.0, 2.0)
            @test length(nodes) == 3
            @test sum(weights) ≈ 4.0
            @test all(-2 .< nodes .< 2)
        end

        @testset "_contract_with_weights" begin
            cores = [
                reshape([1.0, 2.0], 2, 1, 1),
                reshape([1.0, 2.0, 3.0], 3, 1, 1),
            ]
            weights = [[0.5, 0.5], [1 / 3, 1 / 3, 1 / 3]]

            result = TensorTrainNumerics._contract_with_weights(cores, weights)
            @test result ≈ 1.5 * 2.0
        end

        @testset "tt_integrate simple" begin
            f(x) = ones(size(x, 1))

            result = tt_integrate(f, 3; alg = DMRG(verbose = false))
            @test result ≈ 1.0 atol = 1.0e-6
        end

        @testset "tt_integrate with bounds" begin
            f(x) = ones(size(x, 1))
            lower = [0.0, 0.0]
            upper = [2.0, 3.0]

            result = tt_integrate(f, lower, upper; alg = DMRG(verbose = false))
            @test result ≈ 6.0 atol = 1.0e-6
        end

        @testset "tt_integrate polynomial" begin
            f(x) = x[:, 1] .^ 2

            result = tt_integrate(f, 1; alg = DMRG(verbose = false), nquad = 10)
            @test result ≈ 1 / 3 atol = 1.0e-6
        end

    end
end
