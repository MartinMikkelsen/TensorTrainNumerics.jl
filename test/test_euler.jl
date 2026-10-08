using Test
using Random
using TensorTrainNumerics
using LinearAlgebra

@testset "euler_method basic tests" begin
    d = 4
    h = 1 / d^2
    A = -h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]
    u₀ = rand_tt(tt_dims, tt_rks)

    steps = [0.05]  # single Euler step

    # Run Euler method in TT
    solution_tt = euler_method(A, u₀, steps; normalize = false)

    # Convert to dense
    A_dense = qtto_to_matrix(A)
    u_dense = qtt_to_function(u₀)

    # Explicit Euler in dense
    sol_dense = u_dense + steps[1] * (A_dense * u_dense)

    # Compare
    sol_tt_vec = qtt_to_function(solution_tt)
    rel_error = norm(sol_tt_vec - sol_dense) / norm(sol_dense)

    @test rel_error < 1.0e-6
    println("Test passed with relative error: ", rel_error)

end

@testset "implicit_euler_method basic test" begin
    d = 4
    h = 1 / d^2
    A = -h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]

    u₀ = rand_tt(tt_dims, tt_rks)
    guess = (u₀)
    steps = [0.05]

    # Run the implicit Euler solver
    sol_tt = implicit_euler_method(A, u₀, guess, steps; normalize = false, alg = DMRG())

    # Convert to dense for validation
    A_dense = qtto_to_matrix(A)
    u_dense = qtt_to_function(u₀)
    I = qtto_to_matrix(id_tto(nsites(A)))
    sol_dense = (I - steps[1] * A_dense) \ u_dense

    sol_tt_vec = qtt_to_function(sol_tt)
    rel_error = norm(sol_tt_vec - sol_dense) / norm(sol_dense)

    @test rel_error < 1.0e-5
    println("Implicit test passed with relative error: ", rel_error)
end

@testset "implicit_euler_method Krylov solver" begin
    d = 4
    h = 1 / d^2
    A = -h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]

    u₀ = rand_tt(tt_dims, tt_rks)
    guess = u₀
    steps = [0.05]

    sol_tt = implicit_euler_method(
        A, u₀, guess, steps;
        normalize = false, alg = Krylov(), tol = 1.0e-12
    )

    A_dense = qtto_to_matrix(A)
    u_dense = qtt_to_function(u₀)
    I = qtto_to_matrix(id_tto(nsites(A)))
    sol_dense = (I - steps[1] * A_dense) \ u_dense

    sol_tt_vec = qtt_to_function(sol_tt)
    rel_error = norm(sol_tt_vec - sol_dense) / norm(sol_dense)

    @test rel_error < 1.0e-8
end

@testset "Crank-Nicolson method basic test" begin
    d = 4
    h = 1 / d^2
    A = -h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]

    u₀ = rand_tt(tt_dims, tt_rks)
    guess = u₀
    steps = [0.05]

    sol_tt = crank_nicolson_method(A, u₀, guess, steps; normalize = false, alg = MALS())

    A_dense = qtto_to_matrix(A)
    u_dense = qtt_to_function(u₀)
    I = qtto_to_matrix(id_tto(nsites(A)))
    sol_dense = (I - 0.5 * steps[1] * A_dense) \ ((I + 0.5 * steps[1] * A_dense) * u_dense)

    sol_tt_vec = qtt_to_function(sol_tt)
    rel_error = norm(sol_tt_vec - sol_dense) / norm(sol_dense)

    @test rel_error < 1.0e-5
end

@testset "Crank-Nicolson method Krylov solver" begin
    d = 4
    h = 1 / d^2
    A = -h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]

    u₀ = rand_tt(tt_dims, tt_rks)
    guess = u₀
    steps = [0.05]

    sol_tt = crank_nicolson_method(
        A, u₀, guess, steps;
        normalize = false, alg = Krylov(), tol = 1.0e-12
    )

    A_dense = qtto_to_matrix(A)
    u_dense = qtt_to_function(u₀)
    I = qtto_to_matrix(id_tto(nsites(A)))
    sol_dense = (I - 0.5 * steps[1] * A_dense) \ ((I + 0.5 * steps[1] * A_dense) * u_dense)

    sol_tt_vec = qtt_to_function(sol_tt)
    rel_error = norm(sol_tt_vec - sol_dense) / norm(sol_dense)

    @test rel_error < 1.0e-8
end

@testset "Crank-Nicolson Krylov solver handles non-symmetric operators" begin
    d = 4
    A = 0.1 * ∇(d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]

    u₀ = rand_tt(tt_dims, tt_rks)
    guess = u₀
    steps = [0.05]

    sol_tt = crank_nicolson_method(
        A, u₀, guess, steps;
        normalize = false, alg = Krylov(), tol = 1.0e-12
    )

    A_dense = qtto_to_matrix(A)
    @test !issymmetric(A_dense)

    u_dense = qtt_to_function(u₀)
    I = qtto_to_matrix(id_tto(nsites(A)))
    sol_dense = (I - 0.5 * steps[1] * A_dense) \ ((I + 0.5 * steps[1] * A_dense) * u_dense)

    sol_tt_vec = qtt_to_function(sol_tt)
    rel_error = norm(sol_tt_vec - sol_dense) / norm(sol_dense)

    @test rel_error < 1.0e-8
end

@testset "Crank-Nicolson Krylov solver supports bounded BiCGStab" begin
    d = 5
    A = 0.1 * ∇(d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]
    max_bond = 8

    u₀ = rand_tt(tt_dims, tt_rks)
    guess = u₀
    steps = [0.05]

    sol_tt = crank_nicolson_method(
        A, u₀, guess, steps;
        normalize = false,
        alg = Krylov(),
        max_bond = max_bond,
        krylov_solver = :bicgstab,
        maxiter = 30,
        rtol = 1.0e-10,
        atol = 1.0e-12,
        verbosity = 0
    )

    A_dense = qtto_to_matrix(A)
    u_dense = qtt_to_function(u₀)
    I = qtto_to_matrix(id_tto(nsites(A)))
    sol_dense = (I - 0.5 * steps[1] * A_dense) \ ((I + 0.5 * steps[1] * A_dense) * u_dense)

    sol_tt_vec = qtt_to_function(sol_tt)
    rel_error = norm(sol_tt_vec - sol_dense) / norm(sol_dense)

    @test rel_error < 1.0e-7
    @test maximum(sol_tt.ranks) <= max_bond
end

@testset "Krylov solver supports CG selection and rejects unknown solvers" begin
    d = 3
    A = 0.1 * id_tto(d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]
    u₀ = rand_tt(tt_dims, tt_rks)
    guess = u₀
    steps = [0.05]

    sol_tt = implicit_euler_method(
        A, u₀, guess, steps;
        normalize = false,
        alg = Krylov(),
        isposdef = true,
        issymmetric = true,
        tol = 1.0e-12
    )

    A_dense = qtto_to_matrix(A)
    u_dense = qtt_to_function(u₀)
    I = qtto_to_matrix(id_tto(nsites(A)))
    sol_dense = (I - steps[1] * A_dense) \ u_dense

    rel_error = norm(qtt_to_function(sol_tt) - sol_dense) / norm(sol_dense)
    @test rel_error < 1.0e-8

    @test_throws ArgumentError implicit_euler_method(
        A, u₀, guess, steps;
        normalize = false,
        alg = Krylov(),
        krylov_solver = :unknown
    )
end

@testset "Euler-family methods cover normalize and return_info options" begin
    d = 3
    h = 1 / d^2
    A = -h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]
    u₀ = rand_tt(tt_dims, tt_rks)
    guess = u₀
    steps = [0.02]

    sol_euler, info_euler = euler_method(A, u₀, steps; normalize = true, return_info = true)
    @test isapprox(norm(sol_euler), 1.0; atol = 1.0e-10)
    @test isfinite(info_euler.error)

    sol_impl, info_impl = implicit_euler_method(
        A, u₀, guess, steps;
        normalize = true, return_info = true, alg = Krylov(), tol = 1.0e-10
    )
    @test isapprox(norm(sol_impl), 1.0; atol = 1.0e-10)
    @test isfinite(info_impl.error)

    sol_cn, info_cn = crank_nicolson_method(
        A, u₀, guess, steps;
        normalize = true, return_info = true, alg = Krylov(), tol = 1.0e-10
    )
    @test isapprox(norm(sol_cn), 1.0; atol = 1.0e-10)
    @test isfinite(info_cn.error)

    sol_rk = rk4_method(A, u₀, steps; max_bond = 6, normalize = true)
    @test isapprox(norm(sol_rk), 1.0; atol = 1.0e-10)
end

@testset "rk4_method basic test" begin
    d = 4
    h = 1 / d^2
    A = -h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]

    u₀ = rand_tt(tt_dims, tt_rks)
    steps = [0.05]
    max_bond = 8

    sol_tt = rk4_method(A, u₀, steps; max_bond, normalize = false)

    A_dense = qtto_to_matrix(A)
    u_dense = qtt_to_function(u₀)
    h_step = steps[1]

    k1 = A_dense * u_dense
    k2 = A_dense * (u_dense + (h_step / 2) * k1)
    k3 = A_dense * (u_dense + (h_step / 2) * k2)
    k4 = A_dense * (u_dense + h_step * k3)
    incr = (h_step / 6) * (k1 + 2k2 + 2k3 + k4)
    sol_dense = u_dense + incr

    sol_tt_vec = qtt_to_function(sol_tt)
    rel_error = norm(sol_tt_vec - sol_dense) / norm(sol_dense)

    @test rel_error < 1.0e-6
    println("RK4 test passed with relative error: ", rel_error)
end

@testset "rk4_method return_info consistency" begin
    d = 4
    h = 1 / d^2
    A = -h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]

    u₀ = rand_tt(tt_dims, tt_rks)
    steps = [0.05]
    max_bond = 8

    sol_tt, info = rk4_method(A, u₀, steps; max_bond, normalize = false, return_info = true)

    @test info.error < 1.0e-10

    A_dense = qtto_to_matrix(A)
    u_dense = qtt_to_function(u₀)
    h_step = steps[1]

    k1 = A_dense * u_dense
    k2 = A_dense * (u_dense + (h_step / 2) * k1)
    k3 = A_dense * (u_dense + (h_step / 2) * k2)
    k4 = A_dense * (u_dense + h_step * k3)
    incr = (h_step / 6) * (k1 + 2k2 + 2k3 + k4)
    sol_dense = u_dense + incr

    sol_tt_vec = qtt_to_function(sol_tt)
    rel_error = norm(sol_tt_vec - sol_dense) / norm(sol_dense)
    @test rel_error < 1.0e-6
end

@testset "euler_method return_info measures the last-step defect" begin
    d = 5
    A = toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    u₀ = qtt_sin(d)
    steps = fill(1.0e-2, 5)
    sol, info = euler_method(A, u₀, steps; normalize = false, return_info = true)
    # Exact explicit step with no truncation: the defect of the last step is zero.
    @test info.error < 1.0e-10
end

@testset "rk4_method return_info reflects truncation error" begin
    d = 5
    A = ∇(d)
    u₀ = qtt_sin(d)
    steps = fill(0.5, 3)
    _, info_tight = rk4_method(A, u₀, steps; max_bond = 32, normalize = false, return_info = true)
    _, info_loose = rk4_method(A, u₀, steps; max_bond = 1, normalize = false, return_info = true)
    @test info_tight.error < 1.0e-10   # no truncation at max_bond = 32 for d = 5
    @test info_loose.error > 1.0e-6    # rank-1 cap must show up as compression error
end

@testset "solver-type dispatch for implicit time steppers" begin
    d = 5
    h_grid = 1 / 2^d
    A = -h_grid^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    u₀ = qtt_sin(d)
    Random.seed!(17)
    guess = rand_tt(u₀.dims, 4)
    steps = fill(0.05, 3)

    sol_str = implicit_euler_method(A, u₀, guess, steps; alg = ALS(), normalize = false, max_sweeps = 2)
    sol_typ = implicit_euler_method(A, u₀, guess, steps; alg = ALSSolver(), normalize = false, max_sweeps = 2)
    @test qtt_to_vector(sol_typ) ≈ qtt_to_vector(sol_str)

    cn_str = crank_nicolson_method(A, u₀, guess, steps; alg = MALS(), normalize = false)
    cn_typ = crank_nicolson_method(A, u₀, guess, steps; alg = MALSSolver(), normalize = false)
    @test qtt_to_vector(cn_typ) ≈ qtt_to_vector(cn_str)

    kr = crank_nicolson_method(A, u₀, guess, steps; alg = KrylovSolver(), normalize = false, max_bond = 6)
    @test kr isa TTVector
    @test qtt_to_vector(kr) ≈ qtt_to_vector(cn_str) rtol = 1.0e-5

    dm = implicit_euler_method(A, u₀, guess, steps; alg = DMRGSolver(), normalize = false)
    @test dm isa TTVector
    @test qtt_to_vector(dm) ≈ qtt_to_vector(sol_str) rtol = 1.0e-5
end

@testset "time-stepper solver kwargs preserve wrapper parity" begin
    d = 4
    h = 1 / d^2
    A = -h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    dims = ntuple(_ -> 2, d)
    ranks = [1; fill(2, d - 1); 1]
    u₀ = rand_tt(dims, ranks)
    guess = u₀
    steps = [0.05]

    @test_throws MethodError implicit_euler_method(
        A, u₀, guess, steps;
        normalize = false,
        alg = ALSSolver(),
        tol = 1.0e-8
    )

    @test_throws MethodError crank_nicolson_method(
        A, u₀, guess, steps;
        normalize = false,
        alg = MALSSolver(),
        nsites = 2
    )

    @test implicit_euler_method(
        A, u₀, guess, steps;
        normalize = false,
        alg = DMRGSolver(),
        local_maxiter = 50
    ) isa TTVector

    @test crank_nicolson_method(
        A, u₀, guess, steps;
        normalize = false,
        alg = KrylovSolver(),
        max_bond = 6,
        krylovdim = 10
    ) isa TTVector
end

@testset "time steppers accept top-level show_progress option" begin
    d = 3
    h = 1 / 2^d
    A = -h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    u₀ = qtt_sin(d)
    guess = rand_tt(u₀.dims, u₀.ranks)
    steps = [0.01]

    @test euler_method(A, u₀, steps; normalize = false, show_progress = false) isa TTVector
    @test implicit_euler_method(A, u₀, guess, steps; normalize = false, show_progress = false, alg = ALS(max_sweeps = 1)) isa TTVector
    @test crank_nicolson_method(A, u₀, guess, steps; normalize = false, show_progress = false, alg = MALS()) isa TTVector
    @test rk4_method(A, u₀, steps; max_bond = 4, normalize = false, show_progress = false) isa TTVector
end

@testset "time steppers do not normalize by default" begin
    d = 4
    A = -1.0 * Δ(d)          # dissipative: the norm of the solution decays
    u₀ = qtt_sin(d)
    steps = fill(0.01, 3)
    for f in (
            kw -> euler_method(A, u₀, steps; show_progress = false, kw...),
            kw -> implicit_euler_method(A, u₀, u₀, steps; alg = ALS(), show_progress = false, kw...),
            kw -> crank_nicolson_method(A, u₀, u₀, steps; alg = ALS(), show_progress = false, kw...),
            kw -> rk4_method(A, u₀, steps; max_bond = 4, show_progress = false, kw...),
            kw -> tdvp(A, u₀, steps; imaginary_time = true, show_progress = false, kw...),
            kw -> tdvp2(A, u₀, steps; imaginary_time = true, show_progress = false, kw...),
        )
        default = f((;))
        @test norm(default) ≈ norm(f((; normalize = false)))
        @test norm(default) < norm(u₀)
    end
end

@testset "time steppers accept non-binary physical dimensions" begin
    Random.seed!(3)
    dims = (3, 2, 3)
    B = rand_tto(dims, 2)
    A = -1.0 * (B' * B)                       # symmetric negative semidefinite
    Ad = reshape(tto_to_tensor(A), prod(dims), :)
    u₀ = rand_tt(dims, [1, 3, 3, 1])          # full TT ranks: ALS solves exactly
    v₀ = vec(tt_to_tensor(u₀))
    h = 0.1
    Id = Matrix(1.0I, prod(dims), prod(dims))
    ie = implicit_euler_method(A, u₀, u₀, [h]; alg = ALS(max_sweeps = 3), show_progress = false)
    @test vec(tt_to_tensor(ie)) ≈ (Id - h * Ad) \ v₀
    cn = crank_nicolson_method(A, u₀, u₀, [h]; alg = ALS(max_sweeps = 3), show_progress = false)
    @test vec(tt_to_tensor(cn)) ≈ (Id - (h / 2) * Ad) \ ((Id + (h / 2) * Ad) * v₀)
    ee, info = euler_method(A, u₀, [h]; return_info = true, show_progress = false)
    @test vec(tt_to_tensor(ee)) ≈ (Id + h * Ad) * v₀
    @test info.error < 1.0e-12
end

@testset "stepper keyword names" begin
    TTN = TensorTrainNumerics
    Random.seed!(71)
    d = 4
    A = -1.0 * Δ(d)
    u₀ = rand_tt(ntuple(_ -> 2, d), 2; normalize = true)
    steps = fill(1.0e-3, 3)

    u, info = implicit_euler_method(A, u₀, u₀, steps; alg = ALS(; max_sweeps = 2), return_info = true, show_progress = false)
    @test info.error isa Real
    # An inner solver built with return_info = true still hands the stepper a TTVector.
    @test implicit_euler_method(A, u₀, u₀, steps; alg = ALS(; return_info = true), show_progress = false) isa TTVector
    u, info = crank_nicolson_method(A, u₀, u₀, steps; alg = Krylov(), tol = 1.0e-12, return_info = true, show_progress = false)
    @test info.error isa Real
    u, info = euler_method(A, u₀, steps; return_info = true, show_progress = false)
    @test info.error isa Real
    u = rk4_method(A, u₀, steps; max_bond = 4, show_progress = false)
    @test maximum(u.ranks) ≤ 4

    @test_throws MethodError implicit_euler_method(A, u₀, u₀, steps; tt_solver = MALS())
    @test_throws TypeError implicit_euler_method(A, u₀, u₀, steps; alg = "mals")
    @test_throws MethodError rk4_method(A, u₀, steps, 4)
    @test_throws MethodError euler_method(A, u₀, steps; return_error = true)

    inner = TTN._stepper_algorithm(DMRG(; max_sweeps = 3, show_progress = true, return_info = true); max_bond = 5)
    @test inner.max_sweeps == 3 && inner.max_bond == 5
    @test !inner.show_progress && !inner.return_info
    @test Krylov().show_progress
end

@testset "Krylov keeps its own max_bond unless the stepper sets one" begin
    TTN = TensorTrainNumerics
    alg = Krylov(; max_bond = 8)
    @test TTN._stepper_algorithm(alg; TTN._stepper_overrides(alg, 0)...).max_bond == 8
    @test TTN._stepper_algorithm(alg; TTN._stepper_overrides(alg, 5)...).max_bond == 5
end
