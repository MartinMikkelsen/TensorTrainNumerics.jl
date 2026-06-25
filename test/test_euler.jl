using Test
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
    sol_tt = implicit_euler_method(A, u₀, guess, steps; normalize = false, tt_solver = "dmrg")

    # Convert to dense for validation
    A_dense = qtto_to_matrix(A)
    u_dense = qtt_to_function(u₀)
    I = qtto_to_matrix(id_tto(A.N))
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
        normalize = false, tt_solver = "krylov", tol = 1.0e-12
    )

    A_dense = qtto_to_matrix(A)
    u_dense = qtt_to_function(u₀)
    I = qtto_to_matrix(id_tto(A.N))
    sol_dense = (I - steps[1] * A_dense) \ u_dense

    sol_tt_vec = qtt_to_function(sol_tt)
    rel_error = norm(sol_tt_vec - sol_dense) / norm(sol_dense)

    @test rel_error < 1.0e-8
end

@testset "Crank-Nicholson method basic test" begin
    d = 4
    h = 1 / d^2
    A = -h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]

    u₀ = rand_tt(tt_dims, tt_rks)
    guess = u₀
    steps = [0.05]

    sol_tt = crank_nicholson_method(A, u₀, guess, steps; normalize = false, tt_solver = "mals")

    A_dense = qtto_to_matrix(A)
    u_dense = qtt_to_function(u₀)
    I = qtto_to_matrix(id_tto(A.N))
    sol_dense = (I - 0.5 * steps[1] * A_dense) \ ((I + 0.5 * steps[1] * A_dense) * u_dense)

    sol_tt_vec = qtt_to_function(sol_tt)
    rel_error = norm(sol_tt_vec - sol_dense) / norm(sol_dense)

    @test rel_error < 1.0e-5
end

@testset "Crank-Nicholson method Krylov solver" begin
    d = 4
    h = 1 / d^2
    A = -h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]

    u₀ = rand_tt(tt_dims, tt_rks)
    guess = u₀
    steps = [0.05]

    sol_tt = crank_nicholson_method(
        A, u₀, guess, steps;
        normalize = false, tt_solver = "krylov", tol = 1.0e-12
    )

    A_dense = qtto_to_matrix(A)
    u_dense = qtt_to_function(u₀)
    I = qtto_to_matrix(id_tto(A.N))
    sol_dense = (I - 0.5 * steps[1] * A_dense) \ ((I + 0.5 * steps[1] * A_dense) * u_dense)

    sol_tt_vec = qtt_to_function(sol_tt)
    rel_error = norm(sol_tt_vec - sol_dense) / norm(sol_dense)

    @test rel_error < 1.0e-8
end

@testset "Crank-Nicholson Krylov solver handles non-symmetric operators" begin
    d = 4
    A = 0.1 * ∇(d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]

    u₀ = rand_tt(tt_dims, tt_rks)
    guess = u₀
    steps = [0.05]

    sol_tt = crank_nicholson_method(
        A, u₀, guess, steps;
        normalize = false, tt_solver = "krylov", tol = 1.0e-12
    )

    A_dense = qtto_to_matrix(A)
    @test !issymmetric(A_dense)

    u_dense = qtt_to_function(u₀)
    I = qtto_to_matrix(id_tto(A.N))
    sol_dense = (I - 0.5 * steps[1] * A_dense) \ ((I + 0.5 * steps[1] * A_dense) * u_dense)

    sol_tt_vec = qtt_to_function(sol_tt)
    rel_error = norm(sol_tt_vec - sol_dense) / norm(sol_dense)

    @test rel_error < 1.0e-8
end

@testset "Crank-Nicholson Krylov solver supports bounded BiCGStab" begin
    d = 5
    A = 0.1 * ∇(d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]
    max_bond = 8

    u₀ = rand_tt(tt_dims, tt_rks)
    guess = u₀
    steps = [0.05]

    sol_tt = crank_nicholson_method(
        A, u₀, guess, steps;
        normalize = false,
        tt_solver = "krylov",
        max_bond = max_bond,
        krylov_solver = :bicgstab,
        maxiter = 30,
        rtol = 1.0e-10,
        atol = 1.0e-12,
        verbosity = 0
    )

    A_dense = qtto_to_matrix(A)
    u_dense = qtt_to_function(u₀)
    I = qtto_to_matrix(id_tto(A.N))
    sol_dense = (I - 0.5 * steps[1] * A_dense) \ ((I + 0.5 * steps[1] * A_dense) * u_dense)

    sol_tt_vec = qtt_to_function(sol_tt)
    rel_error = norm(sol_tt_vec - sol_dense) / norm(sol_dense)

    @test rel_error < 1.0e-7
    @test maximum(sol_tt.ttv_rks) <= max_bond
    @test TensorTrainNumerics.KRYLOV_ROUND_RANK[] == 0
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
        tt_solver = "krylov",
        isposdef = true,
        issymmetric = true,
        tol = 1.0e-12
    )

    A_dense = qtto_to_matrix(A)
    u_dense = qtt_to_function(u₀)
    I = qtto_to_matrix(id_tto(A.N))
    sol_dense = (I - steps[1] * A_dense) \ u_dense

    rel_error = norm(qtt_to_function(sol_tt) - sol_dense) / norm(sol_dense)
    @test rel_error < 1.0e-8

    @test_throws ArgumentError implicit_euler_method(
        A, u₀, guess, steps;
        normalize = false,
        tt_solver = "krylov",
        krylov_solver = :unknown
    )
end

@testset "Euler-family methods cover normalize and return_error options" begin
    d = 3
    h = 1 / d^2
    A = -h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]
    u₀ = rand_tt(tt_dims, tt_rks)
    guess = u₀
    steps = [0.02]

    sol_euler, err_euler = euler_method(A, u₀, steps; normalize = true, return_error = true)
    @test isapprox(norm(sol_euler), 1.0; atol = 1.0e-10)
    @test isfinite(err_euler)

    sol_impl, err_impl = implicit_euler_method(
        A, u₀, guess, steps;
        normalize = true, return_error = true, tt_solver = "krylov", tol = 1.0e-10
    )
    @test isapprox(norm(sol_impl), 1.0; atol = 1.0e-10)
    @test isfinite(err_impl)

    sol_cn, err_cn = crank_nicholson_method(
        A, u₀, guess, steps;
        normalize = true, return_error = true, tt_solver = "krylov", tol = 1.0e-10
    )
    @test isapprox(norm(sol_cn), 1.0; atol = 1.0e-10)
    @test isfinite(err_cn)

    sol_rk = rk4_method(A, u₀, steps, 6; normalize = true)
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

    sol_tt = rk4_method(A, u₀, steps, max_bond; normalize = false)

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

@testset "rk4_method return_error consistency" begin
    d = 4
    h = 1 / d^2
    A = -h^2 * toeplitz_to_qtto(-2.0, 1.0, 1.0, d)
    tt_dims = ntuple(_ -> 2, d)
    tt_rks = [1; fill(2, d - 1); 1]

    u₀ = rand_tt(tt_dims, tt_rks)
    steps = [0.05]
    max_bond = 8

    sol_tt, rel_err = rk4_method(A, u₀, steps, max_bond; normalize = false, return_error = true)

    @test rel_err < 1.0e-10

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
