using Test
using TensorTrainNumerics

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
