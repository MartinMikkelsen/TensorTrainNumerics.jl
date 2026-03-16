using Test
using TensorTrainNumerics

@testset "QTTGrid" begin

    @testset "valid construction" begin
        g = QTTGrid(2, 8)
        @test g.levels == 8
        @test g isa QTTGrid{2}  # N is a type parameter, not a field
        @test g.domain == ((0.0, 1.0), (0.0, 1.0))
        @test g.bc == (:dn, :dn)
        @test g.ordering == :serial
    end

    @testset "custom kwargs" begin
        g = QTTGrid(3, 5; domain = [(0.0, 2.0), (-1.0, 1.0), (0.0, π)], bc = (:dn, :nn, :periodic))
        @test g.domain == ((0.0, 2.0), (-1.0, 1.0), (0.0, Float64(π)))
        @test g.bc == (:dn, :nn, :periodic)
    end

    @testset "validation errors" begin
        @test_throws ArgumentError QTTGrid(2, 3)              # levels < 4
        @test_throws ArgumentError QTTGrid(0, 4)              # N < 1
        @test_throws ArgumentError QTTGrid(2, 4; ordering = :interleaved)
        @test_throws ArgumentError QTTGrid(2, 4; bc = (:dn, :foo))
        @test_throws ArgumentError QTTGrid(1, 4; domain = [(1.0, 0.0)])  # lo >= hi
    end

    @testset "accessors" begin
        g = QTTGrid(1, 8)                         # 1D, [0,1], 256 points
        @test npoints(g) == 256
        @test gridstep(g, 1) ≈ 1.0 / 256
        nd = nodes(g, 1)
        @test length(nd) == 256
        @test nd[1] == 0.0
        @test nd[end] ≈ 255.0 / 256             # upper NOT included
        @test nd[2] - nd[1] ≈ gridstep(g, 1)
    end

    @testset "accessors — non-unit domain" begin
        g = QTTGrid(1, 4; domain = [(2.0, 4.0)])
        h = gridstep(g, 1)
        @test h ≈ 2.0 / 16                       # L/n = 2/16
        nd = nodes(g, 1)
        @test nd[1] == 2.0
        @test nd[end] ≈ 2.0 + 15 * h
    end

end

@testset "Operator builders" begin

    # Helper: convert QTT operator to dense matrix for comparison
    # qtto_to_matrix converts a TToperator to a dense matrix
    levels = 4          # 2^4 = 16 points; kept small for matrix comparisons
    n = 2^levels
    h = 1.0 / n

    @testset "_laplacian_1d — shape and BC dispatch" begin
        for bc_sym in (:dn, :nd, :nn, :periodic)
            g = QTTGrid(1, levels; bc = (bc_sym,))
            L = TensorTrainNumerics._laplacian_1d(g, 1)
            M = qtto_to_matrix(L)
            @test size(M) == (n, n)
        end
    end

    @testset "diffusion_operator 1D matches (1/h²)·Δ_DN" begin
        g = QTTGrid(1, levels; bc = (:dn,))
        A = diffusion_operator(g; κ = 1.0)
        ref = (1.0 / h^2) * qtto_to_matrix(Δ_DN(levels))
        @test qtto_to_matrix(A) ≈ ref rtol=1e-12
    end

    @testset "diffusion_operator 2D matches manual Kronecker" begin
        g = QTTGrid(2, levels; bc = (:dn, :dn))
        A = diffusion_operator(g; κ = 1.0)
        D  = (1.0 / h^2) * qtto_to_matrix(Δ_DN(levels))
        Id = qtto_to_matrix(id_tto(levels))
        ref = kron(D, Id) + kron(Id, D)
        @test qtto_to_matrix(A) ≈ ref rtol=1e-10
    end

    @testset "diffusion_operator κ scaling" begin
        g = QTTGrid(1, levels; bc = (:dn,))
        A2 = diffusion_operator(g; κ = 2.5)
        A1 = diffusion_operator(g; κ = 1.0)
        @test qtto_to_matrix(A2) ≈ 2.5 .* qtto_to_matrix(A1) rtol=1e-12
    end

    @testset "advection_operator 1D matches (v/h)·∇" begin
        g = QTTGrid(1, levels; bc = (:dn,))
        A = advection_operator(g; v = [1.0])
        ref = (1.0 / h) * qtto_to_matrix(∇(levels))
        @test qtto_to_matrix(A) ≈ ref rtol=1e-12
    end

    @testset "advection_operator — all-zero velocity gives zero matrix" begin
        g = QTTGrid(1, levels; bc = (:dn,))
        A = advection_operator(g; v = [0.0])
        M = qtto_to_matrix(A)
        @test all(iszero, M)
    end

    @testset "reaction_operator 2D matches σ·(I⊗I)" begin
        g = QTTGrid(2, levels; bc = (:dn, :dn))
        A = reaction_operator(g; σ = 3.7)
        Id = qtto_to_matrix(id_tto(levels))
        ref = 3.7 .* kron(Id, Id)
        @test qtto_to_matrix(A) ≈ ref rtol=1e-12
    end

    @testset "laplacian is alias for diffusion_operator(κ=1)" begin
        g = QTTGrid(1, levels)
        @test qtto_to_matrix(laplacian(g)) ≈ qtto_to_matrix(diffusion_operator(g; κ=1.0)) rtol=1e-14
    end

    @testset "operators are composable" begin
        g = QTTGrid(1, levels)
        L = diffusion_operator(g; κ=0.1) + reaction_operator(g; σ=2.0)
        ref = 0.1 .* qtto_to_matrix(diffusion_operator(g)) .+ 2.0 .* qtto_to_matrix(reaction_operator(g; σ=1.0))
        @test qtto_to_matrix(L) ≈ ref rtol=1e-12
    end

end

@testset "source" begin

    @testset "function form — 1D" begin
        # 2^4 = 16 points; f(x) = sin(πx), rank-1 in QTT
        g  = QTTGrid(1, 4)
        tt = source(g, x -> sin(π * x[1]))
        # Reconstruct and compare to exact
        approx = real.(qtt_to_function(tt))
        exact  = sin.(π .* nodes(g, 1))
        @test norm(approx - exact) / norm(exact) < 1e-6
    end

    @testset "function form — 2D separable" begin
        g  = QTTGrid(2, 4)
        tt = source(g, x -> sin(π * x[1]) * cos(π * x[2]))
        approx = reshape(real.(qtt_to_function(tt)), npoints(g), npoints(g))
        exact  = sin.(π .* nodes(g, 1)) * cos.(π .* nodes(g, 2))'
        @test norm(approx - exact) / norm(exact) < 1e-5
    end

    @testset "TTvector passthrough" begin
        g  = QTTGrid(1, 4)
        b  = rand_tt(ntuple(_ -> 2, 4), 2)   # rank-2 random TT over 2^4 = 16 points
        b2 = source(g, b)
        @test b2 === b                         # reference equality, no copy
    end

end

@testset "EllipticPDE + solve" begin

    # 1D Poisson: laplacian(grid) * u = (π/2)²·sin(πx/2)
    # Exact solution: u(x) = sin(πx/2), satisfies u(0)=0 (Dirichlet), u'(1)=0 (Neumann)
    # laplacian = (1/h²)*Δ_DN is positive-definite (diagonal +2, discretises -∂²/∂x²)
    # Δ_DN uses a ghost-point Neumann BC giving O(h) global error; for n=64 this is ~2–3%.
    # Tolerances below reflect the discretisation error, not solver inaccuracy.

    g     = QTTGrid(1, 6; bc = (:dn,))       # 64 points, h = 1/64
    A     = laplacian(g)
    b     = source(g, x -> (π / 2)^2 * sin(π * x[1] / 2))
    u_ex  = sin.(π / 2 .* nodes(g, 1))

    @testset "ALSLinsolve" begin
        sol = solve(EllipticPDE(A, b, g), ALSLinsolve(maxiter = 20))
        arr = to_array(sol)
        @test size(arr) == (64,)
        @test norm(arr - u_ex) / norm(u_ex) < 0.05   # O(h) discretisation: ~2.25% at n=64
    end

    @testset "MALSLinsolve" begin
        sol = solve(EllipticPDE(A, b, g), MALSLinsolve(tol = 1e-8, rmax = 20))
        arr = to_array(sol)
        @test norm(arr - u_ex) / norm(u_ex) < 0.05
    end

    @testset "DMRGLinsolve" begin
        sol = solve(EllipticPDE(A, b, g), DMRGLinsolve(maxiter = 10, rmax = 20))
        arr = to_array(sol)
        @test norm(arr - u_ex) / norm(u_ex) < 0.05
    end

    @testset "to_array shape — 2D" begin
        g2  = QTTGrid(2, 4; bc = (:dn, :dn))
        A2  = laplacian(g2)
        b2  = source(g2, x -> sin(π * x[1]) * sin(π * x[2]))
        sol2 = solve(EllipticPDE(A2, b2, g2), MALSLinsolve())
        @test size(to_array(sol2)) == (16, 16)
    end

end

@testset "ParabolicPDE + solve" begin

    # 1D heat: ∂ₜu = 0.1·Δu, u₀ = sin(πx/2)
    # Imaginary-time TDVP with normalize=false: norm should decrease
    g  = QTTGrid(1, 5; bc = (:dn,))          # 32 points
    L  = diffusion_operator(g; κ = 0.1)
    u0 = source(g, x -> sin(π * x[1] / 2))
    norm_u0 = norm(real.(qtt_to_function(u0)))

    @testset "TDVP2Solver — norm decreases (imaginary-time, normalize=false)" begin
        sol = solve(
            ParabolicPDE(L, u0, g; tspan = (0.0, 0.5), dt = 0.05),
            TDVP2Solver(truncerr = 1e-6, rmax = 20, imaginary_time = true, normalize = false),
        )
        @test norm(to_array(sol)) < norm_u0
    end

    @testset "TDVP1Solver — returns PDESolution with correct shape" begin
        sol = solve(
            ParabolicPDE(L, u0, g; tspan = (0.0, 0.1), dt = 0.05),
            TDVP1Solver(imaginary_time = true, normalize = false),
        )
        @test size(to_array(sol)) == (32,)
    end

    @testset "to_array shape — 3D" begin
        # Note: Δ_P has non-unit boundary ranks so periodic BC cannot be mixed with
        # other BCs in the TToperator + when N ≥ 3.  Use non-periodic BCs here.
        g3 = QTTGrid(3, 4; bc = (:dn, :dn, :nd))
        L3 = diffusion_operator(g3; κ = 0.1)
        u03 = source(g3, x -> sin(π * x[1] / 2))
        sol3 = solve(
            ParabolicPDE(L3, u03, g3; tspan = (0.0, 0.1), dt = 0.05),
            TDVP2Solver(imaginary_time = true, normalize = false),
        )
        @test size(to_array(sol3)) == (16, 16, 16)
    end

    @testset "ParabolicPDE — validation errors" begin
        g  = QTTGrid(1, 5; bc = (:dn,))
        L  = diffusion_operator(g; κ = 0.1)
        u0 = source(g, x -> sin(π * x[1] / 2))
        @test_throws ArgumentError ParabolicPDE(L, u0, g; tspan = (1.0, 0.0), dt = 0.05)  # reversed tspan
        @test_throws ArgumentError ParabolicPDE(L, u0, g; tspan = (0.0, 1.0), dt = -0.1)  # negative dt
    end

end

@testset "nodes delegate on PDESolution" begin
    g   = QTTGrid(1, 4)
    A   = laplacian(g)
    b   = source(g, x -> (π / 2)^2 * sin(π * x[1] / 2))
    sol = solve(EllipticPDE(A, b, g), ALSLinsolve())
    @test nodes(sol, 1) == nodes(g, 1)
end
