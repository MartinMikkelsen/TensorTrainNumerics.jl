using Test
using TensorTrainNumerics
using LinearAlgebra
using Random

import TensorTrainNumerics: dot

Random.seed!(20260512)

nonlinear_spd_op(d, shift = 3.0) = Δ(d) + shift * id_tto(d)

function nonlinear_chemical_potential(ψ, H_lin, g)
    ρ = hadamard(conj(ψ), ψ)
    return real(dot(ψ, H_lin * ψ)) + g * real(dot(ρ, ρ))
end

@testset "nls_energy handles complex TT states" begin
    d = 4
    g = 1.7
    ψ = complex(rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]))
    for (k, core) in pairs(ψ.ttv_vec)
        ψ.ttv_vec[k] .= cis(0.2 * k) .* core
    end

    H_lin = complex(id_tto(d))
    v = qtt_to_function(ψ)
    expected = real(LinearAlgebra.dot(v, v)) + (g / 2) * sum(abs2.(v) .^ 2)

    @test isapprox(nls_energy(ψ, H_lin, g), expected; rtol = 1.0e-10, atol = 1.0e-10)
end

@testset "nonlinear_als_eigsolve returns global sweep-wise μ history" begin
    d = 4
    g = 0.8
    H_lin = nonlinear_spd_op(d, 4.0)
    ψ0 = rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true)

    μ_hist, ψ = nonlinear_als_eigsolve(H_lin, g, ψ0; sweep_count = 3, verbose = false)

    @test length(μ_hist) == 3
    @test isapprox(μ_hist[end], nonlinear_chemical_potential(ψ, H_lin, g); rtol = 1.0e-10, atol = 1.0e-10)

    ψ_step = ψ0
    μ_expected = Float64[]
    for _ in 1:3
        μ_one, ψ_step = nonlinear_als_eigsolve(H_lin, g, ψ_step; sweep_count = 1, verbose = false)
        push!(μ_expected, only(μ_one))
    end

    @test isapprox(μ_hist, μ_expected; rtol = 1.0e-10, atol = 1.0e-10)
end

@testset "nonlinear_als_eigsolve handles complex TT states" begin
    d = 4
    g = 0.6
    H_lin = complex(nonlinear_spd_op(d, 3.5))
    ψ0 = complex(rand_tt(ntuple(_ -> 2, d), [1, 2, 2, 2, 1]; normalise = true))
    for (k, core) in pairs(ψ0.ttv_vec)
        ψ0.ttv_vec[k] .= cis(0.15 * k) .* core
    end

    μ_hist, ψ = nonlinear_als_eigsolve(H_lin, g, ψ0; sweep_count = 2, verbose = false)

    @test length(μ_hist) == 2
    @test all(isfinite, μ_hist)
    @test isapprox(μ_hist[end], nonlinear_chemical_potential(ψ, H_lin, g); rtol = 1.0e-10, atol = 1.0e-10)
end
