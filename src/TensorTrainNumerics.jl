module TensorTrainNumerics

export AbstractTTvector, AbstractTToperator, TTvector, TToperator, eltype, complex, ttv_decomp, tto_decomp, ttv_to_tensor, tto_to_tensor, tto_to_ttv, visualize, matricize, concatenate, orthogonalize, entanglemententropy, copy, r_and_d_to_rks, tt_compress!, tt_round!, tt_round, ttvector_manifold
"""
    ttvector_manifold(x::TTvector)

Return a ManifoldsBase.jl manifold whose points are `TTvector`s with the
physical dimensions of `x`, for use with Manopt.jl solvers. The space is flat:
the inner product is `real(dot(X, Y))`, and the retraction is
`orthogonalize(p + t X)`, which does not truncate ranks.

Defined in the extension that loads when ManifoldsBase.jl and Manopt.jl are
loaded; real element types only.
"""
function ttvector_manifold end
include("tt_tools.jl")

export *, +, dot, -, /, add!, outer_product, hadamard, hadamard_ttm, kron, ⊕, ⊗, ⨝, ∙, euclidean_distance, euclidean_distance_normalized, ttv_to_diag_tto, norm
include("tt_operations.jl")

export LinearSolverAlgorithm, EigenSolverAlgorithm
export ALS, MALS, DMRG, Krylov
export TTLinearSolver, ALSSolver, MALSSolver, DMRGSolver, KrylovSolver
export linear_solve, eigen_solve
include("solvers/linear_solver.jl")

export als_linsolve, als_eigsolve, als_gen_eigsolv
include("solvers/als.jl")

export NonLinearSolverAlgorithm, PenaltyALS, MGR, non_linear_solve, gpe_energy
include("solvers/non_linear_solver.jl")

export mals_eigsolve, mals_linsolve
include("solvers/mals.jl")

export dmrg_linsolve, dmrg_eigsolve
include("solvers/dmrg.jl")

include("solvers/eigen_solver.jl")

export tdvp, tdvp2
include("solvers/tdvp.jl")

export to_ttvector
"""
    to_ttvector(tt::TensorCrossInterpolation.TensorTrain) -> TTvector

Convert a tensor train from TensorCrossInterpolation.jl to a [`TTvector`](@ref).
Defined in the extension that loads when InterpolativeQTT.jl and
TensorCrossInterpolation.jl are loaded.
"""
function to_ttvector end

export toeplitz_to_qtto, qtto_prolongation, qtto_constant_prolongation, qtto_linear_prolongation, ∇, Δ_DN, Δ_ND, Δ_NN, Δ_P, Δ, Δ⁻¹_DN, shift, pauli_matrix, pauli_sum_tto, pauli_pair_sum_tto, H_μ, H_μν, heisenberg_xyz_tto, ising_tto, xxz_tto, xxx_tto, xy_tto, zeros_tt, zeros_tto, rand_tt, id_tto, rand_tto, qtt_laplacian
include("tt_operators.jl")

export gauss_chebyshev_lobatto
export index_to_point, tuple_to_index, function_to_tensor, tensor_to_grid, function_to_qtt, qtt_to_function, qtt_to_vector, function_to_qtt_uniform, qtt_polynom, qtt_cos, qtt_sin, qtt_exp, qtto_to_matrix, qtt_basis_vector, qtt_chebyshev, qtt_trapezoidal, to_qtt, to_ttv, QTTvector, QTToperator, check_compat, function_to_qttv, qttv_to_array, reorder
include("qtt_tools.jl")

export euler_method, implicit_euler_method, crank_nicholson_method, rk4_method
include("solvers/time_evolution.jl")

export fourier_qtto, reverse_qtt_bits
include("tt_transformations.jl")

export tt_cross, tt_integrate, MaxVol, DMRGcross, Greedy
include("tt_cross_interpolation.jl")

end
