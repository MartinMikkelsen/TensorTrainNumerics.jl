module TensorTrainNumerics

export TTvector, TToperator, QTTvector, QTToperator, ttv_decomp, tto_decomp, ttv_to_tensor, tto_to_tensor, zeros_tt, zeros_tto, rand_tt, rand_tto, is_qtt, is_qtt_operator, visualize, tt_svdvals, matricize, tt2qtt, concatenate, ones_tt, orthogonalize, id_tto
include("tt_tools.jl")

export *, +, dot, -, /, outer_product, concatenate, permute, hadamard, kron, ⊗, euclidean_distance, euclidean_distance_normalized, TTdiag, norm
include("tt_operations.jl")

export als_linsolve, als_eigsolve, als_gen_eigsolv
include("solvers/als.jl")

export mals_eigsolve, mals_linsolve
include("solvers/mals.jl")

export dmrg_linsolve, dmrg_eigsolve
include("solvers/dmrg.jl")

export chebyshev_lobatto_nodes, gauss_chebyshev_lobatto, equally_spaced_nodes, legendre_nodes, get_nodes, lagrange_basis, interpolating_qtt, lagrange_rank_revealing
include("tt_interpolations.jl")

export toeplitz_to_qtto, qtto_prolongation, ∇, Δ, shift
include("tt_operators.jl")

export index_to_point, tuple_to_index, function_to_tensor, tensor_to_grid, function_to_qtt, qtt_to_function, qtt_polynom, qtt_cos, qtt_sin, qtt_exp, qtto_to_matrix, qtt_basis_vector, qtt_chebyshev, qtt_simpson, qtt_trapezoidal
include("qtt_tools.jl")

export euler_method, implicit_euler_method, crank_nicholson_method
include("solvers/euler.jl")

end
