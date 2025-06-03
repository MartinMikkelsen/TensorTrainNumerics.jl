module TensorTrainNumerics

export TTvector,TToperator,QTTvector,QTToperator,ttv_decomp,tto_decomp,ttv_to_tensor,tto_to_tensor,zeros_tt,zeros_tto,rand_tt,rand_tto, is_qtt, is_qtt_operator,visualize,tt_svdvals, matricize, tt2qtt, concatenate, ones_tt, orthogonalize, id_tto
include("tt_tools.jl")

export *, +, dot, -, /, outer_product, concatenate, permute, hadamard, kron, ⊗
include("tt_operations.jl")

export als_linsolv, als_eigsolv, als_gen_eigsolv
include("solvers/als.jl")

export mals_eigsolv, mals_linsolv
include("solvers/mals.jl")

export dmrg_linsolve, dmrg_eigsolve
include("solvers/dmrg.jl")

export chebyshev_lobatto_nodes, equally_spaced_nodes, legendre_nodes, get_nodes, lagrange_basis, interpolating_qtt, lagrange_rank_revealing
include("tt_interpolations.jl")

export Δ_DD, Δ_NN, Δ_DN, Δ_ND, Δ_Periodic, Δ_tto, QTT_Tridiagonal_Toeplitz, ∇_DD, ∇_tto, Jacobian_tto, shift_tto, toeplitz_to_qtto
include("tt_operators.jl")

export index_to_point, tuple_to_index, function_to_tensor, tensor_to_grid, function_to_qtt, qtt_to_function, qtt_polynom, qtt_cos, qtt_sin, qtt_exp, qtto_to_matrix, qtt_basis_vector
include("qtt_tools.jl")

end
