module TensorTrainNumerics

export TTvector,TToperator,QTTvector,QTToperator,ttv_decomp,tto_decomp,ttv_to_tensor,tto_to_tensor,zeros_tt,zeros_tto,rand_tt,rand_tto,tt_to_vidal,vidal_to_tensor,vidal_to_left_canonical, json_to_mps, json_to_mpo, is_qtt, is_qtt_operator,visualize,tt_svdvals, unfold, matricize
include("tt_tools.jl")

export *, +, dot, -, /, outer_product, concatenate
include("tt_operations.jl")

export als_linsolv, als_eigsolv, als_gen_eigsolv
include("solvers/als.jl")

export mals_eigsolv, mals_linsolv
include("solvers/mals.jl")

export dmrg_linsolv, dmrg_eigsolv
include("solvers/dmrg.jl")

export chebyshev_lobatto_nodes, equally_spaced_nodes, legendre_nodes, get_nodes, lagrange_basis, interpolating_qtt, lagrange_rank_revealing
include("tt_interpolations.jl")

export Δ_DD, Δ_NN, Δ_DN, Δ_ND, Δ_Periodic, Δ_tto, QTT_Tridiagonal_Toeplitz, ∇_DD, ∇_tto, Jacobian_tto, matricize, tt2qtt, χ, shift_tto
include("tt_operators.jl")

end
