module TensorTrainNumerics

export AbstractTTvector, AbstractTToperator, TTvector, TToperator, eltype, complex, ttv_decomp, tto_decomp, ttv_to_tensor, tto_to_tensor, tto_to_ttv, visualize, matricize, concatenate, orthogonalize, copy, r_and_d_to_rks, tt_compress!, _svdtrunc
include("tt_tools.jl")

export *, +, dot, -, /, add!, outer_product, hadamard, hadamard_ttm, kron, ⊕, ⊗, euclidean_distance, euclidean_distance_normalized, ttv_to_diag_tto, norm
include("tt_operations.jl")

export als_linsolve, als_eigsolve, als_gen_eigsolv
include("solvers/als.jl")

export nonlinear_als_eigsolve, nonlinear_mals_eigsolve, nonlinear_tdvp_imagtime, nls_energy, burgers_scf_als_step, burgers_scf_als, burgers_scf_mals_step, burgers_scf_mals, allen_cahn_als_step, allen_cahn_als, allen_cahn_mals_step, allen_cahn_mals, allen_cahn_2d_mals_step, allen_cahn_2d_mals
include("solvers/nonlinear.jl")

export mals_eigsolve, mals_linsolve
include("solvers/mals.jl")

export dmrg_linsolve, dmrg_eigsolve
include("solvers/dmrg.jl")

export tdvp, tdvp2
include("solvers/tdvp.jl")

export kdv_als_step, kdv_als, kdv_mals_step, kdv_mals, kdv_cn_mals_step, kdv_cn_mals
include("solvers/kdv.jl")

export to_ttvector
function to_ttvector end

_missing_interpolativeqtt_extension(name::Symbol) =
    throw(ArgumentError("`$name` requires loading InterpolativeQTT first; run `using InterpolativeQTT` before calling it."))

export interpolative_qtt, interpolative_qttv, invert_interpolative_qtt, project_nonlinearity
function interpolative_qtt end
function interpolative_qttv end
function invert_interpolative_qtt end
function project_nonlinearity end
interpolative_qtt(args...; kwargs...) = _missing_interpolativeqtt_extension(:interpolative_qtt)
interpolative_qttv(args...; kwargs...) = _missing_interpolativeqtt_extension(:interpolative_qttv)
invert_interpolative_qtt(args...; kwargs...) = _missing_interpolativeqtt_extension(:invert_interpolative_qtt)
project_nonlinearity(args...; kwargs...) = _missing_interpolativeqtt_extension(:project_nonlinearity)

export toeplitz_to_qtto, qtto_prolongation, ∇, ∇_c, ∇_c_P, ∇3, ∇3_P, Δ_DN, Δ_ND, Δ_NN, Δ_P, Δ, Δ⁻¹_DN, shift, zeros_tt, zeros_tto, ones_tt, rand_tt, id_tto, rand_tto, qtt_laplacian
include("tt_operators.jl")

export gauss_chebyshev_lobatto
export index_to_point, tuple_to_index, function_to_tensor, tensor_to_grid, function_to_qtt, qtt_to_function, qtt_to_vector, function_to_qtt_uniform, qtt_polynom, qtt_cos, qtt_sin, qtt_exp, qtto_to_matrix, qtt_basis_vector, qtt_chebyshev, qtt_trapezoidal, to_qtt, to_ttv, QTTvector, QTToperator, check_compat, function_to_qttv, qttv_to_array, reorder
include("qtt_tools.jl")

export euler_method, implicit_euler_method, crank_nicholson_method, rk4_method
include("solvers/euler.jl")

export fourier_qtto, reverse_qtt_bits
include("tt_transformations.jl")

export tt_cross, tt_integrate, MaxVol, DMRG, Greedy
include("tt_cross_interpolation.jl")

end
