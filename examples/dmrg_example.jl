using TensorTrainNumerics

dims = (2, 2, 2)
rks = [1, 2, 2, 1]

tt_start = rand_tt(dims, rks)

A_dims = (2, 2, 2)
A_rks = [1, 2, 2, 1]
A = rand_tto(A_dims, 3)

b = rand_tt(dims, rks)

tt_opt = linear_solve(A, b, tt_start, DMRG(max_sweeps = 1, nsites = 2, trunc_tol = 1.0e-12))

max_sweeps = [1, 2]
max_bond = [2, 3]

eigenvalues, tt_eigvec, r_hist = eigen_solve(
    A,
    tt_start,
    DMRG(; nsites = 2, trunc_tol = 1.0e-12, max_sweeps, max_bond),
)

println("Lowest eigenvalue: ", eigenvalues[end])
tt_eigvec
