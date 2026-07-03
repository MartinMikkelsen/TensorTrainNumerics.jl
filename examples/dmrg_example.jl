using TensorTrainNumerics

dims = (2, 2, 2)
rks = [1, 2, 2, 1]

tt_start = rand_tt(dims, rks)

A_dims = (2, 2, 2)
A_rks = [1, 2, 2, 1]
A = rand_tto(A_dims, 3)

b = rand_tt(dims, rks)

tt_opt = linear_solve(A, b, tt_start, DMRG(sweep_count = 2, N = 2, tol = 1.0e-12))

sweep_schedule = [2, 4]
rmax_schedule = [2, 3]

eigenvalues, tt_eigvec, r_hist = eigen_solve(
    A,
    tt_start,
    DMRG(N = 2, tol = 1.0e-12, sweep_schedule = sweep_schedule, rmax_schedule = rmax_schedule),
)

println("Lowest eigenvalue: ", eigenvalues[end])
tt_eigvec
