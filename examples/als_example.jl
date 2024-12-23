using TensorTrainNumerics

dims = (2, 2, 2)
rks = [1, 2, 2, 1]

# Create a random TTvector for the initial guess
tt_start = rand_tt(dims, rks)

# Create a random TToperator for the matrix A
A_dims = (2, 2, 2)
A_rks = [1, 2, 2, 1]
A = rand_tto(A_dims, 3)

# Create a random TTvector for the right-hand side b
b = rand_tt(dims, rks)

# Solve the linear system Ax = b using the DMRG algorithm
tt_opt = dmrg_linsolv(A, b, tt_start; sweep_count=2, N=2, tol=1e-12)

# Print the optimized TTvector
println(tt_opt)

# Define the sweep schedule and rank schedule for the eigenvalue problem
sweep_schedule = [2, 4]
rmax_schedule = [2, 3]

# Solve the eigenvalue problem using the DMRG algorithm
eigenvalues, tt_eigvec, r_hist = dmrg_eigsolv(A, tt_start; N=2, tol=1e-12, sweep_schedule=sweep_schedule, rmax_schedule=rmax_schedule)

# Print the lowest eigenvalue and the corresponding eigenvector
println("Lowest eigenvalue: ", eigenvalues[end])
println("Corresponding eigenvector: ", tt_eigvec)
println("Rank history: ", r_hist)