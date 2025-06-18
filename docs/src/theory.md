# Tensor trains

## Table of Contents
- [Basics](#basics)
    - [Multiplication](#multiplication)
    - [Tensor Train Operator times Tensor Train Vector](#tensor-train-operator-and-tensor-train-vector)
    - [Tensor Train Operator times Tensor Train Operator](#tensor-train-operator-and-tensor-train-operator)
    - [Addition](#addition)
    - [Concatenation](#concatenation)
    - [Matricization](#matricization)
    - [Visualization](#visualization)
- [Tensor Train Decomposition](#tensor-train-decomposition)
  - [Example with Tolerance](#example-with-tolerance)
- [Optimization](#optimization)
    - [ALS](#als)
    - [MALS](#mals)
    - [DMRG](#dmrg)

# Basics

## Multiplication

### Tensor train operator times tensor train vector

To multiply a tensor train operator (`TToperator`) by a tensor train vector (`TTvector`), use the `*` operator.

```@example 2
using TensorTrainNumerics

# Define the dimensions and ranks for the TTvector
dims = (2, 2, 2)
rks = [1, 2, 2, 1]

# Create a random TTvector
tt_vec = rand_tt(dims, rks)

# Define the dimensions and ranks for the TToperator
op_dims = (2, 2, 2)
op_rks = [1, 2, 2, 1]

# Create a random TToperator
tt_op = rand_tto(op_dims, 3)

# Perform the multiplication
result = tt_op * tt_vec

# Visualize the result
visualize(result)
```

### Tensor train operator times tensor train operator

To multiply two tensor train operators, use the `*` operator.

```@example 2
# Create another random TToperator
tt_op2 = rand_tto(op_dims, 3)

# Perform the multiplication
result_op = tt_op * tt_op2

# Visualize the result
visualize(result_op)
```

## Addition

To add two tensor train vectors or operators, use the `+` operator.

```@example 2
# Create another random TTvector
tt_vec2 = rand_tt(dims, rks)

# Perform the addition
result_add = tt_vec + tt_vec2

# Visualize the result
visualize(result_add)
```

## Concatenation

To concatenate two tensor train vectors or operators, use the `concatenate` function.

```@example 2
# Concatenate two TTvectors
result_concat = concatenate(tt_vec, tt_vec2)

# Visualize the result
visualize(result_concat)
```

## Matricization

To convert a tensor train vector or operator into its matrix form, use the `matricize` function.

```@example 2
# Matricize the TTvector
result_matrix = matricize(tt_vec)

# Print the result
println(result_matrix)
```

## Visualization

To visualize a tensor train vector or operator, use the `visualize` function.

```@example 2
# Visualize the TTvector
visualize(tt_vec)
```

## Tensor Train Decomposition

The `ttv_decomp` function performs a tensor train decomposition on a given tensor.


```@example 4
using TensorTrainNumerics

# Define a 3-dimensional tensor
tensor = rand(2, 3, 4)

# Perform the tensor train decomposition
ttv = ttv_decomp(tensor)

# Print the TTvector ranks
println(ttv.ttv_rks)
```

### Explanation

The `ttv_decomp` function takes a tensor as input and returns its tensor train decomposition in the form of a `TTvector`. The decomposition is performed using the Hierarchical SVD algorithm, which decomposes the tensor into a series of smaller tensors (cores) connected by ranks.

### Example with Tolerance

```@example 5
using TensorTrainNumerics

# Define a 3-dimensional tensor
tensor = rand(2, 3, 4)

# Perform the tensor train decomposition with a custom tolerance
ttv = ttv_decomp(tensor, tol=1e-10)

# Print the TTvector ranks
println(ttv.ttv_rks)
```
# Optimization

## ALS

```@example 6
using TensorTrainNumerics

# Define the dimensions and ranks for the TTvector
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

# Solve the linear system Ax = b using the ALS algorithm
tt_opt = als_linsolve(A, b, tt_start; sweep_count=2)

# Print the optimized TTvector
println(tt_opt)

# Define the sweep schedule and rank schedule for the eigenvalue problem
sweep_schedule = [2, 4]
rmax_schedule = [2, 3]

# Solve the eigenvalue problem using the ALS algorithm
eigenvalues, tt_eigvec = als_eigsolve(A, tt_start; sweep_schedule=sweep_schedule, rmax_schedule=rmax_schedule)

# Print the lowest eigenvalue and the corresponding eigenvector
println("Lowest eigenvalue: ", eigenvalues[end])
println("Corresponding eigenvector: ", tt_eigvec)
```

## MALS

```@example 7
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

# Solve the linear system Ax = b using the MALS algorithm
tt_opt = mals_linsolve(A, b, tt_start; tol=1e-12, rmax=4)

# Print the optimized TTvector
println(tt_opt)

# Define the sweep schedule and rank schedule for the eigenvalue problem
sweep_schedule = [2, 4]
rmax_schedule = [2, 3]

# Solve the eigenvalue problem using the MALS algorithm
eigenvalues, tt_eigvec, r_hist = mals_eigsolve(A, tt_start; tol=1e-12, sweep_schedule=sweep_schedule, rmax_schedule=rmax_schedule)

# Print the lowest eigenvalue and the corresponding eigenvector
println("Lowest eigenvalue: ", eigenvalues[end])
println("Corresponding eigenvector: ", tt_eigvec)
println("Rank history: ", r_hist)
```

## DMRG 
```@example 8 
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
tt_opt = dmrg_linsolvee(A, b, tt_start; sweep_count=2, N=2, tol=1e-12)

# Print the optimized TTvector
println(tt_opt)

# Define the sweep schedule and rank schedule for the eigenvalue problem
sweep_schedule = [2, 4]
rmax_schedule = [2, 3]

# Solve the eigenvalue problem using the DMRG algorithm
eigenvalues, tt_eigvec, r_hist = dmrg_eigsolve(A, tt_start; N=2, tol=1e-12, sweep_schedule=sweep_schedule, rmax_schedule=rmax_schedule)

# Print the lowest eigenvalue and the corresponding eigenvector
println("Lowest eigenvalue: ", eigenvalues[end])
println("Corresponding eigenvector: ", tt_eigvec)
println("Rank history: ", r_hist)
```