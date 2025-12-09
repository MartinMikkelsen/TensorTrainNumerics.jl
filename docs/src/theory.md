# Quantics tensor trains

Quantics tensor trains (QTT) are a powerful tool for representing and manipulating high-dimensional data in a compressed format. They are particularly useful for solving high-dimensional partial differential equations and large-scale linear algebra problems, where traditional methods may be computationally infeasible due to the curse of dimensionality.

## Defining tensor train vectors

You can construct a quantics tensor train in several ways. The most straightforward way is to use the function `function_to_qtt`, which takes a function and a discretization level as input and returns a QTT representation of the function evaluated on a uniform grid.


### Mathematical functions in QTT format

You can also use the built-in functions for common mathematical functions, such as `qtt_exp`, `qtt_sin`, `qtt_cos`, `qtt_polynom`, and `qtt_chebyshev`. These functions provide efficient QTT representations of the corresponding mathematical functions [Khoromskij](@cite). These are illustrated in the following 

```@example 3
using TensorTrainNumerics
using CairoMakie

d = 8

A1 = qtt_exp(d)
A2 = qtt_sin(d, λ = π)
A3 = qtt_cos(d, λ = π)
A4 = qtt_polynom([0.0, 2.0, 3.0, -8.0, -5.0], d; a = 0.0, b = 1.0)


qtt_values_exponential = qtt_to_function(A1)
qtt_values_sin = qtt_to_function(A2)
qtt_values_cos = qtt_to_function(A3)
qtt_values_polynom = qtt_to_function(A4)


values_exp(x) = exp(x)
values_sin(x) = sin(x * π^2)
values_cos(x) = cos(x * π^2)
values_polynom(x) = 2 * x + 3 * x^2 - 8 * x^3 - 5 * x^4

x_points = LinRange(0, 1, 2^8)
original_values_exponential = values_exp.(x_points)
original_values_sin = values_sin.(x_points)
original_values_cos = values_cos.(x_points)
original_values_polynom = values_polynom.(x_points)

let
    fig = Figure()
    ax1 = Axis(fig[2, 2], title = "Exp Approximation", xlabel = "x", ylabel = "f(x)")
    ax2 = Axis(fig[1, 1], title = "Sin Approximation", xlabel = "x", ylabel = "f(x)")
    ax3 = Axis(fig[1, 2], title = "Cos Approximation", xlabel = "x", ylabel = "f(x)")
    ax4 = Axis(fig[2, 1], title = "Polynomial Approximation", xlabel = "x", ylabel = "f(x)")


    lines!(ax1, x_points, original_values_exponential, label = "Exponential function")
    lines!(ax1, x_points, qtt_values_exponential, label = "QTT exponential function", linestyle = :dash, color = :green)

    lines!(ax2, x_points, original_values_sin, label = "Sine function")
    lines!(ax2, x_points, qtt_values_sin, label = "QTT sine function", linestyle = :dash, color = :red)

    lines!(ax3, x_points, original_values_cos, label = "Sine function")
    lines!(ax3, x_points, qtt_values_cos, label = "QTT sine function", linestyle = :dash, color = :red)

    lines!(ax4, x_points, original_values_polynom, label = "Sine function")
    lines!(ax4, x_points, qtt_values_polynom, label = "QTT sine function", linestyle = :dash, color = :red)

    fig
end
```

### Interpolation-based QTT construction

You can also build a QTT based on interpolation techniques based on [lindsey2023multiscale](@cite). The function `interpolating_qtt` constructs a QTT representation of a given function using polynomial interpolation at Chebyshev nodes. Another option is to use the function `lagrange_rank_revealing`, which constructs a QTT representation using a rank-revealing approach based on Lagrange interpolation as shown in this example

```@example 2
using CairoMakie
using TensorTrainNumerics

f = x -> cos(1 / (x^3 + 0.01)) + sin(π * x)
num_cores = 10  
N = 150 

qtt = interpolating_qtt(f, num_cores, N)
qtt_rank_revealing = lagrange_rank_revealing(f, num_cores, N)

qtt_values = matricize(qtt, num_cores)
qtt_values_rank_revealing = matricize(qtt_rank_revealing, num_cores)

x_points = LinRange(0, 1, 2^num_cores)
original_values = f.(x_points)

fig = Figure()
ax = Axis(fig[1, 1], title="Function Approximation", xlabel="x", ylabel="f(x)")

lines!(ax, x_points, original_values, label="Original Function")
lines!(ax, x_points, qtt_values_rank_revealing, label="QTT, rank rev.", linestyle=:dash, color=:green)
lines!(ax, x_points, qtt_values, label="QTT", linestyle=:dash, color=:red)

axislegend(ax)
fig
```
We can visualize the interpolating QTT as 
```@example 2
visualize(qtt)
```
And similarly for the rank-revealing
```@example 2
visualize(qtt_rank_revealing)
```

## Defining tensor train operators

Suppose we want to construct the finite difference discretization of the second derivative operator with Dirichlet-Neumann boundary conditions on the interval \([0, 1]\). We can use the function `Δ_DN` to create a QTT representation of the corresponding tridiagonal matrix. 

```@example QTT_operators
using TensorTrainNumerics

d = 8
A = Δ_DN(d)
```

And we can check the matrix representation by matricizing it
```@example QTT_operators
mat_A = qtto_to_matrix(A)
```

## Solving equations in the QTT framework

To illustrate how to solve a differential equation consider the following 1D heat/diffusion equation in semi-discrete form with Dirichlet-Dirichlet boundary conditions
```math
\begin{aligned}
u_t(x,t) &= u_{xx}(x,t),\qquad x\in(0,1),\ t\ge 0,\\
u(x,0) &= \sin(\pi x),\\
u(0,t)&=0,\quad u(1,t)=0.
\end{aligned}
```
We define the operator

```@example heat
using TensorTrainNumerics
using CairoMakie
using KrylovKit

d = 8
N = 2^d
h = 1 / (N-1)
xes = collect(range(0.0, 1.0, 2^d))
A = h^2 * toeplitz_to_qtto(-2, 1.0, 1.0, d)
```
We then define some initial condition, random guess and some time steps
```@example heat
u₀ = qtt_sin(d, λ = π) # sin(π^2 x)
steps = collect(range(0.0, 10.0, 1000))
init = rand_tt(u₀.ttv_dims, u₀.ttv_rks)
```
Finally, we can solve the problem using the explicit Euler method, the implicit Euler method, the Crank-Nicolson scheme and a Krylov-based exponential integrator
```@example heat

solution_implicit, rel_implicit = implicit_euler_method(A, u₀, init, steps; return_error = true, normalize = true)

solution_crank, rel_crank = crank_nicholson_method(A, u₀, init, steps; return_error = true, tt_solver = "mals")

solution_krylov, rel_krylov = expintegrator(A, last(steps), u₀)
```
And we can evaluate the solution on the grid at the final time step for visualization

```@example heat

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "Comparison of Time-Stepping Methods")
lines!(ax, xes, qtt_to_function(solution_implicit), label = "Implicit Euler", linestyle = :dot, linewidth = 3)
lines!(ax, xes, qtt_to_function(solution_crank), label = "Crank-Nicolson", linestyle = :dash, linewidth = 3)
axislegend(ax)
fig
```
And similarly for the Krylov-based exponential integrator
```@example heat
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "Comparison of Time-Stepping Methods")
lines!(ax, xes, qtt_to_function(solution_krylov), label = "Krylov", linestyle = :solid, linewidth = 3)
axislegend(ax)
fig
```
For more solvers check out the [solvers](https://github.com/MartinMikkelsen/TensorTrainNumerics.jl/tree/main/src/solvers).