using TensorTrainNumerics
using CairoMakie

d = 8

A = qtt_chebyshev(0, 8)
B = qtt_chebyshev(1, 8)
C = qtt_chebyshev(2, 8)
D = qtt_chebyshev(3, 8)
E = qtt_chebyshev(4, 8)

let
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "T(x)", title = "Chebyshev Polynomials [0,1]")
    xes = LinRange(0, 1, 2^d)
    lines!(ax, xes, matricize(A, d), label = "T₀(x)", linestyle = :solid, linewidth = 3)
    lines!(ax, xes, matricize(B, d), label = "T₁(x)", linestyle = :solid, linewidth = 3)
    lines!(ax, xes, matricize(C, d), label = "T₂(x)", linestyle = :solid, linewidth = 3)
    lines!(ax, xes, matricize(D, d), label = "T₃(x)", linestyle = :solid, linewidth = 3)
    lines!(ax, xes, matricize(E, d), label = "T₄(x)", linestyle = :solid, linewidth = 3)
    axislegend(ax)
    display(fig)
end
