using TensorTrainNumerics
using CairoMakie
using SpecialFunctions
using Struve

function inte(x,d,N,ν=1.0)
    integrant(t) = (1-t^2)^(ν-0.5)*sin(x*t)
    return (2*(x/2)^ν)/(sqrt(π)*gamma(ν+1.5)).*integrating_qtt(integrant, d, N, method="Rank-Revealing")
end

d = 14
N = 75

xes = LinRange(-15.0, 15.0, 2^d)

A = [inte(x,d,N) for x in xes];

let
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Function Approximation", xlabel = "x", ylabel = "f(x)")
    lines!(ax, xes, 1.5*A, label = "QTT Function", color = :red)
    lines!(ax, xes, struveh.(1,xes), label = "Exact.", color = :green,linestyle = :dash)
    axislegend(ax)
    fig
end

1.5*A-(struveh.(1,xes))