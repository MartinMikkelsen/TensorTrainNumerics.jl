using CairoMakie
using TensorTrainNumerics
using SpecialFunctions

h(x;ν=2.0) = 2*(x/2)^ν/(sqrt(π)*gamma(ν+0.5))

function integrant(x; ν=2.0, d=12)

  fqtt = function_to_qtt(t -> (1 - t^2)^(ν - 0.5) * sin(x*t), d; a=0, b=1, tol=1e-25)

  h = 1/(2^d-1)
  wqtt = function_to_qtt(t -> (isapprox(t,0) || isapprox(t,1) ? h/2 : h), d; a=0, b=1, tol=1e-25)

  return TensorTrainNumerics.dot(wqtt, fqtt)
end

Q_test = [h(xs).*integrant(xs) for xs in collect(range(-5, 5, 2^12))]

using Struve

let
    fig = Figure()
    xes = collect(range(-5, 5, 2^12))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "Comparison of Time-Stepping Methods")
    lines!(ax, xes, Q_test, label = "Explicit Euler", linestyle = :solid, linewidth=3)
    lines!(ax, xes, struveh.(2,xes), label = "Explicit Euler", linestyle = :dash, linewidth=3)
    display(fig)
end

Q_test - struveh.(2, collect(range(-5, 5, 2^12)))