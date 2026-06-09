using LinearAlgebra
using TensorTrainNumerics

function dense_linear_prolongation(values)
    n = length(values)
    out = zeros(eltype(values), 2n)
    @inbounds for alpha in 0:(n - 1)
        out[2alpha + 1] = values[alpha + 1]
        out[2alpha + 2] += 0.5 * values[alpha + 1]
        if alpha + 1 < n
            out[2alpha + 2] += 0.5 * values[alpha + 2]
        end
    end
    return out
end

d = 6
u = function_to_qtt(x -> sin(2pi * x) + 0.25 * cos(6pi * x), d)
P = qtto_linear_prolongation(d)
u_fine = P * u

coarse_values = qtt_to_function(u)
fine_values = qtt_to_function(u_fine)
reference = dense_linear_prolongation(coarse_values)
relative_error = norm(fine_values - reference) / max(norm(reference), eps(Float64))

@info "QTT linear prolongation" coarse_sites=d fine_sites=(d + 1) coarse_points=length(coarse_values) fine_points=length(fine_values) relative_error=relative_error max_rank=maximum(u_fine.ttv_rks)
