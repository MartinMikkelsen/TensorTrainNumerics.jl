using LinearAlgebra
using TensorTrainNumerics

function qtt_from_vector(v::AbstractVector)
    d = Int(log2(length(v)))
    @assert 2^d == length(v) "vector length must be a power of two"

    tensor = zeros(eltype(v), ntuple(_ -> 2, d))
    @inbounds for I in CartesianIndices(tensor)
        tensor[I] = v[tuple_to_index(Tuple(I))]
    end
    return ttv_decomp(tensor)
end

function dense_cconv(kernel::AbstractVector, x::AbstractVector)
    n = length(kernel)
    return [sum(kernel[mod(i - j, n) + 1] * x[j + 1] for j in 0:(n - 1)) for i in 0:(n - 1)]
end

function dense_conv(kernel::AbstractVector, x::AbstractVector)
    n = length(kernel)
    y = zeros(promote_type(eltype(kernel), eltype(x)), 2n)
    @inbounds for i in 0:(n - 1), j in 0:(n - 1)
        y[i + j + 1] += kernel[i + 1] * x[j + 1]
    end
    return y
end

d = 6
N = 2^d
grid = (0:(N - 1)) ./ N

kernel = exp.(-40 .* min.(grid, 1 .- grid).^2)
kernel ./= sum(kernel)
signal = sin.(2π .* grid) .+ 0.25 .* cos.(6π .* grid)

kernel_qtt = QTTvector(qtt_from_vector(kernel), 1, d, :serial)
signal_qtt = QTTvector(qtt_from_vector(signal), 1, d, :serial)

circulant = circulant_qtto(kernel_qtt)
via_operator = circulant * signal_qtt
via_cconv = qtt_cconv(kernel_qtt, signal_qtt)
via_conv = qtt_conv(kernel_qtt, signal_qtt)

circular_reference = dense_cconv(kernel, signal)
linear_reference = dense_conv(kernel, signal)

operator_error = norm(qtt_to_vector(via_operator) - circular_reference) / norm(circular_reference)
circular_error = norm(qtt_to_vector(via_cconv) - circular_reference) / norm(circular_reference)
linear_error = norm(qtt_to_vector(via_conv) - linear_reference) / norm(linear_reference)

@info "QTT convolution" points=N operator_error circular_error linear_error circulant_rank=maximum(circulant.tto_rks) circular_result_rank=maximum(via_cconv.ttv_rks) linear_result_rank=maximum(via_conv.ttv_rks)
