module TensorTrainNumericsInterpolativeQTTExt

using TensorTrainNumerics
using InterpolativeQTT
import TensorCrossInterpolation as TCI

"""
    to_ttvector(tt::TCI.TensorTrain{V,3}) -> TTvector{V,N}

Convert a TensorCrossInterpolation `TensorTrain` to a `TTvector`.

TCI cores have layout `(left_rank, phys_dim, right_rank)` while TTN cores use
`(phys_dim, left_rank, right_rank)`, so each core is permuted accordingly.
"""
function TensorTrainNumerics.to_ttvector(tt::TCI.TensorTrain{V, 3}) where {V}
    sites = TCI.sitetensors(tt)
    N = length(sites)
    ttv_vec = [permutedims(c, (2, 1, 3)) for c in sites]
    ttv_dims = ntuple(i -> size(ttv_vec[i], 1), N)
    ttv_rks = vcat([1], [size(c, 3) for c in sites])
    ttv_ot = zeros(Int, N)
    return TTvector{V, N}(N, ttv_vec, ttv_dims, ttv_rks, ttv_ot)
end

end
