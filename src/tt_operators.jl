function toeplitz_to_qtto(α,β,γ,d)
  out = zeros_tto(2,d,3)
  id = Matrix{Float64}(I,2,2)
  J = zeros(2,2)
  J[1,2] = 1
  for i in 1:2 
    for j in 1:2
      out.tto_vec[1][i,j,1,:] = [id[i,j];J[j,i];J[i,j]]
      for k in 2:d-1
        out.tto_vec[k][i,j,:,:] = [id[i,j] J[j,i] J[i,j]; 0 J[i,j] 0 ; 0 0 J[j,i]]
      end
      out.tto_vec[d][i,j,:,1] = [α*id[i,j] + β*J[i,j] + γ*J[j,i]; γ*J[i,j] ; β*J[j,i]]
    end
  end
  return out
end

function shift(d::Int)
    return toeplitz_to_qtto(0,1,0,d)
end

function ∇(d::Int)
    return toeplitz_to_qtto(1,-1,0,d)
end

function Δ(d::Int)
    return toeplitz_to_qtto(2,-1,-1,d)
end

function qtto_prolongation(d)
    out = zeros_tto(Float64, ntuple(_->2, d), fill(2, d+1))
    for j in 1:d
        out.tto_vec[j][1,1,1,1] = 1.0
        out.tto_vec[j][1,1,2,2] = 1.0
        out.tto_vec[j][1,2,2,1] = 1.0
        out.tto_vec[j][2,2,1,2] = 1.0
    end
    return out
end