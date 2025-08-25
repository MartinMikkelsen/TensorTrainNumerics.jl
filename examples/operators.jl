using TensorTrainNumerics

function Δ_DN(d::Int)
    out = zeros_tto(2, d, 4)
    id = [1 0; 0 1]
    J = [0 1; 0 0]
    I₂ = [0 0; 0 1]
    for i in 1:2
        for j in 1:2
            out.tto_vec[1][i, j, 1, :] = [id[i, j]; J[j, i]; J[i, j]; I₂[i, j]]
            for k in 2:(d - 1)
                out.tto_vec[k][i, j, :, :] = [id[i, j] J[j, i] J[i, j] 0; 0 J[i, j] 0 0; 0 0 J[j, i] 0; 0 0 0 I₂[i, j]]
            end
            out.tto_vec[d][i, j, :, 1] = [2 * id[i, j] - J[i, j] - J[j, i]; -J[i, j]; -J[j, i]; -I₂[i, j]]
        end
    end
    return out
end

function Δ_ND(d::Int)
    out = zeros_tto(2, d, 4)
    id = [1 0; 0 1]
    J = [0 1; 0 0]
    I₁ = [1 0; 0 0]
    for i in 1:2
        for j in 1:2
            out.tto_vec[1][i, j, 1, :] = [id[i, j]; J[j, i]; J[i, j]; I₁[i, j]]
            for k in 2:(d - 1)
                out.tto_vec[k][i, j, :, :] = [id[i, j] J[j, i] J[i, j] 0; 0 J[i, j] 0 0; 0 0 J[j, i] 0; 0 0 0 I₁[i, j]]
            end
            out.tto_vec[d][i, j, :, 1] = [2 * id[i, j] - J[i, j] - J[j, i]; -J[i, j]; -J[j, i]; -I₁[i, j]]
        end
    end
    return out
end

function Δ_NN(d)
    out = zeros_tto(ntuple(_ -> 2, d), [4; fill(5, d - 1); 4])
    id = [1 0; 0 1]
    J = [0 1; 0 0]
    I₁ = [1 0; 0 0]
    I₂ = [0 0; 0 1]
    for i in 1:2
        for j in 1:2
            out.tto_vec[1][i, j, 1, :] = [id[i, j]; J[j, i]; J[i, j]; I₂[i, j]; I₁[i, j]]
            for k in 2:(d - 1)
                out.tto_vec[k][i, j, :, :] = [id[i, j] J[j, i] J[i, j] 0 0; 0 J[i, j] 0 0 0; 0 0 J[j, i] 0 0; 0 0 0 I₂[i, j] 0; 0 0 0 0 -I₁[i, j]]
            end
            out.tto_vec[d][i, j, :, 1] = [2 * id[i, j] - J[i, j] - J[j, i]; -J[i, j]; -J[j, i]; -I₂[i, j]; -I₁[i, j]]
        end
    end
    return out
end

function Δ_P(d)
    out = zeros_tto(ntuple(_ -> 2, d), fill(5, d + 1))
    id = [1 0; 0 1]
    J = [0 1; 0 0]
    for i in 1:2
        for j in 1:2
            out.tto_vec[1][i, j, 1, :] = [id[i, j], J[j, i], J[i, j], J[i, j], J[j, i]]
            for k in 2:(d - 1)
                out.tto_vec[k][i, j, :, :] = [
                    id[i, j] J[j, i] J[i, j] 0 0;
                    0 J[i, j] 0 0 0;
                    0 0 J[j, i] 0 0;
                    0 0 0 J[i, j] 0;
                    0 0 0 0 J[j, i]
                ]
            end
            out.tto_vec[d][i, j, :, 1] = [
                2 * id[i, j] - J[i, j] - J[j, i];
                -J[i, j];
                -J[j, i];
                -J[i, j];
                -J[j, i]
            ]
        end
    end
    return out
end
