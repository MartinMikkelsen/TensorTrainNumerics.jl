using LinearMaps
using TensorOperations

"""
Implementation based on the presentation in 
Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider. "The alternating linear scheme for tensor optimization in the tensor train format." SIAM Journal on Scientific Computing 34.2 (2012): A683-A713.
"""

function init_H(x_tt::TTvector{T}, A_tto::TToperator{T}) where {T <: Number}
    d = x_tt.N
    H = Array{Array{T}}(undef, d)
    H[d] = ones(T, 1, 1, 1)
    for i in d:-1:2
        H[i - 1] = zeros(T, A_tto.tto_rks[i], x_tt.ttv_rks[i], x_tt.ttv_rks[i])
        x_vec = x_tt.ttv_vec[i]
        A_vec = A_tto.tto_vec[i]
        update_H!(x_vec, A_vec, H[i], H[i - 1])
    end
    return H
end

function update_H!(x_vec::Array{T, 3}, A_vec::Array{T, 4}, Hi::Array{T, 3}, Him::Array{T, 3}) where {T <: Number}
    @tensoropt((ϕ, χ), Him[a, α, β] = conj.(x_vec)[j, α, ϕ] * Hi[z, ϕ, χ] * x_vec[k, β, χ] * A_vec[j, k, a, z]) #size (rim, rim, rAim)
    return nothing
end

function init_Hb(x_tt::TTvector{T}, b_tt::TTvector{T}) where {T <: Number}
    d = x_tt.N
    H_b = Array{Array{T}}(undef, d)
    H_b[d] = ones(T, 1, 1)
    for i in d:-1:2
        H_b[i - 1] = zeros(T, x_tt.ttv_rks[i], b_tt.ttv_rks[i])
        b_vec = b_tt.ttv_vec[i]
        x_vec = x_tt.ttv_vec[i]
        update_Hb!(x_vec, b_vec, H_b[i], H_b[i - 1]) #size(rbim, rim)
    end
    return H_b
end

function update_Hb!(x_vec::Array{T, 3}, b_vec::Array{T, 3}, H_bi::Array{T, 2}, H_bim::Array{T, 2}) where {T <: Number}
    @tensoropt((ϕ, χ), H_bim[α, β] = H_bi[ϕ, χ] * b_vec[i, β, χ] * conj.(x_vec)[i, α, ϕ])
    return nothing
end

function update_G!(x_vec::Array{T, 3}, A_vec::Array{T, 4}, Gi::AbstractArray{T, 5}, Gip::AbstractArray{T, 5}) where {T <: Number}
    @tensor Gip[j, α, k, β, J] = (conj.(x_vec)[l, ϕ, α] * (Gi[l, ϕ, m, χ, L] * x_vec[m, χ, β])) * A_vec[j, k, L, J]
    return nothing
end

function update_Gb!(x_vec::Array{T, 3}, b_vec::Array{T, 3}, G_bi::AbstractArray{T, 3}, G_bip::AbstractArray{T, 3}) where {T <: Number}
    @tensoropt((ϕ, χ), G_bip[i, α, β] = b_vec[i, ϕ, β] * G_bi[j, χ, ϕ] * conj.(x_vec)[j, χ, α])
    return nothing
end

#full assemble of matrix K
function K_full(Gi::Array{T, 5}, Hi::Array{T, 3}, K_dims::NTuple{3, Int}) where {T <: Number}
    K = zeros(T, prod(K_dims), prod(K_dims))
    Krshp = reshape(K, (K_dims..., K_dims...))
    @tensor Krshp[a, b, c, d, e, f] = Gi[a, b, d, e, z] * Hi[z, c, f] #size (ni,rim,ri,ni,rim,ri)
    return K
end

function Ksolve(Gi::Array{T, 5}, G_bi::Array{T, 3}, Hi::Array{T, 3}, H_bi::Array{T, 2}) where {T <: Number}
    K_dims = (size(Gi, 1), size(Gi, 2), size(Hi, 2))
    K = K_full(Gi, Hi, K_dims)
    @tensor Pb[i, α1, α2] := G_bi[i, α1, β] * H_bi[α2, β] #size (ni,rim,ri)
    return reshape(K \ Pb[:], K_dims)
end

function K_eigmin(Gi::Array{T, 5}, Hi::Array{T, 3}, ttv_vec::Array{T, 3}; it_solver = false, itslv_thresh = 256::Int64, maxiter = 200::Int64, tol = 1.0e-6::Float64) where {T <: Number}
    K_dims = (size(Gi, 1), size(Gi, 2), size(Hi, 2))
    if it_solver && prod(K_dims) > itslv_thresh
        H = zeros(T, prod(K_dims))
        function K_matfree(V::AbstractArray{T, 1}; Gi = Gi::Array{T, 5}, Hi = Hi::Array{T, 3}, K_dims = K_dims, H = H::AbstractArray{T, 1})
            Hrshp = reshape(H, K_dims)
            @tensoropt((b, c, e, f), Hrshp[a, b, c] = Gi[a, b, d, e, z] * reshape(V, K_dims)[d, e, f] * Hi[z, c, f])
            return H::AbstractArray{T, 1}
        end
        r = lobpcg(LinearMap(K_matfree, prod(K_dims); ishermitian = true), false, ttv_vec[:], 1; maxiter = maxiter, tol = tol)
        return r.λ[1]::Real, reshape(r.X[:, 1], K_dims)::Array{T, 3}
    else
        K = K_full(Gi, Hi, K_dims)
        F = eigen(Hermitian(K), 1:1)
        return real(F.values[1])::Real, reshape(F.vectors[:, 1], K_dims)::Array{T, 3}
    end
end

function K_eiggenmin(Gi, Hi, Ki, Li, ttv_vec; it_solver = false, itslv_thresh = 2500)
    @tensor begin
        K[a, b, c, d, e, f] := Gi[d, e, a, b, z] * Hi[z, f, c] #size (ni,rim,ri,ni,rim,ri)
        S[a, b, c, d, e, f] := Ki[d, e, a, b, z] * Li[z, f, c] #size (ni,rim,ri,ni,rim,ri)
    end
    if it_solver || prod(size(K)[1:3]) > itslv_thresh
        r = lobpcg(reshape(K, prod(size(K)[1:3]), :), reshape(S, prod(size(S)[1:3]), :), false, ttv_vec[:], 1; maxiter = 500, tol = 1.0e-8)
        return r.λ[1], reshape(r.X[:, 1], size(K)[1:3])
    else
        F = eigen(reshape(K, prod(size(K)[1:3]), :), reshape(S, prod(size(K)[1:3]), :))
        return real(F.values[1]), reshape(F.vectors[:, 1], size(K)[1:3])
    end
end

function left_core_move(x_tt::TTvector{T}, V::Array{T, 3}, i::Int, x_rks) where {T <: Number}
    rim, ri = x_rks[i], x_rks[i + 1]
    ni = x_tt.ttv_dims[i]

    # Prepare core movements
    QV, RV = qr(reshape(permutedims(V, [1 3 2]), ni * ri, :)) #QV: ni*ri x ni*ri; RV ni*ri x rim

    # Apply core movement 3.1
    x_tt.ttv_vec[i] = permutedims(reshape(Matrix(QV)[:, 1:rim], ni, ri, :), [1 3 2])
    x_tt.ttv_ot[i] = 1

    # Apply core movement 3.2
    @tensoropt((b, c, z), Xim[a, b, c] := x_tt.ttv_vec[i - 1][a, b, z] * RV[1:rim, :][c, z]) #size (nim,rim2,rim_new)
    x_tt.ttv_vec[i - 1] = Xim
    x_tt.ttv_ot[i - 1] = 0
    return x_tt
end

function right_core_move(x_tt::TTvector{T}, V::Array{T, 3}, i::Int, x_rks) where {T <: Number}
    rim, ri = x_rks[i], x_rks[i + 1]
    ni = x_tt.ttv_dims[i]
    QV, RV = qr(reshape(V, ni * rim, :)) #QV: ni*rim x ni*rim; RV ni*rim x ri

    # Apply core movement 3.1
    x_tt.ttv_vec[i] = reshape(Matrix(QV)[:, 1:ri], ni, rim, :)
    x_tt.ttv_ot[i] = -1

    # Apply core movement 3.2
    @tensoropt((b, c, z), Xip[a, b, c] := RV[1:ri, :][b, z] * x_tt.ttv_vec[i + 1][a, z, c]) #size (nip,ri,rip)
    x_tt.ttv_vec[i + 1] = Xip
    x_tt.ttv_ot[i + 1] = 0
    return x_tt
end


"""
Solve Ax=b using the ALS algorithm where A is given as `TToperator` and `b`, `tt_start` are `TTvector`.
The ranks of the solution is the same as `tt_start`.
`sweep_count` is the number of total sweeps in the ALS.
"""
function als_linsolve(A::TToperator{T}, b::TTvector{T}, tt_start::TTvector{T}; sweep_count = 2, it_solver = false, r_itsolver = 5000) where {T <: Number}
    # als finds the minimum of the operator J:1/2*<Ax,Ax> - <x,b>
    # input:
    # 	A: the tensor operator in its tensor train format
    #   b: the tensor in its tensor train format
    #	tt_start: start value in its tensor train format
    #	opt_rks: rank vector considered to be optimal enough
    # output:
    #	tt_opt: stationary point of J up to tolerated rank opt_rks
    # 			in its tensor train format

    d = A.N
    # Initialize the to be returned tensor in its tensor train format
    tt_opt = orthogonalize(tt_start)
    dims = tt_start.ttv_dims
    # Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
    rks = copy(tt_start.ttv_rks)

    # Initialize the arrays of G and G_b
    G = Array{Array{T}}(undef, d)
    G_b = Array{Array{T}}(undef, d)

    # Initialize G[1], G_b[1], H[d] and H_b[d]
    for i in 1:d
        G[i] = zeros(T, dims[i], rks[i], dims[i], rks[i], A.tto_rks[i + 1])
        G_b[i] = zeros(dims[i], rks[i], b.ttv_rks[i + 1])
    end
    G[1] = reshape(A.tto_vec[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)
    G_b[1] = reshape(b.ttv_vec[1], dims[1], 1, :)

    #Initialize H and H_b
    H = init_H(tt_opt, A)
    H_b = init_Hb(tt_opt, b)

    nsweeps = 0 #sweeps counter

    while nsweeps < sweep_count
        nsweeps += 1
        # First half sweep
        for i in 1:(d - 1)
            # Define V as solution of K*x=Pb in x
            V = Ksolve(G[i], G_b[i], H[i], H_b[i])
            tt_opt = right_core_move(tt_opt, V, i, rks)
            #update G,G_b
            update_G!(tt_opt.ttv_vec[i], A.tto_vec[i + 1], G[i], G[i + 1])
            update_Gb!(tt_opt.ttv_vec[i], b.ttv_vec[i + 1], G_b[i], G_b[i + 1])
        end

        if nsweeps == sweep_count
            return tt_opt
        else
            nsweeps += 1
            # Second half sweep
            for i in d:(-1):2
                # Define V as solution of K*x=Pb in x
                V = Ksolve(G[i], G_b[i], H[i], H_b[i])
                tt_opt = left_core_move(tt_opt, V, i, rks)
                update_H!(tt_opt.ttv_vec[i], A.tto_vec[i], H[i], H[i - 1])
                update_Hb!(tt_opt.ttv_vec[i], b.ttv_vec[i], H_b[i], H_b[i - 1])
            end
        end
    end
    return tt_opt
end

"""
Returns the lowest eigenvalue of A by minimizing the Rayleigh quotient in the ALS algorithm.

The ranks can be increased in the course of the ALS: if `sweep_schedule[k] ≤ i <sweep_schedule[k+1]` is the current number of sweeps then the ranks is given by `rmax_schedule[k]`.
"""
function als_eigsolve(
        A::TToperator{T},
        tt_start::TTvector{T}; #TT initial guess
        sweep_schedule = [2]::Array{Int64, 1}, #Number of sweeps for each bond dimension in rmax_schedule
        rmax_schedule = [maximum(tt_start.ttv_rks)]::Array{Int64, 1}, #bond dimension at each sweep
        noise_schedule = zeros(length(rmax_schedule))::Array{Float64, 1}, #noise at each bond dimension increase
        it_solver = false::Bool, #linear solver for the microstep
        itslv_thresh = 1024::Int64, #switch from full to iterative
        maxiter = 200::Int64, #maximum of iterations for the iterative solver
        linsolv_tol = 1.0e-8::Float64
    ) where {T <: Number} #tolerance of the iterative linear solver
    @assert(length(rmax_schedule) == length(sweep_schedule) == length(noise_schedule), "Sweep schedule error")
    d = A.N
    # Initialize the to be returned tensor in its tensor train format
    tt_opt = orthogonalize(tt_start)
    dims = tt_start.ttv_dims
    E = zeros(Float64, 2d * (sweep_schedule[end] + 1)) #output eigenvalue
    # Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
    rks = copy(tt_start.ttv_rks)

    # Initialize the array G
    G = Array{Array{T}}(undef, d)
    for i in 1:d
        G[i] = zeros(T, dims[i], rks[i], dims[i], rks[i], A.tto_rks[i + 1])
    end
    G[1] = reshape(A.tto_vec[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)

    #Initialize H and H_b
    H = init_H(tt_opt, A)

    nsweeps = 0 #sweeps counter
    i_schedule, i_μit = 1, 0
    while i_schedule <= length(sweep_schedule)
        nsweeps += 1
        if nsweeps == sweep_schedule[i_schedule]
            i_schedule += 1
            if i_schedule > length(sweep_schedule)
                return E[1:i_μit]::Array{Float64, 1}, tt_opt::TTvector{T}
            else
                tt_opt = tt_up_rks(tt_opt, rmax_schedule[i_schedule]; ϵ_wn = noise_schedule[i_schedule])
                tt_opt = orthogonalize(tt_opt)
                H = init_H(tt_opt, A)
                for i in 1:(d - 1)
                    Gtemp = zeros(dims[i + 1], tt_opt.ttv_rks[i + 1], dims[i + 1], tt_opt.ttv_rks[i + 1], A.tto_rks[i + 2])
                    Gtemp[1:size(G[i + 1], 1), 1:size(G[i + 1], 2), 1:size(G[i + 1], 3), 1:size(G[i + 1], 4), 1:size(G[i + 1], 5)] = G[i + 1]
                    G[i + 1] = Gtemp
                end
            end
        end
        # First half sweep
        for i in 1:(d - 1)
            # Define V as solution of K*x=Pb in x
            i_μit += 1
            E[i_μit], V = K_eigmin(G[i], H[i], tt_opt.ttv_vec[i]; it_solver = it_solver, itslv_thresh = itslv_thresh, maxiter = maxiter, tol = linsolv_tol)
            tt_opt = right_core_move(tt_opt, V, i, tt_opt.ttv_rks)
            #update G
            update_G!(tt_opt.ttv_vec[i], A.tto_vec[i + 1], G[i], G[i + 1])
        end

        # Second half sweep
        for i in d:(-1):2
            # Define V as solution of K*x=Pb in x
            i_μit += 1
            E[i_μit], V = K_eigmin(G[i], H[i], tt_opt.ttv_vec[i]; it_solver = it_solver, itslv_thresh = itslv_thresh, maxiter = maxiter, tol = linsolv_tol)
            tt_opt = left_core_move(tt_opt, V, i, tt_opt.ttv_rks)
            update_H!(tt_opt.ttv_vec[i], A.tto_vec[i], H[i], H[i - 1])
        end
    end
    return E[1:i_μit]::Array{Float64, 1}, tt_opt::TTvector{T}
end

"""
returns the smallest eigenpair Ax = Sx
"""
function als_gen_eigsolv(A::TToperator{T}, S::TToperator{T}, tt_start::TTvector{T}; sweep_schedule = [2], rmax_schedule = [maximum(tt_start.ttv_rks)], tol = 1.0e-10, it_solver = false, itslv_thresh = 2500) where {T <: Number}
    d = A.N
    # Initialize the to be returned tensor in its tensor train format
    tt_opt = orthogonalize(tt_start)
    dims = tt_start.ttv_dims
    E = zeros(Float64, d * sweep_schedule[end]) #output eigenvalue
    # Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
    rks = tt_start.ttv_rks

    # Initialize the arrays of G and K
    G = Array{Array{T}}(undef, d)
    K = Array{Array{T}}(undef, d)

    # Initialize G[1]
    for i in 1:d
        G[i] = zeros(dims[i], rks[i], dims[i], rks[i], A.tto_rks[i + 1])
        K[i] = zeros(dims[i], rks[i], dims[i], rks[i], S.tto_rks[i + 1])
    end
    G[1] = reshape(A.tto_vec[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)
    K[1] = reshape(S.tto_vec[1][:, :, 1, :], dims[1], 1, dims[1], 1, :)

    #Initialize H and H_b
    H = init_H(tt_opt, A)
    L = init_H(tt_opt, S)

    nsweeps = 0 #sweeps counter
    i_schedule, i_μit = 1, 0
    while i_schedule <= length(sweep_schedule)
        nsweeps += 1
        if nsweeps == sweep_schedule[i_schedule]
            i_schedule += 1
            if i_schedule > length(sweep_schedule)
                return E[1:i_μit], tt_opt
            else
                tt_opt = tt_up_rks(tt_opt, rmax_schedule[i_schedule])
                for i in 1:(d - 1)
                    Htemp = zeros(tt_opt.ttv_rks[i], tt_opt.ttv_rks[i], A.tto_rks[i])
                    Ltemp = zeros(tt_opt.ttv_rks[i], tt_opt.ttv_rks[i], S.tto_rks[i])
                    Htemp[1:size(H[i], 1), 1:size(H[i], 2), 1:size(H[i], 3)] = H[i]
                    Ltemp[1:size(L[i], 1), 1:size(L[i], 2), 1:size(L[i], 3)] = L[i]
                    H[i] = Htemp
                    L[i] = Ltemp
                end
            end
        end

        # First half sweep
        for i in 1:(d - 1)

            # If i is the index of the core matrices do the optimization
            if tt_opt.ttv_ot[i] == 0
                # Define V as solution of K*x=Pb in x
                i_μit += 1
                E[i_μit], V = K_eiggenmin(G[i], H[i], K[i], L[i], tt_opt.ttv_vec[i]; it_solver = it_solver, itslv_thresh = itslv_thresh)
                tt_opt = right_core_move(tt_opt, V, i, rks)
            end

            #update G and K
            update_G!(tt_opt.ttv_vec[i], A.tto_vec[i + 1], G[i], G[i + 1])
            update_G!(tt_opt.ttv_vec[i], S.tto_vec[i + 1], K[i], K[i + 1])
        end

        # Second half sweep
        for i in d:(-1):2
            # Define V as solution of K*x=Pb in x
            i_μit += 1
            E[i_μit], V = K_eiggenmin(G[i], H[i], K[i], L[i], tt_opt.ttv_vec[i]; it_solver = it_solver, itslv_thresh = itslv_thresh)
            tt_opt = left_core_move(tt_opt, V, i, rks)
            update_H!(tt_opt.ttv_vec[i], A.tto_vec[i], H[i], H[i - 1])
            update_H!(tt_opt.ttv_vec[i], S.tto_vec[i], L[i], L[i - 1])
        end
    end
    return
end
