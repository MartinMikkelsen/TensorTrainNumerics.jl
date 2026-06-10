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
function TensorTrainNumerics.to_ttvector(tt::TCI.TensorTrain{V,3}) where {V}
    sites = TCI.sitetensors(tt)
    N = length(sites)
    ttv_vec = [permutedims(c, (2, 1, 3)) for c in sites]
    ttv_dims = ntuple(i -> size(ttv_vec[i], 1), N)
    ttv_rks = vcat([1], [size(c, 3) for c in sites])
    ttv_ot = zeros(Int, N)
    return TTvector{V,N}(N, ttv_vec, ttv_dims, ttv_rks, ttv_ot)
end

function _domain_tuple(x, n_dims::Int, name::Symbol)
    if x isa Number
        return ntuple(_ -> Float64(x), n_dims)
    elseif x isa Tuple || x isa AbstractVector
        length(x) == n_dims ||
            throw(ArgumentError("`$name` must have length $n_dims, got $(length(x))"))
        return ntuple(i -> Float64(x[i]), n_dims)
    else
        throw(ArgumentError("`$name` must be a scalar or length-$n_dims tuple/vector"))
    end
end

function _little_for_big_permutation(n_dims::Int)
    perm = Vector{Int}(undef, 2^n_dims)
    for σ in Iterators.product(ntuple(_ -> 0:1, n_dims)...)
        little = 1 + sum(σ[dim] * 2^(dim - 1) for dim in 1:n_dims)
        big = 1 + sum(σ[dim] * 2^(n_dims - dim) for dim in 1:n_dims)
        perm[big] = little
    end
    return perm
end

function _permute_fused_physical(u::TensorTrainNumerics.TTvector{T,N}, perm) where {T,N}
    ttv_vec = [core[perm, :, :] for core in u.ttv_vec]
    return TTvector{T,N}(u.N, ttv_vec, u.ttv_dims, copy(u.ttv_rks), copy(u.ttv_ot))
end

function _fused_little_to_big(u::TensorTrainNumerics.TTvector, n_dims::Int)
    n_dims == 1 && return TensorTrainNumerics.copy(u)
    return _permute_fused_physical(u, _little_for_big_permutation(n_dims))
end

function _fused_big_to_little(u::TensorTrainNumerics.TTvector, n_dims::Int)
    n_dims == 1 && return TensorTrainNumerics.copy(u)
    return _permute_fused_physical(u, invperm(_little_for_big_permutation(n_dims)))
end

function _fused_to_qttvector(
        u::TensorTrainNumerics.TTvector,
        n_dims::Int,
        bits::Int,
        ordering::Symbol
    )
    ordering ∈ (:interleaved, :serial) ||
        throw(ArgumentError("ordering must be :interleaved or :serial, got $ordering"))
    split = [fill(2, n_dims) for _ in 1:bits]
    u_big = _fused_little_to_big(u, n_dims)
    q_interleaved = TensorTrainNumerics.QTTvector(
        TensorTrainNumerics.to_qtt(u_big, split),
        n_dims,
        bits,
        :interleaved,
    )
    return ordering == :interleaved ? q_interleaved :
        TensorTrainNumerics.reorder(q_interleaved, :serial)
end

function _qttvector_to_fused(q::TensorTrainNumerics.QTTvector)
    q_interleaved = q.ordering == :interleaved ? q : TensorTrainNumerics.reorder(q, :interleaved)
    fused_big = TensorTrainNumerics.to_ttv(
        TensorTrainNumerics.TTvector(q_interleaved),
        fill(q.n_dims, q.bits_per_dim),
    )
    return _fused_big_to_little(fused_big, q.n_dims)
end

function _to_tci_tensortrain(u::TensorTrainNumerics.TTvector{V,N}) where {V,N}
    phys_dim = first(u.ttv_dims)
    all(==(phys_dim), u.ttv_dims) ||
        throw(ArgumentError("InterpolativeQTT conversion expects equal physical dimensions, got $(u.ttv_dims)"))
    ispow2(phys_dim) ||
        throw(ArgumentError("InterpolativeQTT conversion expects power-of-two physical dimensions, got $phys_dim"))
    sites = [permutedims(c, (2, 1, 3)) for c in u.ttv_vec]
    return TCI.TensorTrain{V,3}(sites)
end

_check_interpolation_mode(mode::Symbol) =
    mode ∈ (:singlescale, :adaptive) ||
    throw(ArgumentError("mode must be :singlescale or :adaptive, got $mode"))

"""
    interpolative_qtt(f, bits; degree, tolerance, maxbonddim, a, b, mode, adaptive_tolerance)

Build a one-dimensional binary `TTvector` for `f` on `[a,b]` using Lindsey's
InterpolativeQTT construction. `mode = :singlescale` (default) interpolates with
degree-`degree` Chebyshev pieces at the coarsest scale only; `mode = :adaptive`
refines sub-intervals whose local interpolation error exceeds
`adaptive_tolerance`, which is required for spectral accuracy on localized
features.
"""
function TensorTrainNumerics.interpolative_qtt(
        f::Function,
        bits::Int;
        degree::Int = 6,
        tolerance::Float64 = 1.0e-12,
        maxbonddim::Int = typemax(Int),
        a::Float64 = 0.0,
        b::Float64 = 1.0,
        mode::Symbol = :singlescale,
        adaptive_tolerance::Float64 = 1.0e-8
    )
    bits >= 2 || throw(ArgumentError("bits must be at least 2, got $bits"))
    _check_interpolation_mode(mode)
    # Adaptive trains can transiently need rank ~2^(bits-1); the upstream cap
    # truncates mid-train without orthogonalization, which destroys the result.
    # Build uncapped and enforce the cap afterwards with a gauged compression.
    tt_tci = mode == :adaptive ?
        InterpolativeQTT.interpolateadaptive(
            f, a, b, bits, degree;
            tolerance = tolerance,
            maxbonddim = typemax(Int),
            adaptiveTol = adaptive_tolerance,
        ) :
        InterpolativeQTT.interpolatesinglescale(
            f, a, b, bits, degree;
            tolerance = tolerance,
            maxbonddim = maxbonddim,
        )
    out = TensorTrainNumerics.to_ttvector(tt_tci)
    if mode == :adaptive && maxbonddim < typemax(Int) && maximum(out.ttv_rks) > maxbonddim
        out = TensorTrainNumerics.tt_compress!(TensorTrainNumerics.orthogonalize(out), maxbonddim)
    end
    return out
end

"""
    interpolative_qttv(f, n_dims, bits; ordering, degree, tolerance, maxbonddim, a, b)

Build an `n_dims`-dimensional `QTTvector` with explicit `:interleaved` or `:serial`
ordering from Lindsey's fused multivariate InterpolativeQTT construction.
"""
function TensorTrainNumerics.interpolative_qttv(
        f::Function,
        n_dims::Int,
        bits::Int;
        ordering::Symbol = :interleaved,
        degree::Int = 6,
        tolerance::Float64 = 1.0e-12,
        maxbonddim::Int = typemax(Int),
        a = 0.0,
        b = 1.0,
        mode::Symbol = :singlescale,
        adaptive_tolerance::Float64 = 1.0e-8
    )
    n_dims >= 1 || throw(ArgumentError("n_dims must be positive, got $n_dims"))
    bits >= 2 || throw(ArgumentError("bits must be at least 2, got $bits"))
    ordering ∈ (:interleaved, :serial) ||
        throw(ArgumentError("ordering must be :interleaved or :serial, got $ordering"))
    _check_interpolation_mode(mode)

    if n_dims == 1
        a1 = _domain_tuple(a, 1, :a)[1]
        b1 = _domain_tuple(b, 1, :b)[1]
        tt = TensorTrainNumerics.interpolative_qtt(f, bits;
            degree = degree,
            tolerance = tolerance,
            maxbonddim = maxbonddim,
            a = a1,
            b = b1,
            mode = mode,
            adaptive_tolerance = adaptive_tolerance,
        )
        return TensorTrainNumerics.QTTvector(tt, 1, bits, ordering)
    end

    a_tuple = _domain_tuple(a, n_dims, :a)
    b_tuple = _domain_tuple(b, n_dims, :b)
    tt_tci = mode == :adaptive ?
        InterpolativeQTT.interpolateadaptive(
            f, a_tuple, b_tuple, bits, degree;
            tolerance = tolerance,
            maxbonddim = typemax(Int),
            adaptiveTol = adaptive_tolerance,
        ) :
        InterpolativeQTT.interpolatesinglescale(
            f, a_tuple, b_tuple, bits, degree;
            tolerance = tolerance,
            maxbonddim = maxbonddim,
        )
    q_out = _fused_to_qttvector(TensorTrainNumerics.to_ttvector(tt_tci), n_dims, bits, ordering)
    # The binary split and ordering permutation can inflate bond dimensions past
    # the cap that interpolatesinglescale enforced on the fused representation.
    if maxbonddim < typemax(Int) && maximum(q_out.ttv_rks) > maxbonddim
        # Orthogonalize first: bond-wise SVD truncation is only near-optimal
        # with an orthogonal environment, and the split/reorder leaves a bad gauge.
        compressed = TensorTrainNumerics.tt_compress!(
            TensorTrainNumerics.orthogonalize(TensorTrainNumerics.TTvector(q_out)),
            maxbonddim,
        )
        q_out = TensorTrainNumerics.QTTvector(compressed, n_dims, bits, ordering)
    end
    return q_out
end

"""
    invert_interpolative_qtt(u; degree, q)

Invert a one-dimensional binary `TTvector` to multiresolution Chebyshev-Lobatto
tables using `InterpolativeQTT.invertqtt`.
"""
function TensorTrainNumerics.invert_interpolative_qtt(
        u::TensorTrainNumerics.TTvector;
        degree::Int = 6,
        q::Int = 1
    )
    P = InterpolativeQTT.getChebyshevGrid(degree)
    return InterpolativeQTT.invertqtt(_to_tci_tensortrain(u), P; q = q)
end

function TensorTrainNumerics.invert_interpolative_qtt(
        u::TensorTrainNumerics.QTTvector;
        degree::Int = 6,
        q::Int = 1
    )
    fused = _qttvector_to_fused(u)
    return TensorTrainNumerics.invert_interpolative_qtt(fused; degree = degree, q = q)
end

function _chebyshev_table_evaluator(
        table::AbstractMatrix,
        P,
        a::Float64,
        b::Float64
    )
    b > a || throw(ArgumentError("expected a < b, got a=$a and b=$b"))
    nintervals = size(table, 1)
    nnodes = length(P.grid)
    size(table, 2) == nnodes ||
        throw(ArgumentError("table has $(size(table, 2)) nodes, but grid has $nnodes nodes"))

    return function (x)
        ξ = clamp((Float64(x) - a) / (b - a), 0.0, 1.0)
        scaled = ξ * nintervals
        interval = min(floor(Int, scaled) + 1, nintervals)
        local_x = interval == nintervals && scaled >= nintervals ? 1.0 : scaled - (interval - 1)

        value = zero(eltype(table))
        for α in 0:(nnodes - 1)
            value += table[interval, α + 1] * P(α, local_x)
        end
        return value
    end
end

function _mv_row_index(intervals::AbstractVector{Int}, n_dims::Int, levels::Int)
    row = 0
    for level in 1:levels
        σ = 0
        for dim in 1:n_dims
            bit = (intervals[dim] >> (levels - level)) & 1
            σ += bit * 2^(dim - 1)
        end
        row = row * 2^n_dims + σ
    end
    return row + 1
end

function _point_tuple(xs, n_dims::Int)
    if length(xs) == 1 && xs[1] isa AbstractVector
        length(xs[1]) == n_dims ||
            throw(ArgumentError("expected $n_dims coordinates, got $(length(xs[1]))"))
        return Tuple(xs[1])
    end
    length(xs) == n_dims ||
        throw(ArgumentError("expected $n_dims coordinates, got $(length(xs))"))
    return Tuple(xs)
end

function _chebyshev_table_evaluator_mv(
        table::AbstractMatrix,
        P,
        n_dims::Int,
        levels::Int,
        a::NTuple,
        b::NTuple
    )
    n_intervals = 2^levels
    n_nodes = length(P.grid)
    size(table, 1) == 2^(n_dims * levels) ||
        throw(ArgumentError("table has incompatible interval count $(size(table, 1))"))
    size(table, 2) == n_nodes^n_dims ||
        throw(ArgumentError("table has incompatible Chebyshev-node count $(size(table, 2))"))

    return function (xs...)
        point = _point_tuple(xs, n_dims)
        intervals = Vector{Int}(undef, n_dims)
        locals = Vector{Float64}(undef, n_dims)
        for dim in 1:n_dims
            b[dim] > a[dim] || throw(ArgumentError("expected a < b in dimension $dim"))
            ξ = clamp((Float64(point[dim]) - a[dim]) / (b[dim] - a[dim]), 0.0, 1.0)
            scaled = ξ * n_intervals
            interval = min(floor(Int, scaled), n_intervals - 1)
            intervals[dim] = interval
            locals[dim] = interval == n_intervals - 1 && scaled >= n_intervals ?
                1.0 : scaled - interval
        end

        row = _mv_row_index(intervals, n_dims, levels)
        value = zero(eltype(table))
        for (col, β) in enumerate(Iterators.product(ntuple(_ -> 0:(n_nodes - 1), n_dims)...))
            weight = one(Float64)
            for dim in 1:n_dims
                weight *= P(β[dim], locals[dim])
            end
            value += table[row, col] * weight
        end
        return value
    end
end

function _check_projection_fields(fields::Tuple)
    isempty(fields) && throw(ArgumentError("project_nonlinearity needs at least one TTvector field"))
    all(field -> field isa TensorTrainNumerics.TTvector, fields) ||
        throw(ArgumentError("all projected fields must be TTvector instances"))

    reference = first(fields)
    all(==(2), reference.ttv_dims) ||
        throw(ArgumentError("project_nonlinearity expects binary QTT cores, got dimensions $(reference.ttv_dims)"))
    for field in Base.tail(fields)
        field.N == reference.N ||
            throw(DimensionMismatch("all projected fields must have the same number of bits"))
        field.ttv_dims == reference.ttv_dims ||
            throw(DimensionMismatch("all projected fields must have matching QTT dimensions"))
    end
    return reference
end

"""
    project_nonlinearity(u, Φ; degree, tolerance, maxbonddim, q, a, b)

Form a QTT coefficient approximation of `x -> Φ(u(x))` without dense sampling:
`u` is inverted to Chebyshev-Lobatto tables, locally evaluated through `Φ`, and
then interpolated back with `interpolative_qtt`.
"""
function TensorTrainNumerics.project_nonlinearity(
        fields::Tuple{Vararg{T}},
        Φ::Function;
        degree::Int = 6,
        tolerance::Float64 = 1.0e-10,
        maxbonddim::Int = typemax(Int),
        q::Int = 1,
        a::Float64 = 0.0,
        b::Float64 = 1.0,
        mode::Symbol = :singlescale,
        adaptive_tolerance::Float64 = 1.0e-8
    ) where {T <: TensorTrainNumerics.TTvector}
    reference = _check_projection_fields(fields)
    P = InterpolativeQTT.getChebyshevGrid(degree)
    evaluators = map(fields) do field
        tables = TensorTrainNumerics.invert_interpolative_qtt(field; degree = degree, q = q)
        _chebyshev_table_evaluator(last(tables), P, a, b)
    end
    return TensorTrainNumerics.interpolative_qtt(
        x -> Φ((evaluator(x) for evaluator in evaluators)...),
        reference.N;
        degree = degree,
        tolerance = tolerance,
        maxbonddim = maxbonddim,
        a = a,
        b = b,
        mode = mode,
        adaptive_tolerance = adaptive_tolerance,
    )
end

function TensorTrainNumerics.project_nonlinearity(
        u::TensorTrainNumerics.TTvector,
        Φ::Function;
        kwargs...
    )
    return TensorTrainNumerics.project_nonlinearity((u,), Φ; kwargs...)
end

function TensorTrainNumerics.project_nonlinearity(
        u::TensorTrainNumerics.QTTvector,
        Φ::Function;
        kwargs...
    )
    return TensorTrainNumerics.project_nonlinearity((u,), Φ; kwargs...)
end

function TensorTrainNumerics.project_nonlinearity(
        fields::Tuple{Vararg{Q}},
        Φ::Function;
        degree::Int = 6,
        tolerance::Float64 = 1.0e-10,
        maxbonddim::Int = typemax(Int),
        q::Int = 1,
        a = 0.0,
        b = 1.0,
        mode::Symbol = :singlescale,
        adaptive_tolerance::Float64 = 1.0e-8
    ) where {Q <: TensorTrainNumerics.QTTvector}
    isempty(fields) && throw(ArgumentError("project_nonlinearity needs at least one QTTvector field"))

    reference = first(fields)
    for field in Base.tail(fields)
        field.n_dims == reference.n_dims ||
            throw(DimensionMismatch("all projected QTTvectors must have matching n_dims"))
        field.bits_per_dim == reference.bits_per_dim ||
            throw(DimensionMismatch("all projected QTTvectors must have matching bits_per_dim"))
        field.ordering == reference.ordering ||
            throw(DimensionMismatch("all projected QTTvectors must have matching ordering"))
    end

    if reference.n_dims == 1
        tt_fields = Tuple(TensorTrainNumerics.TTvector(field) for field in fields)
        coeff = TensorTrainNumerics.project_nonlinearity(tt_fields, Φ;
            degree = degree,
            tolerance = tolerance,
            maxbonddim = maxbonddim,
            q = q,
            a = _domain_tuple(a, 1, :a)[1],
            b = _domain_tuple(b, 1, :b)[1],
            mode = mode,
            adaptive_tolerance = adaptive_tolerance,
        )
        return TensorTrainNumerics.QTTvector(coeff, 1, reference.bits_per_dim, reference.ordering)
    end

    P = InterpolativeQTT.getChebyshevGrid(degree)
    levels = reference.bits_per_dim - q
    levels >= 1 || throw(ArgumentError("q must be smaller than bits_per_dim=$(reference.bits_per_dim)"))
    a_tuple = _domain_tuple(a, reference.n_dims, :a)
    b_tuple = _domain_tuple(b, reference.n_dims, :b)

    evaluators = map(fields) do field
        tables = TensorTrainNumerics.invert_interpolative_qtt(field; degree = degree, q = q)
        _chebyshev_table_evaluator_mv(last(tables), P, reference.n_dims, levels, a_tuple, b_tuple)
    end

    return TensorTrainNumerics.interpolative_qttv(
        (xs...) -> Φ((evaluator(xs...) for evaluator in evaluators)...),
        reference.n_dims,
        reference.bits_per_dim;
        ordering = reference.ordering,
        degree = degree,
        tolerance = tolerance,
        maxbonddim = maxbonddim,
        a = a_tuple,
        b = b_tuple,
        mode = mode,
        adaptive_tolerance = adaptive_tolerance,
    )
end

end
