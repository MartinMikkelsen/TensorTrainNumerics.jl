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

function _to_tci_tensortrain(u::TensorTrainNumerics.TTvector{V,N}) where {V,N}
    all(==(2), u.ttv_dims) ||
        throw(ArgumentError("invert_interpolative_qtt expects binary QTT cores, got dimensions $(u.ttv_dims)"))
    sites = [permutedims(c, (2, 1, 3)) for c in u.ttv_vec]
    return TCI.TensorTrain{V,3}(sites)
end

"""
    interpolative_qtt(f, bits; degree, tolerance, maxbonddim, a, b)

Build a one-dimensional binary `TTvector` for `f` on `[a,b]` using Lindsey's
single-scale InterpolativeQTT construction.
"""
function TensorTrainNumerics.interpolative_qtt(
        f::Function,
        bits::Int;
        degree::Int = 6,
        tolerance::Float64 = 1.0e-12,
        maxbonddim::Int = typemax(Int),
        a::Float64 = 0.0,
        b::Float64 = 1.0
    )
    bits >= 2 || throw(ArgumentError("bits must be at least 2, got $bits"))
    tt_tci = InterpolativeQTT.interpolatesinglescale(
        f, a, b, bits, degree;
        tolerance = tolerance,
        maxbonddim = maxbonddim,
    )
    return TensorTrainNumerics.to_ttvector(tt_tci)
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
    u.n_dims == 1 ||
        throw(ArgumentError("invert_interpolative_qtt currently supports 1D QTTvectors, got n_dims=$(u.n_dims)"))
    return TensorTrainNumerics.invert_interpolative_qtt(TensorTrainNumerics.TTvector(u);
        degree = degree,
        q = q,
    )
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
        b::Float64 = 1.0
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
    )
end

function TensorTrainNumerics.project_nonlinearity(
        u::TensorTrainNumerics.TTvector,
        Φ::Function;
        kwargs...
    )
    return TensorTrainNumerics.project_nonlinearity((u,), Φ; kwargs...)
end

function _project_nonlinearity_2d_dense(
        u::TensorTrainNumerics.TTvector,
        Φ::Function;
        maxbonddim::Int,
        tol::Float64
    )::TensorTrainNumerics.TTvector
    vals = real.(TensorTrainNumerics.qtt_to_function(u))
    phi_vals = map(Φ, vals)
    N_total = length(phi_vals)
    d_total = round(Int, log2(N_total))
    tensor = zeros(eltype(phi_vals), ntuple(_ -> 2, d_total))
    for pos in 1:N_total
        bits = reverse(digits(pos - 1; base = 2, pad = d_total)) .+ 1
        tensor[CartesianIndex(Tuple(bits))] = phi_vals[pos]
    end
    result = TensorTrainNumerics.ttv_decomp(tensor; tol = tol)
    return maxbonddim < typemax(Int) ? TensorTrainNumerics.tt_compress!(result, maxbonddim) : result
end

function TensorTrainNumerics.project_nonlinearity(
        u::TensorTrainNumerics.QTTvector,
        Φ::Function;
        maxbonddim::Int = typemax(Int),
        tolerance::Float64 = 1.0e-10,
        kwargs...
    )
    if u.n_dims == 1
        coeff = TensorTrainNumerics.project_nonlinearity(TensorTrainNumerics.TTvector(u), Φ;
            maxbonddim = maxbonddim, tolerance = tolerance, kwargs...)
        return TensorTrainNumerics.QTTvector(coeff, u.n_dims, u.bits_per_dim, u.ordering)
    elseif u.n_dims == 2
        coeff = _project_nonlinearity_2d_dense(TensorTrainNumerics.TTvector(u), Φ;
            maxbonddim = maxbonddim, tol = tolerance)
        return TensorTrainNumerics.QTTvector(coeff, u.n_dims, u.bits_per_dim, u.ordering)
    else
        throw(ArgumentError("project_nonlinearity supports QTTvectors with n_dims ∈ {1,2}, got n_dims=$(u.n_dims)"))
    end
end

function TensorTrainNumerics.project_nonlinearity(
        fields::Tuple{Vararg{Q}},
        Φ::Function;
        kwargs...
    ) where {Q <: TensorTrainNumerics.QTTvector}
    isempty(fields) && throw(ArgumentError("project_nonlinearity needs at least one QTTvector field"))

    reference = first(fields)
    reference.n_dims == 1 ||
        throw(ArgumentError("project_nonlinearity currently supports 1D QTTvectors, got n_dims=$(reference.n_dims)"))
    for field in Base.tail(fields)
        field.n_dims == reference.n_dims ||
            throw(DimensionMismatch("all projected QTTvectors must have matching n_dims"))
        field.bits_per_dim == reference.bits_per_dim ||
            throw(DimensionMismatch("all projected QTTvectors must have matching bits_per_dim"))
        field.ordering == reference.ordering ||
            throw(DimensionMismatch("all projected QTTvectors must have matching ordering"))
    end

    tt_fields = Tuple(TensorTrainNumerics.TTvector(field) for field in fields)
    coeff = TensorTrainNumerics.project_nonlinearity(tt_fields, Φ; kwargs...)
    return TensorTrainNumerics.QTTvector(coeff, reference.n_dims, reference.bits_per_dim, reference.ordering)
end

end
