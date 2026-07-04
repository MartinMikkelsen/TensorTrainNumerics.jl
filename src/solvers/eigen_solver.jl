"""
    eigen_solve(A, guess; alg=MALS())
    eigen_solve(A, guess, alg)

Solve a TT eigenvalue problem with the algorithm object `alg`.
"""
eigen_solve(A, guess; alg::EigenSolverAlgorithm = MALS()) = eigen_solve(A, guess, alg)

"""
    als_eigsolve(A, guess; kwargs...)

Compatibility wrapper for `eigen_solve(A, guess, ALS(; kwargs...))`.
"""
function als_eigsolve(A, guess; kwargs...)
    return eigen_solve(A, guess, ALS(; kwargs...))
end

"""
    mals_eigsolve(A, guess; kwargs...)

Compatibility wrapper for `eigen_solve(A, guess, MALS(; kwargs...))`.
"""
function mals_eigsolve(A, guess; kwargs...)
    return eigen_solve(A, guess, MALS(; kwargs...))
end

"""
    dmrg_eigsolve(A, guess; kwargs...)

Compatibility wrapper for `eigen_solve(A, guess, DMRG(; kwargs...))`.
"""
function dmrg_eigsolve(A, guess; kwargs...)
    return eigen_solve(A, guess, DMRG(; kwargs...))
end
