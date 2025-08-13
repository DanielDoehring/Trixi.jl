using Trixi

using LinearSolve
#using Sparspak
using LineSearch, NonlinearSolve

# Functionality for automatic sparsity detection
using SparseDiffTools, Symbolics

###############################################################################################
### Overloads to construct the `LobattoLegendreBasis` with `Real` type (supertype of `Num`) ###

# Required for setting up the Lobatto-Legendre basis for abstract `Real` type.
# Constructing the Lobatto-Legendre basis with `Real` instead of `Num` is 
# significantly easier as we do not have to care about e.g. if-clauses.
# As a consequence, we need to provide some overloads hinting towards the intended behavior.

const float_type = Float64 # Actual floating point type for the simulation

# Newton tolerance for finding LGL nodes & weights
Trixi.eps(::Type{Real}) = Base.eps(float_type)
# There are some places where `one(RealT)` or `zero(uEltype)` is called where `RealT` or `uEltype` is `Real`.
# This returns an `Int64`, i.e., `1` or `0`, respectively which gives errors when a floating-point alike type is expected.
Trixi.one(::Type{Real}) = Base.one(float_type)
Trixi.zero(::Type{Real}) = Base.zero(float_type)

module RealMatMulOverload

# Multiplying two Matrix{Real}s gives a Matrix{Any}.
# This causes problems when instantiating the Legendre basis, which calls
# `calc_{forward,reverse}_{upper, lower}` which in turn uses the matrix multiplication
# which is overloaded here in construction of the interpolation/projection operators 
# required for mortars.
function Base.:*(A::Matrix{Real}, B::Matrix{Real})::Matrix{Real}
    m, n = size(A, 1), size(B, 2)
    kA = size(A, 2)
    kB = size(B, 1)
    @assert kA==kB "Matrix dimensions must match for multiplication"

    C = Matrix{Real}(undef, m, n)
    for i in 1:m, j in 1:n
        #acc::Real = zero(promote_type(typeof(A[i,1]), typeof(B[1,j])))
        acc = zero(Real)
        for k in 1:kA
            acc += A[i, k] * B[k, j]
        end
        C[i, j] = acc
    end
    return C
end
end

import .RealMatMulOverload

###############################################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

num_flux = flux_godunov
#num_flux = flux_lax_friedrichs

# `RealT = Real` requires fewer overloads than the more explicit `RealT = Num` from Symbolics
# `solver_real` is used for computing the Jacobian sparsity pattern
solver_real = DGSEM(polydeg = 3, surface_flux = num_flux, RealT = Real)
# `solver_float` is  used for the subsequent simulation
solver_float = DGSEM(polydeg = 3, surface_flux = num_flux)

coordinates_min = -1.0
coordinates_max = 1.0

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                n_cells_max = 30_000)

# `semi_real` is used for computing the Jacobian sparsity pattern
semi_real = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                         solver_real)

# `semi_float` is used for the subsequent simulation
semi_float = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                          solver_float)

###############################################################################
# ODE & callbacks

t0 = 0.0
t_end = 2.0
t_span = (t0, t_end)

ode = semidiscretize(semi_float, t_span)
u0_ode = ode.u0
du_ode = similar(u0_ode)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi_float, interval = 100,
                                     extra_analysis_errors = (:conservation_error,))

callbacks = CallbackSet(summary_callback,
                        analysis_callback)

###############################################################################
# Set up integrator

path = "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/"

# Two refinements
ode_alg = Trixi.PairedExplicitRK2IMEXMulti([16, 8], path, [1, 1])

dt = 0.0125 / (2^0) # 0.0125 for explicit 8-16 pair

integrator = Trixi.init(ode, ode_alg; dt = dt, callback = callbacks);

element_indices = integrator.level_info_elements_acc[ode_alg.num_methods]
interface_indices = integrator.level_info_interfaces_acc[ode_alg.num_methods]
boundary_indices = integrator.level_info_boundaries_acc[ode_alg.num_methods]
mortar_indices = integrator.level_info_mortars_acc[ode_alg.num_methods]
u_indices = integrator.level_u_indices_elements[ode_alg.num_methods]

###############################################################################################
### Compute the Jacobian with SparseDiffTools ###

sd = SymbolicsSparsityDetection()
ad_type = AutoFiniteDiff()
sparse_adtype = AutoSparse(ad_type)

# For the operator split employed here, the implicit equation is the parabolic part only
# The sparsity pattern of the residual function is identical with the one from the rhs
rhs_im = (du_ode, u0_ode) -> Trixi.rhs!(du_ode, u0_ode, semi_real, t0,
                                        element_indices, interface_indices, boundary_indices, mortar_indices)

sparse_cache = sparse_jacobian_cache(sparse_adtype, sd, rhs_im, du_ode, u0_ode)

# Full-dimensional sparsity information
jac_prototype = sparse_cache.jac_prototype
colorvec = sparse_cache.coloring.colorvec

# Truncate to relevant entries
# CARE: Not sure if this works for non-continuous `u_indices`!
# I am especially worried about the columns
subset_jac = jac_prototype[u_indices, u_indices]
subset_colorvec = colorvec[u_indices] # This should be safe

###############################################################################
# run the simulation

### Linesearch ###
# See https://docs.sciml.ai/LineSearch/dev/api/native/

#linesearch = BackTracking(autodiff = AutoFiniteDiff(), order = 3, maxstep = 10)
linesearch = LiFukushimaLineSearch()
#linesearch = nothing

### Linear Solver ###
# See https://docs.sciml.ai/LinearSolve/stable/solvers/solvers/

#linsolve = KLUFactorization()
#linsolve = UMFPACKFactorization()

linsolve = SimpleGMRES()
#linsolve = KrylovJL_GMRES()

# TODO: Could try algorithms from IterativeSolvers, KrylovKit

#linsolve = SparspakFactorization() # requires Sparspak.jl

# HYPRE & MKL do not work with sparsity structure of the Jacobian

#linsolve = nothing

nonlin_solver = NewtonRaphson(autodiff = AutoFiniteDiff(),
                              linesearch = linesearch, linsolve = linsolve)

#nonlin_solver = Broyden(autodiff = AutoFiniteDiff(), linesearch = linesearch)
# Could also check the advanced solvers: https://docs.sciml.ai/NonlinearSolve/stable/native/solvers/#Advanced-Solvers

n_conv = 1
dt = 0.0125/2^n_conv
integrator = Trixi.init(ode, ode_alg; dt = dt, callback = callbacks,
                        jac_prototype = subset_jac, colorvec = subset_colorvec,
                        nonlin_solver = nonlin_solver,
                        abstol = 1e-8, reltol = 1e-8);

sol = Trixi.solve!(integrator);

