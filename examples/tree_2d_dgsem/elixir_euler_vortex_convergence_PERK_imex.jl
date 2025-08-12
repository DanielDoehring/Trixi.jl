using Trixi
using OrdinaryDiffEqLowStorageRK

using LinearSolve
#using Sparspak
using LineSearch, NonlinearSolve

# Functionality for automatic sparsity detection
using SparseDiffTools, Symbolics

###############################################################################
# semidiscretization of the compressible Euler equations

# define new structs inside a module to allow re-evaluating the file
module TrixiExtension
using Trixi

struct IndicatorVortex{Cache <: NamedTuple} <: Trixi.AbstractIndicator
    cache::Cache
end

function IndicatorVortex(semi)
    basis = semi.solver.basis
    alpha = Vector{real(basis)}()
    A = Array{real(basis), 2}
    indicator_threaded = [A(undef, nnodes(basis), nnodes(basis))
                          for _ in 1:Threads.nthreads()]
    cache = (; semi.mesh, alpha, indicator_threaded) # "Leading semicolon" makes this a named tuple

    return IndicatorVortex{typeof(cache)}(cache)
end

function (indicator_vortex::IndicatorVortex)(u::AbstractArray{<:Any, 4},
                                             mesh, equations, dg, cache;
                                             t, kwargs...)
    mesh = indicator_vortex.cache.mesh
    alpha = indicator_vortex.cache.alpha
    indicator_threaded = indicator_vortex.cache.indicator_threaded
    resize!(alpha, nelements(dg, cache))

    # get analytical vortex center (based on assumption that center=[0.0,0.0]
    # at t=0.0 and that we stop after one period)
    domain_length = mesh.tree.length_level_0
    if t < 0.5 * domain_length
        center = (t, t)
    else
        center = (t - domain_length, t - domain_length)
    end

    Threads.@threads for element in eachelement(dg, cache)
        cell_id = cache.elements.cell_ids[element]
        coordinates = (mesh.tree.coordinates[1, cell_id], mesh.tree.coordinates[2, cell_id])
        # use the negative radius as indicator since the AMR controller increases
        # the level with increasing value of the indicator and we want to use
        # high levels near the vortex center
        alpha[element] = -periodic_distance_2d(coordinates, center, domain_length)
    end

    return alpha
end

function periodic_distance_2d(coordinates, center, domain_length)
    dx = @. abs(coordinates - center)
    dx_periodic = @. min(dx, domain_length - dx)
    return sqrt(sum(abs2, dx_periodic))
end
end # module TrixiExtension

import .TrixiExtension

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

# We need to avoid if-clauses to be able to use `Num` type from Symbolics without additional hassle.
# In the Trixi implementation, we overload the sqrt function to first check if the argument 
# is < 0 and then return NaN instead of an error.
# To turn off this behaviour, we switch back to the Base implementation here which does not contain an if-clause.
Trixi.sqrt(x::Num) = Base.sqrt(x)

# Ratio of specific heats
gamma = 1.4

equations = CompressibleEulerEquations2D(gamma)

EdgeLength = 20.0

N_passes = 1
T_end = EdgeLength * N_passes
t0 = 0.0
tspan = (t0, T_end)
#tspan = (0.0, 1.0)

"""
    initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)

The classical isentropic vortex test case as presented in 
https://spectrum.library.concordia.ca/id/eprint/985444/1/Paired-explicit-Runge-Kutta-schemes-for-stiff-sy_2019_Journal-of-Computation.pdf
"""
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
    # Evaluate error after full domain traversion
    if t == T_end
        t = 0
    end

    # initial center of the vortex
    inicenter = SVector(0.0, 0.0)
    # strength of the vortex
    S = 13.5
    # Radius of vortex
    R = 1.5
    # Free-stream Mach 
    M = 0.4
    # base flow
    v1 = 1.0
    v2 = 1.0
    vel = SVector(v1, v2)

    cent = inicenter + vel * t      # advection of center
    cent = x - cent               # distance to centerpoint
    cent = SVector(cent[2], -cent[1])
    r2 = cent[1]^2 + cent[2]^2

    f = (1 - r2) / (2 * R^2)

    rho = (1 - (S * M / pi)^2 * (gamma - 1) * exp(2 * f) / 8)^(1 / (gamma - 1))

    du = S / (2 * π * R) * exp(f) # vel. perturbation
    vel = vel + du * cent
    v1, v2 = vel

    p = rho^gamma / (gamma * M^2)
    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_isentropic_vortex

surf_flux = flux_hllc # Better flux, allows much larger timesteps
surf_flux = flux_lax_friedrichs # if-clause free

PolyDeg = 6
solver_real = DGSEM(RealT = Real, polydeg = PolyDeg, surface_flux = surf_flux)
solver_float = DGSEM(RealT = Float64, polydeg = PolyDeg, surface_flux = surf_flux)

coordinates_min = (-EdgeLength / 2, -EdgeLength / 2)
coordinates_max = (EdgeLength / 2, EdgeLength / 2)

Refinement = 5
refinement_patches = ((type = "sphere", center = (0.0, 0.0), radius = 3.0),
                      (type = "sphere", center = (0.0, 0.0), radius = 2.0),
                      (type = "sphere", center = (0.0, 0.0), radius = 0.5))

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = Refinement,
                refinement_patches = refinement_patches,
                n_cells_max = 100_000)

semi_real = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver_real)
semi_float = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver_float)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi_float, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10^6
analysis_callback = AnalysisCallback(semi_float, interval = analysis_interval,
                                     extra_analysis_errors = (:l1_error,),
                                     analysis_integrals = (;))


N_Convergence = 0 # up to 7 for p2, 3 for p3/p4
CFL_Convergence = 1.0 / (2^N_Convergence)

alive_callback = AliveCallback(alive_interval = 100 * Int((2^N_Convergence)))

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback)

###############################################################################
# run the simulation

dtRatios = [1, 0.5, 0.25]
#basepath = "/home/daniel/git/Paper_PERRK/Data/IsentropicVortex/IsentropicVortex/k6/"
basepath = "/storage/home/daniel/PERRK/Data/IsentropicVortex/IsentropicVortex/k6/"

# p = 2

Stages = [12, 6, 3]
path = basepath * "p2/"
ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)

dt = 0.008 * CFL_Convergence # Timestep in asymptotic regime

#=
# NOTE: For some reason `prolong2mortars` massive allocates if the standalone version is not executed before
sol = solve(ode, CarpenterKennedy2N54(thread = Trixi.True()),
            dt = 0.001,
            save_everystep = false, callback = callbacks);
=#

#=
sol = Trixi.solve(ode, ode_alg,
                  dt = dt,
                  save_everystep = false, callback = callbacks);
=#

###############################################################################
# IMEX

ode_alg = Trixi.PairedExplicitRK2IMEXMulti(Stages, path, dtRatios)

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

u0_ode = ode.u0
du_ode = similar(u0_ode)
sparse_cache = sparse_jacobian_cache(sparse_adtype, sd, rhs_im, du_ode, u0_ode)

linesearch = BackTracking(autodiff = AutoFiniteDiff(), order = 3, maxstep = 10)
#linesearch = LiFukushimaLineSearch()
#linesearch = nothing

### Linear Solver ###
# See https://docs.sciml.ai/LinearSolve/stable/solvers/solvers/

#linsolve = KLUFactorization()
#linsolve = UMFPACKFactorization()

#linsolve = SimpleGMRES()
linsolve = KrylovJL_GMRES()

# TODO: Could try algorithms from IterativeSolvers, KrylovKit

#linsolve = SparspakFactorization() # requires Sparspak.jl

# HYPRE & MKL do not work with sparsity structure of the Jacobian

#linsolve = nothing

nonlin_solver = NewtonRaphson(autodiff = AutoFiniteDiff(),
                              linesearch = linesearch, linsolve = linsolve)

#nonlin_solver = Broyden(autodiff = AutoFiniteDiff(), linesearch = linesearch)
# Could also check the advanced solvers: https://docs.sciml.ai/NonlinearSolve/stable/native/solvers/#Advanced-Solvers


integrator = Trixi.init(ode, ode_alg; dt = dt, callback = callbacks,
                        #jac_prototype = subset_jac, colorvec = subset_colorvec,
                        nonlin_solver = nonlin_solver,
                        abstol = 1e-4, reltol = 1e-4);

sol = Trixi.solve!(integrator);