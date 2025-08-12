using Trixi

using LinearSolve
#using Sparspak
using LineSearch, NonlinearSolve

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

num_flux = flux_godunov
#num_flux = flux_lax_friedrichs

solver = DGSEM(polydeg = 3, surface_flux = num_flux)

coordinates_min = -1.0
coordinates_max = 1.0

# One refinement
refinement_patches = ((type = "box", coordinates_min = (-0.5,),
                       coordinates_max = (0.5,)),)

# Two refinements

refinement_patches = ((type = "box", coordinates_min = (-0.5,),
                       coordinates_max = (0.5,)),
                       (type = "box", coordinates_min = (-0.25,),
                       coordinates_max = (0.25,)))

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                refinement_patches = refinement_patches,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

# TODO: Sparsity detection for partitioned RHS

###############################################################################
# ODE solvers, callbacks etc.

t_end = 2.0

ode = semidiscretize(semi, (0.0, t_end))

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 100,
                                     extra_analysis_errors = (:conservation_error,))

callbacks = CallbackSet(summary_callback,
                        analysis_callback)

###############################################################################
# run the simulation

path = "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/"

dtRatios = [1, 0.5]
Stages = [16, 8]

# One refinement
ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)
ode_alg = Trixi.PairedExplicitRK2IMEXMulti([8], path, [1])

# Two refinements
ode_alg = Trixi.PairedExplicitRK2IMEXMulti([16, 8], path, [1, 1])

dt = 0.05 / (2^0) # 0.05 for explicit 8-16 pair

### Linesearch ###
# See https://docs.sciml.ai/LineSearch/dev/api/native/

#linesearch = BackTracking(autodiff = AutoFiniteDiff(), order = 3, maxstep = 10)
linesearch = LiFukushimaLineSearch()
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
                        jac_prototype = nothing, colorvec = nothing,
                        nonlin_solver = nonlin_solver,
                        abstol = 1e-4, reltol = 1e-4);

sol = Trixi.solve!(integrator);

