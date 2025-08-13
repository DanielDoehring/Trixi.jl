using Trixi

using LinearSolve
#using Sparspak
using LineSearch, NonlinearSolve

###############################################################################
# semidiscretization of the linear advection equation

equations = InviscidBurgersEquation1D()

num_flux = flux_godunov
#num_flux = flux_lax_friedrichs

solver = DGSEM(polydeg = 3, surface_flux = num_flux)

coordinates_min = 0.0
coordinates_max = 1.0

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver,
                                    source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

t0 = 0.0
t_end = 2.0
t_span = (t0, t_end)

ode = semidiscretize(semi, t_span)
u0_ode = ode.u0
du_ode = similar(u0_ode)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 10_000,
                                     extra_analysis_errors = (:conservation_error,))

callbacks = CallbackSet(summary_callback,
                        analysis_callback)

###############################################################################
# Set up integrator

# TODO: Burgers has (potentially) different spectrum
path = "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/"

#=
ode_alg = Trixi.PairedExplicitRK2Multi([16, 8], path, [1, 1])

n_conv = 2
dt = (2.5e-2)/2^n_conv
sol = Trixi.solve(ode, ode_alg, dt = dt, 
                  save_everystep = false, callback = callbacks);
=#

ode_alg = Trixi.PairedExplicitRK2IMEXMulti([16], path, [1])
#ode_alg = Trixi.PairedExplicitRK2IMEXMulti([16, 8], path, [1, 1])

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

n_conv = 0
dt = (2e-3)/2^n_conv
integrator = Trixi.init(ode, ode_alg; dt = dt, callback = callbacks,
                        nonlin_solver = nonlin_solver,
                        abstol = 1e-8, reltol = 1e-8,
                        maxiters_nonlin = 100);

sol = Trixi.solve!(integrator);
