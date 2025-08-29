using Trixi

using NonlinearSolve, LinearSolve, ADTypes

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
t_end = 0.0
t_span = (t0, t_end)

ode = semidiscretize(semi, t_span)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 10_000,
                                     extra_analysis_errors = (:conservation_error,))

callbacks = CallbackSet(summary_callback,
                        analysis_callback)

###############################################################################
# Set up integrator

# TODO: Burgers has (potentially) different spectrum
path = "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/"
#path = "/storage/home/daniel/PERRK/Data/IsentropicVortex/IsentropicVortex/k6/p2/"

#=
ode_alg = Trixi.PairedExplicitRK2Multi([16, 8], path, [1, 1])

n_conv = 2
dt = (2.5e-2)/2^n_conv
sol = Trixi.solve(ode, ode_alg, dt = dt, 
                  save_everystep = false, callback = callbacks);
=#

#ode_alg = Trixi.PairedExplicitRK2IMEXMulti([16], path, [1])
ode_alg = Trixi.PairedExplicitRK2IMEXMulti([16, 8], path, [1, 1])
#ode_alg = Trixi.PairedExplicitRK2IMEXMulti([12, 6], path, [1, 1])

atol_lin = 1e-8
rtol_lin = 1e-6
#maxiters_lin = 50

linsolve = KrylovJL_GMRES(atol = atol_lin, rtol = rtol_lin)

# For Krylov.jl kwargs see https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/#Krylov.gmres
nonlin_solver = NewtonRaphson(autodiff = AutoFiniteDiff(), 
                              linsolve = linsolve)

atol_nonlin = atol_lin
rtol_nonlin = rtol_lin
maxiters_nonlin = 20

n_conv = 3
dt = (8e-3)/2^n_conv
integrator = Trixi.init(ode, ode_alg;
                        dt = dt, callback = callbacks,
                        # IMEX-specific kwargs
                        nonlin_solver = nonlin_solver,
                        abstol = atol_nonlin, reltol = rtol_nonlin,
                        maxiters_nonlin = maxiters_nonlin);

sol = Trixi.solve!(integrator);
