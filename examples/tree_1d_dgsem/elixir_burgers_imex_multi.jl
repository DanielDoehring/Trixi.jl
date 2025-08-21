using Trixi

using LinearSolve
#using Sparspak

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

### Linear Solver ###
# See https://docs.sciml.ai/LinearSolve/stable/solvers/solvers/

linear_solver = KLUFactorization()
#linear_solver = UMFPACKFactorization()

#linear_solver = SimpleGMRES()
#linear_solver = KrylovJL_GMRES()

# TODO: Could try algorithms from IterativeSolvers, KrylovKit

#linear_solver = SparspakFactorization() # requires Sparspak.jl


n_conv = 0
dt = (8e-3)/2^n_conv
integrator = Trixi.init(ode, ode_alg; dt = dt, callback = callbacks,
                        linear_solver = linear_solver,
                        atol_newton = 1e-8, maxits_newton = 100);

sol = Trixi.solve!(integrator);
