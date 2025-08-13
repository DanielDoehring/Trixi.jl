using Trixi

using LinearSolve
#using Sparspak
using LineSearch, NonlinearSolve

###############################################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

num_flux = flux_godunov
#num_flux = flux_lax_friedrichs

solver = DGSEM(polydeg = 3, surface_flux = num_flux)

coordinates_min = -4.0
coordinates_max = 4.0
length = coordinates_max - coordinates_min

function ic_gauss(x, t, equation::LinearScalarAdvectionEquation1D)
    scalar = exp(-(x[1]^2) * 5)
    return SVector(scalar)
end

refinement_patches = ((type = "box", coordinates_min = (-2.0,), coordinates_max = (2.0,)),
                      (type = "box", coordinates_min = (-1.0,), coordinates_max = (1.0,)))

#refinement_patches = ((type = "box", coordinates_min = (-2.0,), coordinates_max = (2.0,)),)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                refinement_patches = refinement_patches,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, ic_gauss, solver)

###############################################################################
# ODE & callbacks

t0 = 0.0
t_end = length
t_end = 1.0
t_span = (t0, t_end)

ode = semidiscretize(semi, t_span)
u0_ode = ode.u0
du_ode = similar(u0_ode)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 1,
                                     extra_analysis_errors = (:conservation_error,))

# 2-Levels
#=
amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                      base_level = 4,
                                      med_level = 4, med_threshold = 0.05,
                                      max_level = 5, max_threshold = 0.1)
=#
# 3-Levels
amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                      base_level = 4,
                                      med_level = 5, med_threshold = 0.2,
                                      max_level = 6, max_threshold = 0.4)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 1,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

callbacks = CallbackSet(summary_callback,
                        amr_callback,
                        analysis_callback)

###############################################################################
# Set up integrator

path = "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/"

dtRatios = [1, 0.5]
Stages = [16, 8]

# One refinement
ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)
#ode_alg = Trixi.PairedExplicitRK2IMEXMulti([8], path, [1])

#=
n_conv = 2
dt = 0.2/2^n_conv
sol = Trixi.solve(ode, ode_alg;
                  dt = dt,
                  ode_default_options()..., callback = callbacks);
=#

###############################################################################
# run the simulation

# Two refinements
ode_alg = Trixi.PairedExplicitRK2IMEXMulti([16, 8], path, [1, 1])

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
dt = 0.2/2^n_conv
integrator = Trixi.init(ode, ode_alg; dt = dt, callback = callbacks,
                        nonlin_solver = nonlin_solver,
                        abstol = 1e-4, reltol = 1e-4);

sol = Trixi.solve!(integrator);