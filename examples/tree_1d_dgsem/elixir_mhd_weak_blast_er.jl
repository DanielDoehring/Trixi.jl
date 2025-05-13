using Trixi

###############################################################################
# semidiscretization of the ideal MHD equations
gamma = 2
equations = IdealGlmMhdEquations1D(gamma)

initial_condition = initial_condition_weak_blast_wave

volume_flux = flux_hindenlang_gassner
solver = DGSEM(polydeg = 3, surface_flux = flux_hindenlang_gassner,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = -2.0
coordinates_max = 2.0

BaseLevel = 5
# Test PERK on non-uniform mesh
refinement_patches = ((type = "box", coordinates_min = (-1.0,),
                       coordinates_max = (1.0,)),
                      (type = "box", coordinates_min = (-0.5,),
                       coordinates_max = (0.5,)))

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = BaseLevel,
                refinement_patches = refinement_patches,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     # Note: entropy defaults to mathematical entropy
                                     analysis_integrals = (entropy,),
                                     #analysis_filename = "entropy_standard.dat",
                                     analysis_filename = "entropy_ER.dat",
                                     save_analysis = true)

cfl = 1.0 # Probably not maxed out (PERRK schemes)
#cfl = 0.25 # RK4 etc.

stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

# TODO: Do optimization for the MHD case?
basepath = "/home/daniel/git/Paper_PERRK/Data/WeakBlastWave/"
dtRatios = [1, 0.5, 0.25]

relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 5, root_tol = eps(Float64), gamma_tol = eps(Float64))

# p = 2
#=
Stages = [9, 5, 3]
path = basepath * "p2/"

#ode_alg = Trixi.PairedExplicitRK2(Stages[1], path)
#ode_alg = Trixi.PairedExplicitRelaxationRK2(Stages[1], path, relaxation_solver = relaxation_solver)

#ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)
ode_alg = Trixi.PairedExplicitRelaxationRK2Multi(Stages, path, dtRatios, relaxation_solver = relaxation_solver)
=#
# p = 3
#=
Stages = [13, 7, 4]
path = basepath * "p3/"

#ode_alg = Trixi.PairedExplicitRK3(Stages[1], path)
#ode_alg = Trixi.PairedExplicitRelaxationRK3(Stages[1], path, relaxation_solver = relaxation_solver)

#ode_alg = Trixi.PairedExplicitRK3Multi(Stages, path, dtRatios)
ode_alg = Trixi.PairedExplicitRelaxationRK3Multi(Stages, path, dtRatios, relaxation_solver = relaxation_solver)
=#

# p = 4

Stages = [18, 10, 6]
path = basepath * "p4/"

#ode_alg = Trixi.PairedExplicitRK4(Stages[1], path)
#ode_alg = Trixi.PairedExplicitRelaxationRK4(Stages[1], path, relaxation_solver = relaxation_solver)

#ode_alg = Trixi.PairedExplicitRK4Multi(Stages, path, dtRatios)
ode_alg = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios, relaxation_solver = relaxation_solver)


# Test comparison algorithms for entropy conservation property
#ode_alg = Trixi.RK44()
#ode_alg = Trixi.TS64()
#ode_alg = Trixi.CKL54()

#ode_alg = Trixi.RelaxationRK44()
#ode_alg = Trixi.RelaxationTS64()
#ode_alg = Trixi.RelaxationCKL54()

sol = Trixi.solve(ode, ode_alg,
                  dt = 42.0,
                  save_everystep = false, callback = callbacks);
