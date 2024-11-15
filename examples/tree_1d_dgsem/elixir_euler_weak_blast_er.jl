
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

# NOTE: Tied to 1D somewhat as mortars are not EC (yet)!

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_weak_blast_wave

# Volume flux adds some (minimal) disspation, thus stabilizing the simulation - 
# in contrast to standard DGSEM only
volume_flux = flux_ranocha
solver = DGSEM(polydeg = 3, surface_flux = flux_ranocha,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = -2.0
coordinates_max = 2.0

BaseLevel = 7
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
                                     analysis_integrals = (entropy,),
                                     analysis_filename = "analysis_ER.dat",
                                     #analysis_filename = "analysis_standard.dat",
                                     save_analysis = true)

cfl = 1.7

stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

Stages = 14

#=
ode_alg = Trixi.PairedExplicitRK4(Stages,
                                  "/home/daniel/git/Paper-EntropyStabPERK/Data/IsentropicVortex_EC/")

ode_alg = Trixi.PairedExplicitERRK4(Stages,
                                    "/home/daniel/git/Paper-EntropyStabPERK/Data/IsentropicVortex_EC/")
=#

dtRatios = [1, 0.5, 0.25]
Stages = [14, 8, 5]

ode_alg = Trixi.PairedExplicitRK4Multi(Stages, "/home/daniel/git/Paper-EntropyStabPERK/Data/IsentropicVortex_EC/", dtRatios)


# NOTE: 3 Newton iterations suffice to ensure exact entropy conservation!
ode_alg = Trixi.PairedExplicitERRK4Multi(Stages,
                                         "/home/daniel/git/Paper-EntropyStabPERK/Data/IsentropicVortex_EC/",
                                         dtRatios)


sol = Trixi.solve(ode, ode_alg,
                  dt = 42.0,
                  save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary

using Plots
plot(sol)