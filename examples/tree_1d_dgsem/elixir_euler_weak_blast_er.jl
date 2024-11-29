
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
                                     #analysis_filename = "analysis_ER.dat",
                                     analysis_filename = "analysis_standard.dat",
                                     save_analysis = true)

cfl = 1.0

stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

# NOTE: The methods are not optimized for this testcase!
basepath = "/home/daniel/git/Paper_PERRK/Data/IsentropicVortex_Conv_Test_AMR_k6/"
dtRatios = [1, 0.5, 0.25]

#=
# p = 2
Stages = [12, 6, 3]
path = basepath * "p2/"

#ode_alg = Trixi.PairedExplicitRK2(12, path)
#ode_alg = Trixi.PairedExplicitRelaxationRK2(12, path)

#ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)
ode_alg = Trixi.PairedExplicitRelaxationRK2Multi(Stages, path, dtRatios)
=#

# p = 3
#=
Stages = [16, 8, 4]
path = basepath * "p3/"

#ode_alg = Trixi.PairedExplicitRK3(16, path)
ode_alg = Trixi.PairedExplicitRK3Multi(Stages, path, dtRatios)
=#

# p = 4

Stages = [15, 9, 5]
path = basepath * "p4/"

#ode_alg = Trixi.PairedExplicitRK4(15, path)
#ode_alg = Trixi.PairedExplicitRelaxationRK4(15, path)

ode_alg = Trixi.PairedExplicitRK4Multi(Stages, path, dtRatios)
#ode_alg = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios)


#=
# Note: This is actually optimized!
path = "/home/daniel/git/Paper_PERRK/Data/IsentropicVortex_EC/k3/"
Stages = 14

ode_alg = Trixi.PairedExplicitRK4(Stages, path)
ode_alg = Trixi.PairedExplicitRelaxationRK4(Stages, path)

Stages = [14, 8, 5]

#ode_alg = Trixi.PairedExplicitRK4Multi(Stages, path, dtRatios)

# NOTE: 3 Newton iterations suffice to ensure exact entropy conservation!
ode_alg = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios)
=#

sol = Trixi.solve(ode, ode_alg,
                  dt = 42.0,
                  save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
