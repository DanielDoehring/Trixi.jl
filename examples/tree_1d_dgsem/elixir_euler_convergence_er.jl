
using OrdinaryDiffEq
using Trixi

#using DoubleFloats

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_convergence_test

volume_flux = flux_ranocha
solver = DGSEM(RealT = Float64, polydeg = 3, surface_flux = flux_ranocha,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = 0.0
coordinates_max = 2.0

BaseLevel = 5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = BaseLevel,
                n_cells_max = 10_000)



# Test PERK on non-uniform mesh
refinement_patches = ((type = "box", coordinates_min = (0.5,),
                       coordinates_max = (1.5,)),
                      (type = "box", coordinates_min = (0.75,),
                       coordinates_max = (1.25,)))

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = BaseLevel,
                refinement_patches = refinement_patches,
                n_cells_max = 10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0) # 2.0
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 5
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (entropy,),
                                     save_analysis = true)

cfl = 1.5 # Non-uniform
cfl = 1.4 # AMR              
stepsize_callback = StepsizeCallback(cfl = cfl)

### Test with AMR: ###
amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = Trixi.density),
                                      base_level = BaseLevel,
                                      med_level = BaseLevel+1, med_threshold = 2.05,
                                      max_level = BaseLevel+2, max_threshold = 2.075)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        #amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

Stages = 14

ode_algorithm = Trixi.PairedExplicitRK4(Stages, "/home/daniel/git/Paper-EntropyStabPERK/Data/IsentropicVortex_EC/")
ode_algorithm = Trixi.PairedExplicitERRK4(Stages, "/home/daniel/git/Paper-EntropyStabPERK/Data/IsentropicVortex_EC/")

dtRatios = [1, 0.5, 0.25]
Stages = [14, 8, 5]

#ode_algorithm = Trixi.PairedExplicitERRK4Multi(Stages, "/home/daniel/git/Paper-EntropyStabPERK/Data/IsentropicVortex_EC/", dtRatios)
#ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages, "/home/daniel/git/Paper-EntropyStabPERK/Data/IsentropicVortex_EC/", dtRatios)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep=false, callback=callbacks);


#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
=#

summary_callback() # print the timer summary