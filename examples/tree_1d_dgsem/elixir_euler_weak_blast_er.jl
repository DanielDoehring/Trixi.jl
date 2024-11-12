
using OrdinaryDiffEq
using Trixi

using DoubleFloats
RealT = Double64

RealT = Float64

###############################################################################
# semidiscretization of the compressible Euler equations

# NOTE: Copied setup from "euler_ec.jl"

equations = CompressibleEulerEquations1D(RealT(14) / 10)

initial_condition = initial_condition_weak_blast_wave

# Volume flux adds some (minimal) disspation, thus stabilizing the simulation - 
# in contrast to standard DGSEM only
volume_flux = flux_ranocha
solver = DGSEM(RealT = RealT, polydeg = 3, surface_flux = flux_ranocha,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = -RealT(2)
coordinates_max = RealT(2)

BaseLevel = 7
# Test PERK on non-uniform mesh
refinement_patches = ((type = "box", coordinates_min = (0.5,),
                       coordinates_max = (1.5,)),
                      (type = "box", coordinates_min = (0.75,),
                       coordinates_max = (1.25,)))

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = BaseLevel,
                refinement_patches = refinement_patches,
                n_cells_max = 10_000,
                RealT = RealT)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (zero(RealT), RealT(4) / 10)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (entropy,),
                                     analysis_filename = "analysis_ER.dat",
                                     #analysis_filename = "analysis_standard.dat",
                                     save_analysis = true)

### Standalone ###
# Double 64:                                     
# PERK4 14 standalone                                     
cfl = 2.2 # Uniform
cfl = 2.2 # Non-uniform

# Float64:
cfl = 1.7 # Uniform
#cfl = 2.2 # Non-uniform

### Multi ###
# Float64:
cfl = 2.2 # Non-uniform

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

#ode_alg = Trixi.PairedExplicitRK4Multi(Stages, "/home/daniel/git/Paper-EntropyStabPERK/Data/IsentropicVortex_EC/", dtRatios)
ode_alg = Trixi.PairedExplicitERRK4Multi(Stages,
                                         "/home/daniel/git/Paper-EntropyStabPERK/Data/IsentropicVortex_EC/",
                                         dtRatios)

sol = Trixi.solve(ode, ode_alg,
                  dt = RealT(42),
                  save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
