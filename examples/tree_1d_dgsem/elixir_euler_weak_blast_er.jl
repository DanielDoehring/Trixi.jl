
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

# NOTE: Tied to 1D somewhat as mortars are not EC (yet)!

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_weak_blast_wave

# Volume flux adds some (minimal) disspation, thus stabilizing the simulation - 
# in contrast to standard DGSEM only
solver = DGSEM(polydeg = 3, surface_flux = flux_ranocha,
               volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

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

tspan = (0.0, 0.0) # For plotting of IC
#tspan = (0.0, 1.0)

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

cfl = 1.0 # Probably not maxed out

cfl = 0.25 # RK4

stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

basepath = "/home/daniel/git/Paper_PERRK/Data/WeakBlastWave/"
dtRatios = [1, 0.5, 0.25]

relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 3)
#relaxation_solver = Trixi.RelaxationSolverSecant()

#=
# p = 2
Stages = [9, 5, 3]
path = basepath * "p2/"

#ode_alg = Trixi.PairedExplicitRK2(Stages[1], path)
#ode_alg = Trixi.PairedExplicitRelaxationRK2(Stages[1], path, relaxation_solver = relaxation_solver)

#ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)
ode_alg = Trixi.PairedExplicitRelaxationRK2Multi(Stages, path, dtRatios, relaxation_solver = relaxation_solver)
=#

# p = 3

Stages = [13, 7, 4]
path = basepath * "p3/"

#ode_alg = Trixi.PairedExplicitRK3(Stages[1], path)
#ode_alg = Trixi.PairedExplicitRelaxationRK3(Stages[1], path, relaxation_solver = relaxation_solver)

#ode_alg = Trixi.PairedExplicitRK3Multi(Stages, path, dtRatios)
ode_alg = Trixi.PairedExplicitRelaxationRK3Multi(Stages, path, dtRatios,
                                                 relaxation_solver = relaxation_solver)

# p = 4
#=
Stages = [18, 10, 6]
path = basepath * "p4/"

#ode_alg = Trixi.PairedExplicitRK4(Stages[1], path)
#ode_alg = Trixi.PairedExplicitRelaxationRK4(Stages[1], path, relaxation_solver = relaxation_solver)

#ode_alg = Trixi.PairedExplicitRK4Multi(Stages, path, dtRatios)
ode_alg = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios, relaxation_solver = relaxation_solver)
=#

# Test comparison algorithms for entropy conservation property
ode_alg = Trixi.RK44()
ode_alg = Trixi.TS64()
ode_alg = Trixi.CKL54()

#ode_alg = Trixi.RelaxationRK44()
#ode_alg = Trixi.RelaxationTS64()
ode_alg = Trixi.RelaxationCKL54()

sol = Trixi.solve(ode, ode_alg,
                  dt = 42.0,
                  save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary

###############################################################################

# Plot IC & grid

using Plots

pd = PlotData1D(sol)

plot(getmesh(pd), label = "", ylim = (-0.3, 1.4))

plot!(pd["rho"], xlabel = "\$x\$",
      label = "\$\\rho\$",
      linewidth = 3, color = RGB(0, 84 / 256, 159 / 256),
      guidefont = font("Computer Modern", 16), tickfont = font("Computer Modern", 14),
      yticks = -0.2:0.4:1.4,
      legend = true)

plot!(pd["v1"],
      label = "\$v\$",
      linewidth = 3, color = RGB(246 / 256, 169 / 256, 0), legend = true)

plot!(pd["p"], title = "Weak Blast Wave: Initial Condition",
      label = "\$p\$",
      linewidth = 3, color = RGB(70 / 256, 171 / 256, 39 / 256),
      titlefont = font("Computer Modern", 18),
      legendfont = font("Computer Modern", 16),
      legend = :left)
