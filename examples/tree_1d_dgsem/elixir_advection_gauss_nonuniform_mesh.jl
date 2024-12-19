
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

num_flux = flux_lax_friedrichs
num_flux = flux_godunov
solver = DGSEM(polydeg = 3, surface_flux = num_flux)

coordinates_min = -5.0 # minimum coordinate
coordinates_max = 5.0 # maximum coordinate

refinement_patches = ((type = "box", coordinates_min = (-1.0,),
                       coordinates_max = (1.0,)),)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                refinement_patches = refinement_patches,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_gauss,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, (0.0, 11.0))

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 1,
                                     extra_analysis_errors = (:conservation_error,),
                                     extra_analysis_integrals = (Trixi.entropy_math,),
                                     analysis_filename = "1D_Adv_NonUni_Std.dat",
                                     #analysis_filename = "1D_Adv_NonUni_Rel.dat",
                                     save_analysis = true)
cfl = 3.5 # [16, 8]
#cfl = 5.5 # [32, 16]

#cfl = 2.0 # CarpenterKennedy2N54
stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback, 
                        analysis_callback,
                        #stepsize_callback
                        )

###############################################################################
# run the simulation

path = "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/"
#path = "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/Joint/"

dtRatios = [1, 0.5]
Stages = [16, 8]
#Stages = [32, 16]

relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 8)
#relaxation_solver = Trixi.RelaxationSolverBisection()

ode_alg = Trixi.PairedExplicitRelaxationRK2Multi(Stages, path, dtRatios; 
                                                 relaxation_solver = relaxation_solver)

#ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)

sol = Trixi.solve(ode, ode_alg,
                  dt = 0.25,
                  save_everystep = false, callback = callbacks);


#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 42.0,
            save_everystep = false, callback = callbacks);
=#

# Print the timer summary
summary_callback()

#scatter(sol.u[end])
scatter!(sol.u[end])