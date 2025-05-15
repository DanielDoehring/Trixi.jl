using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

gamma = 5 / 3
equations = IdealGlmMhdEquations2D(gamma)

initial_condition = initial_condition_convergence_test

volume_flux = (flux_central, flux_nonconservative_powell)
solver = DGSEM(polydeg = 2, # 2, 3, 4
               surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

function mapping(xi_, eta_)
    # Transform input variables between -1 and 1 onto [0, sqrt(2)]
    # Note, we use the domain [0, sqrt(2)]^2 for the Alfvén wave convergence test case
    x = 0.5 * sqrt(2) * (xi_ + 1)
    y = 0.5 * sqrt(2) * (eta_ + 1)

    return SVector(x, y)
end

N = 64
cells_per_dimension = (N, N)
mesh = StructuredMesh(cells_per_dimension, mapping)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = false,
                                     analysis_errors = [:l2_error, :l1_error, :linf_error],
                                     extra_analysis_integrals = (entropy, energy_total,
                                                                 energy_kinetic,
                                                                 energy_internal,
                                                                 energy_magnetic,
                                                                 cross_helicity))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = 4.0 # p2
#cfl = 2.5 # p3
#cfl = 1.0 # p4

stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

basepath = "/home/daniel/git/Paper_PERRK/Data/Alfven_Wave/"

Stages = [14, 13, 12, 11, 10]
dtRatios = [42, 42, 42, 42, 42] # Not relevant (random assignment)
relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 10, root_tol = 1e-15, gamma_tol = eps(Float64))


path = basepath * "k2/p2/"
ode_algorithm = Trixi.PairedExplicitRelaxationRK2Multi(Stages, path, dtRatios; relaxation_solver = relaxation_solver)


#=
path = basepath * "k3/p3/"
ode_algorithm = Trixi.PairedExplicitRelaxationRK3Multi(Stages, path, dtRatios; relaxation_solver = relaxation_solver)
=#

#=
path = basepath * "k4/p4/"
ode_algorithm = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios; relaxation_solver = relaxation_solver)
=#

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep = false, callback = callbacks);

###############################################################################

#=
using Plots

pd = PlotData2D(sol)

plot(pd["B1"])
plot!(getmesh(pd))
=#