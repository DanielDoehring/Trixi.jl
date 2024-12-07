
using OrdinaryDiffEq
using Trixi

# TODO: Try 3D version to be more expensive!

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations_euler = CompressibleEulerEquations2D(gamma)

function initial_condition_weakblast_self_gravity(x, t,
                                                  equations::CompressibleEulerEquations2D)
    # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) (WEAK Blast wave!)
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    # Calculate primitive variables
    rho = r > 0.5 ? 1.0 : 1.1691
    v1 = r > 0.5 ? 0.0 : 0.1882 * cos_phi
    v2 = r > 0.5 ? 0.0 : 0.1882 * sin_phi
    p = r > 0.5 ? 1.0 : 1.245

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_weakblast_self_gravity

function boundary_condition_weakblast_self_gravity(u_inner, orientation, direction, x, t,
                                                   surface_flux_function,
                                                   equations::CompressibleEulerEquations2D)
    # velocities are zero, density/pressure are ambient values according to
    # initial_condition_weakblast_self_gravity
    rho = 1.0
    v1 = 0.0
    v2 = 0.0
    p = 1.0

    u_boundary = prim2cons(SVector(rho, v1, v2, p), equations)

    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end

    return flux
end
boundary_conditions = boundary_condition_weakblast_self_gravity

surface_flux = flux_hll
volume_flux = flux_chandrashekar
polydeg = 3
basis = LobattoLegendreBasis(polydeg)

volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver_euler = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-4, -4)
coordinates_max = (4, 4)
mesh = TreeMesh(coordinates_min, coordinates_max,
                # Increase initial resolution to stability PERK
                initial_refinement_level = 7, # 2
                n_cells_max = 100_000,
                periodicity = false)

semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_condition,
                                          solver_euler,
                                          boundary_conditions = boundary_conditions)

###############################################################################
# semidiscretization of the hyperbolic diffusion equations
equations_gravity = HyperbolicDiffusionEquations2D()

function initial_condition_weakblast_self_gravity(x, t,
                                                  equations::HyperbolicDiffusionEquations2D)
    # for now just use constant initial condition for sedov blast wave (can likely be improved)
    phi = 0.0
    q1 = 0.0
    q2 = 0.0
    return SVector(phi, q1, q2)
end

function boundary_condition_weakblast_self_gravity(u_inner, orientation, direction, x, t,
                                                   surface_flux_function,
                                                   equations::HyperbolicDiffusionEquations2D)
    u_boundary = initial_condition_weakblast_self_gravity(x, t, equations)

    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end

    return flux
end

solver_gravity = DGSEM(polydeg, flux_lax_friedrichs)

semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity, initial_condition,
                                            solver_gravity,
                                            boundary_conditions = boundary_conditions,
                                            source_terms = source_terms_harmonic)

###############################################################################

# combining both semidiscretizations for Euler + self-gravity

cfl_gravity = 2.4 # p = 2

parameters = ParametersEulerGravity(background_density = 0.0, # aka rho0
                                    gravitational_constant = 6.674e-8, # aka G
                                    cfl = cfl_gravity,
                                    resid_tol = 1.0e-4,
                                    n_iterations_max = 100,
                                    timestep_gravity = timestep_gravity_erk52_3Sstar!)

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters)


#=
b1   = 0.0
bS   = 1.0 - b1
cEnd = 0.5/bS

StagesGravity = 5
cfl_gravity = 2.1

alg_gravity = PERK(StagesGravity, 
                   "/home/daniel/git/MA/EigenspectraGeneration/PERK4/EulerGravity/WeakBlastWave/HypDiff/p2/", 
                   bS, cEnd)
=#

#=
StagesGravity = 14
cfl_gravity = 4.7

alg_gravity = PERK4(StagesGravity, "/home/daniel/git/MA/EigenspectraGeneration/PERK4/EulerGravity/WeakBlastWave/HypDiff/p4/")
=#

#=
parameters = ParametersEulerGravity(background_density = 0.0, # aka rho0
                                    gravitational_constant = 6.674e-8, # aka G
                                    cfl = cfl_gravity,
                                    resid_tol = 1.0e-4,
                                    n_iterations_max = 100,

                                    timestep_gravity = timestep_gravity_PERK2!
                                    #timestep_gravity = timestep_gravity_PERK4!
                                    )

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters, alg_gravity)
=#

###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 2.5)

ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 1.0,
                                          alpha_min = 0.0,
                                          alpha_smooth = false,
                                          variable = density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 4, # 2
                                      max_level = 8, max_threshold = 0.00013) # Heavily tailored, REVISIT!
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 1,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = false #= true =#)

# CFL values for 4th order gravity solver!                           
#cfl = 1.7 # CarpenterKennedy2N54
cfl = 4.5 # SSPRK104 5.0 stable, but artifacts in solution
#cfl = 1.6 # SSPRK54 1.7 stable, but artifacts in solution
#cfl = 3.2 # DGLDDRK84_F
#cfl = 2.1 # ParsaniKetchesonDeconinck3S94
#cfl = 1.8 # NDBLSRK124

stepsize_callback = StepsizeCallback(cfl = cfl)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval = analysis_interval)

analysis_callback = AnalysisCallback(semi_euler, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     save_analysis = false,
                                     analysis_integrals = ())

callbacks = CallbackSet(summary_callback, amr_callback, stepsize_callback,
                        #save_solution,
                        analysis_callback, 
                        alive_callback)

###############################################################################
# run the simulation

#ode_alg = CarpenterKennedy2N54(thread = OrdinaryDiffEq.True())
ode_alg = SSPRK104(thread = OrdinaryDiffEq.True())
#ode_alg = SSPRK54(thread = OrdinaryDiffEq.True())
#ode_alg = DGLDDRK84_F(thread = OrdinaryDiffEq.True())
#ode_alg = ParsaniKetchesonDeconinck3S94(thread = OrdinaryDiffEq.True())
#ode_alg = NDBLSRK124(thread = OrdinaryDiffEq.True())

sol = solve(ode, ode_alg,
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);


summary_callback() # print the timer summary
println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)
