
using OrdinaryDiffEq
using Trixi

# TODO: Try 3D version to be more expensive!

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations_euler = CompressibleEulerEquations2D(gamma)

function initial_condition_weak_blast(x, t,
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
initial_condition = initial_condition_weak_blast

function boundary_condition_weak_blast(u_inner, orientation, direction, x, t,
                                               surface_flux_function,
                                               equations::CompressibleEulerEquations2D)
    # velocities are zero, density/pressure are ambient values according to
    # initial_condition_weak_blast
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
boundary_conditions = boundary_condition_weak_blast

surface_flux = flux_hll
volume_flux = flux_chandrashekar

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

#=
indicator_sc = IndicatorHennemannGassner(equations_euler, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
=#

volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver_euler = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-4, -4)
coordinates_max = (4, 4)
mesh = TreeMesh(coordinates_min, coordinates_max,
                # Increase initial resolution to stability PERK
                initial_refinement_level = 7, # 2
                n_cells_max = 100_000,
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations_euler, initial_condition,
                                          solver_euler,
                                          boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 2.0)

ode = semidiscretize(semi, tspan); # Run euler only (testcase)

summary_callback = SummaryCallback()

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 1.0,
                                          alpha_min = 0.0,
                                          alpha_smooth = false,
                                          variable = density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 4, # 2
                                      max_level = 8, 
                                      max_threshold = 0.00013) # Heavily tailored

# TODO: Something seems to be not working with initial condition adapt!
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 1,
                           adapt_initial_condition = true, # true
                           adapt_initial_condition_only_refine = false #= true =#)

# initial ref = 7

# Single PERK 
# E = 8
cfl = 2.4 # Euler only
# E = 13
cfl = 3.1

# PERK Multi
# E = 8, 6, 5
#cfl = 2.3 # Euler only

# CarpenterKennedy2N54
#cfl = 1.7

stepsize_callback = StepsizeCallback(cfl = cfl)

analysis_interval = 50

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     save_analysis = false,
                                     analysis_integrals = ())

callbacks = CallbackSet(summary_callback, 
                        amr_callback, 
                        stepsize_callback,
                        analysis_callback)

###############################################################################
# run the simulation


Stages = 8
#Stages = 13

ode_algorithm = PERK4(Stages, "/home/daniel/git/MA/EigenspectraGeneration/PERK4/EulerGravity/WeakBlastWave/")


#=
dtRatios = [1, 0.5, 0.25, 0.125]
Stages = [13, 8, 6, 5]

dtRatios = [1, 0.5, 0.25]
Stages = [8, 6, 5]

ode_algorithm = PERK4_Multi(Stages, "/home/daniel/git/MA/EigenspectraGeneration/PERK4/EulerGravity/WeakBlastWave/", dtRatios)
=#

sol = Trixi.solve(ode, ode_algorithm, dt = 1.0, save_everystep = false, callback = callbacks);


sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);


summary_callback() # print the timer summary
println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)
