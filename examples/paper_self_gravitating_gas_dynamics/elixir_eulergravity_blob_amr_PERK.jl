
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 5 / 3
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_blob(x, t, equations::CompressibleEulerEquations2D)

The blob test case taken from
- Agertz et al. (2006)
  Fundamental differences between SPH and grid methods
  [arXiv: astro-ph/0610051](https://arxiv.org/abs/astro-ph/0610051)
"""
function initial_condition_blob(x, t, equations::CompressibleEulerEquations2D)
    # blob test case, see Agertz et al. https://arxiv.org/pdf/astro-ph/0610051.pdf
    # other reference: https://arxiv.org/pdf/astro-ph/0610051.pdf
    # change discontinuity to tanh
    # typical domain is rectangular, we change it to a square
    # resolution 128^2, 256^2
    # domain size is [-20.0,20.0]^2
    # gamma = 5/3 for this test case
    R = 1.0 # radius of the blob
    # background density
    dens0 = 1.0
    Chi = 10.0 # density contrast (initial)
    # reference time of characteristic growth of KH instability equal to 1.0
    tau_kh = 1.0
    tau_cr = tau_kh / 1.6 # crushing time
    # determine background velocity
    velx0 = 2 * R * sqrt(Chi) / tau_cr
    vely0 = 0.0
    Ma0 = 2.7 # background flow Mach number Ma=v/c
    c = velx0 / Ma0 # sound speed
    # use perfect gas assumption to compute background pressure via the sound speed c^2 = gamma * pressure/density
    p0 = c * c * dens0 / equations.gamma

    # initial center of the blob (Trixi style)
    inicenter = [-15, 0]
    # Paper setup:
    #inicenter = [5, 5]
    # Try some middle ground:
    #inicenter = [-5, 0]

    x_rel = x - inicenter
    r = sqrt(x_rel[1]^2 + x_rel[2]^2)
    # steepness of the tanh transition zone
    slope = 2
    # density blob
    dens = dens0 +
           (Chi - 1) * 0.5 * (1 + (tanh(slope * (r + R)) - (tanh(slope * (r - R)) + 1)))
    # velocity blob is zero
    velx = velx0 - velx0 * 0.5 * (1 + (tanh(slope * (r + R)) - (tanh(slope * (r - R)) + 1)))
    return prim2cons(SVector(dens, velx, vely0, p0), equations)
end
initial_condition = initial_condition_blob

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 4
basis = LobattoLegendreBasis(polydeg)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.4,
                                         alpha_min = 0.0001,
                                         alpha_smooth = true,
                                         #variable = pressure
                                         variable = density_pressure
                                         )

volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

# Domain size as present in Trixi
coordinates_min = (-20.0, -20.0)
coordinates_max = (20.0, 20.0)

# Paper suggestion:
#coordinates_min = (0.0, 0.0)
#coordinates_max = (10.0, 10.0)

# Try some middle-ground:
#coordinates_min = (-10.0, -10.0)
#coordinates_max = (10.0, 10.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 100_000)

semi_euler = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# semidiscretization of the hyperbolic diffusion equations
equations_gravity = HyperbolicDiffusionEquations2D()

function initial_condition_blob_self_gravity(x, t,
                                             equations::HyperbolicDiffusionEquations2D)
    # for now just use constant initial condition for sedov blast wave (can likely be improved)
    phi = 0.0
    q1 = 0.0
    q2 = 0.0
    return SVector(phi, q1, q2)
end

solver_gravity = DGSEM(polydeg, flux_lax_friedrichs)

semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity,
                                            initial_condition_blob_self_gravity,
                                            solver_gravity,
                                            source_terms = source_terms_harmonic)

###############################################################################
# combining both semidiscretizations for Euler + self-gravity

base_path = "/home/daniel/git/MA/EigenspectraGeneration/PERK4/EulerGravity/Blob/2D/"

dtRatios = [1, 0.5, 0.25]
StagesGravity = [5, 3, 2]

cfl_gravity = 1.4
alg_gravity = Trixi.PairedExplicitRK2Multi(StagesGravity, base_path * "HypDiff/p2/", dtRatios)


#cfl_gravity = 1.4
cfl_gravity = 1.3
alg_gravity = Trixi.PairedExplicitRK2(5, base_path * "HypDiff/p2/")                         


parameters = ParametersEulerGravity(background_density = 0.0, # aka rho0
                                    gravitational_constant = 6.674e-8, # aka G
                                    cfl = cfl_gravity,
                                    resid_tol = 1.0e-4,
                                    n_iterations_max = 500,

                                    timestep_gravity = Trixi.timestep_gravity_PERK2!
                                    #timestep_gravity = Trixi.timestep_gravity_PERK2_Multi!
                                    )

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters, alg_gravity)

###############################################################################
# ODE solvers, callbacks etc.

# t_f = 8.0
tspan = (0.0, 8.0) # As tau_KH = 1.0, this is "nondimensional" time

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     save_analysis = false,
                                     analysis_integrals = ())

alive_callback = AliveCallback(analysis_interval = analysis_interval)

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 1.0,
                                          alpha_min = 0.0001,
                                          alpha_smooth = false,
                                          #variable = Trixi.density
                                          variable = Trixi.density_pressure
                                          )

amr_controller = ControllerThreeLevelCombined(semi, amr_indicator, indicator_sc,
                                              base_level = 4, # 5

                                              med_level = 0, # 0
                                              med_threshold = 0.0003, 

                                              max_level = 8, # 7
                                              max_threshold = 0.0003, # 0.0003 when max_level = 7
                                              
                                              max_threshold_secondary = indicator_sc.alpha_max)

#cfl = 1.1 # PERK 4 Multi, S_max = 9
cfl = 1.1 # PERK 4 Standalone, S_max = 9

amr_callback = AMRCallback(semi, amr_controller,
                           interval = Int(floor(15 * 1.1 / cfl)),
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        amr_callback, stepsize_callback)

###############################################################################
# run the simulation

dtRatios = [1, 0.5, 0.25]
Stages = [9, 6, 5]

#ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages, base_path * "Euler_only/", dtRatios)

ode_algorithm = Trixi.PairedExplicitRK4(Stages[1], base_path * "Euler_only/")

sol = Trixi.solve(ode, ode_algorithm, dt = 1.0, save_everystep = false,
                  callback = callbacks);

summary_callback() # print the timer summary
