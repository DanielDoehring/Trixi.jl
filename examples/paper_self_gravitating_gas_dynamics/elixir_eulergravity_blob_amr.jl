
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
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
                                         variable = density_pressure)

volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

# Domain size as present in Trixi
coordinates_min = (-20.0, -20.0)
coordinates_max = (20.0, 20.0)

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

cfl_gravity = 1.8 # CarpenterKennedy2N54
cfl_gravity = 1.9 # SSPRK104
cfl_gravity = 1.9 # SSPRK54
cfl_gravity = 1.8 # DGLDDRK84_F
cfl_gravity = 1.8 # ParsaniKetchesonDeconinck3S94
cfl_gravity = 1.8 # NDBLSRK124

parameters = ParametersEulerGravity(background_density = 0.0, # taken from above
                                    gravitational_constant = 6.674e-8, # aka G
                                    cfl = cfl_gravity, # Stability limit (at least for largest CFL = 3.2 for SSPRK104)
                                    resid_tol = 1.0e-4,
                                    n_iterations_max = 500,
                                    timestep_gravity = timestep_gravity_erk52_3Sstar!)

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters)

###############################################################################
# ODE solvers, callbacks etc.

# t_f = 8.0
tspan = (0.0, 8.0) # As tau_KH = 1.0, this is "nondimensional" time

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 20_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     save_analysis = false,
                                     analysis_integrals = ())

alive_callback = AliveCallback(alive_interval = 200)

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 1.0,
                                          alpha_min = 0.0001,
                                          alpha_smooth = false,
                                          variable = Trixi.density_pressure)
amr_controller = ControllerThreeLevelCombined(semi, amr_indicator, indicator_sc,
                                              base_level = 4,
                                              max_level = 8,
                                              max_threshold = 0.0003,
                                              max_threshold_secondary = indicator_sc.alpha_max)

###############################################################################
# run the simulation

cfl_ref = 1.1
N_AMR_ref = 15

cfl = 0.7 # CarpenterKennedy2N54 # tested
cfl = 3.2 # SSPRK104 # tested
cfl = 1.2  # SSPRK54 # tested 
cfl = 0.8 # DGLDDRK84_F # tested
cfl = 0.6 # ParsaniKetchesonDeconinck3S94
cfl = 0.6 # NDBLSRK124 # tested

amr_callback = AMRCallback(semi, amr_controller,
                           interval = Int(floor(N_AMR_ref * cfl_ref / cfl)),
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = cfl)

save_solution = SaveSolutionCallback(interval = 10_000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        amr_callback, stepsize_callback,
                        save_solution)

ode_alg = CarpenterKennedy2N54(thread = OrdinaryDiffEq.True())
ode_alg = SSPRK104(thread = OrdinaryDiffEq.True())
ode_alg = SSPRK54(thread = OrdinaryDiffEq.True())
ode_alg = DGLDDRK84_F(thread = OrdinaryDiffEq.True())
ode_alg = ParsaniKetchesonDeconinck3S94(thread = OrdinaryDiffEq.True())
ode_alg = NDBLSRK124(thread = OrdinaryDiffEq.True())

sol = solve(ode, ode_alg,
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
