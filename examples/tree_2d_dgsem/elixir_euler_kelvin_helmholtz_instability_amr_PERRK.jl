using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-1,+1]^2
  slope = 15
  amplitude = 0.02
  B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
  rho = 0.5 + 0.75 * B
  v1 = 0.5 * (B - 1)
  v2 = 0.1 * sin(2 * pi * x[1])
  p = 1.0
  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability

surface_flux = flux_hlle
volume_flux  = flux_ranocha

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.002,
                                         alpha_min=0.0001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
Refinement = 4
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=Refinement,
                n_cells_max=100_000)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

# PERK stable
tspan = (0.0, 2.73)
# PERRK stable
#tspan = (0.0, 3.2)

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=1.0,
                                          alpha_min=0.0001,
                                          alpha_smooth=false,
                                          variable=Trixi.density)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=Refinement,
                                      med_level=Refinement+3, med_threshold=0.7, # med_level = current level
                                      max_level=Refinement+5, max_threshold=0.9)

amr_interval = 7 # PERK p3 4, 6, 11
amr_callback = AMRCallback(semi, amr_controller,
                           interval=amr_interval,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

analysis_interval = 5 # For entropy write-out
analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     analysis_errors = Symbol[],
                                     #analysis_integrals = Symbol[]
                                     analysis_integrals = (entropy,),
                                     analysis_filename = "entropy_standard.dat",
                                     #analysis_filename = "entropy_ER.dat",
                                     save_analysis = true
                                     )

alive_interval = 1 # For finding crash time standard scheme
alive_interval = 50
alive_callback = AliveCallback(alive_interval = alive_interval)

cfl = 3.8 # p = 3, E = 4, 6, 11
stepsize_callback = StepsizeCallback(cfl=cfl)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        amr_callback,
                        #save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

Stages = [11, 6, 4]
dtRatios = [42.0, 42.0, 42.0]

path = "/home/daniel/git/Paper_PERRK/Data/Kelvin_Helmholtz/Viscous_Discretization/k3/"

#ode_algorithm = Trixi.PairedExplicitRK3Multi(Stages, path, dtRatios)
ode_algorithm = Trixi.PairedExplicitRelaxationRK3Multi(Stages, path, dtRatios)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep=false, callback=callbacks);

