
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
prandtl_number() = 1.0

# Make less diffusive to still have convection-dominated spectra
mu() = 1e-3
eta() = 1e-3

gamma = 5/3
equations = IdealGlmMhdEquations2D(gamma)
equations_parabolic = ViscoResistiveMhdDiffusion2D(equations, mu = mu(),
                                                   Prandtl = prandtl_number(),
                                                   eta = eta(),
                                                   gradient_variables = GradientVariablesPrimitive())

"""
    initial_condition_orszag_tang(x, t, equations::IdealGlmMhdEquations2D)

The classical Orszag-Tang vortex test case. Here, the setup is taken from
- https://onlinelibrary.wiley.com/doi/pdf/10.1002/fld.4681

# Note: In their MHD equations some Lundquist scaling factors are included, see also 
https://doi.org/10.1006/jcph.1999.6248
"""
function initial_condition_orszag_tang(x, t, equations::IdealGlmMhdEquations2D)
  rho = 1.0
  v1 = -2 * sqrt(pi) * sin(x[2])
  v2 =  2 * sqrt(pi) * sin(x[1])
  v3 = 0.0
  p = 15/4 + 0.25 * cos(4*x[1]) + 0.8 * cos(2*x[1])*cos(x[2]) - cos(x[1])*cos(x[2]) + 0.25 * cos(2*x[2])
  
  B1 = -sin(x[2])
  B2 =  sin(2.0*x[1])

  B3 = 0.0
  psi = 0.0
  return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

initial_condition = initial_condition_orszag_tang

surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell)
volume_flux  = (flux_central, flux_nonconservative_powell)

basis = LobattoLegendreBasis(3)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (2*pi, 2*pi)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=100000)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
#tspan = (0.0, 1.0) # For plotting

ode = semidiscretize(semi, tspan; split_problem = false)

summary_callback = SummaryCallback()

analysis_interval = 10000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     #analysis_errors = Symbol[],
                                     analysis_integrals = Symbol[])

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=0.5,
                                          alpha_min=0.001,
                                          alpha_smooth=false,
                                          variable=density_pressure)
#=                                 
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=5,
                                      med_level =7, med_threshold=0.04,
                                      max_level =9, max_threshold=0.4)
=#

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=4,
                                      med_level =6, med_threshold=0.04,
                                      max_level =9, max_threshold=0.4)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=10, # PERK
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)
                                   
CFL = 1.9 # PERK 3, 4, 6
CFL = 1.9 # PERRK 3, 4, 6

stepsize_callback = StepsizeCallback(cfl=CFL)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=CFL)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        amr_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

Stages = [6, 4, 3]
dtRatios = [1.0, 0.5, 0.25]

path = "/home/daniel/git/paper-2024-amr-paired-rk/elixirs/sec7_applications/sec_7.1_hyperbolic_parabolic/visco_resistive_orszag_tang/data/"

#ode_algorithm = Trixi.PairedExplicitRK3(Stages[1], path)

ode_algorithm = Trixi.PairedExplicitRK3Multi(Stages, path, dtRatios)

#ode_algorithm = Trixi.PairedExplicitRelaxationRK3Multi(Stages, path, dtRatios)

sol = Trixi.solve(ode, ode_algorithm, dt = 42.0,
                  save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary


pd = PlotData2D(sol)

plot(pd["rho"], c = :jet, title = "\$ ρ, t_f = 3.0 \$", 
           xticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]),
           yticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]))

plot(pd["p"], c = :jet, title = "\$ p, t = 1.0 \$",
     xticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]),
     yticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]))

plot(getmesh(pd), xlabel = "\$x\$", ylabel="\$y\$", title = "Mesh at \$t = 1.0\$",
           xticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]),
           yticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]))