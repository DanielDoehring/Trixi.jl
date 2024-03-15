
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal MHD equations
gamma = 1.4
equations = CompressibleEulerEquations1D(gamma)

"""
    initial_condition_sod_shock_tube(x, t, equations::CompressibleEulerEquations1D)

Setup of the classic Sod shock tube problem following the setup of test 1 from
- Valerie Kulka, Patrick Jenny (2022)
  Temporally adaptive conservative scheme for unsteady compressible flow
  [DOI: 10.1016/j.jcp.2021.110918](https://doi.org/10.1016/j.jcp.2021.110918)
"""
function initial_condition_sod_shock_tube(x, t, equations::CompressibleEulerEquations1D)
    rho = x[1] < 0.5 ? 2.0 : 1.0
    v1 = 0.0
    p = x[1] < 0.5 ? 2.0e5 : 1.0e5
    return prim2cons(SVector(rho, v1, p), equations)
end
initial_condition = initial_condition_sod_shock_tube

# These BC are only valid for short simulation times
boundary_conditions = BoundaryConditionDirichlet(initial_condition)

surface_flux = flux_hllc
#surface_flux = flux_hlle
#surface_flux = flux_lax_friedrichs

volume_flux = flux_ranocha
basis = LobattoLegendreBasis(3)

# Shock-capturing
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = 0.0
coordinates_max = 1.0

InitialRef = 7
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = InitialRef,
                n_cells_max = 10_000,
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.8e-3)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10^5
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:l2_error_primitive,
                                                              :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 0.5,
                                          alpha_min = 0.001,
                                          alpha_smooth = true,
                                          variable = density_pressure)
                                          #variable = Trixi.density)
                                          
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = InitialRef,
                                      med_level = 6, med_threshold = 1e-6,
                                      max_level = 10, 
                                      max_threshold = 1e-5)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 1.6) # hllc, hlle
#stepsize_callback = StepsizeCallback(cfl = 1.2) # flux_lax_friedrichs

callbacks = CallbackSet(summary_callback,
                        analysis_callback, 
                        alive_callback,
                        #amr_callback, 
                        stepsize_callback)

###############################################################################
# run the simulation

# positivity limiter necessary for this example with strong shocks
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-6, 5.0e-6),
                                                     variables = (Trixi.density, pressure))

sol = solve(ode, 
            SSPRK54(),
            #SSPRK54(stage_limiter! = stage_limiter!),
            dt = 1e-4, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary

using Plots
plot(sol)

pd = PlotData1D(sol)

plot(pd["rho"])

plot!(getmesh(pd))