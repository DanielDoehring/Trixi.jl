
using OrdinaryDiffEq, Plots
using Trixi

using DoubleFloats

###############################################################################
# semidiscretization of the compressible Euler equations

# Ratio of specific heats
gamma = 1.4

equations = CompressibleEulerEquations3D(gamma)

EdgeLength = 5

N_passes = 1
T_end = 2*EdgeLength * N_passes
tspan = (0.0, T_end)

"""
    initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations3D)

Condition from: Relaxation Runge-Kutta Methods: Entropy Stability
# TODO: Something is wrong with this IC!
"""
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations3D)
  # Evaluate error after full domain traversion
  if t == T_end
    t = 0
  end
  
  # Free-stream values
  c = sqrt(1.4) # T_inf = 1
  M = 0.5
  U = M * c
  alpha = deg2rad(45.0)

  G = 1 - ((x[1] - U * cos(alpha) * t)^2 + (x[2] - U * sin(alpha) * t)^2)

  eps_v = 5
  T = 1 - (eps_v * M / pi)^2 * 0.4/8 * exp(G)

  rho = T^(1/0.4)

  v1 = U * cos(alpha) - eps_v * (x[2] - U * sin(alpha) * t) * exp(G/2) / (2 * pi)
  v2 = U * sin(alpha) - eps_v * (x[1] - U * cos(alpha) * t) * exp(G/2) / (2 * pi)
  v3 = 0.0

  p = T * rho

  prim = SVector(rho, v1, v2, v3, p)
  return prim2cons(prim, equations)
end
initial_condition = initial_condition_isentropic_vortex

# For real entropy conservation!
volume_flux = flux_ranocha
solver = DGSEM(RealT = Float64, polydeg = 3, surface_flux = flux_ranocha,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)               

coordinates_min = (-EdgeLength, -EdgeLength, -EdgeLength)
coordinates_max = ( EdgeLength,  EdgeLength, EdgeLength)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 20
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     #analysis_errors = Symbol[],
                                     analysis_integrals = (entropy,),
                                     save_analysis = true)

### AMR, Ref_Lvl = 6, Standard DGSEM ###

stepsize_callback = StepsizeCallback(cfl = 0.3) 
callbacks = CallbackSet(summary_callback,
                        #amr_callback, # Not sure if AMR is entropy stable
                        stepsize_callback,
                        analysis_callback)                

###############################################################################
# run the simulation

sol = solve(ode, SSPRK33(), dt = 42.0, save_everystep = false, callback=callbacks);

Stages = 5

#ode_algorithm = PERK4_ER(Stages, "/home/daniel/git/MA/EigenspectraGeneration/PERK4/IsentropicVortex_c1/")
ode_algorithm = PERK4(Stages, "/home/daniel/git/MA/EigenspectraGeneration/PERK4/IsentropicVortex_c1/")

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
