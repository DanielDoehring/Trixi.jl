using OrdinaryDiffEq
using Trixi
using LinearAlgebra:norm

###############################################################################
# semidiscretization of the linearized Euler equations

equations = LinearizedEulerEquationsVarSoS3D(v_mean_global = (1.0, 1.0, 1.0), rho_mean_global = 1.0)

function initial_condition_acoustic_wave(x, t, equations::LinearizedEulerEquationsVarSoS3D)
  # Parameters
  alpha = 1.0
  beta = 30.0

  # Distance from center of domain
  dist = norm(x)

  # Clip distance at corners
  if dist > 1.0
    dist = 1.0
  end

  c_mean = 10.0 - 9.0 * dist

  v1_prime = alpha * exp(-beta * (x[1]^2 + x[2]^2 + x[3]^2))
  
  #rho_prime = -v1_prime / c_mean
  rho_prime = -v1_prime

  v2_prime = v1_prime
  
  v3_prime = v1_prime

  #p_prime = -c_mean * v1_prime
  p_prime = -v1_prime

  return SVector(rho_prime, v1_prime, v2_prime, v3_prime, p_prime, c_mean)
end

initial_condition = initial_condition_acoustic_wave

solver = DGSEM(polydeg = 3, surface_flux = flux_hll)

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 300_000)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 0.2
tspan = (0.0, 1.0) 
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

analysis_interval = 5000 # High enough to just eval at end

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = [])

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(alive_interval = 50)

# Use CFL callback only for finding timestep, then turn off (linear equation)
stepsize_callback = StepsizeCallback(cfl = 6.0) # PERK4_Multi
#stepsize_callback = StepsizeCallback(cfl = 7.5) # PERK4_Single S 15

#stepsize_callback = StepsizeCallback(cfl = 1.7) # CarpenterKennedy2N54
#stepsize_callback = StepsizeCallback(cfl = 6.8) # NDBLSRK144
#stepsize_callback = StepsizeCallback(cfl = 7.6) # ParsaniKetchesonDeconinck3S184

#stepsize_callback = StepsizeCallback(cfl = 1.05) # RK4

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, 
                        #stepsize_callback,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation

Stages = [15, 14, 13, 12, 11, 10, 9, 7, 6, 5]
corr_c = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

ode_algorithm = PERK4_Multi(Stages, "/home/daniel/git/MA/EigenspectraGeneration/PERK4/LinEuler2D_var_c/", corr_c)
#ode_algorithm = PERK4(15, "/home/daniel/git/MA/EigenspectraGeneration/PERK4/LinEuler2D_var_c/")

sol = Trixi.solve(ode, ode_algorithm, 
                  dt = 1.4205e-03, # Multi PERK
                  #dt = 1.77556818e-03, # Single PERK 15
                  save_everystep = false, callback = callbacks);


         
# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false, thread = OrdinaryDiffEq.True()),
            dt = 4.0246e-04, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

sol = solve(ode, NDBLSRK144(williamson_condition = false, thread = OrdinaryDiffEq.True()),
            dt = 1.6098e-03, 
            save_everystep = false, callback = callbacks);

sol = solve(ode, ParsaniKetchesonDeconinck3S184(thread = OrdinaryDiffEq.True()),
            dt = 1.32575758e-03, 
            save_everystep = false, callback = callbacks);

sol = solve(ode, RK4(thread = OrdinaryDiffEq.True()),
            dt = 2.4272e-04, 
            adaptive = false, # Ref level = 6 
            save_everystep = false, callback = callbacks);


# print the timer summary
summary_callback() # print the timer summary