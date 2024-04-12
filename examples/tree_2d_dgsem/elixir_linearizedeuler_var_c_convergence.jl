using OrdinaryDiffEq
using Trixi
using LinearAlgebra:norm

###############################################################################
# semidiscretization of the linearized Euler equations

equations = LinearizedEulerEquationsVarSoS2D(v_mean_global = (1.0, 1.0), rho_mean_global = 1.0)

function initial_condition_acoustic_wave(x, t, equations::LinearizedEulerEquationsVarSoS2D)
  # Parameters
  alpha = 1.0
  beta = 30.0

  # Distance from center of domain
  dist = norm(x)

  # Clip distance at corners
  if dist > 1.0
    dist = 1.0
  end

  c_mean = 5.0 - 4.0 * dist

  v1_prime = alpha * exp(-beta * (x[1]^2 + x[2]^2))
  
  #rho_prime = -v1_prime / c_mean
  rho_prime = -v1_prime

  v2_prime = alpha * exp(-beta * (x[1]^2 + x[2]^2))
  
  #p_prime = -c_mean * v1_prime
  p_prime = -v1_prime

  return SVector(rho_prime, v1_prime, v2_prime, p_prime, c_mean)
end

initial_condition = initial_condition_acoustic_wave

solver = DGSEM(polydeg = 3, surface_flux = flux_hll)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                n_cells_max = 30_000)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 0.2
tspan = (0.0, 0.5) 
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

analysis_interval = 100

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval = analysis_interval)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.8)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

# print the timer summary
summary_callback() # print the timer summary

using Plots
plot(sol)