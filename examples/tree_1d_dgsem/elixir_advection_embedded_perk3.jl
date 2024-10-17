
# Convex and ECOS are imported because they are used for finding the optimal time step and optimal 
# monomial coefficients in the stability polynomial of P-ERK time integrators.
using Convex, ECOS, Clarabel

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0 # minimum coordinate
coordinates_max = 1.0 # maximum coordinate

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000) # set maximum capacity of tree data structure

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 20.0
tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan);

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step

alive_callback = AliveCallback(alive_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 0.1,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver

# Construct embedded order paired explicit Runge-Kutta method with 10 stages and 7 evaluation stages for given simulation setup.
# Pass `tspan` to calculate maximum time step allowed for the bisection algorithm used 
# in calculating the polynomial coefficients in the ODE algorithm.
ode_algorithm = Trixi.EmbeddedPairedRK3(10, 10, tspan, semi)

# Calculate the CFL number for the given ODE algorithm and ODE problem (cfl_number calculate from dt_opt of the optimization of
# b values in the Butcher tableau of the ODE algorithm).
#cfl_number = Trixi.calculate_cfl(ode_algorithm, ode)
stepsize_callback = StepsizeCallback(cfl = 0.75) # Warisa: This number is quite small in contrast the other one from optimizing A
# I've tried using cfl of 1.5 and the error is very similar.

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        save_solution,
                        analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation
sol = Trixi.solve(ode, ode_algorithm,
                  dt = 1.0, # Manual time step value, will be overwritten by the stepsize_callback when it is specified.
                  save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()

# Some function defined so that I can check if the second order condition is met. This will be removed later.
function construct_b_vector(b_unknown, num_stages_embedded, num_stage_evals_embedded)
    # Construct the b vector
    b = [
        1 - sum(b_unknown),
        zeros(Float64, num_stages_embedded - num_stage_evals_embedded)...,
        b_unknown...,
        0
    ]
    return b
end

b = construct_b_vector(ode_algorithm.b, ode_algorithm.num_stages - 1,
                       ode_algorithm.num_stage_evals - 1)
println("dot(b, c) = ", dot(b, ode_algorithm.c))
println("sum(b) = ", sum(b))

println("dt_opt_a = ", ode_algorithm.dt_opt_a)
println("dt_opt_b = ", ode_algorithm.dt_opt_b)
println("ratio = ", ode_algorithm.dt_opt_b / ode_algorithm.dt_opt_a * 100)