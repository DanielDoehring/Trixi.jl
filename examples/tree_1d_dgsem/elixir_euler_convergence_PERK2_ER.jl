
using Convex, ECOS
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

prandtl_number() = 0.72
mu() = 6.25e-4 # equivalent to Re = 1600

equations = CompressibleEulerEquations1D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion1D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

# This convergence test setup was originally derived by Andrew Winters (@andrewwinters5000)
# (Simplified version of the 2D)
function initial_condition_navier_stokes_convergence_test(x, t, equations)
    # Amplitude and shift
    A = 0.5
    c = 2.0

    # convenience values for trig. functions
    pi_x = pi * x[1]
    pi_t = pi * t

    rho = c + A * sin(pi_x) * cos(pi_t)
    v1 = sin(pi_x) * cos(pi_t)
    p = rho^2

    return prim2cons(SVector(rho, v1, p), equations)
end
initial_condition = initial_condition_navier_stokes_convergence_test

@inline function source_terms_navier_stokes_convergence_test(u, x, t, equations)
    # we currently need to hardcode these parameters until we fix the "combined equation" issue
    # see also https://github.com/trixi-framework/Trixi.jl/pull/1160
    inv_gamma_minus_one = inv(equations.gamma - 1)
    Pr = prandtl_number()
    mu_ = mu()

    # Same settings as in `initial_condition`
    # Amplitude and shift
    A = 0.5
    c = 2.0

    # convenience values for trig. functions
    pi_x = pi * x[1]
    pi_t = pi * t

    # compute the manufactured solution and all necessary derivatives
    rho = c + A * sin(pi_x) * cos(pi_t)
    rho_t = -pi * A * sin(pi_x) * sin(pi_t)
    rho_x = pi * A * cos(pi_x) * cos(pi_t)
    rho_xx = -pi * pi * A * sin(pi_x) * cos(pi_t)

    v1 = sin(pi_x) * cos(pi_t)
    v1_t = -pi * sin(pi_x) * sin(pi_t)
    v1_x = pi * cos(pi_x) * cos(pi_t)
    v1_xx = -pi * pi * sin(pi_x) * cos(pi_t)

    p = rho * rho
    p_t = 2.0 * rho * rho_t
    p_x = 2.0 * rho * rho_x
    p_xx = 2.0 * rho * rho_xx + 2.0 * rho_x * rho_x

    E = p * inv_gamma_minus_one + 0.5 * rho * v1^2
    E_t = p_t * inv_gamma_minus_one + 0.5 * rho_t * v1^2 + rho * v1 * v1_t
    E_x = p_x * inv_gamma_minus_one + 0.5 * rho_x * v1^2 + rho * v1 * v1_x

    # Some convenience constants
    T_const = equations.gamma * inv_gamma_minus_one / Pr
    inv_rho_cubed = 1.0 / (rho^3)

    # compute the source terms
    # density equation
    du1 = rho_t + rho_x * v1 + rho * v1_x

    # x-momentum equation
    du2 = (rho_t * v1 + rho * v1_t
           + p_x + rho_x * v1^2 + 2.0 * rho * v1 * v1_x -
           # stress tensor from x-direction
           v1_xx * mu_)

    # total energy equation
    du3 = (E_t + v1_x * (E + p) + v1 * (E_x + p_x) -
           # stress tensor and temperature gradient terms from x-direction
           v1_xx * v1 * mu_ -
           v1_x * v1_x * mu_ -
           T_const * inv_rho_cubed *
           (p_xx * rho * rho -
            2.0 * p_x * rho * rho_x +
            2.0 * p * rho_x * rho_x -
            p * rho * rho_xx) * mu_)

    return SVector(du1, du2, du3)
end

volume_flux = flux_ranocha
solver = DGSEM(polydeg = 3, surface_flux = flux_hllc,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver,
                                             source_terms = source_terms_navier_stokes_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 20.0
tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan);

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(alive_interval = analysis_interval)

ode_algorithm = Trixi.PairedExplicitRK2(6, tspan, semi)

cfl_number = Trixi.calculate_cfl(ode_algorithm, ode)
stepsize_callback = StepsizeCallback(cfl = 0.5 * cfl_number)

entropy_relaxation_callback = EntropyRelaxationCallback()

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        stepsize_callback,
                        #entropy_relaxation_callback
                        )

###############################################################################
# run the simulation
sol = Trixi.solve(ode, ode_algorithm,
                  dt = 1.0, # Manual time step value, will be overwritten by the stepsize_callback when it is specified.
                  save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()
