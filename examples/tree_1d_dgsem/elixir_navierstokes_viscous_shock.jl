using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

# This is the classic 1D viscous shock wave problem with analytical solution 
# for a special value of the Prandtl number.
# The original references are:
#
# - R. Becker (1922)
#   Stoßwelle und Detonation.
#   [DOI: 10.1007/BF01329605](https://doi.org/10.1007/BF01329605)
#
#   English translations:
#   Impact waves and detonation. Part I.
#   https://ntrs.nasa.gov/api/citations/19930090862/downloads/19930090862.pdf
#
#   Impact waves and detonation. Part II.
#   https://ntrs.nasa.gov/api/citations/19930090863/downloads/19930090863.pdf
#
# - M. Morduchow, P. A. Libby (1949)
#   On a Complete Solution of the One-Dimensional Flow Equations 
#   of a Viscous, Head-Conducting, Compressible Gas
#   [DOI: 10.2514/8.11882](https://doi.org/10.2514/8.11882)
#
# The particular problem considered here is described in
# - L. G. Margolin, J. M. Reisner, P. M. Jordan (2017) 
#   Entropy in self-similar shock profiles
#   [DOI: 10.1016/j.ijnonlinmec.2017.07.003](https://doi.org/10.1016/j.ijnonlinmec.2017.07.003)

### Fixed parameters ###
# Special value for which nonlinear solver can be omitted
# Corresponds essentially to fixing the Mach number
alpha = 0.5
# We want kappa = cp * mu = mu_bar to ensure constant enthalpy
prandtl_number() = 1

### Free choices: ###
gamma = 5 / 3
rho_0 = 1
mu() = 0.1
v = 1 # Shock speed

domain_length = 5.0

### Derived quantities ###
Ma = 2 / sqrt(3 - gamma) # Mach number for alpha = 0.5
c_0 = v / Ma # Speed of sound ahead of the shock

# From constant enthalpy condition
p_0 = c_0^2 * rho_0 / gamma

l = mu() / (rho_0 * v) * 2 * gamma / (gamma + 1) # Appropriate lenght scale

# Helper function for coordinate transformation. See eq. (33) in Margolin et al. (2017)
chi_of_y(y) = 2 * exp(y / (2 * l))

"""
    initial_condition_viscous_shock(x, t, equations)

Classic 1D viscous shock wave problem with analytical solution 
for a special value of the Prandtl number.
The version implemented here is described in
- L. G. Margolin, J. M. Reisner, P. M. Jordan (2017)
  Entropy in self-similar shock profiles
  [DOI: 10.1016/j.ijnonlinmec.2017.07.003](https://doi.org/10.1016/j.ijnonlinmec.2017.07.003)
"""
function initial_condition_viscous_shock(x, t, equations)
    y = x[1] - v * t

    chi = chi_of_y(y)
    w = 1 + 1 / (2 * chi^2) * (1 - sqrt(1 + 2 * chi^2))

    rho = rho_0 / w
    u = v * (1 - w)
    p = p_0 * 1 / w * (1 + (gamma - 1) / 2 * Ma^2 * (1 - w^2))

    return prim2cons(SVector(rho, u, p), equations)
end

equations = CompressibleEulerEquations1D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion1D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())

solver = DGSEM(polydeg = 3, surface_flux = flux_hlle)

coordinates_min = -domain_length / 2
coordinates_max = domain_length / 2

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                periodicity = false,
                n_cells_max = 30_000)

initial_condition = initial_condition_viscous_shock

### Inviscid boundary conditions ###

# Prescribe pure influx based on initial conditions
function boundary_condition_inflow(u_inner, orientation, normal_direction, x, t,
                                   surface_flux_function, equations)
    u_cons = initial_condition(x, t, equations)
    flux = Trixi.flux(u_cons, normal_direction, equations)

    return flux
end

# Completely free outflow
function boundary_condition_outflow(u_inner, orientation, normal_direction, x, t,
                                    surface_flux_function, equations)
    # Calculate the boundary flux entirely from the internal solution state
    flux = Trixi.flux(u_inner, normal_direction, equations)

    return flux
end

boundary_conditions = (; x_neg = boundary_condition_inflow,
                       x_pos = boundary_condition_outflow)

### Viscous boundary conditions ###
# For the viscous BCs, we use the known analytical solution
velocity_bc = NoSlip((x, t, equations) -> Trixi.velocity(initial_condition(x,
                                                                           t,
                                                                           equations),
                                                         equations))

heat_bc = Isothermal((x, t, equations) -> Trixi.temperature(initial_condition(x,
                                                                              t,
                                                                              equations),
                                                            equations_parabolic))

boundary_condition_parabolic = BoundaryConditionNavierStokesWall(velocity_bc, heat_bc)

boundary_conditions_parabolic = (; x_neg = boundary_condition_parabolic,
                                 x_pos = boundary_condition_parabolic)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

alive_callback = AliveCallback(alive_interval = 100)

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            dt = 1e-5, ode_default_options()..., callback = callbacks)

summary_callback() # print the timer summary