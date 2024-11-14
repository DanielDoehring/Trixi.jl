using OrdinaryDiffEq
using Trixi

using NLsolve # Need nonlinear solver to solve for the IC/true solution

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

# TODO: Reference

prandtl_number() = 3/4
Re = 10  # Reynolds number
Ma = 2.5 # Mach number

gamma = 1.4

# TODO: Check if these lead to constant mass/enthalpy!
rho_ref = 1.0

L_ref = 1.0 # TODO: Not sure if this is the right reference length (= domain size)
Mass_flow = 0.1

U_L = 2.0
U_R = 1.0
V_f = U_L/U_R

c = max(U_L, U_R) / Ma
p_ref = rho_ref * c^2 / gamma

h_ref = p_ref / (gamma - 1) + 0.5 * U_L^2

mu() = rho_ref * U_L * L_ref / Re

alpha = 2 * gamma/(gamma + 1) * mu() / (prandtl_number() * Mass_flow)

function momemtum_ode_sol(V, x)
  x - 0.5 * alpha * (log(abs((V - 1) * (V - V_f))) + (1 + V_f)/(1 - V_f) * log(abs((V - 1)/(V - V_f))))
end

function dV(V)
  return - alpha * V[1] / ((V[1] - 1) * (V[1] - V_f))
end

coordinates_min = -1.0
coordinates_max = 1.0

function initial_condition_viscous_shock(x, t, equations)
  V_0 = (U_L + U_R) / 2
  # Turn into momentum
  V_0 *= rho_ref

  # TODO: Try out bisection for IC
  momentum = nlsolve(V -> momemtum_ode_sol(V[1], x[1]), 
                     dV, [V_0], 
                     ftol = 1e-14, iterations = 1000,
                     method = :newton).zero[1]

  rho = rho_ref

  v = momentum / rho

  p = (h_ref - 0.5 * v^2) * (gamma - 1)

  return prim2cons(SVector(rho, v, p), equations)
end

equations = CompressibleEulerEquations1D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion1D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_hllc,
               volume_integral = VolumeIntegralWeakForm())

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                periodicity = false,
                n_cells_max = 30_000) # set maximum capacity of tree data structure

initial_condition = initial_condition_viscous_shock

velocity_bc = NoSlip((x, t, equations) -> Trixi.velocity(initial_condition(x,
                                                                           t,
                                                                           equations),
                                                         equations_parabolic))

heat_bc = Isothermal((x, t, equations) -> Trixi.temperature(initial_condition(x,
                                                                              t,
                                                                              equations),
                                                                 equations_parabolic))

boundary_condition_parabolic = BoundaryConditionNavierStokesWall(velocity_bc, heat_bc)                                                                 

boundary_condition_dirichlet = BoundaryConditionDirichlet(initial_condition)

# define inviscid boundary conditions
boundary_conditions = (; x_neg = boundary_condition_dirichlet,
                       x_pos = boundary_condition_dirichlet)

# define viscous boundary conditions
boundary_conditions_parabolic = (; x_neg = boundary_condition_parabolic,
                                 x_pos = boundary_condition_parabolic)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

alive_callback = AliveCallback(alive_interval = 10)

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol, dt = 1e-5,
            ode_default_options()..., callback = callbacks)
summary_callback() # print the timer summary
