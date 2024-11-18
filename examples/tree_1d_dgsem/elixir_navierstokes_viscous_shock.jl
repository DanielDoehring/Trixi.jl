using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

# TODO: Reference for Becker-Morduchow-Libby solution

# "Physics"
gamma = 1.4
prandtl_number() = 3/4 # Strictly required for this testcase!

# "Choices"
u_0 = 1.0
D = u_0

rho_0 = 1.0
# p_0 = 1.0
p_0 = rho_0^gamma

Ma_0 = 2.0
c_0 = u_0 / Ma_0 # Not needed at the moment

eta_1 = (gamma - 1)/(gamma + 1) + 2/((gamma + 1) * Ma_0^2)

mu() = 0.1

# This is valid for constant viscosity mu
function xi_of_x(x)
  3 * (gamma + 1) / (8 * gamma) * rho_0 * D/mu() * x
end

function eta_root_of_x(eta, x)
  xi = xi_of_x(x)
  eta - 1 + ((1 - eta_1)/2)^(1 - eta_1) * exp((1 - eta_1) * xi) * (eta - eta_1)^eta_1
end

function eta_bisection(eta_0, eta_1, x, eta_tol=1e-15)
  @assert eta_root_of_x(eta_0, x) * eta_root_of_x(eta_1, x) < 0

  eta_max = eta_0
  eta_min = eta_1

  eta = (eta_max + eta_min) / 2

  while eta_max - eta_min > eta_tol
    if eta_root_of_x(eta, x) < 0
      eta_min = eta
    else
      eta_max = eta
    end

    eta = (eta_max + eta_min) / 2
  end

  return eta
end

Length = 2.0
coordinates_min = -Length/2
coordinates_max = Length/2

function initial_condition_viscous_shock(x, t, equations)
  eta_sol = eta_bisection(1.0, eta_1, x[1])

  rho = rho_0 / eta_sol

  v = eta_sol * D
  
  p = p_0 * 1/eta_sol * (1 + (gamma - 1)/2 * Ma_0^2 * (1 - eta_sol^2))

  return prim2cons(SVector(rho, v, p), equations)
end

equations = CompressibleEulerEquations1D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion1D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())

solver = DGSEM(polydeg = 3, surface_flux = flux_hlle,
               volume_integral = VolumeIntegralWeakForm())

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                periodicity = false,
                n_cells_max = 30_000) # set maximum capacity of tree data structure

initial_condition = initial_condition_viscous_shock

boundary_condition_dirichlet = BoundaryConditionDirichlet(initial_condition)
# define inviscid boundary conditions

boundary_conditions = (; x_neg = boundary_condition_dirichlet,
                       x_pos = boundary_condition_dirichlet)

velocity_bc = NoSlip((x, t, equations) -> Trixi.velocity(initial_condition(x,
                                                                           t,
                                                                           equations), 
                                                         equations))


heat_bc = Isothermal((x, t, equations) -> Trixi.temperature(initial_condition(x,
                                                                              t,
                                                                              equations),
                                                            equations_parabolic))

boundary_condition_parabolic = BoundaryConditionNavierStokesWall(velocity_bc, heat_bc)

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
tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

alive_callback = AliveCallback(alive_interval = 100)

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

visualization = VisualizationCallback(interval=10, plot_data_creator=PlotData1D)

callbacks = CallbackSet(summary_callback, 
                        alive_callback, 
                        #visualization,
                        analysis_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol, dt = 1e-5,
            ode_default_options()..., callback = callbacks)
summary_callback() # print the timer summary