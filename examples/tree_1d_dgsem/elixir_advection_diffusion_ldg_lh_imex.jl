using Trixi

###############################################################################
# semidiscretization of the linear advection diffusion equation

advection_velocity = 0.01
equations = LinearScalarAdvectionEquation1D(advection_velocity)
diffusivity() = 0.5
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -convert(Float64, pi)
coordinates_max = convert(Float64, pi)

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000)

function x_trans_periodic(x, domain_length = SVector(oftype(x[1], 2 * pi)),
                          center = SVector(oftype(x[1], 0)))
    x_normalized = x .- center
    x_shifted = x_normalized .% domain_length
    x_offset = ((x_shifted .< -0.5f0 * domain_length) -
                (x_shifted .> 0.5f0 * domain_length)) .*
               domain_length
    return center + x_shifted + x_offset
end

# Define initial condition
function initial_condition_diffusive_convergence_test(x, t,
                                                      equation::LinearScalarAdvectionEquation1D)
    # Store translated coordinate for easy use of exact solution
    x_trans = x_trans_periodic(x - equation.advection_velocity * t)

    nu = diffusivity()
    c = 0
    A = 1
    omega = 1
    scalar = c + A * sin(omega * sum(x_trans)) * exp(-nu * omega^2 * t)
    return SVector(scalar)
end
initial_condition = initial_condition_diffusive_convergence_test

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition,
                                             solver; solver_parabolic = ViscousFormulationLocalDG())

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan, split_problem = true)

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 100)

callbacks = CallbackSet(summary_callback, analysis_callback)

###############################################################################
# run the simulation

ode_alg = Trixi.LobattoIIIA_p2_Heun()
sol = Trixi.solve(ode, ode_alg, dt = 0.3,
                  save_everystep = false, callback = callbacks);
