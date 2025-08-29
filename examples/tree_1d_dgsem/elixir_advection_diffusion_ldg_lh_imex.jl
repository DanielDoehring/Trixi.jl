using Trixi

using NonlinearSolve, LinearSolve, ADTypes

# semidiscretization of the linear advection diffusion equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)
diffusivity() = 0.01 # 0.01
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

# `solver_float` is  used for the subsequent simulation
solver_float = DGSEM(polydeg = 5, surface_flux = flux_lax_friedrichs)

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

# `semi_float` is used for the subsequent simulation
semi_float = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                  initial_condition,
                                                  solver_float; solver_parabolic = ViscousFormulationLocalDG())

t0 = 0.0
t_end = 10.0
t_span = (t0, t_end)

#ode = semidiscretize(semi_float, t_span)
ode = semidiscretize(semi_float, t_span, split_problem = false)

###############################################################################
# ODE solvers, callbacks etc.

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi_float, interval = 100,
                                     extra_analysis_errors = (:conservation_error,))

callbacks = CallbackSet(summary_callback, analysis_callback)

###############################################################################
# run the simulation

# NOTE: If weird error messages show up (Intel mkl one) update packages, seemes to solve the thing

#ode_alg = Trixi.IMEX_LobattoIIIAp2_Heun()
#ode_alg = Trixi.IMEX_Midpoint_Midpoint()

#path = "/storage/home/daniel/PERRK/Data/IsentropicVortex/IsentropicVortex/k6/p2/"
path = "/home/daniel/git/Paper_PERRK/Data/IsentropicVortex/IsentropicVortex/k6/p2/"

#ode_alg = Trixi.PairedExplicitRK2Multi([12, 6, 2], path, [1, 1, 1])

#ode_alg = Trixi.PairedExplicitRK2IMEXMulti([12], path, [1])
ode_alg = Trixi.PairedExplicitRK2IMEXMulti([12, 6], path, [1, 1])

atol_lin = 1e-8
rtol_lin = 1e-6
#maxiters_lin = 50

linsolve = KrylovJL_GMRES(atol = atol_lin, rtol = rtol_lin)

# For Krylov.jl kwargs see https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/#Krylov.gmres
nonlin_solver = NewtonRaphson(autodiff = AutoFiniteDiff(), 
                              linsolve = linsolve)

atol_nonlin = atol_lin
rtol_nonlin = rtol_lin
maxiters_nonlin = 20

#dt = 2.0 / (2^3) # Operator-split RHS IMEX algorithms
dt = 0.01 / (2^3) # PERK IMEX test algorithms
integrator = Trixi.init(ode, ode_alg;
                        dt = dt, callback = callbacks,
                        # IMEX-specific kwargs
                        nonlin_solver = nonlin_solver,
                        abstol = atol_nonlin, reltol = rtol_nonlin,
                        maxiters_nonlin = maxiters_nonlin);

sol = Trixi.solve!(integrator);
