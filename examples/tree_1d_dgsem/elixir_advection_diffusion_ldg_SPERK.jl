using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)
diffusivity() = 0.015
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -convert(Float64, pi)
coordinates_max = convert(Float64, pi)

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

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 100)

# p = 2
cfl = 2.5 # Stable for a = 1.0, d = 0.01 with advection-only optimized spectra
cfl = 2.5 # Stable for a = 1.0, d = 0.02 with advection & diffusion optimized spectra

# p = 3
cfl = 2.5 # Stable for a = 1.0, d = 0.01 with advection-only optimized spectra
cfl = 2.5 # Stable for a = 1.0, d = 0.015 with advection & diffusion optimized spectra

stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

path = "/home/daniel/git/Paper_Split_IMEX_PERK/Data/Spectra/1D_Advection/"

#path_para = "/home/daniel/git/Paper_Split_IMEX_PERK/Data/Spectra/1D_Advection/" # For test only
path_para = "/home/daniel/git/Paper_Split_IMEX_PERK/Data/Spectra/1D_Diffusion/"

Stages = 8

#ode_alg = Trixi.PairedExplicitRK2(Stages, path * "p2/")
#ode_alg = Trixi.PairedExplicitRKSplit2(Stages, path * "p2/", path_para * "p2/")

ode_alg = Trixi.PairedExplicitRK3(Stages, path * "p3/")
ode_alg = Trixi.PairedExplicitRKSplit3(Stages, path * "p3/", path_para * "p3/")

sol = Trixi.solve(ode, ode_alg,
                  dt = 0.2,
                  save_everystep = false, callback = callbacks);