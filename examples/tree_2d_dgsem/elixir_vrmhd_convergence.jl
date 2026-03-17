using Trixi
using OrdinaryDiffEqLowStorageRK

###############################################################################
# Semidiscretization of the compressible Euler equations

# Fluid parameters
gamma() = 2.0
prandtl_number() = 0.72

mu() = 5e-3
eta() = 5e-3

equations = IdealGlmMhdEquations2D(gamma())
equations_parabolic = ViscoResistiveMhdDiffusion2D(equations, mu = mu(),
                                                   Prandtl = prandtl_number(),
                                                   eta = eta())

@inline function initial_condition_convergence(x, t, equations)
    h = 0.5 * sinpi(2 * (x[1] + x[2] - t)) + 2

    u_1 = h
    u_2 = h
    u_3 = h
    u_4 = 0.0
    u_5 = 2 * h^2 + h

    u_6 = h
    u_7 = -h
    u_8 = 0.0
    u_9 = 0.0

    return SVector(u_1, u_2, u_3, u_4, u_5, u_6, u_7, u_8, u_9)
end

function source_terms_convergence(u, x, t, equations)
    h = 0.5 * sinpi(2 * (x[1] + x[2] - t)) + 2
    h_x = pi * cospi(2 * (x[1] + x[2] - t))
    h_xx = -2 * pi^2 * sinpi(2 * (x[1] + x[2] - t))

    s_1 = h_x
    s_2 = h_x + 4 * h * h_x
    s_3 = h_x + 4 * h * h_x
    s_4 = 4 * h * h_x
    s_5 = h_x + 12 * h * h_x - 6 * (eta() * (h_x^2 + h * h_xx) + mu() * h_xx / prandtl_number())
    s_6 = h_x - 3 * eta() * h_xx
    s_7 = -h_x + 3 * eta() * h_xx
    s_8 = 0.0
    s_9 = 0.0

    return SVector(s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8, s_9)
end

surface_flux = (flux_hll, flux_nonconservative_powell)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                periodicity = true,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition_convergence, solver;
                                             source_terms = source_terms_convergence,
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 50
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = 0.1
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);