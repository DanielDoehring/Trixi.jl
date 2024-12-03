
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

# Ratio of specific heats
gamma = 1.4

equations = CompressibleEulerEquations2D(gamma)

EdgeLength = 20.0

N_passes = 10
T_end = EdgeLength * N_passes
tspan = (0.0, T_end)

"""
    initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)

The classical isentropic vortex test case as presented in 
https://spectrum.library.concordia.ca/id/eprint/985444/1/Paired-explicit-Runge-Kutta-schemes-for-stiff-sy_2019_Journal-of-Computation.pdf
"""
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
    # Evaluate error after full domain traversion
    if t == T_end
        t = 0
    end

    # initial center of the vortex
    inicenter = SVector(0.0, 0.0)
    # strength of the vortex
    S = 13.5
    # Radius of vortex
    R = 1.5
    # Free-stream Mach 
    M = 0.4
    # base flow
    v1 = 1.0
    v2 = 1.0
    vel = SVector(v1, v2)

    cent = inicenter + vel * t      # advection of center
    cent = x - cent               # distance to centerpoint
    cent = SVector(cent[2], -cent[1])
    r2 = cent[1]^2 + cent[2]^2

    f = (1 - r2) / (2 * R^2)

    rho = (1 - (S * M / pi)^2 * (gamma - 1) * exp(2 * f) / 8)^(1 / (gamma - 1))

    du = S / (2 * π * R) * exp(f) # vel. perturbation
    vel = vel + du * cent
    v1, v2 = vel

    p = rho^gamma / (gamma * M^2)
    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_isentropic_vortex

surf_flux = flux_hllc # Better flux, allows much larger timesteps
PolyDeg = 3
solver = DGSEM(RealT = Float64, polydeg = PolyDeg, surface_flux = surf_flux)

coordinates_min = (-EdgeLength / 2, -EdgeLength / 2)
coordinates_max = (EdgeLength / 2, EdgeLength / 2)

Refinement = 5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = Refinement,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10^6
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = [:conservation_error],
                                     analysis_integrals = (;))

alive_callback = AliveCallback(alive_interval = 1000)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback)

###############################################################################
# run the simulation

dtRatios = [1, 0.5, 0.25]
path = "/home/daniel/git/Paper_PERRK/Data/IsentropicVortex/IsentropicVortex/k3/p4/"

Stages = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages, path, dtRatios)
#ode_algorithm = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios)

dt = 0.01

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep = false, callback = callbacks);

sol = solve(ode, SSPRK104(thread = OrdinaryDiffEq.True()),
            dt = dt, save_everystep = false, callback = callbacks,
            adaptive = false);

summary_callback() # print the timer summary
