using OrdinaryDiffEq
using Trixi

# Ratio of specific heats
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

polydeg = 3

# Volume flux adds some (minimal) disspation, thus stabilizing the simulation -
# in contrast to standard DGSEM only
volume_flux = flux_ranocha
solver = DGSEM(polydeg = polydeg, surface_flux = flux_ranocha,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

EdgeLength = 20.0
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

N_passes = 4
T_end = EdgeLength * N_passes
tspan = (0.0, T_end)

function mapping(xi_, eta_)
    exponent = 1.4

    # Apply a non-linear transformation to refine towards the center
    xi_transformed = sign(xi_) * abs(xi_)^(exponent + abs(xi_))
    eta_transformed = sign(eta_) * abs(eta_)^(exponent + abs(eta_))

    # Scale the transformed coordinates to maintain the original domain size
    x = xi_transformed * EdgeLength / 2
    y = eta_transformed * EdgeLength / 2

    return SVector(x, y)
end

cells_per_dimension = (32, 32)
mesh = StructuredMesh(cells_per_dimension, mapping)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 40
analysis_cb_entropy = AnalysisCallback(semi, interval = analysis_interval,
                                       analysis_errors = Symbol[],
                                       # Note: entropy defaults to mathematical entropy
                                       analysis_integrals = (entropy,),
                                       #analysis_filename = "entropy_standard.dat",
                                       analysis_filename = "entropy_ER.dat",
                                       save_analysis = true)

# NOTE: Not really well-suited for convergence test                                       
analysis_callback = AnalysisCallback(semi, interval = 1_000_000,
                                     analysis_errors = [:conservation_error],
                                     analysis_integrals = (;))

alive_callback = AliveCallback(alive_interval = 1000)

callbacks = CallbackSet(summary_callback,
                        analysis_cb_entropy,
                        #analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

basepath = "/home/daniel/git/Paper_PERRK/Data/IsentropicVortex/IsentropicVortex_EC/k3/"
relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 3, root_tol = 1e-14)

# p = 2
path = basepath * "p2/"

Stages = [16, 12, 10, 8, 6, 4]
dtRatios = [
    0.631627607345581,
    0.485828685760498,
    0.366690540313721,
    0.282330989837646,
    0.197234153747559,
    0.124999046325684
] ./ 0.631627607345581

#ode_algorithm = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)
ode_algorithm = Trixi.PairedExplicitRelaxationRK2Multi(Stages, path, dtRatios, relaxation_solver = relaxation_solver)

#=
# p = 3
path = basepath * "p3/"

Stages = [16, 12, 10, 9, 8, 7, 6, 5, 4]
dtRatios = [
    0.675333578553955,
    0.494580285519851,
    0.405339434131065,
    0.359326166425581,
    0.307904954277865,
    0.264428701231645,
    0.216773986597445,
    0.176284400979739,
    0.12732553184037
] ./ 0.675333578553955

#ode_algorithm = Trixi.PairedExplicitRK3Multi(Stages, path, dtRatios)
ode_algorithm = Trixi.PairedExplicitRelaxationRK3Multi(Stages, path, dtRatios, relaxation_solver = relaxation_solver)
=#

#=
# p = 4
path = basepath * "p4/"

Stages = [16, 11, 9, 7, 6, 5]
dtRatios = [
    0.636282563128043,
    0.412078842462506,
    0.31982226180844,
    0.22663973638555,
    0.160154267621692,
    0.130952239152975
] ./ 0.636282563128043

#ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages, path, dtRatios)
ode_algorithm = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios, relaxation_solver = relaxation_solver)
=#

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 7.25e-3,
                  save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
