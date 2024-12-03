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
    inicenter = SVector(EdgeLength / 2, EdgeLength / 2)
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

N_passes = 1
T_end = EdgeLength * N_passes
tspan = (0.0, T_end)

function mapping(xi_, eta_)
    exponent = 1.4

    # Apply a non-linear transformation to refine towards the center
    xi_transformed = sign(xi_) * abs(xi_)^(exponent + abs(xi_)) + 1
    eta_transformed = sign(eta_) * abs(eta_)^(exponent + abs(eta_)) + 1

    # Scale the transformed coordinates to maintain the original domain size
    x = xi_transformed * EdgeLength / 2
    y = eta_transformed * EdgeLength / 2

    return SVector(x, y)
end

cells_per_dimension = (16, 16)
mesh = StructuredMesh(cells_per_dimension, mapping)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10
analysis_cb_entropy = AnalysisCallback(semi, interval = analysis_interval,
                                       analysis_errors = [:conservation_error],
                                       analysis_integrals = (entropy,),
                                       analysis_filename = "analysis_standard.dat",
                                       #analysis_filename = "analysis_ER.dat",
                                       save_analysis = true)

# NOTE: Not really well-suited for convergence test                                       
analysis_callback = AnalysisCallback(semi, interval = 1_000_000,
                                     analysis_integrals = (;))

cfl = 5.0 # Multi # 8
cfl = 5.0 # p = S = 3

stepsize_callback = StepsizeCallback(cfl = cfl)

alive_callback = AliveCallback(alive_interval = 1000)

callbacks = CallbackSet(summary_callback,
                        #analysis_cb_entropy,
                        analysis_callback,
                        stepsize_callback,
                        alive_callback)

###############################################################################
# run the simulation

# p = 3
path = "/home/daniel/git/Paper_PERRK/Data/IsentropicVortex_EC/k2/"

Stages = [17, 13, 11, 9, 7, 6, 5, 4, 3]
dtRatios = [
    1.43509674072266,
    1.07526779174805,
    0.894473266601563,
    0.714339447021484,
    0.532713890075684,
    0.439394950866699,
    0.350951194763184,
    0.253698348999023,
    0.155333518981934
] / 1.43509674072266

ode_algorithm = Trixi.PairedExplicitRK3Multi(Stages, path, dtRatios)

#=
Stages = [16, 11, 9, 7, 6, 5]
dtRatios = [
    0.636282563128043,
    0.412078842462506,
    0.31982226180844,
    0.22663973638555,
    0.160154267621692,
    0.130952239152975
] ./ 0.636282563128043

#=
ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages,
                                             "/home/daniel/git/Paper_PEERRK/Data/IsentropicVortex_EC/",
                                             dtRatios)
=#

ode_algorithm = Trixi.PairedExplicitRelaxationRK4Multi(Stages,
                                               "/home/daniel/git/Paper_PEERRK/Data/IsentropicVortex_EC/",
                                               dtRatios)

=#

#=
_, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)

nnodes = length(dg.basis.nodes)
n_elements = nelements(dg, cache)
elements = cache.elements

h_min = floatmax(Float64)

for element_id in 1:n_elements
    # pull the four corners numbered as right-handed
    P0 = elements.node_coordinates[:, 1, 1, element_id]
    P1 = elements.node_coordinates[:, nnodes, 1, element_id]
    P2 = elements.node_coordinates[:, nnodes, nnodes, element_id]
    P3 = elements.node_coordinates[:, 1, nnodes, element_id]
    # compute the four side lengths and get the smallest
    L0 = sqrt(sum((P1 - P0) .^ 2))
    L1 = sqrt(sum((P2 - P1) .^ 2))
    L2 = sqrt(sum((P3 - P2) .^ 2))
    L3 = sqrt(sum((P0 - P3) .^ 2))
    h = min(L0, L1, L2, L3)

    if h < h_min
        global h_min = h
    end
end

dtRef = 0.03333333333333333
CFLREF = 2.383483940996325

dt = CFLREF * dtRef * h_min
=#

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
