using OrdinaryDiffEq
using Trixi

# Ratio of specific heats
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

polydeg = 2
# For some reason we lose an order of convergence when using this solver,
# when being used with an ODD polynomial degree
solver = DGSEM(polydeg = polydeg, surface_flux = flux_ranocha,
               volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))


#solver = DGSEM(polydeg = polydeg, surface_flux = flux_hllc)

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

N_passes = 1
T_end = EdgeLength * N_passes
tspan = (0.0, T_end)
#tspan = (0.0, 2.0)

coordinates_min = (-EdgeLength/2, -EdgeLength/2)
coordinates_max = (EdgeLength/2, EdgeLength/2)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 100_000)

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

analysis_callback = AnalysisCallback(semi, interval = 10_000,
                                     analysis_integrals = (;))

cfl = 2.0 # S = 5 # 1.0 for HLLC convergence
cfl = 1.5 # S = 4, p = 3

stepsize_callback = StepsizeCallback(cfl = cfl)

alive_callback = AliveCallback(alive_interval = 1000)

callbacks = CallbackSet(summary_callback,
                        #analysis_cb_entropy,
                        analysis_callback,
                        stepsize_callback,
                        alive_callback)

###############################################################################
# run the simulation

Stages = 5

# p = 4
path = "/home/daniel/git/MA/EigenspectraGeneration/PERK4/IsentropicVortex_c1/"
#path = "/home/daniel/git/Paper_PEERRK/Data/IsentropicVortex_EC/"

# p = 3
path = "/home/daniel/git/Paper_AMR_PERK/Data/Isentropic_Vortex/PolyDeg3/"

#ode_algorithm = Trixi.PairedExplicitRK4(Stages, path)
#ode_algorithm = Trixi.PairedExplicitERRK4(Stages, path)

# NOTE: Multi on uniform mesh: For random distribution of methods

dtRatios = [1, 0.5, 0.25, 0.125]
Stages = [19, 11, 7, 5]

dtRatios = [1, 0.5, 0.25]
Stages = [11, 7, 5]

Stages = [16, 8, 4]

#ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages, path, dtRatios)
#ode_algorithm = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios)                                             

#ode_algorithm = Trixi.PairedExplicitRK3(7, path)
ode_algorithm = Trixi.PairedExplicitRK3Multi(Stages, path, dtRatios)


sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep = false, callback = callbacks);


#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false, thread = OrdinaryDiffEq.True()),
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks);
=#

summary_callback() # print the timer summary
