using OrdinaryDiffEq
using Trixi

# Ratio of specific heats
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

# Volume flux adds some (minimal) disspation, thus stabilizing the simulation -
# in contrast to standard DGSEM only
volume_flux = flux_ranocha
solver = DGSEM(polydeg = 3, surface_flux = flux_ranocha,
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
tspan = (0.0, 0.0)

# Mapping as described in https://arxiv.org/abs/2012.12040, but reduced to 2D
function mapping(xi_, eta_)
    exponent = 1.4
    
    #=
    # Apply a non-linear transformation to refine towards the center
    xi_transformed = sign(xi_) * abs(xi_)^(exponent + abs(xi_)) + 1
    eta_transformed = sign(eta_) * abs(eta_)^(exponent + abs(eta_)) + 1

    # Scale the transformed coordinates to maintain the original domain size
    x = xi_transformed * EdgeLength / 2
    y = eta_transformed * EdgeLength / 2
    =#

    # Compute the combined coordinate
    combined = (xi_ + eta_) / sqrt(2)

    # Apply a non-linear transformation to refine towards the line x = y
    combined_transformed = sign(combined) * abs(combined)^(exponent + abs(combined)) + 1

    # Scale the transformed combined coordinate to maintain the original domain size
    x = combined_transformed * EdgeLength / 2
    y = combined_transformed * EdgeLength / 2

    return SVector(x, y)
end

# For convergence test
cells_per_dimension = (10, 10) # Low resolution for convergence test

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

# For convergence test                                       
analysis_callback = AnalysisCallback(semi, interval = 1_000_000)

#cfl = 19.2 # Standalone
cfl = 5.0 # Multi # 8

stepsize_callback = StepsizeCallback(cfl = cfl)

alive_callback = AliveCallback(alive_interval = 100)

callbacks = CallbackSet(summary_callback,
                        #analysis_cb_entropy,
                        analysis_callback,
                        #stepsize_callback,
                        alive_callback)

###############################################################################
# run the simulation

#=
Stages_standalone = 16
ode_algorithm = Trixi.PairedExplicitERRK4(Stages_standalone,
                                          "/home/daniel/git/Paper-EntropyStabPERK/Data/IsentropicVortex_EC/")
=#

Stages = [16, 11, 9, 8, 7, 6, 5]
dtRatios = [
    0.636282563128043,
    0.412078842462506,
    0.31982226180844,
    0.26286863302812,
    0.22663973638555,
    0.160154267621692,
    0.130952239152975
] ./ 0.636282563128043


ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages,
                                             "/home/daniel/git/Paper-EntropyStabPERK/Data/IsentropicVortex_EC/",
                                             dtRatios)


#=
ode_algorithm = Trixi.PairedExplicitERRK4Multi(Stages,
                                               "/home/daniel/git/Paper-EntropyStabPERK/Data/IsentropicVortex_EC/",
                                               dtRatios)
=#

# Obtained from finest discretization
cfl_factor = 0.07439492193432441

h_min = -1
if cells_per_dimension[1] == 10
    h_min = 0.7614615754863507
elseif cells_per_dimension[1] == 20
    h_min = 0.3162277660168371
elseif cells_per_dimension[1] == 40
    h_min = 0.12986930197671676
elseif cells_per_dimension[1] == 80
    h_min = 0.05212721378246066
end

dt = cfl_factor * h_min

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
