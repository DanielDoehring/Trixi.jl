using Trixi

using LinearSolve
using LineSearch, NonlinearSolve

# Ratio of specific heats
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

solver = DGSEM(polydeg = 3, surface_flux = flux_hll,
               volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

EdgeLength() = 20.0
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
T_end = EdgeLength() * N_passes
tspan = (0.0, T_end)

function mapping(xi_, eta_)
    exponent = 1.4

    # Apply a non-linear transformation to refine towards the center
    xi_transformed = sign(xi_) * abs(xi_)^(exponent + abs(xi_))
    eta_transformed = sign(eta_) * abs(eta_)^(exponent + abs(eta_))

    # Scale the transformed coordinates to maintain the original domain size
    #x = xi_transformed * EdgeLength() / 2
    x = xi_transformed * 10

    #y = eta_transformed * EdgeLength() / 2
    y = eta_transformed * 10

    return SVector(x, y)
end

cells_per_dimension = (32, 32)
mesh = StructuredMesh(cells_per_dimension, mapping)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

# NOTE: Not really well-suited for convergence test
analysis_callback = AnalysisCallback(semi, interval = 1000,
                                     extra_analysis_errors = (:conservation_error,),
                                     analysis_integrals = (;))

alive_callback = AliveCallback(alive_interval = 100)
                                     
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

basepath = "/home/daniel/git/Paper_PERRK/Data/IsentropicVortex/IsentropicVortex_EC/k3/"

# p = 2
path = basepath * "p2/"

Stages = [16, 12, 10, 8, 6, 4]

timestep_explicit = 0.631627607345581

dtRatios = [
    0.631627607345581,
    0.485828685760498,
    0.366690540313721,
    0.282330989837646,
    0.197234153747559,
    0.124999046325684
] ./ 0.631627607345581

dt_explicit = 5e-3

#=
ode_algorithm = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt_explicit,
                  save_everystep = false, callback = callbacks);
=#
###############################################################################

timestep_implicit = 0.8

dtRatios = [
    timestep_implicit, # Implicit
    0.631627607345581,
    0.485828685760498,
    0.366690540313721,
    0.282330989837646,
    0.197234153747559,
    0.124999046325684
] ./ timestep_implicit

ode_alg = Trixi.PairedExplicitRK2IMEXMulti(Stages, path, dtRatios)

### Linesearch ###
# See https://docs.sciml.ai/LineSearch/dev/api/native/

#linesearch = BackTracking(autodiff = AutoFiniteDiff(), order = 3, maxstep = 10)
#linesearch = LiFukushimaLineSearch()
linesearch = nothing

### Linear Solver ###
# See https://docs.sciml.ai/LinearSolve/stable/solvers/solvers/

#linsolve = SimpleLUFactorization()

# Require sparse matrix
#linsolve = KLUFactorization()
#linsolve = UMFPACKFactorization()

#linsolve = SimpleGMRES()
linsolve = KrylovJL_GMRES()

nonlin_solver = NewtonRaphson(autodiff = AutoFiniteDiff(),
                              linesearch = linesearch, linsolve = linsolve)

#nonlin_solver = Broyden(autodiff = AutoFiniteDiff(), linesearch = linesearch)
#nonlin_solver = DFSane()
# Could also check the advanced solvers: https://docs.sciml.ai/NonlinearSolve/stable/native/solvers/#Advanced-Solvers


dt_implicit = dt_explicit * timestep_implicit / timestep_explicit
dt_implicit = 8e-3

#=
t_ramp_up() = 5.0

cfl_0() = 1.0
cfl_max() = 2.0

cfl(t) = min(cfl_max(), cfl_0() + t/t_ramp_up() * (cfl_max() - cfl_0()))

stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        stepsize_callback,
                        analysis_callback,
                        alive_callback)
=#

integrator = Trixi.init(ode, ode_alg; dt = dt_implicit, callback = callbacks,
                        nonlin_solver = nonlin_solver,
                        abstol = 1e-5, reltol = 1e-3,
                        maxiters_nonlin = 20); # Maxiters should be on the order of the number of stages of the highest explicit method

sol = Trixi.solve!(integrator);