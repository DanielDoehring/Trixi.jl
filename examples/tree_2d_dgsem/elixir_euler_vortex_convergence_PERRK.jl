
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

# define new structs inside a module to allow re-evaluating the file
module TrixiExtension
using Trixi

struct IndicatorVortex{Cache <: NamedTuple} <: Trixi.AbstractIndicator
    cache::Cache
end

function IndicatorVortex(semi)
    basis = semi.solver.basis
    alpha = Vector{real(basis)}()
    A = Array{real(basis), 2}
    indicator_threaded = [A(undef, nnodes(basis), nnodes(basis))
                          for _ in 1:Threads.nthreads()]
    cache = (; semi.mesh, alpha, indicator_threaded) # "Leading semicolon" makes this a named tuple

    return IndicatorVortex{typeof(cache)}(cache)
end

function (indicator_vortex::IndicatorVortex)(u::AbstractArray{<:Any, 4},
                                             mesh, equations, dg, cache;
                                             t, kwargs...)
    mesh = indicator_vortex.cache.mesh
    alpha = indicator_vortex.cache.alpha
    indicator_threaded = indicator_vortex.cache.indicator_threaded
    resize!(alpha, nelements(dg, cache))

    # get analytical vortex center (based on assumption that center=[0.0,0.0]
    # at t=0.0 and that we stop after one period)
    domain_length = mesh.tree.length_level_0
    if t < 0.5 * domain_length
        center = (t, t)
    else
        center = (t - domain_length, t - domain_length)
    end

    Threads.@threads for element in eachelement(dg, cache)
        cell_id = cache.elements.cell_ids[element]
        coordinates = (mesh.tree.coordinates[1, cell_id], mesh.tree.coordinates[2, cell_id])
        # use the negative radius as indicator since the AMR controller increases
        # the level with increasing value of the indicator and we want to use
        # high levels near the vortex center
        alpha[element] = -periodic_distance_2d(coordinates, center, domain_length)
    end

    return alpha
end

function periodic_distance_2d(coordinates, center, domain_length)
    dx = @. abs(coordinates - center)
    dx_periodic = @. min(dx, domain_length - dx)
    return sqrt(sum(abs2, dx_periodic))
end
end # module TrixiExtension

import .TrixiExtension

# Ratio of specific heats
gamma = 1.4

equations = CompressibleEulerEquations2D(gamma)

EdgeLength = 20.0

N_passes = 1
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
PolyDeg = 6
solver = DGSEM(RealT = Float64, polydeg = PolyDeg, surface_flux = surf_flux)

coordinates_min = (-EdgeLength / 2, -EdgeLength / 2)
coordinates_max = (EdgeLength / 2, EdgeLength / 2)

Refinement = 6
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = Refinement,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

#=
analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (entropy,),
                                     save_analysis = true)
=#

analysis_interval = 10^6
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:l1_error,),
                                     analysis_integrals = (;))

amr_controller = ControllerThreeLevel(semi, TrixiExtension.IndicatorVortex(semi),
                                      base_level = Refinement,
                                      med_level = Refinement + 1, med_threshold = -3.0,
                                      max_level = Refinement + 2, max_threshold = -2.0)

N_Convergence = 0 # up to 7 for p2, 3 for p3/p4
CFL_Convergence = 1.0 / (2^N_Convergence)

amr_callback = AMRCallback(semi, amr_controller,
                           # For convergence study
                           interval = Int(50 / CFL_Convergence),
                           adapt_initial_condition = true)

alive_callback = AliveCallback(alive_interval = 100 * Int((2^N_Convergence)))

callbacks = CallbackSet(summary_callback,
                        amr_callback,
                        alive_callback,
                        analysis_callback)

###############################################################################
# run the simulation

dtRatios = [1, 0.5, 0.25]
#basepath = "/home/daniel/git/Paper_PERRK/Data/IsentropicVortex/IsentropicVortex/k6/"
basepath = "/storage/home/daniel/PERRK/Data/IsentropicVortex/IsentropicVortex/k6/"

newton = Trixi.RelaxationSolverNewton(max_iterations = 5, root_tol = 1e-13, gamma_tol = eps(Float64))

# p = 2

Stages = [12, 6, 3]
path = basepath * "p2/"
#ode_algorithm = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)
ode_algorithm = Trixi.PairedExplicitRelaxationRK2Multi(Stages, path, dtRatios,
                                                       relaxation_solver = newton)


# p = 3
#=
Stages = [16, 8, 4]
path = basepath * "p3/"
#ode_algorithm = Trixi.PairedExplicitRK3Multi(Stages, path, dtRatios)
ode_algorithm = Trixi.PairedExplicitRelaxationRK3Multi(Stages, path, dtRatios,
                                                       relaxation_solver = newton)
=#

# p = 4
#=
Stages = [15, 9, 5]
path = basepath * "p4/"
#ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages, path, dtRatios)
ode_algorithm = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios,
                                                       relaxation_solver = newton)
=#

dt = 0.004 * CFL_Convergence # Timestep in asymptotic regime

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep = false, callback = callbacks);
