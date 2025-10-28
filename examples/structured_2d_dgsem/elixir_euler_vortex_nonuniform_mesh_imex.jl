using Trixi

using NonlinearSolve, LinearSolve, ADTypes

using SparseConnectivityTracer # For obtaining the Jacobian sparsity pattern
using SparseMatrixColorings # For obtaining the coloring vector
using JLD2 # For loading/storing sparsity info

# Ratio of specific heats
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

solver = DGSEM(polydeg = 3, surface_flux = flux_hll,
               volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

# For SD: a flux without if-clauses
solver_SD = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
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

t0 = 0.0
N_passes = 1
T_end = EdgeLength() * N_passes
t_span = (t0, T_end)

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

jac_detector = TracerSparsityDetector()
# We need to construct the semidiscretization with the correct
# sparsity-detection ready datatype, which is retrieved here
jac_eltype = jacobian_eltype(real(solver), jac_detector)


@inline function Trixi.ln_mean(x::SparseConnectivityTracer.AbstractTracer, y::SparseConnectivityTracer.AbstractTracer)
    return (y - x) / log(y / x)
end

@inline function Trixi.inv_ln_mean(x::SparseConnectivityTracer.AbstractTracer, y::SparseConnectivityTracer.AbstractTracer)
    return log(y / x) / (y - x)
end


# For sparsity detection
semi_SD = SemidiscretizationHyperbolic(mesh, equations,
                                       initial_condition, solver_SD;
                                       uEltype = jac_eltype)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, t_span)

#=
ode_SD = semidiscretize(semi_SD, t_span)

u = copy(ode_SD.u0)
du = zero(u)
u_tmp = zero(u)

k1 = zero(u) # Additional PERK register

semi = ode_SD.p
cache = semi_SD.cache
=#

Stages = [16, 12, 10, 8, 6, 4]

basepath = "/home/daniel/git/Paper_PERRK/Data/IsentropicVortex/IsentropicVortex_EC/k3/"

# p = 2
path = basepath * "p2/"

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

#=
n_levels = Trixi.get_n_levels(mesh, ode_alg)

level_info_elements = [Vector{Int64}() for _ in 1:n_levels]
level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]

level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]

level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]

level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

Trixi.partition_variables!(level_info_elements,
                           level_info_elements_acc,
                           level_info_interfaces_acc,
                           level_info_boundaries_acc,
                           level_info_mortars_acc,
                           n_levels, mesh, solver, cache, ode_alg)

level_info_u = [Vector{Int64}() for _ in 1:n_levels]

level_info_u_acc = [Vector{Int64}() for _ in 1:n_levels]
Trixi.partition_u!(level_info_u, level_info_u_acc,
                   level_info_elements, n_levels,
                   u, mesh, equations, solver, cache)

# For fixed meshes/no re-partitioning: Allocate only required storage
R = ode_alg.num_methods
u_implicit = level_info_u[R]
N_nonlin = length(u_implicit)
k_nonlin = zeros(eltype(u), N_nonlin)
residual = copy(k_nonlin)

dt = 8e-3
t0 = t_span[1]

p = Trixi.NonlinParams{typeof(t0), typeof(u),
                    typeof(semi), typeof(ode.f)}(t0, dt,
                                                u, du, u_tmp,
                                                semi, ode.f,
                                                level_info_elements_acc[R],
                                                level_info_interfaces_acc[R],
                                                level_info_boundaries_acc[R],
                                                level_info_mortars_acc[R],
                                                level_info_u[R])

res_wrapped! = (residual, k_nonlin) -> Trixi.residual_S_PERK2IMEXMulti!(residual, k_nonlin, p)

ode_SD.f(du, u, semi_SD, t0) # Do one step to prevent e.g. undefined BCs
jac_prototype = jacobian_sparsity(res_wrapped!, residual, k_nonlin, jac_detector)

coloring_prob = ColoringProblem(; structure = :nonsymmetric, partition = :column)
coloring_alg = GreedyColoringAlgorithm(; decompression = :direct)
coloring_result = coloring(jac_prototype, coloring_prob, coloring_alg)
colorvec = column_colors(coloring_result)

maximum(colorvec) + 1 # Number RHS evaluations to get full Jacobian

# Store the sparsity information
jldopen("/home/daniel/git/DissDoc/Data/IMEX_Sparse_Frozen_Jacobian/sparsity_info.jld2", "w") do file
    file["jac_prototype"] = jac_prototype
    file["colorvec"] = colorvec  # Also store the coloring vector
end
=#

# Load the sparsity information
jac_prototype, colorvec = jldopen("/home/daniel/git/DissDoc/Data/IMEX_Sparse_Frozen_Jacobian/sparsity_info.jld2", "r") do file
    return file["jac_prototype"], file["colorvec"]
end

summary_callback = SummaryCallback()

# NOTE: Not really well-suited for convergence test
analysis_callback = AnalysisCallback(semi, interval = 10_000,
                                     extra_analysis_errors = (:conservation_error,),
                                     analysis_integrals = (;))

alive_callback = AliveCallback(alive_interval = 100)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

#=
timestep_explicit = 0.631627607345581

dtRatios = [
    0.631627607345581,
    0.485828685760498,
    0.366690540313721,
    0.282330989837646,
    0.197234153747559,
    0.124999046325684
] ./ 0.631627607345581

dt_explicit = 5e-3 # very close to max dt for HLL


ode_algorithm = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt_explicit,
                  save_everystep = false, callback = callbacks);
=#

###############################################################################

### Frozen Sparse Jacobian setup ###
maxiters_nonlin = 20
abstol = 1e-5 # Should suffice (compare initial errors). 1e-4 gives same behaviour

dt_implicit = 8e-3 # 5e-3 yields identical errors to explicit solve
integrator = Trixi.init(ode, ode_alg;
                        dt = dt_implicit, callback = callbacks,
                        # IMEX-specific kwargs
                        jac_prototype = jac_prototype,
                        colorvec = colorvec,
                        maxiters_nonlin = maxiters_nonlin,
                        abstol = abstol);

sol = Trixi.solve!(integrator);

### Jacobian-Free Newton-Krylov solver setup ###

#=
atol_lin = 1e-5
rtol_lin = 1e-3
#maxiters_lin = 50

linsolve = KrylovJL_GMRES(atol = atol_lin, rtol = rtol_lin)

# For Krylov.jl kwargs see https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/#Krylov.gmres
nonlin_solver = NewtonRaphson(autodiff = AutoFiniteDiff(),
                              linsolve = linsolve)

atol_nonlin = atol_lin
rtol_nonlin = rtol_lin
maxiters_nonlin = 20

dt_implicit = dt_explicit * timestep_implicit / timestep_explicit
dt_implicit = 8e-3
integrator = Trixi.init(ode, ode_alg;
                        dt = dt_implicit, callback = callbacks,
                        # IMEX-specific kwargs
                        nonlin_solver = nonlin_solver,
                        abstol = atol_nonlin, reltol = rtol_nonlin,
                        maxiters_nonlin = maxiters_nonlin);

sol = Trixi.solve!(integrator);
=#