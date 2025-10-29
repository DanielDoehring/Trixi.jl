using Trixi

using NonlinearSolve, LinearSolve, ADTypes

using SparseConnectivityTracer # For obtaining the Jacobian sparsity pattern
using SparseMatrixColorings # For obtaining the coloring vector
using JLD2 # For loading/storing sparsity info

###############################################################################
# semidiscretization of the linear advection equation

equations = InviscidBurgersEquation1D()

num_flux = flux_godunov

solver = DGSEM(polydeg = 3, surface_flux = num_flux)

coordinates_min = 0.0
coordinates_max = 1.0

cells_per_dimension = (32,)
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition_convergence_test, solver,
                                    source_terms = source_terms_convergence_test)

#=
jac_detector = TracerSparsityDetector()
# We need to construct the semidiscretization with the correct
# sparsity-detection ready datatype, which is retrieved here
jac_eltype = jacobian_eltype(real(solver), jac_detector)

semi_SD = SemidiscretizationHyperbolic(mesh, equations,
                                       initial_condition_convergence_test, solver;
                                       uEltype = jac_eltype)
=#

###############################################################################
# ODE solvers, callbacks etc.

t0 = 0.0
t_end = 2.0
t_span = (t0, t_end)

ode = semidiscretize(semi, t_span)

path = "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/"
ode_alg = Trixi.PairedExplicitRK2IMEXMulti([16, 8], path, [1, 1])

#=
ode_SD = semidiscretize(semi_SD, t_span)

u = copy(ode_SD.u0)
du = zero(u)
u_tmp = zero(u)

k1 = zero(u) # Additional PERK register

semi = ode_SD.p
cache = semi_SD.cache

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

# Store the sparsity information
jldopen("/home/daniel/git/DissDoc/Data/IMEX/Burgers/sparsity_info.jld2", "w") do file
    file["jac_prototype"] = jac_prototype
    file["colorvec"] = colorvec  # Also store the coloring vector
end

maximum(colorvec) + 1 # Number RHS evaluations to get full Jacobian
=#

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 10_000,
                                     extra_analysis_errors = (:conservation_error, :l1_error),
                                     analysis_integrals = ())

callbacks = CallbackSet(summary_callback,
                        analysis_callback)

###############################################################################
# Set up integrator

n_conv = 0

atol_nonlin = 1e-7
maxiters_nonlin = 20

dt = (2e-3)/2^n_conv

#=
# Load the sparsity information
jac_prototype, colorvec = jldopen("/home/daniel/git/DissDoc/Data/IMEX/Burgers/sparsity_info.jld2", "r") do file
    return file["jac_prototype"], file["colorvec"]
end

integrator = Trixi.init(ode, ode_alg;
                        dt = dt, callback = callbacks,
                        # IMEX-specific kwargs
                        jac_prototype = jac_prototype,
                        colorvec = colorvec,
                        maxiters_nonlin = maxiters_nonlin,
                        abstol = atol_nonlin);
=#

atol_lin = 1e-8
rtol_lin = 1e-6
#maxiters_lin = 50

linsolve = KrylovJL_GMRES(atol = atol_lin, rtol = rtol_lin)

# For Krylov.jl kwargs see https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/#Krylov.gmres
nonlin_solver = NewtonRaphson(autodiff = AutoFiniteDiff(),
                              linsolve = linsolve)

rtol_nonlin = 1e-4

integrator = Trixi.init(ode, ode_alg;
                        dt = dt, callback = callbacks,
                        # IMEX-specific kwargs
                        nonlin_solver = nonlin_solver,
                        abstol = atol_nonlin, reltol = rtol_nonlin,
                        maxiters_nonlin = maxiters_nonlin);

sol = Trixi.solve!(integrator);