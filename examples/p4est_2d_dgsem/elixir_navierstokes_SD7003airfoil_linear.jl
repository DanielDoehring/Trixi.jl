using Trixi

using SparseConnectivityTracer # For obtaining the Jacobian sparsity pattern
using SparseMatrixColorings # For obtaining the coloring vector
using JLD2 # For loading sparse matrix

import LinearAlgebra:norm
using NonlinearSolve, LinearSolve, LineSearch, ADTypes

###############################################################################
# semidiscretization of the compressible Euler equations

U_inf = 0.2
c_inf = 1.0

rho_inf = 1.4 # with gamma = 1.4 => p_inf = 1.0

Re = 10000.0
airfoil_cord_length = 1.0

t_c = airfoil_cord_length / U_inf

aoa = 4 * pi / 180
u_x = U_inf * cos(aoa)
u_y = U_inf * sin(aoa)

gamma = 1.4
prandtl_number() = 0.72
mu() = rho_inf * U_inf * airfoil_cord_length / Re

equations = CompressibleEulerEquations2D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

@inline function initial_condition_mach02_flow(x, t, equations)
    # set the freestream flow parameters
    rho_freestream = 1.4

    v1 = 0.19951281005196486 # 0.2 * cos(aoa)
    v2 = 0.01395129474882506 # 0.2 * sin(aoa)

    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach02_flow

# Boundary conditions for free-stream testing
boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_airfoil = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)

polydeg = 3

surf_flux = flux_hllc
vol_flux = flux_chandrashekar
solver = DGSEM(polydeg = polydeg, surface_flux = surf_flux,
               volume_integral = VolumeIntegralFluxDifferencing(vol_flux))

# for sparsity [pattern] detection (SD): if-clause free
solver_SD = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs,
                  volume_integral = VolumeIntegralFluxDifferencing(flux_central))

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

#path = "/storage/home/daniel/PERK4/SD7003/"
path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/PERK_mesh/SD7003Laminar/"
mesh_file = path * "sd7003_laminar_straight_sided_Trixi.inp"

boundary_symbols = [:Airfoil, :FarField]

mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:FarField => boundary_condition_free_stream,
                           :Airfoil => boundary_condition_slip_wall)

# For sparsity detection
@inline function bc_symmetry(u_inner, normal_direction::AbstractVector, x, t,
                             surface_flux_function,
                             equations::CompressibleEulerEquations2D)
    norm_ = norm(normal_direction)
    normal = normal_direction / norm_

    # compute the primitive variables
    rho, v1, v2, p = cons2prim(u_inner, equations)

    v_normal = normal[1] * v1 + normal[2] * v2

    u_mirror = prim2cons(SVector(rho,
                                 v1 - 2 * v_normal * normal[1],
                                 v2 - 2 * v_normal * normal[2],
                                 p), equations)

    flux = surface_flux_function(u_inner, u_mirror, normal, equations) * norm_

    return flux
end
boundary_conditions_SD = Dict(:FarField => boundary_condition_free_stream,
                              :Airfoil => bc_symmetry)

boundary_conditions_parabolic = Dict(:FarField => boundary_condition_free_stream,
                                     :Airfoil => boundary_condition_airfoil)

jac_detector = TracerSparsityDetector()
# We need to construct the semidiscretization with the correct
# sparsity-detection ready datatype, which is retrieved here
jac_eltype = jacobian_eltype(real(solver), jac_detector)

Trixi.sqrt(x::SparseConnectivityTracer.AbstractTracer) = Base.sqrt(x)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

semi_SD = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                initial_condition, solver_SD;
                                                boundary_conditions = (boundary_conditions_SD,
                                                                       boundary_conditions_parabolic),
                                                uEltype = jac_eltype)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 30 * t_c) # Try to get into a state where initial pressure wave is gone

#ode = semidiscretize(semi, tspan)
#ode = semidiscretize(semi, tspan; split_problem = false) # for multirate PERK

# For PERK Multi coefficient measurements
restart_file = "restart_000126951.h5"
restart_filename = joinpath("/home/daniel/git/Paper_Split_IMEX_PERK/Data/SD7003/restart_data/", restart_file)

tspan = (30 * t_c, 35 * t_c)
tspan = (30 * t_c, 30.25 * t_c) # For testing only
tspan = (30 * t_c, 30 * t_c) # Plot sol

#ode = semidiscretize(semi, tspan, restart_filename) # For split PERK
ode = semidiscretize(semi, tspan, restart_filename; split_problem = false) # For non-split PERK Multi

ode_SD = semidiscretize(semi_SD, tspan, restart_filename; split_problem = false)

summary_callback = SummaryCallback()

# Choose analysis interval such that roughly every dt_c = 0.005 a record is taken
analysis_interval = 25 # Matches for PERK 4 schemes
analysis_interval = 1_000_000 # Only at end

f_aoa() = aoa
f_rho_inf() = rho_inf
f_U_inf() = U_inf
f_linf() = airfoil_cord_length

drag_coefficient = AnalysisSurfaceIntegral((:Airfoil,),
                                           DragCoefficientPressure2D(f_aoa(), f_rho_inf(),
                                                                   f_U_inf(), f_linf()))

drag_coefficient_shear_force = AnalysisSurfaceIntegral((:Airfoil,),
                                                       DragCoefficientShearStress2D(f_aoa(),
                                                                                  f_rho_inf(),
                                                                                  f_U_inf(),
                                                                                  f_linf()))

lift_coefficient = AnalysisSurfaceIntegral((:Airfoil,),
                                           LiftCoefficientPressure2D(f_aoa(), f_rho_inf(),
                                                                   f_U_inf(), f_linf()))

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     save_analysis = true,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (drag_coefficient,
                                                           drag_coefficient_shear_force,
                                                           lift_coefficient))

cfl = 6.2 # PERK 4 Multi E = 5, ..., 14
#cfl = 6.5 # PERK 4 Single 12

cfl = 7.4 # PERRK_4 Multi E = 5, ..., 14
#cfl = 7.6 # Single PERK 14

#cfl = 1.9 # R-RK44
#cfl = 2.5 # R-TS64
#cfl = 2.6 # R-CKL54

cfl = 7.6 # PERK2 unsplit/standard multi
cfl = 8.7 # PERK2 split multi with same stages & distribution
cfl = 7.9 # PERK2 split multi with different stages (14, 10) & distribution

cfl = 6.2 # PERK 4 Multi E = 5, ..., 14
#cfl = 5.6 # PERK 4 Multi Split E = 5, ..., 10
cfl = 6.1 # PERK 4 Multi Split (14, 10)

#cfl = 7.4 # PEERRKS_4 Multi E = 5, ..., 14
cfl = 6.1 # PERRK 4 Multi Split (14, 10) # Seems to have no benefit of RESTARTED simulation from relaxation

stepsize_callback = StepsizeCallback(cfl = cfl)

### Plot which cells are convection and which are diffusion dominated
extra_node_variables = (:cfl_diffusion, :cfl_convection)

function Trixi.get_node_variable(::Val{:cfl_diffusion}, u, mesh, equations, dg, cache,
                                 equations_parabolic, cache_parabolic)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    cfl_d_array = zeros(eltype(cache.elements),
                            n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                            n_elements)

    @unpack contravariant_vectors, inverse_jacobian = cache.elements

    # We can accelerate the computation by thread-parallelizing the loop over elements
    # by using the `@threaded` macro.
    Trixi.@threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)

            # Speeds
            d = Trixi.max_diffusivity(u_node, equations_parabolic)

            # Mesh data
            Ja11, Ja12 = Trixi.get_contravariant_vector(1, contravariant_vectors, i, j,
                                                  element)
            Ja21, Ja22 = Trixi.get_contravariant_vector(2, contravariant_vectors, i, j,
                                                  element)

            inv_jacobian = abs(inverse_jacobian[i, j, element])

            # See FLUXO:
            # https://github.com/project-fluxo/fluxo/blob/c7e0cc9b7fd4569dcab67bbb6e5a25c0a84859f1/src/equation/navierstokes/calctimestep.f90#L130-L137
            d1_transformed = d * (Ja11^2 + Ja12^2) * inv_jacobian^2
            d2_transformed = d * (Ja21^2 + Ja22^2) * inv_jacobian^2

            cfl_d_array[i, j, element] = d1_transformed + d2_transformed
        end
    end

    return cfl_d_array
end

function Trixi.get_node_variable(::Val{:cfl_convection}, u, mesh, equations, dg, cache,
                                 equations_parabolic, cache_parabolic)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    cfl_a_array = zeros(eltype(cache.elements),
                            n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                            n_elements)

    @unpack contravariant_vectors, inverse_jacobian = cache.elements

    # We can accelerate the computation by thread-parallelizing the loop over elements
    # by using the `@threaded` macro.
    Trixi.@threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)

            # Speeds
            a1, a2 = Trixi.max_abs_speeds(u_node, equations)

            # Mesh data
            Ja11, Ja12 = Trixi.get_contravariant_vector(1, contravariant_vectors, i, j,
                                                  element)
            Ja21, Ja22 = Trixi.get_contravariant_vector(2, contravariant_vectors, i, j,
                                                  element)

            inv_jacobian = abs(inverse_jacobian[i, j, element])

            # Transform
            a1_transformed = abs(Ja11 * a1 + Ja12 * a2) * inv_jacobian
            a2_transformed = abs(Ja21 * a1 + Ja22 * a2) * inv_jacobian

            cfl_a_array[i, j, element] = a1_transformed + a2_transformed
        end
    end

    return cfl_a_array
end

save_solution = SaveSolutionCallback(interval = 1_000_000, # Only at end
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out",
                                     extra_node_variables = extra_node_variables)

alive_callback = AliveCallback(alive_interval = 50) # 500

save_restart = SaveRestartCallback(interval = 1_000_000, # Only at end
                                   save_final_restart = true)

callbacks = CallbackSet(#stepsize_callback, # For measurements: Fixed timestep (do not use this)
                        alive_callback, # Not needed for measurement run
                        save_solution, # For plotting during measurement run
                        #save_restart, # For restart with measurements
                        analysis_callback,
                        summary_callback);

###############################################################################
# run the simulation

#=
relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 5, root_tol = 1e-13)

Stages = [14, 12, 10, 8, 7, 6, 5]
dtRatios = [0.208310160790890, # 14
    0.172356930215766, # 12
    0.129859071602721, # 10
    0.092778774946394, #  8
    0.069255720146485, #  7
    0.049637258180915, #  6
    0.030629777558366] / 0.208310160790890 #= 5 =#

#ode_alg = Trixi.PairedExplicitRelaxationRK4(Stages[1], path; relaxation_solver = relaxation_solver)

ode_alg = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios; relaxation_solver = relaxation_solver)
#ode_alg = Trixi.PairedExplicitRK4Multi(Stages, path, dtRatios)

#ode_alg = Trixi.RelaxationRK44(; relaxation_solver = relaxation_solver)
#ode_alg = Trixi.RelaxationTS64(; relaxation_solver = relaxation_solver)
#ode_alg = Trixi.RelaxationCKL54(; relaxation_solver = relaxation_solver)
=#

### Split-Multi methods ###

# p = 2

Stages = [14, 12, 10, 8, 7, 6, 5, 4, 3, 2]
dtRatios = [0.253144726232790162612, # 14
    0.214041846963368698198,  # 12
    0.177173703567632401246,  # 10
    0.138494092598762108537,  #  8
    0.121607896165869533434,  #  7
    0.0975166462040988335502, #  6
    0.0818171376613463507965, #  5
    0.0656503721211265656166, #  4
    0.0419871921542380732717, #  3
    0.0209738927526359475451] / 0.253144726232790162612 #= 2 =#

path_coeffs = "/home/daniel/git/Paper_Split_IMEX_PERK/Data/SD7003/coeffs_p2/full_rhs/"

# Can run with these stages with cfl 8.7
Stages_para = [14, 12, 10, 7, 6, 5, 4, 3, 2]
dtRatios_para = [3008.7179235408357, # 14
    1762.72919916304,   # 12
    842.7395049162385,  # 10
    375.2614698283665,  # 7
    272.91099615894865, # 6
    186.2882904812082,  # 5
    115.33971622203154, # 4
    59.942706190483364, # 3
    19.14860438454724] / 3008.7179235408357 #= 2 =#

Stages_para = [10, 7, 6, 5, 4, 3, 2]
dtRatios_para = [842.7395049162385,  # 10
    375.2614698283665,  # 7
    272.91099615894865, # 6
    186.2882904812082,  # 5
    115.33971622203154, # 4
    59.942706190483364, # 3
    19.14860438454724] / 842.7395049162385 #= 2 =#

path_coeffs_para = "/home/daniel/git/Paper_Split_IMEX_PERK/Data/SD7003/coeffs_p2/para_rhs/"

ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path_coeffs, dtRatios)

#=
# Version with SAME number of stages for hyperbolic and parabolic part
ode_alg = Trixi.PairedExplicitRK2SplitMulti(Stages,
                                            path_coeffs, path_coeffs_para,
                                            dtRatios)
=#
# Version with DIFFERENT number of stages for hyperbolic and parabolic part
ode_alg = Trixi.PairedExplicitRK2SplitMulti(Stages, Stages_para,
                                            path_coeffs, path_coeffs_para,
                                            dtRatios, dtRatios_para)

# p = 4

Stages = [14, 12, 10, 8, 7, 6, 5]
dtRatios = [0.208310160790890, # 14
    0.172356930215766, # 12
    0.129859071602721, # 10
    0.092778774946394, #  8
    0.069255720146485, #  7
    0.049637258180915, #  6
    0.030629777558366] / 0.208310160790890 #= 5 =#

#=
Stages = [10, 8, 7, 6, 5]
dtRatios = [0.129859071602721, # 10
    0.092778774946394, #  8
    0.069255720146485, #  7
    0.049637258180915, #  6
    0.030629777558366] / 0.129859071602721 #= 5 =#
=#

path_coeffs = "/home/daniel/git/Paper_Split_IMEX_PERK/Data/SD7003/coeffs_p4/full_rhs/"

Stages_para = [10, 8, 7, 6, 5]
dtRatios_para = [312.83020308753, # 10
    152.879047415621, #  8
    102.760239306439, #  7
    61.6046703927691, #  6
    27.9645800405075] / 312.83020308753 #= 5 =#

path_coeffs_para = "/home/daniel/git/Paper_Split_IMEX_PERK/Data/SD7003/coeffs_p4/para_rhs/"

ode_alg = Trixi.PairedExplicitRK4Multi(Stages, path_coeffs, dtRatios)

#=
ode_alg = Trixi.PairedExplicitRK4SplitMulti(Stages,
                                            path_coeffs, path_coeffs_para,
                                            dtRatios)
=#


ode_alg = Trixi.PairedExplicitRK4SplitMulti(Stages, Stages_para,
                                            path_coeffs, path_coeffs_para,
                                            dtRatios, dtRatios_para)


relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 5, root_tol = 1e-13)

ode_alg = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path_coeffs, dtRatios; relaxation_solver = relaxation_solver)

ode_alg = Trixi.PairedExplicitRelaxationRK4SplitMulti(Stages, Stages_para, 
                                                      path_coeffs, path_coeffs_para,
                                                      dtRatios, dtRatios_para; relaxation_solver = relaxation_solver)


# For measurement run with fixed timestep
dt = 1e-3 # PERK4, dt_c = 2e-4

sol = Trixi.solve(ode, ode_alg,
                  dt = dt,
                  save_everystep = false, callback = callbacks);


###############################################################################
# IMEX
path_coeffs = "/home/daniel/git/Paper_Split_IMEX_PERK/Data/SD7003/coeffs_p2/full_rhs/"

Stages = [14, 12, 10, 8, 7, 6, 5, 4, 2]
dtRatios = [0.26, # Implicit
    0.253144726232790162612, # 14
    0.214041846963368698198,  # 12
    0.177173703567632401246,  # 10
    0.138494092598762108537,  #  8
    0.121607896165869533434,  #  7
    0.0975166462040988335502, #  6
    0.0818171376613463507965, #  5
    0.0656503721211265656166, #  4
    0.0209738927526359475451] / 0.26 #= 2 =#

ode_alg = Trixi.PairedExplicitRK2IMEXMulti(Stages, path_coeffs, dtRatios)

### Repeat code from integrator init ###
#=
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

du_para = zero(u)

dt = 7e-4
t0 = tspan[1]
res_wrapped! = (residual, k_nonlin) -> Trixi.residual_S_PERK2IMEXMulti!(residual, k_nonlin,
                                    t0, dt,
                                    u, du, u_tmp, du_para,
                                    semi, ode.f,
                                    level_info_elements_acc[R], level_info_interfaces_acc[R],
                                    level_info_boundaries_acc[R], level_info_mortars_acc[R], level_info_u[R])

ode_SD.f(du, u, semi_SD, t0) # Do one step to prevent e.g. undefined BCs
jac_prototype = jacobian_sparsity(res_wrapped!, residual, k_nonlin, jac_detector)

coloring_prob = ColoringProblem(; structure = :nonsymmetric, partition = :column)
coloring_alg = GreedyColoringAlgorithm(; decompression = :direct)
coloring_result = coloring(jac_prototype, coloring_prob, coloring_alg)
colorvec = column_colors(coloring_result)

maximum(colorvec) + 1 # Number RHS evaluations to get full Jacobian

# Store the sparsity information
jldopen("sparsity_info.jld2", "w") do file
    file["jac_prototype"] = jac_prototype
    file["colorvec"] = colorvec  # Also store the coloring vector
end
=#

# Load the sparsity information
jac_prototype_loaded, colorvec_loaded = jldopen("sparsity_info.jld2", "r") do file
    return file["jac_prototype"], file["colorvec"]
end

maximum(colorvec_loaded) + 1


# TODO: Could try algorithms from IterativeSolvers, KrylovKit (wrappers provided by LinearSolve.jl)

dt = 7e-4

atol_lin = 1e-6
rtol_lin = 1e-4
#maxiters_lin = 50

linsolve = KrylovJL_GMRES(atol = atol_lin, rtol = rtol_lin)

linesearch = LiFukushimaLineSearch()
linesearch = nothing

# For Krylov.jl kwargs see https://jso.dev/Krylov.jl/stable/solvers/unsymmetric/#Krylov.gmres
nonlin_solver = NewtonRaphson(autodiff = AutoFiniteDiff(), 
                              linsolve = linsolve,
                              linesearch = linesearch)

atol_nonlin = 1e-6
rtol_nonlin = 1e-4
maxiters_nonlin = 20

#=
# Factorization-based methods

linear_solver = KLUFactorization()
#linear_solver = UMFPACKFactorization()

#linear_solver = MKLPardisoFactorize() # requires Pardiso.jl; slow

#linear_solver = SparspakFactorization() # requires Sparspak.jl

integrator = Trixi.init(ode, ode_alg;
                        dt = dt, callback = callbacks,
                        # IMEX-specific kwargs
                        #jac_prototype = jac_prototype, colorvec = colorvec,
                        jac_prototype = jac_prototype_loaded, colorvec = colorvec_loaded,
                        linear_solver = linear_solver,
                        abstol = atol_nonlin,
                        maxiters_nonlin = maxiters_nonlin);
=#

integrator = Trixi.init(ode, ode_alg;
                        dt = dt, callback = callbacks,
                        # IMEX-specific kwargs
                        nonlin_solver = nonlin_solver,
                        abstol = atol_nonlin, reltol = rtol_nonlin,
                        maxiters_nonlin = maxiters_nonlin);

sol = Trixi.solve!(integrator);