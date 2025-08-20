using Trixi

using LinearSolve
using LineSearch, NonlinearSolve

# Functionality for automatic sparsity detection
using SparseDiffTools, Symbolics

###############################################################################################
### Overloads to construct the `LobattoLegendreBasis` with `Real` type (supertype of `Num`) ###

# Required for setting up the Lobatto-Legendre basis for abstract `Real` type.
# Constructing the Lobatto-Legendre basis with `Real` instead of `Num` is 
# significantly easier as we do not have to care about e.g. if-clauses.
# As a consequence, we need to provide some overloads hinting towards the intended behavior.

const float_type = Float64 # Actual floating point type for the simulation

# Newton tolerance for finding LGL nodes & weights
Trixi.eps(::Type{Real}) = Base.eps(float_type)
# There are some places where `one(RealT)` or `zero(uEltype)` is called where `RealT` or `uEltype` is `Real`.
# This returns an `Int64`, i.e., `1` or `0`, respectively which gives errors when a floating-point alike type is expected.
Trixi.one(::Type{Real}) = Base.one(float_type)
Trixi.zero(::Type{Real}) = Base.zero(float_type)

module RealMatMulOverload

# Multiplying two Matrix{Real}s gives a Matrix{Any}.
# This causes problems when instantiating the Legendre basis, which calls
# `calc_{forward,reverse}_{upper, lower}` which in turn uses the matrix multiplication
# which is overloaded here in construction of the interpolation/projection operators 
# required for mortars.
function Base.:*(A::Matrix{Real}, B::Matrix{Real})::Matrix{Real}
    m, n = size(A, 1), size(B, 2)
    kA = size(A, 2)
    kB = size(B, 1)
    @assert kA==kB "Matrix dimensions must match for multiplication"

    C = Matrix{Real}(undef, m, n)
    for i in 1:m, j in 1:n
        #acc::Real = zero(promote_type(typeof(A[i,1]), typeof(B[1,j])))
        acc = zero(Real)
        for k in 1:kA
            acc += A[i, k] * B[k, j]
        end
        C[i, j] = acc
    end
    return C
end
end

import .RealMatMulOverload

Trixi.sqrt(x::Num) = Base.sqrt(x)

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

# For sparsity detection
solver_real = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs, RealT = Real)

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

#path = "/storage/home/daniel/PERK4/SD7003/"
path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/PERK_mesh/SD7003Laminar/"
mesh_file = path * "sd7003_laminar_straight_sided_Trixi.inp"

boundary_symbols = [:Airfoil, :FarField]

mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:FarField => boundary_condition_free_stream,
                           :Airfoil => boundary_condition_slip_wall)

boundary_conditions_parabolic = Dict(:FarField => boundary_condition_free_stream,
                                     :Airfoil => boundary_condition_airfoil)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

semi_real = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                  initial_condition, solver_real;
                                                  boundary_conditions = (boundary_conditions,
                                                                         boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 30 * t_c) # Try to get into a state where initial pressure wave is gone

ode = semidiscretize(semi, tspan)
#ode = semidiscretize(semi, tspan; split_problem = false) # for multirate PERK


# For PERK Multi coefficient measurements
restart_file = "restart_000126951.h5"
restart_filename = joinpath("/home/daniel/git/Paper_Split_IMEX_PERK/Data/SD7003/restart_data/", restart_file)

tspan = (30 * t_c, 35 * t_c)
tspan = (30 * t_c, 31 * t_c) # For testing only

ode = semidiscretize(semi, tspan, restart_filename) # For split PERK
ode = semidiscretize(semi, tspan, restart_filename; split_problem = false) # For non-split PERK Multi


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

# For plots etc
save_solution = SaveSolutionCallback(interval = 1_000_000, # Only at end
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "out")

alive_callback = AliveCallback(alive_interval = 10) # 500

save_restart = SaveRestartCallback(interval = 1_000_000, # Only at end
                                   save_final_restart = true)

callbacks = CallbackSet(#stepsize_callback, # For measurements: Fixed timestep (do not use this)
                        alive_callback, # Not needed for measurement run
                        #save_solution, # For plotting during measurement run
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

#ode_algorithm = Trixi.PairedExplicitRelaxationRK4(Stages[1], path; relaxation_solver = relaxation_solver)

ode_algorithm = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios; relaxation_solver = relaxation_solver)
#ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages, path, dtRatios)

#ode_algorithm = Trixi.RelaxationRK44(; relaxation_solver = relaxation_solver)
#ode_algorithm = Trixi.RelaxationTS64(; relaxation_solver = relaxation_solver)
#ode_algorithm = Trixi.RelaxationCKL54(; relaxation_solver = relaxation_solver)
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

ode_algorithm = Trixi.PairedExplicitRK2Multi(Stages, path_coeffs, dtRatios)

#=
# Version with SAME number of stages for hyperbolic and parabolic part
ode_algorithm = Trixi.PairedExplicitRK2SplitMulti(Stages,
                                                  path_coeffs, path_coeffs_para,
                                                  dtRatios)
=#
# Version with DIFFERENT number of stages for hyperbolic and parabolic part
ode_algorithm = Trixi.PairedExplicitRK2SplitMulti(Stages, Stages_para,
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

ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages, path_coeffs, dtRatios)

#=
ode_algorithm = Trixi.PairedExplicitRK4SplitMulti(Stages,
                                                  path_coeffs, path_coeffs_para,
                                                  dtRatios)
=#


ode_algorithm = Trixi.PairedExplicitRK4SplitMulti(Stages, Stages_para,
                                                  path_coeffs, path_coeffs_para,
                                                  dtRatios, dtRatios_para)


relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 5, root_tol = 1e-13)

ode_algorithm = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path_coeffs, dtRatios; relaxation_solver = relaxation_solver)

ode_algorithm = Trixi.PairedExplicitRelaxationRK4SplitMulti(Stages, Stages_para, 
                                                            path_coeffs, path_coeffs_para,
                                                            dtRatios, dtRatios_para; relaxation_solver = relaxation_solver)


# For measurement run with fixed timestep
dt = 1e-3 # PERK4, dt_c = 2e-4

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep = false, callback = callbacks);

###############################################################################
# IMEX

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

ode_algorithm = Trixi.PairedExplicitRK2IMEXMulti(Stages, path_coeffs, dtRatios)

### Linesearch ###
# See https://docs.sciml.ai/LineSearch/dev/api/native/

#linesearch = BackTracking(autodiff = AutoFiniteDiff(), order = 3, maxstep = 10)
#linesearch = LiFukushimaLineSearch()
linesearch = nothing

### Linear Solver ###
# See https://docs.sciml.ai/LinearSolve/stable/solvers/solvers/

#linsolve = SimpleGMRES()
linsolve = KrylovJL_GMRES()

nonlin_solver = NewtonRaphson(autodiff = AutoFiniteDiff(),
                              linesearch = linesearch, linsolve = linsolve,
                              concrete_jac = false)

dt_init = 7e-4
integrator = Trixi.init(ode, ode_algorithm;
                        dt = 7e-4,
                        callback = callbacks,
                        nonlin_solver = nonlin_solver,
                        abstol = 1e-5, reltol = 1e-3,
                        maxiters_nonlin = 10, # Maxiters should be on the order of the number of stages of the highest explicit method
                        dt_init = dt_init);

###############################################################################################
### Compute the Jacobian with SparseDiffTools ###

sd = SymbolicsSparsityDetection()
ad_type = AutoFiniteDiff()
#ad_type = AutoForwardDiff()
sparse_adtype = AutoSparse(ad_type)

R = ode_algorithm.num_methods
element_indices = integrator.level_info_elements_acc[R]
interface_indices = integrator.level_info_interfaces_acc[R]
boundary_indices = integrator.level_info_boundaries_acc[R]
mortar_indices = integrator.level_info_mortars_acc[R]
u_indices = integrator.level_u_indices_elements[R]

# As for rhs! Jacobian computation: Start with plain float,
# which are automatically promoted to `Dual` type
residual = zeros(Float64, length(u_indices))
k_nonlin = zeros(Float64, length(u_indices))

u_ode = Num.(ode.u0)
u_tmp = copy(u_ode) # Would normally carry the explicit update, here for now simply set to IC
du_ode = zero(u_ode) # Try to avoid "undef"
du_para = zero(u_ode)

t0 = tspan[1]
function stage_residual_PERK2IMEXMultiParabolic!(residual, k_nonlin)
    a_dt = 0.5 * dt_init # Hard-coded for IMEX midpoint method

    # Add implicit contribution
    for i in eachindex(u_indices)
        u_idx = u_indices[i] # Ensure thread safety
        u_tmp[u_idx] = u_ode[u_idx] + a_dt * k_nonlin[i]
    end

    # Evaluate implicit stage
    Trixi.rhs_hyperbolic_parabolic!(du_ode, u_tmp, semi_real,
                                    t0 + 0.5 * dt_init, # Hard-coded for IMEX midpoint method
                                    du_para,
                                    element_indices, interface_indices, boundary_indices, mortar_indices, u_indices)

    # Compute residual
    for i in eachindex(u_indices)
        residual[i] = k_nonlin[i] - du_ode[u_indices[i]]
    end

    return nothing
end

sparse_cache = sparse_jacobian_cache(sparse_adtype, sd, stage_residual_PERK2IMEXMultiParabolic!, residual, k_nonlin)

sol = Trixi.solve!(integrator);