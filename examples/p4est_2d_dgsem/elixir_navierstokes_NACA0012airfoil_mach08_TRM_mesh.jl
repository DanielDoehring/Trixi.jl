# Transonic flow around an airfoil

# This test is taken from the paper below. The values from Case 5 in Table 3 are used to validate
# the scheme and computation of surface forces.

# - Roy Charles Swanson, Stefan Langer (2016)
#   Structured and Unstructured Grid Methods (2016)
#   [https://ntrs.nasa.gov/citations/20160003623] (https://ntrs.nasa.gov/citations/20160003623)

using Downloads: download
using OrdinaryDiffEq
using Trixi

using Trixi: AnalysisSurfaceIntegral, DragCoefficient, LiftCoefficient

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

prandtl_number() = 0.72

#mu() = 0.0031959974968701088 # Re = 500, Ma = 0.8
mu() = 0.0001997498435543818 # Re = 5000, Ma = 0.5

equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

sw_rho_inf() = 1.0
sw_pre_inf() = 2.85

#sw_aoa() = 10.0 * pi / 180.0
sw_aoa() = 0.0

sw_linf() = 1.0

#sw_mach_inf() = 0.8
sw_mach_inf() = 0.5

sw_U_inf(equations) = sw_mach_inf() * sqrt(equations.gamma * sw_pre_inf() / sw_rho_inf())

# Control
Ma = sw_U_inf(equations) / sqrt(equations.gamma * sw_pre_inf() / sw_rho_inf())
Re = sw_rho_inf() * sw_U_inf(equations) * sw_linf() / mu()

@inline function initial_condition_mach08_flow(x, t, equations)
    # set the freestream flow parameters
    gasGam = equations.gamma
    mach_inf = sw_mach_inf()
    aoa = sw_aoa()
    rho_inf = sw_rho_inf()
    pre_inf = sw_pre_inf()
    U_inf = mach_inf * sqrt(gasGam * pre_inf / rho_inf)

    #v1 = U_inf * cos(aoa)
    v1 = U_inf

    #v2 = U_inf * sin(aoa)
    v2 = 0.0

    prim = SVector(rho_inf, v1, v2, pre_inf)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach08_flow

polydeg = 3
solver = DGSEM(polydeg = polydeg, surface_flux = flux_hll)

path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/NASA_meshes/"

mesh = "NACA0012/Family_1/113_33/n0012family_1_7_2D_unique.inp"
#mesh = "NACA0012/Family_1/225_65/n0012familyI_6_2D_unique.inp"
mesh_file = path * mesh

boundary_symbols = [:b2_symmetry_y_strong,
                    :b4_farfield_riem, :b5_farfield_riem, :b7_farfield_riem, :b6_viscous_solid, :b8_to_stitch_a]

mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

# The boundary values across outer boundary are constant but subsonic, so we cannot compute the
# boundary flux from the external information alone. Thus, we use the numerical flux to distinguish
# between inflow and outflow characteristics
@inline function boundary_condition_subsonic_constant(u_inner,
                                                      normal_direction::AbstractVector, x,
                                                      t,
                                                      surface_flux_function,
                                                      equations::CompressibleEulerEquations2D)
    u_boundary = initial_condition_mach08_flow(x, t, equations)

    return Trixi.flux_hll(u_inner, u_boundary, normal_direction, equations)
end

boundary_conditions = Dict(:b2_symmetry_y_strong => boundary_condition_subsonic_constant,
                           :b4_farfield_riem => boundary_condition_subsonic_constant,
                           :b5_farfield_riem => boundary_condition_subsonic_constant,
                           :b7_farfield_riem => boundary_condition_subsonic_constant,
                           :b6_viscous_solid => boundary_condition_slip_wall,
                           :b8_to_stitch_a => boundary_condition_subsonic_constant)

velocity_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))

heat_airfoil = Adiabatic((x, t, equations) -> 0.0)

boundary_conditions_airfoil = BoundaryConditionNavierStokesWall(velocity_airfoil,
                                                                heat_airfoil)

function momenta_initial_condition_mach08_flow(x, t, equations)
  u = initial_condition_mach08_flow(x, t, equations)
  momenta = SVector(u[2], u[3])
end
velocity_bc_square = NoSlip((x, t, equations) -> momenta_initial_condition_mach08_flow(x, t, equations))                                                

heat_bc_square = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_square = BoundaryConditionNavierStokesWall(velocity_bc_square,
                                                              heat_bc_square)

boundary_conditions_parabolic = Dict(:b2_symmetry_y_strong => boundary_condition_square,
                                      :b4_farfield_riem => boundary_condition_square,
                                      :b5_farfield_riem => boundary_condition_square,
                                      :b7_farfield_riem => boundary_condition_square,
                                      :b6_viscous_solid => boundary_conditions_airfoil,
                                      :b8_to_stitch_a => boundary_condition_square)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers

restart_file = "restart_250000.h5"
restart_filename = joinpath("out", restart_file)

# Run for a long time to reach a steady state
tspan = (0.0, 1)
#tspan = (load_time(restart_filename), 1.0)

ode = semidiscretize(semi, tspan)
#ode = semidiscretize(semi, tspan, restart_filename)

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 2000

force_boundary_name = :b6_viscous_solid
drag_coefficient = AnalysisSurfaceIntegral(semi, force_boundary_name,
                                           DragCoefficient(sw_aoa(), sw_rho_inf(),
                                                           sw_U_inf(equations), sw_linf()))

lift_coefficient = AnalysisSurfaceIntegral(semi, force_boundary_name,
                                           LiftCoefficient(sw_aoa(), sw_rho_inf(),
                                                           sw_U_inf(equations), sw_linf()))

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     save_analysis = true,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (drag_coefficient,
                                                           lift_coefficient))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_restart = SaveRestartCallback(interval = 5e4,
                                   save_final_restart = true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_restart)

###############################################################################
# run the simulation

#=
sol = solve(ode, RK4(thread = OrdinaryDiffEq.True()); dt = 5e-10, adaptive = true,
            save_everystep = false, callback = callbacks)
=#

sol = Trixi.solve(ode, Trixi.HypDiffN3Erk3Sstar52(); dt = 1.9e-8, save_everystep = false, callback = callbacks)

summary_callback() # print the timer summary