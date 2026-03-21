using OrdinaryDiffEqSSPRK
using Trixi
using LinearAlgebra

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

aerospike_angle = 15 # degree
sin_angle() = sin(deg2rad(aerospike_angle))
cos_angle() = cos(deg2rad(aerospike_angle))

@inline function state_ambient(x, t, equations::CompressibleEulerEquations2D)
    rho_ = 1.0
    v1_ = 0.0
    v2_ = 0.0
    p_ = 0.18

    prim = SVector(rho_, v1_, v2_, p_)
    return prim2cons(prim, equations)
end
bc_ambient = BoundaryConditionDirichlet(state_ambient)

function state_thruster_nozzle_inlet_top(x, t, equations::CompressibleEulerEquations2D)
    rho_ = 4.18
    v1_ = sin_angle() * 0.19
    v2_ = -cos_angle() * 0.19
    p_ = 1.31

    prim = SVector(rho_, v1_, v2_, p_)
    return prim2cons(prim, equations)
end
bc_nozzle_inlet_top = BoundaryConditionDirichlet(state_thruster_nozzle_inlet_top)

function state_thruster_nozzle_inlet_bottom(x, t, equations::CompressibleEulerEquations2D)
    rho_ = 4.18
    v1_ = sin_angle() * 0.19
    v2_ = cos_angle() * 0.19
    p_ = 1.31

    prim = SVector(rho_, v1_, v2_, p_)
    return prim2cons(prim, equations)
end
bc_nozzle_inlet_bottom = BoundaryConditionDirichlet(state_thruster_nozzle_inlet_bottom)

volume_flux = flux_ranocha_turbo
surface_flux = flux_lax_friedrichs

polydeg = 4
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
# In non-blended/limited regions, we use the cheaper weak form volume integral
volume_integral_default = VolumeIntegralWeakForm()
#volume_integral_default = VolumeIntegralFluxDifferencing(volume_flux)

# For the blended/limited regions, we need to supply high-order and low-order volume integrals.
volume_integral_blend_high_order = VolumeIntegralFluxDifferencing(volume_flux)
volume_integral_blend_low_order = VolumeIntegralPureLGLFiniteVolume(flux_hllc)

volume_integral_blend_low_order = VolumeIntegralPureLGLFiniteVolumeO2(basis;
                                                                      volume_flux_fv = flux_hllc,
                                                                      reconstruction_mode = reconstruction_O2_inner,
                                                                      slope_limiter = minmod)

volume_integral = VolumeIntegralShockCapturingHGType(shock_indicator;
                                                     volume_integral_default = volume_integral_default,
                                                     volume_integral_blend_high_order = volume_integral_blend_high_order,
                                                     volume_integral_blend_low_order = volume_integral_blend_low_order)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

mesh_file = "/storage/home/daniel/Meshes/Aerospike/out/nozzle.inp"

mesh = P4estMesh{2}(mesh_file)

@inline function boundary_condition_outflow_general(u_inner,
                                                    normal_direction::AbstractVector, x, t,
                                                    surface_flux_function,
                                                    equations::CompressibleEulerEquations2D)

    # This would be for the general case where we need to check the magnitude of the local Mach number
    norm_ = norm(normal_direction)
    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    normal = normal_direction / norm_

    # Rotate the internal solution state
    u_local = Trixi.rotate_to_x(u_inner, normal, equations)

    # Compute the primitive variables
    rho_local, v_normal, v_tangent, p_local = cons2prim(u_local, equations)

    # Compute local Mach number
    a_local = sqrt(equations.gamma * p_local / rho_local)
    Mach_local = abs(v_normal / a_local)
    if Mach_local <= 1.0
        p_local = pressure(state_ambient(x, t, equations), equations)
    end

    # Create the `u_surface` solution state where the local pressure is possibly set from an external value
    prim = SVector(rho_local, v_normal, v_tangent, p_local)
    u_boundary = prim2cons(prim, equations)
    u_surface = Trixi.rotate_from_x(u_boundary, normal, equations)

    # Compute the flux using the appropriate mixture of internal / external solution states
    return flux(u_surface, normal_direction, equations)
end

#bc_right_top = boundary_condition_do_nothing
bc_right_top = boundary_condition_outflow_general

boundary_conditions = (; NozzleWallTop = boundary_condition_slip_wall,
                       Inlet = bc_nozzle_inlet_top,
                       NozzleWallBottom = boundary_condition_slip_wall,
                       ExtBottom = boundary_condition_slip_wall,
                       ExtLeft = boundary_condition_slip_wall,
                       Right = bc_right_top,
                       Top = bc_right_top,
                       Left = bc_ambient,
                       # Symmetry boundaries
                       NozzleWallTop_R = boundary_condition_slip_wall,
                       Inlet_R = bc_nozzle_inlet_bottom,
                       NozzleWallBottom_R = boundary_condition_slip_wall,
                       ExtBottom_R = boundary_condition_slip_wall,
                       ExtLeft_R = boundary_condition_slip_wall,
                       Right_R = bc_right_top,
                       Top_R = bc_right_top,
                       Left_R = bc_ambient)

function initial_condition(x, t, equations::CompressibleEulerEquations2D)
    if x[1] > 0.1
        return state_ambient(x, t, equations)
    else
        if x[2] > 0.0
            return state_thruster_nozzle_inlet_top(x, t, equations)
        else
            return state_thruster_nozzle_inlet_bottom(x, t, equations)
        end
    end
end

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers

tspan = (0.0, 25.0)
dt0 = 1e-4
ode = semidiscretize(semi, tspan)

#=
restart_file = "out/restart_000027000.h5"

mesh = load_mesh(restart_file)

semi = SemidiscretizationHyperbolic(mesh, equations, state_ambient, solver;
                                    boundary_conditions = boundary_conditions)

tspan = (load_time(restart_file), 100.0)
dt0 = load_dt(restart_file)
ode = semidiscretize(semi, tspan, restart_file)
=#

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

#amr_indicator = IndicatorLöhner(semi, variable = Trixi.density_pressure)
amr_indicator = shock_indicator

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      med_level = 2, med_threshold = 0.05,
                                      max_level = 3, max_threshold = 0.1)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 20,
                           adapt_initial_condition = false)

save_restart = SaveRestartCallback(interval = 1000)

stepsize_callback = StepsizeCallback(cfl = 3.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution, save_restart,
                        amr_callback,
                        #stepsize_callback
                        )

###############################################################################
# run the simulation

#=
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1.0e-4, 1.0e-4),
                                                     variables = (Trixi.density, pressure))
ode_alg = SSPRK43(stage_limiter! = stage_limiter!, thread = Trixi.True())
=#

ode_alg = SSPRK43(thread = Trixi.True())

sol = solve(ode, ode_alg;
            #adaptive = false, dt = dt0, #abstol = 1e-5,
            adaptive = true, dt = dt0,
            ode_default_options()..., callback = callbacks);
