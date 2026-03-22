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

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
# In non-blended/limited regions, we use the cheaper weak form volume integral
volume_integral_default = VolumeIntegralWeakForm()

# For the blended/limited regions, we need to supply high-order and low-order volume integrals.
volume_integral_blend_high_order = VolumeIntegralFluxDifferencing(volume_flux)

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

mesh_file = "/storage/home/daniel/Meshes/Aerospike/out/Aerospike.inp"
#mesh_file = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/Aerospike/out/Aerospike.inp"

mesh = P4estMesh{2}(mesh_file)

# Calculate the boundary flux from the inner state while
# using the pressure from the ambient state when the flow is subsonic.
#
# See the reference below for a discussion on inflow/outflow boundary conditions. The subsonic
# outflow boundary conditions are discussed in Section 2.3.
#
# - Jan-Reneé Carlson (2011)
#   Inflow/Outflow Boundary Conditions with Application to FUN3D.
#   [NASA TM 20110022658](https://ntrs.nasa.gov/citations/20110022658)
@inline function boundary_condition_outflow_general(u_inner,
                                                    normal_direction::AbstractVector, x, t,
                                                    surface_flux_function,
                                                    equations::CompressibleEulerEquations2D)
    norm_ = norm(normal_direction)
    normal = normal_direction / norm_

    # Rotate the internal solution state to have normal and tangential components
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
bc_right_top = boundary_condition_outflow_general

boundary_conditions = (; NozzleWallTop = boundary_condition_slip_wall,
                       Inlet = bc_nozzle_inlet_top,
                       NozzleWallBottom = boundary_condition_slip_wall,
                       ExtBottom = boundary_condition_slip_wall,
                       ExtLeft = boundary_condition_slip_wall,
                       Right = bc_right_top,
                       Top = bc_right_top,
                       Left = bc_ambient,
                       # Boundaries on flipped/symmetric part
                       NozzleWallTop_R = boundary_condition_slip_wall,
                       Inlet_R = bc_nozzle_inlet_bottom,
                       NozzleWallBottom_R = boundary_condition_slip_wall,
                       ExtBottom_R = boundary_condition_slip_wall,
                       ExtLeft_R = boundary_condition_slip_wall,
                       Right_R = bc_right_top,
                       Top_R = bc_right_top,
                       Left_R = bc_ambient)

# Initialize nozzles already with the inlet state, rest of the domain with ambient state.
function initial_condition(x, t, equations::CompressibleEulerEquations2D)
    if x[1] > 0.1 # Outside nozzles
        return state_ambient(x, t, equations)
    else # Inside nozzles
        if x[2] > 0.0 # Top nozzle
            return state_thruster_nozzle_inlet_top(x, t, equations)
        else # Bottom nozzle
            return state_thruster_nozzle_inlet_bottom(x, t, equations)
        end
    end
end

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################

tspan = (0.0, 25.0)
ode = semidiscretize(semi, tspan)

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

amr_indicator = shock_indicator
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      med_level = 2, med_threshold = 0.1,
                                      max_level = 3, max_threshold = 0.2)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 20,
                           adapt_initial_condition = false)

save_restart = SaveRestartCallback(interval = 1000)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution, save_restart,
                        amr_callback)

###############################################################################
# run the simulation

ode_alg = SSPRK43(thread = Trixi.True())
sol = solve(ode, ode_alg;
            adaptive = true, dt = 1e-4,
            ode_default_options()..., callback = callbacks);
