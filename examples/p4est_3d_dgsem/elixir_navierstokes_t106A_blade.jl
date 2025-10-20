using OrdinaryDiffEqLowStorageRK
using Trixi
import LinearAlgebra: norm

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

function initial_condition_Ma015(x, t, equations::CompressibleEulerEquations3D)
    # Choose rho, p s.t. c = 1
    rho = 1
    p = 1.4

    v1 = 0.15
    v2 = 0
    v3 = 0

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end
initial_condition = initial_condition_Ma015

polydeg = 2
basis = LobattoLegendreBasis(polydeg)

surface_flux = flux_hll
volume_flux = flux_ranocha
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 1.0,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)


case_path = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/CS2/T106A_HOW4/"

#mesh_file = case_path * "coarse/t106A_3D_coarse_GE_fixed.inp"
mesh_file = case_path * "baseline/t106A_3D_baseline_GE_fixed.inp"

# 13: Front (z+) x-y plane
#  1: Back (z-) x-y plane

#  2: Blade

## From here, the faces are numbered counter-clockwise ##

#  3: Top-left right-tilted;     x-z plane
#  4: Left vertical, "inlet";    y-z plane
#  5: Bottom-left right-tilted;  x-z plane
#  6: Bottom, horizontal;        x-z plane
#  7: Bottom-right, left-tilted; x-z plane
#  8: Bottom, horizontal;        x-z plane
#  9: Right vertical, "outlet";  y-z plane
# 10: Top, horizontal;           x-z plane
# 11: Top-right left-tilted;     x-z plane
# 12: Top, horizontal;           x-z plane

boundary_symbols = [:Front, :Back, :Blade, 
                    :PhysicalSurface4, :PhysicalSurface9,
                    :PhysicalSurface3, :PhysicalSurface5, :PhysicalSurface6, :PhysicalSurface7,
                    :PhysicalSurface8, :PhysicalSurface10, :PhysicalSurface11, :PhysicalSurface12]

mesh = P4estMesh{3}(mesh_file; boundary_symbols = boundary_symbols)

# TODO: Bundle nodesets with same boundary condition for efficiency ?

bc_inlet = BoundaryConditionDirichlet(initial_condition)

# Calculate the boundary flux from the inner state while using the pressure from the outer state
# when the flow is subsonic.
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
                                                    equations::CompressibleEulerEquations3D)
    # Rotate the internal solution state
    #=
    norm_ = norm(normal_direction)
    normal = normal_direction / norm_
    u_local = Trixi.rotate_to_x(u_inner, normal, equations)
    rho_local, v_x, v_y, v_z, p_local = cons2prim(u_local, equations)
    =#

    rho_local, v_x, v_y, v_z, p_local = cons2prim(u_inner, equations)

    # Compute local Mach number
    a_local = sqrt(equations.gamma * p_local / rho_local)
    Mach_local = abs(v_x / a_local)
    if Mach_local <= 1.0
        # In general, `p_local` need not be available from the initial condition
        p_local = pressure(initial_condition_Ma015(x, t, equations), equations)
    end

    # Create the `u_surface` solution state where the local pressure is possibly set from an external value
    prim = SVector(rho_local, v_x, v_y, v_z, p_local)
    u_boundary = prim2cons(prim, equations)

    #u_surface = Trixi.rotate_from_x(u_boundary, normal, equations)
    u_surface = u_boundary

    return flux(u_surface, normal_direction, equations)
end
boundary_condition_outflow = boundary_condition_outflow_general

# Ensure that rho and p are the same across symmetry line and allow only 
# tangential velocity
@inline function bc_symmetry(u_inner, normal_direction::AbstractVector, x, t,
                             surface_flux_function, equations)
    norm_ = norm(normal_direction)
    normal = normal_direction / norm_

    # compute the primitive variables
    rho, v1, v2, v3, p = cons2prim(u_inner, equations)

    v_normal = normal[1] * v1 + normal[2] * v2 + normal[3] * v3

    u_mirror = prim2cons(SVector(rho,
                                 v1 - 2 * v_normal * normal[1],
                                 v2 - 2 * v_normal * normal[2],
                                 v3 - 2 * v_normal * normal[3],
                                 p), equations)

    flux = surface_flux_function(u_inner, u_mirror, normal, equations) * norm_

    return flux
end

boundary_conditions = Dict(:PhysicalSurface4 => bc_inlet,
                           :Blade => boundary_condition_slip_wall,

                           :Front => bc_symmetry,
                           :Back => bc_symmetry,

                           :PhysicalSurface3 => boundary_condition_outflow,
                           :PhysicalSurface5 => boundary_condition_outflow,
                           :PhysicalSurface6 => boundary_condition_outflow,
                           :PhysicalSurface7 => boundary_condition_outflow,
                           :PhysicalSurface8 => boundary_condition_outflow,
                           :PhysicalSurface9 => boundary_condition_outflow,
                           :PhysicalSurface10 => boundary_condition_outflow,
                           :PhysicalSurface11 => boundary_condition_outflow,
                           :PhysicalSurface12 => boundary_condition_outflow)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

# TODO: Figure out relevant length scale to define convective time
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(alive_interval = 100)

stepsize_callback = StepsizeCallback(cfl = 2.0)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     save_initial_solution = false)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);

