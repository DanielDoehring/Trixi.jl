using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the compressible Euler equations

# Fluid parameters
gamma() = 5 / 3

# Parameters for compressible von-Karman vortex street
Ma() = 0.5f0
D() = 1 # Diameter of the cylinder as in the mesh file

# Parameters that can be freely chosen
v_in() = 1
p_in() = 1

# Parameters that follow from Reynolds and Mach number + adiabatic index gamma
c() = v_in() / Ma()
p_over_rho() = c()^2 / gamma()
rho_in() = p_in() / p_over_rho()

# Equations for this configuration
equations = CompressibleEulerEquations2D(gamma())

# Freestream configuration
@inline function initial_condition(x, t, equations::CompressibleEulerEquations2D)
    rho = rho_in()
    v1 = v_in()
    v2 = 0.0
    p = p_in()

    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end

# Mesh which is refined around the cylinder and the wake region
mesh_file = Trixi.download("https://gist.githubusercontent.com/DanielDoehring/7312faba9a50ef506b13f01716b4ec26/raw/8e68f9006e634905544207ca322bc0a03a9313ad/cylinder_vortex_street.inp",
                           joinpath(@__DIR__, "cylinder_vortex_street.inp"))
mesh = P4estMesh{2}(mesh_file)

bc_freestream = BoundaryConditionDirichlet(initial_condition)

using LinearAlgebra: norm, dot # for use in the MHD boundary condition
function boundary_condition_velocity_slip_wall(u_inner, normal_direction::AbstractVector,
                                               x, t, surface_flux_function,
                                               equations::CompressibleEulerEquations2D)

    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    norm_ = norm(normal_direction)
    normal = normal_direction / norm_

    # compute the primitive variables
    rho, v1, v2, p = cons2prim(u_inner, equations)

    # Impose no magnetic field on cylinder
    B1 = B2 = B3 = 0.0

    v_normal = dot(normal, SVector(v1, v2))
    u_mirror = prim2cons(SVector(rho, v1 - 2 * v_normal * normal[1],
                                 v2 - 2 * v_normal * normal[2],
                                 p), equations)

    return surface_flux_function(u_inner, u_mirror, normal, equations) * norm_
end

# Boundary names follow from the mesh file
boundary_conditions = Dict(:Bottom => bc_freestream,
                           :Circle => boundary_condition_velocity_slip_wall,
                           :Top => bc_freestream,
                           :Right => bc_freestream,
                           :Left => bc_freestream)

surface_flux = flux_hll
volume_flux = flux_ranocha

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

# For inviscid case: Stabilization required                                                 
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# Setup an ODE problem
tspan = (0, 100)
ode = semidiscretize(semi, tspan)

# Callbacks
summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback
                        #save_solution
                        )

###############################################################################
# run the simulation

time_int_tol = 1e-7
sol = solve(ode,
            # Moderate number of threads (e.g. 4) advisable to speed things up
            RDPK3SpFSAL49(thread = OrdinaryDiffEq.True());
            dt = 1e-4, abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)

summary_callback() # print the timer summary
