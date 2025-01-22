using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the compressible Euler equations

# Setup the governing equations, initial condition and boundary conditions
# Fluid parameters
gamma() = 5 / 3 # Not specified by Warburton and Karniadakis

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

# MHD additions
Alfven() = 0.1
B_in() = Alfven() * rho_in()

equations = IdealGlmMhdEquations2D(gamma())

@inline function initial_condition_mach05_flow(x, t, equations::IdealGlmMhdEquations2D)
    rho = rho_in()
    v1 = v_in()
    v2 = 0.0
    v3 = 0.0
    p = p_in()

    A = 0.1
    B1 = B_in()
    B2 = 0.0
    B3 = 0.0
    psi = 0.0

    prim = SVector(rho, v1, v2, v3, p, B1, B2, B3, psi)
    return prim2cons(prim, equations)
end

# Mesh which is refined around the cylinder and the wake region
mesh_file = Trixi.download("https://gist.githubusercontent.com/DanielDoehring/7312faba9a50ef506b13f01716b4ec26/raw/8e68f9006e634905544207ca322bc0a03a9313ad/cylinder_vortex_street.inp",
                           joinpath(@__DIR__, "cylinder_vortex_street.inp"))
mesh = P4estMesh{2}(mesh_file)

bc_freestream = BoundaryConditionDirichlet(initial_condition_mach05_flow)

using LinearAlgebra: norm, dot # for use in the MHD boundary condition
function boundary_condition_velocity_slip_wall(u_inner, normal_direction::AbstractVector,
                                               x, t, surface_flux_function,
                                               equations::IdealGlmMhdEquations2D)

    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    norm_ = norm(normal_direction)
    normal = normal_direction / norm_

    # compute the primitive variables
    rho, v1, v2, v3, p, _, _, _, psi = cons2prim(u_inner, equations)

    # Impose no magnetic field on cylinder
    B1 = B2 = B3 = 0.0

    v_normal = dot(normal, SVector(v1, v2))
    u_mirror = prim2cons(SVector(rho,
                                 v1 - 2 * v_normal * normal[1],
                                 v2 - 2 * v_normal * normal[2],
                                 v3, p, B1, B2, B3, psi), equations)

    return surface_flux_function(u_inner, u_mirror, normal, equations) * norm_
end

# Boundary names are those we assigned in HOHQMesh.jl
boundary_conditions = Dict(:Bottom => bc_freestream,
                           :Circle => boundary_condition_velocity_slip_wall,
                           :Top => bc_freestream,
                           :Right => bc_freestream,
                           :Left => bc_freestream)

# Set the numerical fluxes for the volume and the surface contributions.
surface_flux = (flux_hll, flux_nonconservative_powell)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
polydeg = 3

# Need stabilization for Mach 0.5 (inviscid)
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

# Combine all the spatial discretization components into a high-level descriptions.
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_mach05_flow, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# Setup an ODE problem
tspan = (0.0, 100.0)
ode = semidiscretize(semi, tspan)

# Callbacks
summary_callback = SummaryCallback()

# Prints solution errors to the screen at check-in intervals.
analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

cfl = 1.5
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

# Combine all the callbacks into a description.
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        glm_speed_callback,
                        stepsize_callback
                        #save_solution
                        )

###############################################################################
# run the simulation

sol = solve(ode, SSPRK54(thread = OrdinaryDiffEq.True()),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
