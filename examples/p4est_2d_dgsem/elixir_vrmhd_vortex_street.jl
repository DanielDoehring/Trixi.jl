using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the compressible Euler equations

# Resource for this testcase:
#=
A Discontinuous Galerkin Method for the Viscous MHD Equations
T. C. Warburton and G. E. Karniadakis
Journal of Computational Physics 152.2, 608-641 (1999)
https://doi.org/10.1006/jcph.1999.6248
=#

# Fluid parameters (Warburton & Karniadakis give no value for these)
gamma() = 5 / 3
prandtl_number() = 0.72

# Parameters for compressible von-Karman vortex street
Re() = 500
Ma() = 0.5f0
D() = 1 # Diameter of the cylinder as in the mesh file

# Parameters that can be freely chosen
v_in() = 1
p_in() = 1

# Parameters that follow from Reynolds and Mach number + adiabatic index gamma
mu() = v_in() * D() / Re()

c() = v_in() / Ma()

p_over_rho() = c()^2 / gamma()
rho_in() = p_in() / p_over_rho()

# MHD additions
Alfven_Mach_number() = 0.1
# Use definition of Alven (Mach) number as in Warburton & Karniadakis
Alfven_speed() = v_in() * Alfven_Mach_number()
B_in() = Alfven_speed() * sqrt(rho_in())

S_v() = Alfven_Mach_number() * Re() # viscous Lundquist number
S_r() = S_v() # Use resistive = viscous Lundquist number as in Warburton & Karniadakis
eta() = D() * Alfven_speed() / S_r()

equations = IdealGlmMhdEquations2D(gamma())
equations_parabolic = ViscoResistiveMhdDiffusion2D(equations, mu = mu(),
                                                   Prandtl = prandtl_number(),
                                                   eta = eta(),
                                                   gradient_variables = GradientVariablesPrimitive())

@inline function initial_condition(x, t, equations)
    # set the freestream flow parameters
    rho = rho_in()
    v1 = v_in()
    v2 = 0.0
    v3 = 0.0
    p = p_in()

    B1 = B_in()
    B2 = 0.0
    B3 = 0.0
    psi = 0.0

    prim = SVector(rho, v1, v2, v3, p, B1, B2, B3, psi)
    return prim2cons(prim, equations)
end

# Mesh which is refined around the cylinder and the wake region

mesh_file = Trixi.download("https://gist.githubusercontent.com/DanielDoehring/7312faba9a50ef506b13f01716b4ec26/raw/f08b4610491637d80947f1f2df483c81bd2cb071/cylinder_vortex_street.inp",
                           joinpath(@__DIR__, "cylinder_vortex_street.inp"))

mesh = P4estMesh{2}(mesh_file)

bc_freestream = BoundaryConditionDirichlet(initial_condition)

using LinearAlgebra: norm, dot # for use in the MHD boundary condition
function bc_slip_wall_no_mag(u_inner, normal_direction::AbstractVector,
                             x, t, surface_flux_function,
                             equations)

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

@inline function boundary_condition_copy(flux_inner,
                                         u_inner,
                                         normal::AbstractVector,
                                         x, t,
                                         operator_type::Trixi.Gradient,
                                         equations::ViscoResistiveMhdDiffusion2D{GradientVariablesPrimitive})
    return u_inner
end
@inline function boundary_condition_copy(flux_inner,
                                         u_inner,
                                         normal::AbstractVector,
                                         x, t,
                                         operator_type::Trixi.Divergence,
                                         equations::ViscoResistiveMhdDiffusion2D{GradientVariablesPrimitive})
    return flux_inner
end

# Boundary names are those we assigned in HOHQMesh.jl
boundary_conditions = Dict(:Circle => bc_slip_wall_no_mag, # top half of the cylinder
                           :Circle_R => bc_slip_wall_no_mag, # bottom half of the cylinder
                           :Top => bc_freestream,
                           :Top_R => bc_freestream, # aka bottom
                           :Right => bc_freestream,
                           :Right_R => bc_freestream, 
                           :Left => bc_freestream,
                           :Left_R => bc_freestream)

velocity_bc_free = NoSlip((x, t, equations) -> SVector(v_in(), 0.0, 0.0))
heat_bc_free = Adiabatic((x, t, equations) -> 0.0)
magnetic_bc_free = Isomagnetic((x, t, equations) -> SVector(B_in(), 0.0, 0.0))
boundary_condition_free = BoundaryConditionVRMHDWall(velocity_bc_free, heat_bc_free,
                                                     magnetic_bc_free)

velocity_bc_cylinder = NoSlip((x, t, equations) -> SVector(0.0, 0.0, 0.0))
heat_bc_cylinder = Adiabatic((x, t, equations) -> 0.0)
magnetic_bc_cylinder = Isomagnetic((x, t, equations) -> SVector(0.0, 0.0, 0.0))

boundary_condition_cylinder = BoundaryConditionVRMHDWall(velocity_bc_cylinder,
                                                         heat_bc_cylinder,
                                                         magnetic_bc_cylinder)

boundary_conditions_para = Dict(:Circle => boundary_condition_cylinder, # top half of the cylinder
                                :Circle_R => boundary_condition_cylinder, # bottom half of the cylinder

                                :Top => boundary_condition_copy, #boundary_condition_free,
                                :Top_R => boundary_condition_copy, #boundary_condition_free, # aka bottom

                                :Right => boundary_condition_copy,
                                :Right_R => boundary_condition_copy,

                                :Left => boundary_condition_free,
                                :Left_R => boundary_condition_free)

# Set the numerical fluxes for the volume and the surface contributions.
surface_flux = (flux_hll, flux_nonconservative_powell)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

# Need stabilization for VRMHD case
solver = DGSEM(basis, surface_flux, volume_integral)

# Combine all the spatial discretization components into a high-level descriptions.
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver,
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_para))

###############################################################################
# Setup an ODE problem

tspan = (0.0, 120.0)
#ode = semidiscretize(semi, tspan)
ode = semidiscretize(semi, tspan; split_problem = false)

# Callbacks
summary_callback = SummaryCallback()

# Prints solution errors to the screen at check-in intervals.
analysis_interval = 10_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(alive_interval = 200)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

#cfl = 2.4 # SSPRK54
cfl(t) = min(3.5, 2.4 + t/10 * 1.1)

#=
#cfl = 1.4 # PE (Relaxation) RK 4 13, 8, 6, 5
#cfl = 1.4 # PE (Relaxation) RK 4 13

# Restarted simulation

#cfl = 6.6 # Restarted PERK3 11, 6, 5, 4, 3
#cfl = 6.7 # Restarted PERK4 13, 8, 6, 5

t_ramp_up() = 5.05 # PE Relaxation RK 4
#t_ramp_up() = 5.4 # PERK 4

cfl(t) = min(6.7, 1.4 + t/t_ramp_up() * 5.3)
=#

stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

# Combine all the callbacks into a description.
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        glm_speed_callback,
                        stepsize_callback,
                        save_solution
                        )

###############################################################################
# run the simulation

base_path = "/home/daniel/git/Paper_PERRK/Data/Cylinder_VortexStreet/VRMHD/"

#=
# p = 4
path = base_path * "p4/"

dtRatios = [0.0771545666269958, # 13
            0.0362618269398808, #  8
            0.0154481055215001, #  6
            0.00702102510258555] / 0.0771545666269958 # 5
Stages = [13, 8, 6, 5]

ode_algorithm = Trixi.PairedExplicitRK4(Stages[1], path)
ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages, path, dtRatios)


relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 3)
#ode_algorithm = Trixi.PairedExplicitRelaxationRK4(Stages[1], path)
ode_algorithm = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios; relaxation_solver = relaxation_solver)


sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep = false, callback = callbacks);
=#

sol = solve(ode, SSPRK54(thread = OrdinaryDiffEq.True()),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);


summary_callback() # print the timer summary

using Plots
pd = PlotData2D(sol);

plot(getmesh(pd), xlabel = "\$x\$", ylabel = "\$y \$")

# For level distribution
Trixi2Vtk.trixi2vtk(semi.cache, dtRatios, Stages, "out/solution_000000001.h5")