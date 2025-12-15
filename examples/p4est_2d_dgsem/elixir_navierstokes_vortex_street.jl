using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Semidiscretization of the compressible Euler equations

# Fluid parameters
const gamma = 5 / 3
const prandtl_number = 0.72

# Parameters for compressible von-Karman vortex street
const Re = 200
const Ma = 0.5f0
const D = 1.0 # Diameter of the cylinder as in the mesh file

# Parameters that can be freely chosen
const v_in = 1.0
const p_in = 1.0

# Parameters that follow from Reynolds and Mach number + adiabatic index gamma
const c = v_in / Ma

const p_over_rho = c^2 / gamma
const rho_in = p_in / p_over_rho

const mu = rho_in * v_in * D / Re

# Equations for this configuration
equations = CompressibleEulerEquations2D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu,
                                                          Prandtl = prandtl_number,
                                                          gradient_variables = GradientVariablesPrimitive())

# Freestream configuration
@inline function initial_condition(x, t, equations::CompressibleEulerEquations2D)
    rho = rho_in
    v1 = v_in
    v2 = 0.0
    p = p_in

    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end

# Symmetric mesh which is refined around the cylinder and in the wake region
mesh_file = Trixi.download("https://gist.githubusercontent.com/DanielDoehring/7312faba9a50ef506b13f01716b4ec26/raw/f08b4610491637d80947f1f2df483c81bd2cb071/cylinder_vortex_street.inp",
                           joinpath(@__DIR__, "cylinder_vortex_street.inp"))
mesh = P4estMesh{2}(mesh_file)

bc_freestream = BoundaryConditionDirichlet(initial_condition)

# Boundary names follow from the mesh file.
# Since this mesh is been generated using the symmetry feature of
# HOHQMesh.jl (https://trixi-framework.github.io/HOHQMesh.jl/stable/tutorials/symmetric_mesh/)
# the mirrored boundaries are named with a "_R" suffix.
boundary_conditions = Dict(:Circle => boundary_condition_slip_wall, # top half of the cylinder
                           :Circle_R => boundary_condition_slip_wall, # bottom half of the cylinder

                           :Top => bc_freestream,
                           :Top_R => bc_freestream, # aka bottom

                           :Right => bc_freestream,
                           :Right_R => bc_freestream,

                           :Left => bc_freestream,
                           :Left_R => bc_freestream)

# Parabolic boundary conditions
velocity_bc_free = NoSlip((x, t, equations) -> SVector(v_in, 0))
# Use adiabatic also on the boundaries to "copy" temperature from the domain
heat_bc_free = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_free = BoundaryConditionNavierStokesWall(velocity_bc_free, heat_bc_free)

velocity_bc_cylinder = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc_cylinder = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_cylinder = BoundaryConditionNavierStokesWall(velocity_bc_cylinder,
                                                                heat_bc_cylinder)

@inline function boundary_condition_copy(flux_inner,
                                         u_inner,
                                         normal::AbstractVector,
                                         x, t,
                                         operator_type::Trixi.Gradient,
                                         equations::CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive})
    return u_inner
end
@inline function boundary_condition_copy(flux_inner,
                                         u_inner,
                                         normal::AbstractVector,
                                         x, t,
                                         operator_type::Trixi.Divergence,
                                         equations::CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive})
    return flux_inner
end

boundary_conditions_para = Dict(:Circle => boundary_condition_cylinder, # top half of the cylinder
                                :Circle_R => boundary_condition_cylinder, # bottom half of the cylinder

                                :Top => boundary_condition_copy, 
                                :Top_R => boundary_condition_copy, # aka bottom

                                :Right => boundary_condition_copy,
                                :Right_R => boundary_condition_copy,

                                :Left => boundary_condition_free,
                                :Left_R => boundary_condition_free)

polydeg = 4 # 3 for PERRK paper
surface_flux = flux_hll
volume_flux = flux_ranocha
solver = DGSEM(polydeg = polydeg, surface_flux = flux_hll)

# NOTE: EC Setup. Gives somewhat oscillatory results, though.
#=
surface_flux = flux_ranocha
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)
=#

#=
# Run also once with SC to compare results
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
=#

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver,
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_para))

###############################################################################
# Setup an ODE problem
tspan = (0.0, 120.0) # For restart file
#ode = semidiscretize(semi, tspan)
ode = semidiscretize(semi, tspan; split_problem = false)

# Callbacks
summary_callback = SummaryCallback()

analysis_interval = 10_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(alive_interval = 200)

# Add `:vorticity` to `extra_node_variables` tuple ...
extra_node_variables = (:vorticity,)

# ... and specify the function `get_node_variable` for this symbol, 
# with first argument matching the symbol (turned into a type via `Val`) for dispatching.
# Note that for parabolic(-extended) equations, `equations_parabolic` and `cache_parabolic`
# must be declared as the last two arguments of the function to match the expected signature.
function Trixi.get_node_variable(::Val{:vorticity}, u, mesh, equations, dg, cache,
                                 equations_parabolic, cache_parabolic)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    vorticity_array = zeros(eltype(cache.elements),
                            n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                            n_elements)

    @unpack viscous_container = cache_parabolic
    @unpack gradients = viscous_container
    gradients_x, gradients_y = gradients

    # We can accelerate the computation by thread-parallelizing the loop over elements
    # by using the `@threaded` macro.
    Trixi.@threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)

            gradients_1 = get_node_vars(gradients_x, equations_parabolic, dg,
                                        i, j, element)
            gradients_2 = get_node_vars(gradients_y, equations_parabolic, dg,
                                        i, j, element)

            vorticity_nodal = vorticity(u_node, (gradients_1, gradients_2),
                                        equations_parabolic)
            vorticity_array[i, j, element] = vorticity_nodal
        end
    end

    return vorticity_array
end

save_solution = SaveSolutionCallback(interval = 100_000,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = extra_node_variables) # Supply the additional `extra_node_variables` here

### k = 3 ###

#cfl = 12.0 # Restarted PERK4 Single 16
#cfl = 7.4 # Restarted PERK4 Multi 16, 10, 7, 6, 5

#cfl = 1.5 # Non-restarted PERK4 Multi 16, 10, 7, 6, 5

# CFL Ramp-up
t_ramp_up() = 4.55 # PE Relaxation RK 4

cfl(t) = min(7.4, 1.5 + t/t_ramp_up() * 5.9)

### k = 4 ###

cfl(t) = min(5.2, 1.0 + t/t_ramp_up() * 4.2)

stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        #analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        save_solution
                        )

###############################################################################
# run the simulation


path = "/home/daniel/git/Paper_PERRK/Data/Cylinder_VortexStreet/NavierStokes/"

dtRatios = [0.0571712958370335, # 16
            0.0279811935336329, # 10
            0.0140505529940128, #  7
            0.00972885200753808,#  6
            0.00620427457615733] / 0.0571712958370335 # 5
Stages = [16, 10, 7, 6, 5]

#ode_algorithm = Trixi.PairedExplicitRK4(Stages[1], path)
#ode_algorithm = Trixi.PairedExplicitRK4Multi(Stages, path, dtRatios)


relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 3)
ode_algorithm = Trixi.PairedExplicitRelaxationRK4Multi(Stages, path, dtRatios; relaxation_solver = relaxation_solver)


sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep = false, callback = callbacks);

#=
using Plots
pd = PlotData2D(sol);
plot(getmesh(pd))
=#
