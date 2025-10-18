using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

gamma() = 1.4
equations = CompressibleEulerEquations2D(gamma())

p_inf() = 1.0
rho_inf() = gamma() # Gives unit speed of sound c_inf = 1.0
mach_inf() = 0.8
aoa() = deg2rad(1.25) # 1.25 Degree angle of attack

@inline function initial_condition_mach08_flow(x, t,
                                               equations::CompressibleEulerEquations2D)
    v1 = 0.7998096216639273   # 0.8 * cos(aoa())
    v2 = 0.017451908027648896 # 0.8 * sin(aoa())

    prim = SVector(1.4, v1, v2, 1.0)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_mach08_flow

surface_flux = flux_lax_friedrichs
volume_flux = flux_chandrashekar

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
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

mesh_file = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/C1.2_quadgrids/naca_ref2_quadr_relabel.inp"

boundary_symbols = [:Airfoil, :Inflow, :Outflow]
mesh = P4estMesh{2}(mesh_file, boundary_symbols = boundary_symbols)

bc_farfield = BoundaryConditionDirichlet(initial_condition)

boundary_conditions = Dict(:Inflow => bc_farfield,
                           :Outflow => bc_farfield,
                           :Airfoil => boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers

restart_file = "restart_ref2_t100.h5"
restart_filename = joinpath("/home/daniel/git/Paper_PERRK/Data/NACA0012/", restart_file)

tspan = (load_time(restart_filename), 200.0)
ode = semidiscretize(semi, tspan, restart_filename)

# Callbacks

summary_callback = SummaryCallback()

save_sol_interval = 500_000

# Add `:Mach` to `extra_node_variables` tuple ...
extra_node_variables = (:Mach,)

# ... and specify the function `get_node_variable` for this symbol, 
# with first argument matching the symbol (turned into a type via `Val`) for dispatching.
# Note that for parabolic(-extended) equations, `equations_parabolic` and `cache_parabolic`
# must be declared as the last two arguments of the function to match the expected signature.
function Trixi.get_node_variable(::Val{:Mach}, u, mesh, equations, dg, cache)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    Mach_array = zeros(eltype(cache.elements),
                            n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                            n_elements)

    # We can accelerate the computation by thread-parallelizing the loop over elements
    # by using the `@threaded` macro.
    Trixi.@threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            prim_node = cons2prim(u_node, equations)
            
            sound_speed = sqrt(equations.gamma * prim_node[4] / prim_node[1])
            velocity_magnitude = sqrt(prim_node[2]^2 + prim_node[3]^2)

            Mach_nodal = velocity_magnitude / sound_speed
            Mach_array[i, j, element] = Mach_nodal
        end
    end

    return Mach_array
end

save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = extra_node_variables)

l_inf = 1.0 # Length of airfoil
force_boundary_names = (:Airfoil,)
u_inf() = mach_inf()
drag_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           DragCoefficientPressure2D(aoa(), rho_inf(),
                                                                   u_inf(), l_inf))

lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           LiftCoefficientPressure2D(aoa(), rho_inf(),
                                                                   u_inf(), l_inf))

pressure_coefficient = AnalysisSurfacePointwise(force_boundary_names,
                                                SurfacePressureCoefficient(p_inf(), rho_inf(),
                                                                           u_inf(), l_inf))

analysis_interval = 500_000 # Only at the end                                                                         
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     analysis_errors = Symbol[],
                                     save_analysis = true,
                                     analysis_integrals = (drag_coefficient,
                                                           lift_coefficient),
                                     analysis_pointwise = (pressure_coefficient,)
                                    )

alive_callback = AliveCallback(alive_interval = 1000)


cfl = 2.8 # Standard PE(R)RK4 Multi/Standalone

#cfl = 0.9 # R-RK44
#cfl = 1.1 # R-TS64
#cfl = 1.5 # R-CKL54

stepsize_callback = StepsizeCallback(cfl = cfl)

amr_indicator = shock_indicator
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.05, # 1
                                      max_level = 3, max_threshold = 0.1)  # 3

amr_ref_interval = 200
cfl_ref = 2.8
amr_interval = Int(ceil(amr_ref_interval * cfl_ref/cfl))

amr_callback = AMRCallback(semi, amr_controller,
                           interval = amr_interval,
                           adapt_initial_condition = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, 
                        alive_callback,
                        save_solution,
                        stepsize_callback,
                        amr_callback
                        )

###############################################################################
# run the simulation

dtRatios_complete_p4 = [ 
    0.653209035337363,
    0.530079549682015,
    0.398295542137155,
    0.326444525366249,
    0.282355465161903,
    0.229828402151329,
    0.163023514708386,
    0.085186504038755] ./ 0.653209035337363
Stages_complete_p4 = [14, 12, 10, 9, 8, 7, 6, 5]

dtRatios_p4 = [ 
    0.653209035337363,
    0.530079549682015,
    0.398295542137155,
    0.282355465161903,
    0.229828402151329,
    0.163023514708386,
    0.085186504038755] ./ 0.653209035337363
Stages_p4 = [14, 12, 10, 8, 7, 6, 5]

path = "/home/daniel/git/MA/EigenspectraGeneration/PERK4/NACA0012_Mach08/rusanov_chandrashekar/"

relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 5, root_tol = 1e-12)

#ode_alg = Trixi.PairedExplicitRK4Multi(Stages_p4, path, dtRatios_p4)
ode_alg = Trixi.PairedExplicitRelaxationRK4Multi(Stages_p4, path, dtRatios_p4; relaxation_solver = relaxation_solver)

#ode_alg = Trixi.PairedExplicitRelaxationRK4(Stages_p4[1], path; relaxation_solver = relaxation_solver)
#ode_alg = Trixi.PairedExplicitRK4(Stages_p4[1], path)

# NOTE: For some reason `prolong2mortars` massive allocates if the multirate version is not executed before

#ode_alg = Trixi.RelaxationRK44(; relaxation_solver = relaxation_solver)
#ode_alg = Trixi.RK44()

#ode_alg = Trixi.RelaxationTS64(; relaxation_solver = relaxation_solver)
#ode_alg = Trixi.TS64()

#ode_alg = Trixi.RelaxationCKL54(; relaxation_solver = relaxation_solver)
#ode_alg = Trixi.CKL54()

sol = Trixi.solve(ode, ode_alg, dt = 42.0, 
                  save_everystep = false, callback = callbacks);