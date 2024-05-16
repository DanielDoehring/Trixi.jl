# Transonic flow around an airfoil

# This test is taken from the paper below. The values from Case 5 in Table 3 are used to validate
# the scheme and computation of surface forces.

# - Roy Charles Swanson, Stefan Langer (2016)
#   Structured and Unstructured Grid Methods (2016)
#   [https://ntrs.nasa.gov/citations/20160003623] (https://ntrs.nasa.gov/citations/20160003623)

using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

prandtl_number() = 0.72

mu() = 1e-5 # TODO: Revisit this value

equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

# This AoA gives shock on airfoil surface for inviscid NACA0012
# Question: Does this also produce a shock for (viscous) NACA4412?                                                          
AoA = 0.02181661564992912 # 1.25 degreee in radians

@inline function initial_condition_mach08_flow(x, t, equations)
    # set the freestream flow parameters
    rho_freestream = 1.4

    #=
    sin_AoA, cos_AoA = sincos(0.02181661564992912)
    v = 0.8

    v1 = cos_AoA * v
    v2 = sin_AoA * v
    =#

    v1 = 0.7998096216639273
    v2 = 0.017451908027648896

    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach08_flow

# Boundary conditions for free-stream testing
boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

polydeg = 3

#surface_flux = flux_lax_friedrichs
#surface_flux = flux_hll
#surface_flux = flux_hlle
surface_flux = flux_hllc

#volume_flux = flux_ranocha
volume_flux = flux_chandrashekar

basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

# DG Solver                                                 
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/NASA_meshes/"

mesh = "NACA4412/NACA4412_1_2D_unique.inp"

#mesh = "NACA0012/Family_1/113_33/n0012family_1_7_2D_unique.inp"
#mesh = "NACA0012/Family_1/225_65/n0012familyI_6_2D_unique.inp"

mesh_file = path * mesh

boundary_symbols = [:b2_symmetry_y_strong,
                    :b4_farfield_riem, :b5_farfield_riem, :b7_farfield_riem, :b6_viscous_solid, :b8_to_stitch_a]

restart_filename = "out/restart_009500.h5"
#mesh = load_mesh(restart_filename)
mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:b2_symmetry_y_strong => boundary_condition_free_stream,
                           :b4_farfield_riem => boundary_condition_free_stream,
                           :b5_farfield_riem => boundary_condition_free_stream,
                           :b7_farfield_riem => boundary_condition_free_stream,
                           :b6_viscous_solid => boundary_condition_slip_wall,
                           :b8_to_stitch_a => boundary_condition_free_stream)

velocity_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_airfoil = Adiabatic((x, t, equations) -> 0.0)

boundary_conditions_airfoil = BoundaryConditionNavierStokesWall(velocity_airfoil,
                                                                heat_airfoil)

boundary_conditions_parabolic = Dict(:b2_symmetry_y_strong => boundary_condition_free_stream,
                                      :b4_farfield_riem => boundary_condition_free_stream,
                                      :b5_farfield_riem => boundary_condition_free_stream,
                                      :b7_farfield_riem => boundary_condition_free_stream,
                                      :b6_viscous_solid => boundary_conditions_airfoil,
                                      :b8_to_stitch_a => boundary_condition_free_stream)
#=
semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions)
=#
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers

# Run for a long time to reach a steady state
tspan = (0.0, 1)
ode = semidiscretize(semi, tspan)

#tspan = (load_time(restart_filename), 1.0)
#ode = semidiscretize(semi, tspan, restart_filename)

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 10000

#=
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
=#

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                           analysis_errors = Symbol[],
                                           analysis_integrals = ())

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_restart = SaveRestartCallback(interval = 9500,
                                   save_final_restart = true)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                    save_initial_solution = true,
                                    save_final_solution = true,
                                    solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 8.5) # 4412
#stepsize_callback = StepsizeCallback(cfl = 4.0) #0012

amr_controller = ControllerThreeLevel(semi, shock_indicator,
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.05,
                                      max_level = 2, max_threshold = 0.1)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 100,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

callbacks = CallbackSet(summary_callback, 
                        alive_callback, 
                        save_solution,
                        amr_callback,
                        #stepsize_callback,
                        #save_restart,
                        analysis_callback)

###############################################################################
# run the simulation
#=
sol = solve(ode, SSPRK104(thread = OrdinaryDiffEq.True()),
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
=#

sol = solve(ode, SSPRK43(thread = OrdinaryDiffEq.True());
            abstol = 1.0e-7, reltol = 1.0e-7,
            ode_default_options()..., callback = callbacks);

summary_callback() # print the timer summary