using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

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

boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

polydeg = 3

surface_flux = flux_hllc
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

path = "/home/daniel/Meshes/NASA_meshes/"

mesh = "NACA4412/225_65/NACA4412_2_2D_unique.inp"

mesh_file = path * mesh

boundary_symbols = [:b2_symmetry_y_strong, :b4_farfield_riem, :b5_farfield_riem, 
                    :b7_farfield_riem, :b6_viscous_solid, :b8_to_stitch_a]
mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:b2_symmetry_y_strong => boundary_condition_free_stream,
                           :b4_farfield_riem => boundary_condition_free_stream,
                           :b5_farfield_riem => boundary_condition_free_stream,
                           :b7_farfield_riem => boundary_condition_free_stream,
                           :b6_viscous_solid => boundary_condition_slip_wall,
                           :b8_to_stitch_a => boundary_condition_free_stream)

semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition, solver;
                                    boundary_conditions = boundary_conditions)


###############################################################################
# ODE solvers

# Run for a long time to reach a steady state
#tspan = (0.0, 10)
#ode = semidiscretize(semi, tspan)
#dt = 1e-6

restart_filename = "out/restart_602504.h5"
tspan = (load_time(restart_filename), 10.5)
ode = semidiscretize(semi, tspan, restart_filename)
dt = load_dt(restart_filename)

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 50000
analysis_interval = 2000 # For AMR run

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                           analysis_errors = Symbol[],
                                           analysis_integrals = ())

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_restart = SaveRestartCallback(interval = analysis_interval+1,
                                   save_final_restart = true)

save_solution = SaveSolutionCallback(interval = Int(10^8), # For AMR run, else = analysis_interval
                                    save_initial_solution = true,
                                    save_final_solution = true,
                                    solution_variables = cons2prim)

amr_controller = ControllerThreeLevel(semi, shock_indicator,
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.05,
                                      max_level = 2, max_threshold = 0.1)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 200,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

callbacks = CallbackSet(summary_callback, 
                        #alive_callback, # Not needed for AMR run (print analysis_callback more often)
                        save_solution,
                        amr_callback,
                        #save_restart,
                        analysis_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(thread = OrdinaryDiffEq.True());
            abstol = 5.0e-7, reltol = 5.0e-7,
            dt = dt,
            ode_default_options()..., callback = callbacks);

summary_callback() # print the timer summary