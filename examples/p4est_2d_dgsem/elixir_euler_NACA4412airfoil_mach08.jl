using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)
                                                      
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

base_path = "/home/daniel/git/Paper_PERRK/Data/NACA4412_Transonic/"

mesh = "NACA4412_2_2D_unique.inp"
mesh_file = base_path * mesh

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

# Run until shock position is stable
tspan = (0.0, 10)
ode = semidiscretize(semi, tspan)

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 1000 # 50_000

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = ())

alive_callback = AliveCallback(alive_interval = 100)

save_restart = SaveRestartCallback(interval = analysis_interval+1,
                                   save_final_restart = true)

save_solution = SaveSolutionCallback(interval = analysis_interval, 
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 3.0) # CarpenterKennedy2N54
stepsize_callback = StepsizeCallback(cfl = 2.0) # PERK4_5

stepsize_callback = StepsizeCallback(cfl = 10.0)

callbacks = CallbackSet(summary_callback, 
                        alive_callback,
                        save_solution,
                        stepsize_callback,
                        #save_restart,
                        analysis_callback)

###############################################################################
# run the simulation

ode_alg = Trixi.PairedExplicitRK4(5, base_path)
#ode_alg = Trixi.PairedExplicitRelaxationRK4(5, base_path)

dtRatios_complete = [ 0.144554254710674,
                      0.135740657877177,
                      0.126721346955746,
                      0.118815420325845,
                      0.107953248750418,
                      0.0999740357324481,
                      0.0918755314312875,
                      0.0847490527667105,
                      0.0752368071116507,
                      0.066301641985774,
                      0.059096378646791,
                      0.0482352262362838,
                      0.0433334739878774,
                      0.0325130925513804,
                      0.0237590761855245,
                      0.0184619000181556
                      ] ./ 0.144554254710674

Stages_complete = reverse(collect(range(5, 20)))

ode_alg = Trixi.PairedExplicitRelaxationRK4Multi(Stages_complete, base_path, dtRatios_complete)

sol = Trixi.solve(ode, ode_alg, dt = 42.0, 
                  save_everystep = false, callback = callbacks);

#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false, thread = OrdinaryDiffEq.True()),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

=#

#=
sol = solve(ode, SSPRK43(thread = OrdinaryDiffEq.True());
            abstol = 5.0e-7, reltol = 5.0e-7,
            dt = 1e-6,
            ode_default_options()..., callback = callbacks);
=#

summary_callback() # print the timer summary