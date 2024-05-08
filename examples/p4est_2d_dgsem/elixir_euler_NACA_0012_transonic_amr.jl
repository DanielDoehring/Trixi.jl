
using Trixi
using OrdinaryDiffEq
using Downloads: download

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

AoA = 0.02181661564992912 # 1.25 degreee in radians

@inline function initial_condition_mach08_flow(x, t, equations::CompressibleEulerEquations2D)
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

surface_flux = flux_lax_friedrichs
#surface_flux = flux_hll
#surface_flux = flux_hlle
#surface_flux = flux_hllc

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

mesh_file = "/home/daniel/PERK4/NACA0012_Mach08/NACA0012_fine.inp"

boundary_symbols = [:PhysicalLine1, :PhysicalLine2, :PhysicalLine3, :PhysicalLine4]

mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:PhysicalLine1 => boundary_condition_free_stream, # Left boundary
                           :PhysicalLine2 => boundary_condition_free_stream, # Right boundary
                           :PhysicalLine3 => boundary_condition_free_stream, # Top and bottom boundary 
                           :PhysicalLine4 => boundary_condition_slip_wall) # Airfoil

#restart_file = "restart_011044.h5"
#restart_filename = joinpath("out", restart_file)
#mesh = load_mesh(restart_filename)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

#tspan = (load_time(restart_filename), 40)
#ode = semidiscretize(semi, tspan, restart_filename)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                      analysis_errors = Symbol[],
                                      analysis_integrals = ())

# HLLC, Ranocha

# PERK Multi 10      
stepsize_callback = StepsizeCallback(cfl = 4.1)

#=
# PERK Single 14
stepsize_callback = StepsizeCallback(cfl = 5.2)
# PERK Single 12
stepsize_callback = StepsizeCallback(cfl = 5.2)
# PERK Single 10
stepsize_callback = StepsizeCallback(cfl = 5.2)
=#

# NDBLSRK144
#stepsize_callback = StepsizeCallback(cfl = 8.5)
# DGLDDRK84_C
#stepsize_callback = StepsizeCallback(cfl = 4.7)
# SSPRK104
#stepsize_callback = StepsizeCallback(cfl = 8.9)
# SSPRK54
#stepsize_callback = StepsizeCallback(cfl = 3.5)
# RK 4
#stepsize_callback = StepsizeCallback(cfl = 1.9)

#=
shock_indicator = IndicatorLöhner(semi, variable = Trixi.density_pressure)

amr_controller = ControllerThreeLevel(semi, shock_indicator,
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.05,
                                      max_level = 3, max_threshold = 0.1)
                                  
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 40,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)
=#

save_solution = SaveSolutionCallback(interval = 5000,
                                    save_initial_solution = false,
                                    save_final_solution = true,
                                    solution_variables = cons2prim,
                                    output_directory="run/out")       
                                    
save_restart = SaveRestartCallback(interval = 1_000_000,
                                    save_final_restart = true)                                    

alive_callback = AliveCallback(alive_interval = 200)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        #analysis_callback,
                        #save_solution,
                        #amr_callback,
                        #save_restart,
                        stepsize_callback)

# Run the simulation
###############################################################################

# HLLC, Ranocha

dtRatios = [1.043512327234554, # 16
            0.952243167620255, # 15
            0.848507513673069, # 14
            0.772868667566991, # 13
            0.678431370242512, # 12
            0.604705493810468, # 11
            0.517535517327684, # 10
            0.434854974637155, #  9
            0.372977221722726, #  8
            0.287058669803322, #  7
            0.231985507171444, #  6
            0.152610467632869] #= 5 =# / 1.043512327234554
Stages = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]

#=
dtRatios = [0.517535517327684, # 10
            0.434854974637155, #  9
            0.372977221722726, #  8
            0.287058669803322, #  7
            0.231985507171444, #  6
            0.152610467632869] #= 5 =# / 0.517535517327684
Stages = [10, 9, 8, 7, 6, 5]
=#

# LLF, Chandrashekar

dtRatios = [0.653209035337363, # 14
            0.530079549682015, # 12
            0.398295542137155, # 10
            0.326444525366249, #  9
            0.282355465161903, #  8
            0.229828402151329, #  7
            0.163023514708386, #  6
            0.085186504038755] #= 5 =# / 0.653209035337363
Stages = [14, 12, 10, 9, 8, 7, 6, 5]

dtRatios = [0.398295542137155, # 10
            0.326444525366249, #  9
            0.282355465161903, #  8
            0.229828402151329, #  7
            0.163023514708386, #  6
            0.085186504038755] #= 5 =# / 0.398295542137155
Stages = [10, 9, 8, 7, 6, 5]

ode_algorithm = PERK4_Multi(Stages, "/home/daniel/PERK4/NACA0012_Mach08/", dtRatios)
#ode_algorithm = PERK4(8, "/home/daniel/PERK4/NACA0012_Mach08/")

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary


#ode_algorithm = NDBLSRK144(williamson_condition = false, thread = OrdinaryDiffEq.True())
#ode_algorithm = DGLDDRK84_C(williamson_condition = false, thread = OrdinaryDiffEq.True())
#ode_algorithm = SSPRK104(; thread = OrdinaryDiffEq.True())
ode_algorithm = SSPRK54(; thread = OrdinaryDiffEq.True())
#ode_algorithm = RK4(; thread = OrdinaryDiffEq.True())

sol = solve(ode, ode_algorithm;
            dt = 1.0, # overwritten by the `stepsize_callback`
            save_everystep = false,
            adaptive = false,
            callback = callbacks);

summary_callback() # print the timer summary