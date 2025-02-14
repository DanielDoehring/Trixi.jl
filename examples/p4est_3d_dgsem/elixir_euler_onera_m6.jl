using Trixi
using OrdinaryDiffEq

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

# TODO: AoA: 3.06 deg or 6.06 deg

@inline function initial_condition(x, t, equations::CompressibleEulerEquations3D)
    # set the freestream flow parameters
    rho_freestream = 1.4
    v1 = 0.1 # 0.84
    v2 = 0.0
    v3 = 0.0
    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, v3, p_freestream)
    return prim2cons(prim, equations)
end

bc_farfield = BoundaryConditionDirichlet(initial_condition)

# Use simple outflow/extended domain at symmetry line
@inline function bc_symmetry(u_inner, normal_direction::AbstractVector, x, t,
                              surface_flux_function,
                              equations::CompressibleEulerEquations3D)

    flux = Trixi.flux(u_inner, normal_direction, equations)
    return flux
end

polydeg = 2

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

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
#=
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)
=#
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)
#mesh_file = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/OneraM6/NASA/m6wing_Trixi_remeshed_bnds.inp"
mesh_file = "/storage/home/daniel/PERRK/Data/OneraM6/m6wing_Trixi_remeshed_bnds.inp"

boundary_symbols = [:PhysicalSurface2, # "symm1"
                    :PhysicalSurface4, # "out1"
                    :PhysicalSurface7, # "far1"
                    :PhysicalSurface8, # "symm2"
                    :PhysicalSurface12, # "bwing" = bottom wing I guess
                    :PhysicalSurface13, # "far2"
                    :PhysicalSurface14, # "symm3"
                    :PhysicalSurface18, # "twing" = top wing I guess
                    :PhysicalSurface19, # "far3"
                    :PhysicalSurface20, # "symm4"
                    :PhysicalSurface23, # "out4"
                    :PhysicalSurface25, # "far4"
                    ]

mesh = P4estMesh{3}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:PhysicalSurface2 => bc_symmetry, # Symmetry: bc_symmetry
                           :PhysicalSurface4 => bc_farfield, # Farfield: bc_farfield
                           :PhysicalSurface7 => bc_farfield, # Farfield: bc_farfield
                           :PhysicalSurface8 => bc_symmetry, # Symmetry: bc_symmetry
                           :PhysicalSurface12 => boundary_condition_slip_wall, # Wing: bc_slip_wall
                           :PhysicalSurface13 => bc_farfield, # Farfield: bc_farfield
                           :PhysicalSurface14 => bc_symmetry, # Symmetry: bc_symmetry
                           :PhysicalSurface18 => boundary_condition_slip_wall, # Wing: bc_slip_wall
                           :PhysicalSurface19 => bc_farfield, # Farfield: bc_farfield
                           :PhysicalSurface20 => bc_symmetry, # Symmetry: bc_symmetry
                           :PhysicalSurface23 => bc_farfield, # Farfield: bc_farfield
                           :PhysicalSurface25 => bc_farfield, # Farfield: bc_farfield
                          )

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(alive_interval = 100)

# Works for SSPRK54 (but probably not maxed out)
stepsize_callback = StepsizeCallback(cfl = 5.0)

save_sol_interval = 1000
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        #stepsize_callback,
                        save_solution
                        )

# Run the simulation
###############################################################################

sol = solve(ode, SSPRK43(; thread = OrdinaryDiffEq.True());
            dt = 1e-10, # overwritten by the `stepsize_callback`
            save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
