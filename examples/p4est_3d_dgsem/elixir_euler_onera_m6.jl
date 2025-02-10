using Trixi
using OrdinaryDiffEq

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

@inline function initial_condition(x, t, equations::CompressibleEulerEquations3D)
    # set the freestream flow parameters
    rho_freestream = 1.4
    v1 = 0.0 # 0.84
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

polydeg = 3

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

#mesh_file = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/OneraM6/mesh_ONERAM6_turb_hexa_43008_Trixi.inp"
mesh_file = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/OneraM6/mesh_ONERAM6_turb_hexa_43008_rev_Trixi.inp"

#mesh_file = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/OneraM6/NASA/m6wing_Trixi.inp"
#mesh_file = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/PERK_mesh/SD7003Turbulent/sd7003_straight_Trixi.inp"

boundary_symbols = [:PhysicalSurface1, :PhysicalSurface2, :PhysicalSurface3]

#mesh = P4estMesh{3}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)
mesh = P4estMesh{3}(mesh_file, polydeg = polydeg)

boundary_conditions = Dict(:PhysicalSurface1 => bc_farfield, # Far-field / outer
                           :PhysicalSurface2 => bc_farfield, # Wing: boundary_condition_slip_wall
                           :PhysicalSurface3 => bc_farfield # Symmetry: bc_symmetry
                          )

boundary_conditions = Dict(:all => bc_farfield)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 1e-5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(alive_interval = 100)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        stepsize_callback
                        )

# Run the simulation
###############################################################################

sol = solve(ode, SSPRK104(; thread = OrdinaryDiffEq.True());
            dt = 1e-10, # overwritten by the `stepsize_callback`
            save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
