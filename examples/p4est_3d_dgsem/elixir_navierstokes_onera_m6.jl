using Trixi
using LinearAlgebra: norm

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

height_ref = 1.1963

mean_aero_chord = 0.64607 / height_ref

rho_inf() = 1.4
u_inf() = 0.84 # NOTE: True Mach = 0.8395
Re = 11.72 * 10^6 # https://www.grc.nasa.gov/www/wind/valid/m6wing/m6wing.html
mu() = rho_inf() * u_inf() * mean_aero_chord / Re
prandtl_number() = 0.72 # or maybe 0.71

equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

@inline function initial_condition(x, t, equations)
    # set the freestream flow parameters
    rho_freestream = 1.4

    # v_total = 0.84 = Mach

    # AoA = 3.06
    v1 = 0.8388023121403883
    v2 = 0.0448406193973588
    v3 = 0.0

    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, v3, p_freestream)
    return prim2cons(prim, equations)
end

bc_farfield = BoundaryConditionDirichlet(initial_condition)

# Ensure that rho and p are the same across symmetry line and allow only 
# tangential velocity
@inline function bc_symmetry(u_inner, normal_direction::AbstractVector, x, t,
                             surface_flux_function, equations)
    norm_ = norm(normal_direction)
    normal = normal_direction / norm_

    # compute the primitive variables
    rho, v1, v2, v3, p = cons2prim(u_inner, equations)

    v_normal = normal[1] * v1 + normal[2] * v2 + normal[3] * v3

    u_mirror = prim2cons(SVector(rho,
                                 v1 - 2 * v_normal * normal[1],
                                 v2 - 2 * v_normal * normal[2],
                                 v3 - 2 * v_normal * normal[3],
                                 p), equations)

    flux = surface_flux_function(u_inner, u_mirror, normal, equations) * norm_

    return flux
end

polydeg = 2
basis = LobattoLegendreBasis(polydeg)

shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true, # true
                                            variable = density_pressure)

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

# NOTE: Flux Differencing is required, shock capturing not (at least not for simply running the code)
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

#mesh_path = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/OneraM6/NASA/"
mesh_path = "/storage/home/daniel/PERRK/Data/OneraM6/"

mesh_file = mesh_path * "m6wing_sanitized.inp"

boundary_symbols = [:Symmetry, :FarField, :BottomWing, :TopWing]

mesh = P4estMesh{3}(mesh_file, boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:Symmetry => bc_symmetry, # Symmetry: bc_symmetry
                           :FarField => bc_farfield, # Farfield: bc_farfield
                           :BottomWing => boundary_condition_slip_wall, # Wing: bc_slip_wall
                           :TopWing => boundary_condition_slip_wall)

velocity_bc_wing = NoSlip((x, t, equations) -> SVector(0.0, 0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
bc_wing = BoundaryConditionNavierStokesWall(velocity_bc_wing, heat_bc)

bc_symmetry_plane_para = BoundaryConditionNavierStokesWall(Slip(), heat_bc)

boundary_conditions_para = Dict(:Symmetry => bc_symmetry_plane_para, # Symmetry: bc_symmetry_plane_para
                                :FarField => bc_farfield, # Farfield: bc_farfield
                                :BottomWing => bc_wing, # Wing: bc_no_slip
                                :TopWing => bc_wing)

semi_hyp_para = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_para))

restart_file = "restart_t605_undamped.h5"

restart_filename = joinpath("/storage/home/daniel/OneraM6/", restart_file)
#restart_filename = joinpath("/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/OneraM6/NASA/restart_files/k2/", restart_file)

tspan = (load_time(restart_filename), 6.0491) # t_c = 9.5
#tspan = (load_time(restart_filename), 6.05) # t_c = 9.5

ode = semidiscretize(semi_hyp_para, tspan, restart_filename) # Split methods
#ode = semidiscretize(semi_hyp_para, tspan, restart_filename; split_problem = false) # Unsplit methods

# Callbacks
###############################################################################

summary_callback = SummaryCallback()

force_boundary_names = (:BottomWing, :TopWing)

aoa() = deg2rad(3.06)

rho_inf() = 1.4
u_inf() = 0.84
# Area calculated from information given at https://www.grc.nasa.gov/www/wind/valid/m6wing/m6wing.html

height = 1.0 # Mesh we use normalizes wing height to one

g_I = tan(deg2rad(30)) * height

#base = 0.8059
base = 0.8059 / height_ref # Mesh we use normalizes wing height to one

g_II = base - g_I
g_III = tan(deg2rad(15.8)) * height
A = height * (0.5 * (g_I + g_III) + g_II)

lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           LiftCoefficientPressure3D(aoa(), 
                                                                     rho_inf(),
                                                                     u_inf(), A))

p_inf() = 1.0
pressure_coefficient = AnalysisSurfacePointwise(force_boundary_names,
                                                SurfacePressureCoefficient(p_inf(),
                                                                           rho_inf(),
                                                                           u_inf()))

analysis_interval = 100 #100_000
analysis_callback = AnalysisCallback(semi_hyp_para, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (lift_coefficient,),
                                     #analysis_pointwise = (pressure_coefficient,)
                                     #output_directory = "/storage/home/daniel/OneraM6/"
                                     )

alive_callback = AliveCallback(alive_interval = 1)

save_sol_interval = analysis_interval
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = true, # false
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "/storage/home/daniel/OneraM6/")

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true,
                                   output_directory = "/storage/home/daniel/OneraM6/")

## k = 2 ##
base_path = "/storage/home/daniel/OneraM6/Spectra_OptimizedCoeffs/LLF_FD_Ranocha/"
#base_path = "/home/daniel/git/Paper_PERRK/Data/OneraM6/Spectra_OptimizedCoeffs/LLF_FD_Ranocha/"

path = base_path * "k1/p2/"

# TODO: Safety factor, see inviscid

dtRatios_complete_p2 = [ 
    0.753155136853456,
    0.695487338849343,
    0.641318947672844,
    0.574993145465851,
    0.503288297653198,
    0.442298481464386,
    0.391183462142944,
    0.346144811809063,
    0.293439486026764,
    0.243663728386164,
    0.184185989908628,
    0.15320873260498,
    0.123865127563477,
    0.0781898498535156,
    0.0436210632324219
                      ] ./ 0.753155136853456
Stages_complete_p2 = reverse(collect(range(2, 16)))

safety_factor = 1.8
dtRatios_complete_p2_mod = [ 
    0.753155136853456,
    0.695487338849343 / safety_factor, 
    0.641318947672844 / safety_factor,
    0.574993145465851 / safety_factor,
    0.503288297653198 / safety_factor,
    0.442298481464386 / safety_factor,
    0.391183462142944 / safety_factor,
    0.346144811809063 / safety_factor,
    0.293439486026764 / safety_factor,
    0.243663728386164 / safety_factor,
    0.184185989908628 / safety_factor,
    0.15320873260498 / safety_factor,
    0.123865127563477 / safety_factor,
    0.0781898498535156 / safety_factor,
    0.0436210632324219 / safety_factor
                      ] ./ 0.753155136853456

## 6.049 -> 6.05 ##

# Only Flux-Differencing #

cfl = 4.0 # PERK p2 2-16, unsplit
cfl = 4.3 # PERK p2 2-16, unsplit

#cfl = 6.2 # PERK p2 16, unsplit, mod

#cfl = 9.4 # PERK p2 16

cfl = 0.5 # SSPRK22
#cfl = 1.0 # ORK
#cfl = 2.5 # PKD

stepsize_callback = StepsizeCallback(cfl = cfl)

#path = base_path * "k2/p3/"

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        #save_solution,
                        #save_restart,
                        stepsize_callback
                        );

# Run the simulation
###############################################################################

## k = 2, p = 2 ##
ode_alg = Trixi.PairedExplicitRK2(16, path)

#ode_alg = Trixi.PairedExplicitRK2Multi(Stages_complete_p2, path, dtRatios_complete_p2)
#ode_alg = Trixi.PairedExplicitRK2Multi(Stages_complete_p2, path, dtRatios_complete_p2_mod)

#relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 5, root_tol = 1e-12)
#ode_alg = Trixi.PairedExplicitRelaxationRK2Multi(Stages_complete_p2, path, dtRatios_complete_p2; relaxation_solver = relaxation_solver)

#=
Stages_para = [10, 9, 8, 7, 6, 5, 4, 3, 2]
dtRatios_para = reverse([19.1486043845472408975
59.9427061904833635708
115.339716222031540838
186.288290481208207439
272.910996158948648826
375.261469828366500678
493.338227837213594285
627.968143075108287121
842.739504916238502119] ./ 842.739504916238502119)

path_coeffs_para = "/storage/home/daniel/OneraM6/Spectra_OptimizedCoeffs/LLF_FD_Ranocha/k1/p2/para/"

ode_alg = Trixi.PairedExplicitRK2SplitMulti(Stages_complete_p2, Stages_para,
                                            path, path_coeffs_para,
                                            dtRatios_complete_p2, dtRatios_para)
=#

#=
sol = Trixi.solve(ode, ode_alg, dt = 42.0, save_start = false,
                  save_everystep = false, callback = callbacks);
=#

using OrdinaryDiffEqSSPRK
using OrdinaryDiffEqLowStorageRK

ode_alg = SSPRK22(thread = Trixi.True())
#ode_alg = ORK256(thread = Trixi.True())
#ode_alg = ParsaniKetchesonDeconinck3S82(thread = Trixi.True())

sol = solve(ode, ode_alg, dt = 42.0, save_start = false, adaptive = false,
            save_everystep = false, callback = callbacks);