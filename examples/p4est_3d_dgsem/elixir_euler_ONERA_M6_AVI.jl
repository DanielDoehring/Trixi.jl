using Trixi
using LinearAlgebra: norm

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

@inline function initial_condition(x, t, equations::CompressibleEulerEquations3D)
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
                              surface_flux_function,
                              equations::CompressibleEulerEquations3D)

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

polydeg = 3 # tested also with 2
basis = LobattoLegendreBasis(polydeg)

if polydeg == 2
    alpha_min = 0.08
elseif polydeg == 3
    alpha_min = 0.01
end

shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 1.0,
                                            alpha_min = alpha_min,
                                            alpha_smooth = true,
                                            variable = density_pressure)

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

volume_integral_stabilized = VolumeIntegralShockCapturingHG(shock_indicator;
                                                            volume_flux_dg = volume_flux,
                                                            volume_flux_fv = surface_flux)

# NOTE: Flux Differencing is required, shock capturing not (at least not for simply running the code)
volume_integral_fluxdiff = VolumeIntegralFluxDifferencing(volume_flux)


# TODO: Quick & dirty hack reusing indicator HG but only FD!
function Trixi.calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{2}, P4estMesh{2}, T8codeMesh{2},
                                           TreeMesh{3}, P4estMesh{3}, T8codeMesh{3}},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralAdaptive{<: Nothing},
                               dg::DGSEM, cache,
                               element_indices = Trixi.eachelement(dg, cache))
    @unpack (indicator, volume_integral_default,
    volume_integral_blend_high_order, volume_integral_blend_low_order) = volume_integral.volume_integral_stabilized

    @unpack alpha = indicator.cache

    # Print info on how many cells the stabilized volume integral is needed
    #println("count(alpha .> 0) = ", count(>(0), alpha))

    # For `Float64`, this gives 1.8189894035458565e-12
    # For `Float32`, this gives 1.1920929f-5
    RealT = eltype(alpha)
    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))
    Trixi.@threaded for element in element_indices
        alpha_element = alpha[element]
        # Clip blending factor for values close to zero (-> pure DG)
        dg_only = isapprox(alpha_element, 0, atol = atol)

        if dg_only
            Trixi.weak_form_kernel!(du, u, element, mesh,
                              nonconservative_terms, equations,
                              dg, cache)
        else
            # CARE: Quick & dirty test for ONERA M6
            # Calculate DG volume integral contribution
            Trixi.flux_differencing_kernel!(du, u, element, mesh,
                                      nonconservative_terms, equations,
                                      volume_integral_blend_high_order.volume_flux, dg, cache)
        end
    end

    return nothing
end

volume_integral = VolumeIntegralAdaptive(volume_integral_default = VolumeIntegralWeakForm(),
                                         volume_integral_stabilized = volume_integral_stabilized,
                                         indicator = nothing) # taken from `volume_integral_stabilized`

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

#mesh_path = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/OneraM6/NASA/"
mesh_path = "/storage/home/daniel/PERRK/Data/OneraM6/"

mesh_file = mesh_path * "m6wing_sanitized.inp"

boundary_symbols = [:Symmetry,
                    :FarField,
                    :BottomWing,
                    :TopWing]

mesh = P4estMesh{3}(mesh_file, boundary_symbols = boundary_symbols)

boundary_conditions = (Symmetry = bc_symmetry, # Symmetry: bc_symmetry
                       FarField = bc_farfield, # Farfield: bc_farfield
                       BottomWing = boundary_condition_slip_wall, # Wing: bc_slip_wall
                       TopWing = boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

#tspan = (0.0, 6.049)
#ode = semidiscretize(semi, tspan)

restart_file = "restart_t605_undamped.h5"

restart_filename = joinpath("/storage/home/daniel/OneraM6/", restart_file)
#restart_filename = joinpath("/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/OneraM6/NASA/restart_files/k2/", restart_file)

tspan = (load_time(restart_filename), 6.05) # 6.05

ode = semidiscretize(semi, tspan, restart_filename)


# Callbacks
###############################################################################

summary_callback = SummaryCallback()

force_boundary_names = (:BottomWing, :TopWing)

aoa() = deg2rad(3.06)

rho_inf() = 1.4
u_inf(equations) = 0.84
# Area calculated from information given at https://www.grc.nasa.gov/www/wind/valid/m6wing/m6wing.html

height_ref = 1.1963
height = 1.0 # Mesh we use normalizes wing height to one

g_I = tan(deg2rad(30)) * height

#base = 0.8059
base = 0.8059 / height_ref # Mesh we use normalizes wing height to one

g_II = base - g_I
g_III = tan(deg2rad(15.8)) * height
A = height * (0.5 * (g_I + g_III) + g_II)

lift_coefficient = AnalysisSurfaceIntegral(force_boundary_names,
                                           LiftCoefficientPressure3D(aoa(), rho_inf(),
                                                                     u_inf(equations), A))

p_inf() = 1.0
pressure_coefficient = AnalysisSurfacePointwise(force_boundary_names,
                                                SurfacePressureCoefficient(p_inf(), rho_inf(),
                                                                        u_inf(equations)))

analysis_interval = 10_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (lift_coefficient,),
                                     analysis_pointwise = (pressure_coefficient,),
                                     save_analysis = true,
                                     output_directory="out/")

alive_callback = AliveCallback(alive_interval = 20)

save_sol_interval = analysis_interval

save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory="out/")

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true,
                                   output_directory="out/")

## 6.049 -> 6.05 ##

base_path = "/storage/home/daniel/OneraM6/Spectra_OptimizedCoeffs/LLF_FD_Ranocha/"
path = base_path * "k3/p4/"

safety_factor = 1.8
dtRatios_complete_p4_mod = [
    0.460652348399162,
    0.416693951487541 / safety_factor,
    0.381378587856889 / safety_factor,
    0.350212497711182 / safety_factor,
    0.324112872034311 / safety_factor,
    0.281054820492864 / safety_factor,
    0.245423326268792 / safety_factor,
    0.211017424166203 / safety_factor,
    0.174579094536602 / safety_factor,
    0.154489312693477 / safety_factor,
    0.11951766833663 / safety_factor,
    0.0794953770935535 / safety_factor,
    0.0439409114420414 / safety_factor
                      ] ./ 0.460652348399162
Stages_complete_p4 = reverse(collect(range(5, 17)))

ode_alg = Trixi.PairedExplicitRK4Multi(Stages_complete_p4, path, dtRatios_complete_p4_mod)
cfl = 9.5

#ode_alg = Trixi.PairedExplicitRK4(12, path)
#cfl = 10.0

stepsize_callback = StepsizeCallback(cfl = cfl, interval = 2)

indicator_hg_callback = Trixi.IndicatorHGCallback(interval = 3)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        #analysis_callback,
                        #save_solution,
                        #save_restart,
                        stepsize_callback,
                        indicator_hg_callback # NOTE: Required for `VolumeIntegralAdaptive`!
                        );

# Run the simulation
###############################################################################

sol = Trixi.solve(ode, ode_alg, dt = 42.0, save_start = false,
                  save_everystep = false, callback = callbacks);

