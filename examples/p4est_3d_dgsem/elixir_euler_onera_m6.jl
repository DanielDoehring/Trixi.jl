using Trixi
using LinearAlgebra: norm

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

# NOTE: True Mach = 0.8395

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

polydeg = 3 # originally 2, tested also with 3
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
                               volume_integral::VolumeIntegralAdaptive{VolumeIntegralWeakForm,
                                                                       VolumeIntegralSC,
                                                                       Indicator},
                               dg::DGSEM, cache,
                               element_indices = Trixi.eachelement(dg, cache),
                               interface_indices = Trixi.eachinterface(dg, cache),
                               mortar_indices = Trixi.eachmortar(dg, cache)) where {Indicator <: Nothing, # Indicator taken from `VolumeIntegralSC`
                                             VolumeIntegralSC <:
                                             VolumeIntegralShockCapturingHG}
    @unpack volume_flux_dg, volume_flux_fv, indicator = volume_integral.volume_integral_stabilized

    @unpack alpha = indicator.cache
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
                                      volume_flux_dg, dg, cache)
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

boundary_conditions = Dict(:Symmetry => bc_symmetry, # Symmetry: bc_symmetry
                           :FarField => bc_farfield, # Farfield: bc_farfield
                           :BottomWing => boundary_condition_slip_wall, # Wing: bc_slip_wall
                           :TopWing => boundary_condition_slip_wall # Wing: bc_slip_wall
                          )

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
                                     output_directory="out/"
                                     )

alive_callback = AliveCallback(alive_interval = 20)

save_sol_interval = analysis_interval

#=
# Add `:T` to `extra_node_variables` tuple ...
extra_node_variables = (:T,)

@inline function temperature(u, equations::CompressibleEulerEquations3D)
    rho, rho_v1, rho_v2, rho_v3, rho_e = u

    p = (equations.gamma - 1) * (rho_e - 0.5f0 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho)
    T = p / rho # Corresponds to a specific gas constant R = 1
    return T
end

# ... and specify the function `get_node_variable` for this symbol, 
# with first argument matching the symbol (turned into a type via `Val`) for dispatching.
# Note that for parabolic(-extended) equations, `equations_parabolic` and `cache_parabolic`
# must be declared as the last two arguments of the function to match the expected signature.
function Trixi.get_node_variable(::Val{:T}, u, mesh, equations, dg, cache)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    T_array = zeros(eltype(cache.elements),
                            n_nodes, n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                            n_elements)

    # We can accelerate the computation by thread-parallelizing the loop over elements
    # by using the `@threaded` macro.
    Trixi.@threaded for element in eachelement(dg, cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            
            T_array[i, j, k, element] = temperature(u_node, equations)
        end
    end

    return T_array
end
=#

save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     #output_directory="/storage/home/daniel/OneraM6/"
                                     output_directory="out/",
                                     #extra_node_variables = extra_node_variables
                                     )

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true,
                                   #output_directory="/storage/home/daniel/OneraM6/"
                                   output_directory="out/"
                                   )

## k = 2 ##

base_path = "/storage/home/daniel/OneraM6/Spectra_OptimizedCoeffs/LLF_FD_Ranocha/"
#base_path = "/home/daniel/git/Paper_PERRK/Data/OneraM6/Spectra_OptimizedCoeffs/LLF_FD_Ranocha/"

path = base_path * "k1/p2/"

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
cfl_interval = 2

cfl = 13.2 # PERK p2 2-16

cfl = 20.0 # PERK p2 2-16 mod2, safety_factor = 1.8

# steady-state near (restarted 6.049)

#cfl = 31.0 # PERK p2 E16

#cfl = 2.0 # SSPRK22
#cfl = 3.6 # ORK256
#cfl = 9.4 # ParsaniKetchesonDeconinck3S82

stepsize_callback = StepsizeCallback(cfl = cfl, interval = cfl_interval)

#path = base_path * "k2/p3/"

#=
stepsize_callback = StepsizeCallback(cfl = 10.0, interval = cfl_interval) # PERRK p3 15 standalone

dtRatios_complete_p3 = [ 
    0.309904923439026,
    0.277295976877213,
    0.250083755254746,
    0.228134118318558,
    0.20889208316803,
    0.185411275029182,
    0.160719511508942,
    0.138943578004837,
    0.111497408151627,
    0.0973129367828369,
    0.0799268364906311,
    0.0501513481140137,
    0.0280734300613403
                      ] ./ 0.309904923439026
Stages_complete_p3 = reverse(collect(range(3, 15)))

## 6.049 -> 6.05 ##

# Only Flux-Differencing #
cfl_interval = 2

stepsize_callback = StepsizeCallback(cfl = 10.0, interval = cfl_interval) # PER(R)K p3 3-15
#stepsize_callback = StepsizeCallback(cfl = 10.7, interval = cfl_interval) # PER(R)K p3 15
#stepsize_callback = StepsizeCallback(cfl = 2.7, interval = cfl_interval) # (R-)CKL43
#stepsize_callback = StepsizeCallback(cfl = 2.8, interval = cfl_interval) # (R-)RK33
=#

path = base_path * "k3/p2/"

dtRatios_complete_p2_mod = [ 
    0.487036598101258259852,
    0.440858029760420306074 / safety_factor,
    0.394032015837728962588 / safety_factor,
    0.342910405620932566377 / safety_factor,
    0.307774601317942131065 / safety_factor,
    0.274541171640157689442 / safety_factor,
    0.238564100302755823847 / safety_factor,
    0.202247733436524860514 / safety_factor,
    0.160481546446681016716 / safety_factor,
    0.134384183585643763331 / safety_factor,
    0.108587534353137012282 / safety_factor,
    0.0718968311324715587742 / safety_factor,
    0.0300888860598206508947 / safety_factor
                      ] ./ 0.487036598101258259852
Stages_complete_p2 = reverse(collect(range(2, 14)))

cfl = 10.0 # k = 3 optimized


path = base_path * "k3/p4/"

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

cfl = 9.5 # k = 3 optimized

stepsize_callback = StepsizeCallback(cfl = cfl, interval = cfl_interval)

indicator_hg_callback = Trixi.IndicatorHGCallback(interval = 3)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        #analysis_callback,
                        save_solution,
                        #save_restart,
                        stepsize_callback,
                        indicator_hg_callback # NOTE: Required for `VolumeIntegralAdaptive`!
                        );

# Run the simulation
###############################################################################

## k = 2, p = 2 ##

#ode_alg = Trixi.PairedExplicitRK2Multi(Stages_complete_p2, path, dtRatios_complete_p2_mod)

#ode_alg = Trixi.PairedExplicitRK2(16, path)

## k = 2, p = 3 ##

#newton = Trixi.RelaxationSolverNewton(max_iterations = 5, root_tol = 1e-12)

#ode_alg = Trixi.PairedExplicitRK3Multi(Stages_complete_p3, path, dtRatios_complete_p3)
#ode_alg = Trixi.PairedExplicitRK3(15, path)
#=
ode_alg = Trixi.PairedExplicitRelaxationRK3Multi(Stages_complete_p3, path, dtRatios_complete_p3;
                                                 relaxation_solver = newton)
=#
#ode_alg = Trixi.PairedExplicitRelaxationRK3(15, path; relaxation_solver = newton)                                                 

#ode_alg = Trixi.RelaxationCKL43(; relaxation_solver = newton)
#ode_alg = Trixi.CKL43()

#ode_alg = Trixi.RelaxationRK33(; relaxation_solver = newton)
#ode_alg = Trixi.RK33()


sol = Trixi.solve(ode, ode_alg, dt = 42.0, save_start = false,
                  save_everystep = false, callback = callbacks);

#=
using OrdinaryDiffEqSSPRK
using OrdinaryDiffEqLowStorageRK

ode_alg = SSPRK22(thread = Trixi.True())
#ode_alg = ORK256(thread = Trixi.True())
#ode_alg = ParsaniKetchesonDeconinck3S82(thread = Trixi.True())

sol = solve(ode, ode_alg, dt = 42.0, save_start = false, adaptive = false,
            save_everystep = false, callback = callbacks);
=#