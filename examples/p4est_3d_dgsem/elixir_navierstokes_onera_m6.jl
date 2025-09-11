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

tspan = (load_time(restart_filename), 6.04901) # 6.05

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
                                                                           u_inf(), A))

analysis_interval = 5 #100_000
analysis_callback = AnalysisCallback(semi_hyp_para, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (lift_coefficient,)
                                     #analysis_pointwise = (pressure_coefficient,)
                                     )

alive_callback = AliveCallback(alive_interval = 50)

extra_node_variables = (:cfl_diffusion, :cfl_convection)

function Trixi.get_node_variable(::Val{:cfl_diffusion}, u, mesh, equations, dg, cache,
                                 equations_parabolic, cache_parabolic)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    cfl_d_array = zeros(eltype(cache.elements),
                        n_nodes, n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                        n_elements)

    @unpack contravariant_vectors, inverse_jacobian = cache.elements

    # We can accelerate the computation by thread-parallelizing the loop over elements
    # by using the `@threaded` macro.
    Trixi.@threaded for element in eachelement(dg, cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)

            # Speeds
            d = Trixi.max_diffusivity(u_node, equations_parabolic)

            # Mesh data
            Ja11, Ja12, Ja13 = Trixi.get_contravariant_vector(1, contravariant_vectors,
                                                              i, j, k, element)
            Ja21, Ja22, Ja23 = Trixi.get_contravariant_vector(2, contravariant_vectors,
                                                              i, j, k, element)
            Ja31, Ja32, Ja33 = Trixi.get_contravariant_vector(3, contravariant_vectors,
                                                              i, j, k, element)

            inv_jacobian = abs(inverse_jacobian[i, j, k, element])

            # See FLUXO:
            # https://github.com/project-fluxo/fluxo/blob/c7e0cc9b7fd4569dcab67bbb6e5a25c0a84859f1/src/equation/navierstokes/calctimestep.f90#L130-L137
            d1_transformed = d * (Ja11^2 + Ja12^2 + Ja13^2) * inv_jacobian^2
            d2_transformed = d * (Ja21^2 + Ja22^2 + Ja23^2) * inv_jacobian^2
            d3_transformed = d * (Ja31^2 + Ja32^2 + Ja33^2) * inv_jacobian^2

            cfl_d_array[i, j, k, element] = d1_transformed + d2_transformed + d3_transformed
        end
    end

    return cfl_d_array
end

function Trixi.get_node_variable(::Val{:cfl_convection}, u, mesh, equations, dg, cache,
                                 equations_parabolic, cache_parabolic)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    cfl_a_array = zeros(eltype(cache.elements),
                            n_nodes, n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                            n_elements)

    @unpack contravariant_vectors, inverse_jacobian = cache.elements

    # We can accelerate the computation by thread-parallelizing the loop over elements
    # by using the `@threaded` macro.
    Trixi.@threaded for element in eachelement(dg, cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)

            # Speeds
            a1, a2, a3 = Trixi.max_abs_speeds(u_node, equations)

            # Mesh data
            Ja11, Ja12, Ja13 = Trixi.get_contravariant_vector(1, contravariant_vectors,
                                                              i, j, k, element)
            Ja21, Ja22, Ja23 = Trixi.get_contravariant_vector(2, contravariant_vectors,
                                                              i, j, k, element)
            Ja31, Ja32, Ja33 = Trixi.get_contravariant_vector(3, contravariant_vectors,
                                                              i, j, k, element)

            inv_jacobian = abs(inverse_jacobian[i, j, k, element])

            # Transform
            a1_transformed = abs(Ja11 * a1 + Ja12 * a2 + Ja13 * a3) * inv_jacobian
            a2_transformed = abs(Ja21 * a1 + Ja22 * a2 + Ja23 * a3) * inv_jacobian
            a3_transformed = abs(Ja31 * a1 + Ja32 * a2 + Ja33 * a3) * inv_jacobian

            cfl_a_array[i, j, k, element] = a1_transformed + a2_transformed + a3_transformed
        end
    end

    return cfl_a_array
end

save_sol_interval = analysis_interval
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = true, # false
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = "/storage/home/daniel/OneraM6/",
                                     extra_node_variables = extra_node_variables)

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   save_final_restart = true,
                                   output_directory = "/storage/home/daniel/OneraM6/")

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

## 6.049 -> 6.05 ##

# Only Flux-Differencing #
cfl_interval = 2

stepsize_callback = StepsizeCallback(cfl = 13.2, interval = cfl_interval) # PERK p2 2-16

#path = base_path * "k2/p3/"

#=

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
=#

callbacks = CallbackSet(summary_callback,
                        #alive_callback,
                        analysis_callback,
                        #save_solution,
                        #save_restart,
                        #stepsize_callback
                        );

# Run the simulation
###############################################################################


## k = 2, p = 2 ##
ode_alg = Trixi.PairedExplicitRK2Multi(Stages_complete_p2, path, dtRatios_complete_p2)

Stages_14_max = reverse(collect(range(2, 14)))

dtRatios_14_max = [
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
] ./ 0.641318947672844

Stages_para = [14, 12, 10, 7, 6, 5, 4, 3, 2]
path_para = path * "para/"

dtRatios_para = [3008.7179235408357, # 14
    1762.72919916304,   # 12
    842.7395049162385,  # 10
    375.2614698283665,  # 7
    272.91099615894865, # 6
    186.2882904812082,  # 5
    115.33971622203154, # 4
    59.942706190483364, # 3
    19.14860438454724] / 3008.7179235408357 #= 2 =#

ode_alg = Trixi.PairedExplicitRK2SplitMulti(Stages_14_max, Stages_para,
                                            path, path_para,
                                            dtRatios_14_max, dtRatios_para)

## k = 2, p = 3 ##

#ode_alg = Trixi.PairedExplicitRK3Multi(Stages_complete_p3, path, dtRatios_complete_p3)
#ode_alg = Trixi.PairedExplicitRK3(15, path)

dt = 9.5e-8 # Hyp-Para without stepsize control; unsplit
dt = 9e-8 # Hyp-Para without stepsize control; split
sol = Trixi.solve(ode, ode_alg, dt = dt,
                  save_everystep = false, callback = callbacks);

