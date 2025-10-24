using Trixi
using OrdinaryDiffEqLowStorageRK
import LinearAlgebra: norm # For vorticity magnitude

###############################################################################
# semidiscretization of the compressible Euler equations

gamma = 1.4
equations = CompressibleEulerEquations3D(gamma)

#=
rho_ref() = 1.255 # kg/m^3
R_specific_air() = 287.052874 # J/(kg K)
T_ref() = 300 # K
p_ref() = rho_ref() * R_specific_air() * T_ref() # Pa = N/m^2

# Speed of sound reference state
c_ref() = sqrt(gamma * p_ref()/rho_ref()) # m/s
Ma_ref() = 0.1
U() = Ma_ref() * c_ref() # m/s
=#


# NOTE: non-dim for more homogeneous setup/better vorticity visualization?
rho_ref() = 1.4
p_ref() = 1.0
c_ref() = 1.0
Ma_ref() = 0.1
U() = Ma_ref() * c_ref()

D() = 1 # Follows from mesh
Re_D() = 3900
mu_ref() = rho_ref() * U() * D()/Re_D() # TODO: Sutherlands law

prandtl_number = 0.72
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu_ref(),
                                                          Prandtl = prandtl_number)

@inline function initial_condition(x, t, equations)
    # set the freestream flow parameters
    #rho_freestream = 1.255
    rho_freestream = 1.4

    # v_total = 0.1 = Mach (for c = 1)
    #v1 = 34.72206893029273
    v1 = 0.1
    v2 = 0.0
    v3 = 0.0

    #p_freestream = 108075.40706099998
    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, v3, p_freestream)
    return prim2cons(prim, equations)
end

bc_farfield = BoundaryConditionDirichlet(initial_condition)

polydeg = 4 # 2, 3, 4

surface_flux = flux_hll
volume_flux = flux_kennedy_gruber
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

#case_path = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/CS1/"
case_path = "/storage/home/daniel/Meshes/HighOrderCFDWorkshop/CS1/"

mesh_file = case_path * "Pointwise/TandemSpheresHexMesh2P2_fixed.inp"

# Boundary symbols follow from nodesets in the mesh file
boundary_symbols = [:FrontSphere, :BackSphere, :FarField]
mesh = P4estMesh{3}(mesh_file; boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:FrontSphere => boundary_condition_slip_wall,
                           :BackSphere => boundary_condition_slip_wall,
                           :FarField => bc_farfield)

velocity_bc = NoSlip((x, t, equations) -> SVector(0.0, 0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
bc_spheres = BoundaryConditionNavierStokesWall(velocity_bc, heat_bc)

boundary_conditions_para = Dict(:FrontSphere => bc_spheres,
                                :BackSphere => bc_spheres,
                                :FarField => bc_farfield)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_para))

# Strategy:
# 1) 0 to 75: k2, p2
# 2) 75 to 100: k3, p2
# 3) 100 to 200: k4, p3
t_star_end = 150
t_end = t_star_end * D()/U()

tspan = (0.0, t_end)
ode = semidiscretize(semi, tspan; split_problem = false)

#restart_path = "out/"
restart_path = "/storage/home/daniel/Meshes/HighOrderCFDWorkshop/CS1/Pointwise/restart_2p2/"

restart_file = "restart_ts100_hp.h5"

restart_filename = joinpath(restart_path, restart_file)

tspan = (load_time(restart_filename), t_end)
ode = semidiscretize(semi, tspan, restart_filename; split_problem = false)


###############################################################################

summary_callback = SummaryCallback()

analysis_interval = 50 # For coefficient recording run

A_sphere() = pi * (D()/2)^2
drag_p_front = AnalysisSurfaceIntegral((:FrontSphere,),
                                        DragCoefficientPressure3D(0.0, rho_ref(),
                                                                  U(), A_sphere()))

drag_p_back = AnalysisSurfaceIntegral((:BackSphere,),
                                      DragCoefficientPressure3D(0.0, rho_ref(),
                                                                U(), A_sphere()))

analysis_callback = AnalysisCallback(semi,
                                     interval = analysis_interval,
                                     save_analysis = true,
                                     output_directory = restart_path,
                                     analysis_errors = Symbol[], # Turn off error computation
                                     analysis_integrals = (drag_p_front,
                                                           drag_p_back,
                                                           ))

#analysis_callback = AnalysisCallback(semi_hyp, interval = analysis_interval)

alive_callback = AliveCallback(alive_interval = 50)

cfl_0() = 5.0 # k = 2

t_ramp_up() = 5.0

# Hyp-Diff
cfl_max() = 13.5 # k = 2 p2
cfl_max() = 9.0 # k = 3, p2
cfl_max() = 6.0 # k = 4, p3

#cfl(t) = min(cfl_max(), cfl_0() + t/t_ramp_up() * (cfl_max() - cfl_0()))

cfl = cfl_max()

stepsize_callback = StepsizeCallback(cfl = cfl)

save_sol_interval = 2000

extra_node_variables = (:vorticity_magnitude,)
function Trixi.get_node_variable(::Val{:vorticity_magnitude}, u, mesh, equations, dg, cache,
                                 equations_parabolic, cache_parabolic)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    vorticity_array = zeros(eltype(cache.elements),
                            n_nodes, n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                            n_elements)

    @unpack viscous_container = cache_parabolic
    @unpack gradients = viscous_container
    gradients_x, gradients_y, gradients_z = gradients

    Trixi.@threaded for element in eachelement(dg, cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)

            gradients_1 = get_node_vars(gradients_x, equations_parabolic, dg,
                                        i, j, k, element)
            gradients_2 = get_node_vars(gradients_y, equations_parabolic, dg,
                                        i, j, k, element)
            gradients_3 = get_node_vars(gradients_z, equations_parabolic, dg,
                                        i, j, k, element)

            vorticity_nodal = vorticity(u_node, 
                                        (gradients_1, gradients_2, gradients_3),
                                        equations_parabolic)
            vorticity_array[i, j, k, element] = norm(vorticity_nodal)
        end
    end

    return vorticity_array
end

save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false,
                                     extra_node_variables = extra_node_variables)

save_restart = SaveRestartCallback(interval = save_sol_interval,
                                   output_directory = restart_path)

callbacks = CallbackSet(summary_callback,
                        #alive_callback,
                        analysis_callback,
                        stepsize_callback,
                        #save_solution,
                        save_restart
                        )

###############################################################################

### p2 ###

#=
# Pointwise
Stages = [14, 13, 12, 11, 10, 8, 7, 5, 4, 3, 2]

dtRatios = reverse([0.07416057586669921875
0.14945125579833984375
0.236209869384765625
0.29239177703857421875
0.447246551513671875
0.51645755767822265625
0.64532947540283203125
0.72096157073974609375
0.7944393157958984375
0.899280548095703125
0.9783477783203125
] ./ 0.9783477783203125)

path_coeffs = "/storage/home/daniel/Meshes/HighOrderCFDWorkshop/CS1/Spectra_Coeffs/hyp_para/k2_hll_fluxdiff/"
#path_coeffs = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/CS1/Spectra_Coeffs/hyp_para/k2_hll_fluxdiff/p2/"

ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path_coeffs, dtRatios)
=#

### p3 ###

Stages = [14, 13, 12, 11, 10, 9, 8, 6, 5, 4, 3]
dtRatios = reverse([0.000120724458014592531464
0.00022078421326354146521
0.000356726163243874915506
0.00049591444368474186554
0.000669575841305777445039
0.000789872415727004425815
0.000921498959874734302254
0.00105325164646841587883
0.00114614301623776557516
0.00124876051938161256589
0.00135157296740449967876] ./ 0.00135157296740449967876)

path_coeffs = "/storage/home/daniel/Meshes/HighOrderCFDWorkshop/CS1/Spectra_Coeffs/hyp_para/k4_hll_fluxdiff/p3/"
ode_alg = Trixi.PairedExplicitRK3Multi(Stages, path_coeffs, dtRatios)


sol = Trixi.solve(ode, ode_alg,
                  dt = 1e-3,
                  save_everystep = false, callback = callbacks);

###############################################################################

#=
callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback)

sol = solve(ode, RDPK3SpFSAL35(thread = Trixi.True());
            abstol = 1.0e-5, reltol = 1.0e-5,
            ode_default_options()..., callback = callbacks);
=#