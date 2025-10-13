using Trixi
using OrdinaryDiffEqLowStorageRK
import LinearAlgebra: norm # For vorticity magnitude

###############################################################################
# semidiscretization of the compressible Euler equations

gamma = 1.4
equations = CompressibleEulerEquations3D(gamma)

rho_ref() = 1.255 # kg/m^3
R_specific_air() = 287.052874 # J/(kg K)
T_ref() = 300 # K
p_ref() = rho_ref() * R_specific_air() * T_ref() # Pa = N/m^2

# Speed of sound reference state
c_ref() = sqrt(gamma * p_ref()/rho_ref()) # m/s
Ma_ref() = 0.1
U() = Ma_ref() * c_ref() # m/s

D() = 1 # Follows from mesh
Re_D() = 3900
mu_ref() = rho_ref() * D() * U()/Re_D() # TODO: Sutherlands law

prandtl_number = 0.72
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu_ref(),
                                                          Prandtl = prandtl_number)

@inline function initial_condition(x, t, equations)
    # set the freestream flow parameters
    rho_freestream = 1.255

    # v_total = 0.1 = Mach (for c = 1)
    v1 = 34.72206893029273
    v2 = 0.0
    v3 = 0.0

    p_freestream = 108075.40706099998

    prim = SVector(rho_freestream, v1, v2, v3, p_freestream)
    return prim2cons(prim, equations)
end

bc_farfield = BoundaryConditionDirichlet(initial_condition)

polydeg = 2

#surface_flux = flux_hll
surface_flux = FluxLMARS(c_ref())

volume_flux = flux_ranocha
volume_flux = flux_kennedy_gruber
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

case_path = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/CS1/"
case_path = "/storage/home/daniel/Meshes/HighOrderCFDWorkshop/CS1/"

mesh_file = case_path * "Pointwise/TandemSpheresHexMesh2P2_fixed.inp"

# Boundary symbols follow from nodesets in the mesh file
boundary_symbols = [:FrontSphere, :BackSphere, :FarField]
mesh = P4estMesh{3}(mesh_file; boundary_symbols = boundary_symbols)

boundary_conditions = Dict(:FrontSphere => boundary_condition_slip_wall,
                           :BackSphere => boundary_condition_slip_wall,
                           :FarField => bc_farfield)

#=
semi_hyp = SemidiscretizationHyperbolic(mesh, equations,
                                        initial_condition, solver;
                                        boundary_conditions = boundary_conditions)
=#

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
# 1) 0 to 10: Inviscid, k = 2
# 2) 50 to 100: Viscous, k = 2
# 3) 100 to 150: Viscous, k = 4
# 4) 150 to 200: Viscous, k = 4, statistics
t_star_end = 150
t_end = t_star_end * D()/U()
tspan = (0.0, t_end)

#ode = semidiscretize(semi_hyp, tspan)
#ode = semidiscretize(semi, tspan; split_problem = false)


restart_file = "restart_ts50_hyp.h5"
#restart_file = "restart_ts100_hyp_para.h5"
#restart_file = "restart_000010000.h5"

restart_path = "out/"
#restart_path = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/CS1/Pointwise/restart_2p2/"

restart_filename = joinpath(restart_path, restart_file)

tspan = (load_time(restart_filename), t_end)
ode = semidiscretize(semi, tspan, restart_filename; split_problem = false)


###############################################################################

summary_callback = SummaryCallback()

analysis_interval = 50_000
#analysis_interval = 50

A_sphere() = pi * (D()/2)^2
drag_p_front = AnalysisSurfaceIntegral((:FrontSphere,),
                                        DragCoefficientPressure3D(0.0, rho_ref(),
                                                                  U(), A_sphere()))

drag_p_back = AnalysisSurfaceIntegral((:BackSphere,),
                                      DragCoefficientPressure3D(0.0, rho_ref(),
                                                                U(), A_sphere()))

drag_f_front = AnalysisSurfaceIntegral((:FrontSphere,),
                                       DragCoefficientShearStress3D(0.0, rho_ref(),
                                                                    U(), A_sphere()))

analysis_callback = AnalysisCallback(semi,
                                     interval = analysis_interval,
                                     save_analysis = true,
                                     analysis_errors = Symbol[], # Turn off error computation
                                     analysis_integrals = (drag_p_front,
                                                           #drag_p_back,
                                                           drag_f_front))

analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(alive_interval = 50)

t_ramp_up() = 1e-2 # For dimensionalized units

# Hyp
#cfl_max() = 17.0
# Hyp-Diff
cfl_max() = 13.5 # k = 2; 14.0 stable for a long time
#cfl_max() = 9.0 # k = 3
#cfl_max() = 6.0 # k = 4

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

save_restart = SaveRestartCallback(interval = save_sol_interval)

callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        analysis_callback,
                        stepsize_callback,
                        #save_solution,
                        save_restart
                        )

###############################################################################

### p2 ###

# Pointwise:
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
#path_coeffs = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/CS1/Spectra_Coeffs/hyp_para/k2_hll_fluxdiff/"


ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path_coeffs, dtRatios)

### p3 ###

#=
Stages = [14, 13, 12, 11, 10, 8, 7, 6, 4, 3]
dtRatios = reverse([0.00918807983398437551025
0.0122711181640625006812
0.0167221069335937509287
0.0271781921386718765098
0.0316619873046875017584
0.0463378906250000025723
0.0513610839843750028528
0.061634063720703128422
0.0701828002929687538964
0.0815246582031250045238] ./ 0.0815246582031250045238)

path_coeffs = "/storage/home/daniel/Meshes/HighOrderCFDWorkshop/CS1/Spectra_Coeffs/hyp_para/k4_hll_fluxdiff/"
ode_alg = Trixi.PairedExplicitRK3Multi(Stages, path_coeffs, dtRatios)
=#

sol = Trixi.solve(ode, ode_alg,
                  dt = 1e-5,
                  save_everystep = false, callback = callbacks);

###############################################################################

#=
sol = solve(ode, RDPK3SpFSAL35(thread = Trixi.True());
            abstol = 1.0e-5, reltol = 1.0e-5,
            ode_default_options()..., callback = callbacks);
=#