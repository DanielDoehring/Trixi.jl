using Trixi
using OrdinaryDiffEqLowStorageRK

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

D = 1 # Follows from mesh
Re_D = 3900
mu() = rho_ref() * D * U()/Re_D # TODO: Sutherlands law

prandtl_number = 0.72
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu(),
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

surface_flux = flux_hll
volume_flux = flux_ranocha
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

semi_hyp = SemidiscretizationHyperbolic(mesh, equations,
                                        initial_condition, solver;
                                        boundary_conditions = boundary_conditions)

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

t_star_end = 50 # 100
t_end = t_star_end * D/U()
tspan = (0.0, t_end)

ode = semidiscretize(semi_hyp, tspan)
#ode = semidiscretize(semi, tspan)

###############################################################################

summary_callback = SummaryCallback()

# TODO: Lift/Drag coefficients
analysis_interval = 50_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(alive_interval = 50)

t_ramp_up() = 5e-2 # For dimensionalized units
cfl_0() = 10.0
cfl_max() = 17.0

cfl(t) = min(cfl_max(), cfl_0() + t/t_ramp_up() * (cfl_max() - cfl_0()))

stepsize_callback = StepsizeCallback(cfl = cfl)

save_sol_interval = 2000
save_solution = SaveSolutionCallback(interval = save_sol_interval,
                                     save_initial_solution = false)

save_restart = SaveRestartCallback(interval = save_sol_interval)

callbacks = CallbackSet(summary_callback,
                        alive_callback, 
                        analysis_callback,
                        stepsize_callback,
                        #save_solution,
                        save_restart
                        )

###############################################################################

Stages = reverse(collect(range(2, 15)))

Stages = [15, 14, 13, 12, 11, 9, 8, 7, 5, 4, 3, 2]

dtRatios = reverse([0.0717675685882568359375
0.147277712821960449219
0.235483050346374511719
0.291365385055541992188
0.44527530670166015625
0.515322089195251464844
0.579870939254760742188
0.71626186370849609375
0.790532231330871582031
0.891929268836975097656
0.974652767181396484375
1.04339599609375
] ./ 1.04339599609375)

path_coeffs = "/home/daniel/Sciebo/Job/Doktorand/Content/Meshes/HighOrderCFDWorkshop/CS1/Spectra_Coeffs/hyp/k2_LLF_weakform/"
path_coeffs = "/storage/home/daniel/Meshes/HighOrderCFDWorkshop/CS1/Spectra_Coeffs/hyp/k2_LLF_weakform"

ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path_coeffs, dtRatios)

sol = Trixi.solve(ode, ode_alg,
                  dt = 2.5e-5,
                  save_everystep = false, callback = callbacks);

###############################################################################

#=
sol = solve(ode, RDPK3SpFSAL35(thread = Trixi.True());
            abstol = 1.0e-5, reltol = 1.0e-5,
            ode_default_options()..., callback = callbacks);
=#