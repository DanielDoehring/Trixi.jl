using OrdinaryDiffEqSSPRK
using Trixi
using Trixi: StartUpDG

polydeg = 3
basis = DGMultiBasis(Tri(), polydeg, approximation_type = SBP())

# Import mesh consisting of triangles
mesh_file = "/home/daniel/Desktop/ONERA_OAT15A_airfoil.msh"
VXY, EToV = StartUpDG.read_Gmsh_2D(mesh_file)

# tag different boundary conditions
function freestream(x)
    r = sqrt((x[1] - 0.0)^2 + (x[2] - 0.0)^2)
    isapprox(r, 50.0, atol = 1e-6)
end
airfoil(x) = !freestream(x)
is_on_boundary = (; freestream = freestream, wall = airfoil)

equations = CompressibleEulerEquations2D(1.4)

# Conditions taken from https://www.nas.nasa.gov/LAVA/curv_docs/tutorials/oat15a_hrles/oat15a_hrles/
# Pressure, temperature and density correspond roughly to an altitude of 2800 m
P_inf() = 7.1840768006645056e+04 # [Pa]
T_inf() = 271.0 # [K]
R_specific_air() = 287.058 # [J/(kg*K)]
rho_inf() = P_inf() / (R_specific_air() * T_inf()) # [kg/m^3]

AoA() = deg2rad(3.5)
U_mag() = 2.4094645006722962e+02 # [m/s]
v1() = U_mag() * cos(AoA())
v2() = U_mag() * sin(AoA())

@inline function initial_condition_mach2_flow(x, t, equations::CompressibleEulerEquations2D)
    prim = SVector(rho_inf(), v1(), v2(), P_inf())
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_mach2_flow

volume_flux = flux_ranocha
surface_flux = flux_lax_friedrichs
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

dg = DGMulti(basis,
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = volume_integral)
mesh = DGMultiMesh(dg, VXY, EToV; is_on_boundary)

boundary_conditions = (; freestream = BoundaryConditionDirichlet(initial_condition),
                       wall = boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    boundary_conditions = boundary_conditions)

chord() = 1.0 # [m]
t_convective() = chord() / U_mag() # [s]
#tspan = (0.0, 25.0 * t_convective())
tspan = (0.0, 0.5 * t_convective())
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 50)
analysis_interval = 10_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     solution_variables = cons2prim)
callbacks = CallbackSet(summary_callback, alive_callback,
                        analysis_callback, save_solution)

###############################################################################
# run the simulation

solver = SSPRK43(thread = Trixi.Threaded())

sol = solve(ode, solver; dt = 1e-7, adaptive = true, abstol = 1e-4, reltol = 1e-3,
            ode_default_options()...,
            callback = callbacks);
