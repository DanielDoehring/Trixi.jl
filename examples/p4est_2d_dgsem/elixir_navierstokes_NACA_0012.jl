using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

# Set-up described as on https://turbmodels.larc.nasa.gov/naca0012numerics_val.html

Ma = 0.15
Re = 6.0 * 10^6
gamma = 1.4

# Reference parameters in Rankine "R" units
T0_R = 491.6
S_R = 198.6

T_ref_R = 540

# Values from CFD Online https://www.cfd-online.com/Wiki/Sutherland's_law linked on the TMR page
T0 = 273.15
S = 110.4

T_Ref_K = T_ref_R / 1.8 # 300 K

R_spec_air = 287.052874 # TODO: Not sure if this is correct, leads to high density and pressure

#u_Ref = Ma * sqrt(gamma * R_spec_air * T_Ref_K)
#u_Ref = Ma * sqrt(gamma * T_Ref_K)
u_Ref = 52.08

angle = 5 * pi/180
#angle = 0

mu_0 = 1.716 * 10^-5
mu(T) = mu_0 * (T/T0)^(3/2) * (T0 + S) / (T + S)
mu_Ref = mu(T_Ref_K)

L = 1.0 # Airfoil chord length
rho_Ref = Re * mu_Ref / (u_Ref * L)
#rho_Ref = Re * mu_Ref / (u_Ref * L * sqrt(R_spec_air))
rho_Ref = 1.1767527315296566

#p_Ref = rho_Ref * R_spec_air * T_Ref_K
#p_Ref = rho_Ref * T_Ref_K
p_Ref = 101325

prim = SVector(rho_Ref, u_Ref * cos(angle), u_Ref * sin(angle), p_Ref)
cons = prim2cons(prim, CompressibleEulerEquations2D(gamma))
p = (gamma - 1) * (cons[4] - 0.5 * (cons[1]^2 + cons[2]^2) / cons[1])
T = p / cons[1]

prandtl_number() = 0.72
#mu() = 1.8459162511975808e-5
mu() = 1.0214213709677419e-5

equations = CompressibleEulerEquations2D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

@inline function initial_condition_freestream(x, t, equations)
  # set the freestream flow parameters
  #=
  rho_freestream = rho_Ref
  v1 = u_Ref * cos(angle)
  v2 = u_Ref * sin(angle)
  p_freestream = p_Ref
  =#

  rho_freestream = 1.9769585253456223
  v1 = 51.88181987661811
  v2 = 4.539071082298038
  p_freestream = 101325

  prim = SVector(rho_freestream, v1, v2, p_freestream)
  return prim2cons(prim, equations)
end

initial_condition = initial_condition_freestream

polydeg = 3

surface_flux = flux_hllc
volume_flux = flux_ranocha

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

path = "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/NASA_meshes/"

#mesh = "NACA0012/Family_1/113_33/n0012family_1_7_2D_unique.inp"
mesh = "NACA0012/Family_1/225_65/n0012familyI_6_2D_unique.inp"
mesh_file = path * mesh

boundary_symbols = [:b2_symmetry_y_strong,
                    :b4_farfield_riem, :b5_farfield_riem, :b7_farfield_riem, :b6_viscous_solid, :b8_to_stitch_a]

mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols)

# Boundary conditions for free-stream testing
boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

boundary_conditions = Dict(:b2_symmetry_y_strong => boundary_condition_free_stream,
                           :b4_farfield_riem => boundary_condition_free_stream,
                           :b5_farfield_riem => boundary_condition_free_stream,
                           :b7_farfield_riem => boundary_condition_free_stream,
                           :b6_viscous_solid => boundary_condition_slip_wall,
                           :b8_to_stitch_a => boundary_condition_free_stream)

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_airfoil = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)

boundary_conditions_parabolic = Dict(:b2_symmetry_y_strong => boundary_condition_free_stream,
                                     :b4_farfield_riem => boundary_condition_free_stream,
                                     :b5_farfield_riem => boundary_condition_free_stream,
                                     :b7_farfield_riem => boundary_condition_free_stream,
                                     :b6_viscous_solid => boundary_condition_airfoil,
                                     :b8_to_stitch_a => boundary_condition_free_stream)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

t_c = L / u_Ref
tspan = (0.0, 1 * t_c)

ode = semidiscretize(semi, tspan) # ODE Integrators
ode = semidiscretize(semi, tspan; split_form = false) # PERK

summary_callback = SummaryCallback()

analysis_interval = 2000

aoa() = angle
rho_inf() = rho_Ref
U_inf(equations) = u_Ref
linf() = 1.0

using Trixi: AnalysisSurfaceIntegral, DragCoefficient, LiftCoefficient

drag_coefficient = AnalysisSurfaceIntegral(semi, boundary_condition_slip_wall,
                                           DragCoefficient(aoa(), rho_inf(),
                                                           U_inf(equations), linf()))

lift_coefficient = AnalysisSurfaceIntegral(semi, boundary_condition_slip_wall,
                                           LiftCoefficient(aoa(), rho_inf(),
                                                           U_inf(equations), linf()))

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     analysis_errors = Symbol[], # Turn off expensive error computation
                                     save_analysis = true,
                                     analysis_integrals = (drag_coefficient,
                                                           lift_coefficient))

alive_callback = AliveCallback(alive_interval = 1000)

stepsize_callback = StepsizeCallback(cfl = 5.5) # PERK4, Family 1 225_65

save_restart = SaveRestartCallback(interval = 5e4,
                                   save_final_restart = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
			                  #save_restart,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation


dtRatios = [8.167437333950148e-04, # 16
            6.768165362700527e-04, # 14
            5.439326213454478e-04, # 12
            4.178899203398032e-04, # 10
            3.097859435001737e-04, # 8
            2.360805942444131e-04, # 7
            1.880221871215326e-04, # 6
            1.032772630933323e-04] / 8.167437333950148e-04

Stages = [16, 14, 12, 10, 8, 7, 6, 5]
ode_algorithm = PERK4_Multi(Stages, "/home/daniel/ownCloud - Döhring, Daniel (1MH1D4@rwth-aachen.de)@rwth-aachen.sciebo.de/Job/Doktorand/Content/Meshes/NASA_meshes/NACA0012/Family_1/", dtRatios)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep=false, callback=callbacks);

#=
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false, thread = OrdinaryDiffEq.True()),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
=#

summary_callback() # print the timer summary

#=
using Plots

pd = PlotData2D(sol)
plot(sol)

plot(pd["rho"])

plot(getmesh(pd), xlim = [-1, 2], ylim = [-0.5, 0.5])
=#
