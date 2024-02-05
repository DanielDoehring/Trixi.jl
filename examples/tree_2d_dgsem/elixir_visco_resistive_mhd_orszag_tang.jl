
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
prandtl_number() = 1.0

# Make less diffusive to still have convection-dominated spectra
mu() = 1e-3
eta() = 1e-3

gamma = 5/3
equations = IdealGlmMhdEquations2D(gamma)
equations_parabolic = ViscoResistiveMhdDiffusion2D(equations, mu = mu(),
                                                   Prandtl = prandtl_number(),
                                                   eta = eta(),
                                                   gradient_variables = GradientVariablesPrimitive())

"""
    initial_condition_orszag_tang(x, t, equations::IdealGlmMhdEquations2D)

The classical Orszag-Tang vortex test case. Here, the setup is taken from
- https://onlinelibrary.wiley.com/doi/pdf/10.1002/fld.4681

# Note: In their MHD equations some Lundquist scaling factors are included, see also 
https://doi.org/10.1006/jcph.1999.6248
"""
function initial_condition_orszag_tang(x, t, equations::IdealGlmMhdEquations2D)
  rho = 1.0
  v1 = -2 * sqrt(pi) * sin(x[2])
  v2 =  2 * sqrt(pi) * sin(x[1])
  v3 = 0.0
  p = 15/4 + 0.25 * cos(4*x[1]) + 0.8 * cos(2*x[1])*cos(x[2]) - cos(x[1])*cos(x[2]) + 0.25 * cos(2*x[2])
  
  B1 = -sin(x[2])
  B2 =  sin(2.0*x[1])
  
  # Due to missing Lundquist numbers in the governing PDEs
  #B1 = -sin(x[2]) / sqrt(2 * pi)
  #B2 =  sin(2.0*x[1]) / sqrt(2 * pi)

  B3 = 0.0
  psi = 0.0
  return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end

initial_condition = initial_condition_orszag_tang

surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell)
volume_flux  = (flux_central, flux_nonconservative_powell)

basis = LobattoLegendreBasis(3)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (2*pi, 2*pi)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=100000)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
#tspan = (0.0, 1.0) # For plotting

ode = semidiscretize(semi, tspan; split_form = false)
#ode = semidiscretize(semi, tspan) # For ODE.jl integrators


summary_callback = SummaryCallback()

analysis_interval = 20000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, analysis_errors=Symbol[])

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=0.5,
                                          alpha_min=0.001,
                                          alpha_smooth=false,
                                          variable=density_pressure)

#For meaningful streamline plot at end                                      
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=5,
                                      med_level =7, med_threshold=0.04,
                                      max_level =9, max_threshold=0.4)


amr_callback = AMRCallback(semi, amr_controller,
                           interval=10, # PERK
                           #interval = 11, # DGLDDRK73_C
                           #interval = 5, # base_level = changed
                           #interval=33, # SSPRK33
                           #interval = 16, # ParsaniKetchesonDeconinck3S53
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)
                      
cfl = 2.0 # PERK 3, 4, 6
cfl = 1.9 # Perk Single

stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        amr_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

# S = 3, p = 2 Ref = 4
dt = 0.0161709425439767083

b1   = 0.0
bS   = 1.0 - b1
cEnd = 0.5/bS

# TODO: Do also p=3

# S = 4, p = 3, Ref = 4
dt = 0.0155320306832436481
# Series: 4, 6, 11

#=
ode_algorithm = PERK_Multi(3, 2, "/home/daniel/git/Paper_AMR_PERK/Data/ViscousOrszagTang/p2/",
                           bS, cEnd,
                           LevelCFL, Integrator_Mesh_Level_Dict)


ode_algorithm = PERK(12, "/home/daniel/git/Paper_AMR_PERK/Data/ViscousOrszagTang/p2/",
                     bS, cEnd)
=#

Stages = [6, 4, 3]

cS2 = 1.0
ode_algorithm = PERK3_Multi(Stages, "/home/daniel/git/Paper_AMR_PERK/Data/ViscousOrszagTang/p3/Central/", cS2)

ode_algorithm = PERK3(6, "/home/daniel/git/Paper_AMR_PERK/Data/ViscousOrszagTang/p3/Central/")
#=
for i = 1:1
  mesh = TreeMesh(coordinates_min, coordinates_max,
                  initial_refinement_level=4,
                  n_cells_max=100000)
  
  semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver) 
  
  ode = semidiscretize(semi, tspan; split_form = false)
  =#
  
  sol = Trixi.solve(ode, ode_algorithm, dt = dt,
                    save_everystep=false, callback=callbacks);
  
#end


#cfl = 1.8 # DGLDDRK73_C Max Level 9, base lvl = 3
cfl = 0.6 # SSPRK33 Max Level 9, base lvl = 3
cfl = 1.2 # ParsaniKetchesonDeconinck3S53
cfl = 1.2 # RDPK3SpFSAL35

stepsize_callback = StepsizeCallback(cfl=cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale=0.5, cfl=cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        amr_callback,
                        stepsize_callback,
                        glm_speed_callback)
#=
for i = 1:10
  mesh = TreeMesh(coordinates_min, coordinates_max,
                  initial_refinement_level=4,
                  n_cells_max=100000)
  
  semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver) 
  
  ode = semidiscretize(semi, tspan) # For ODE.jl integrators
=#
  #=
  sol = solve(ode, DGLDDRK73_C(;thread = OrdinaryDiffEq.True());
                dt = 1.0,
                ode_default_options()..., callback=callbacks);
  =#

  #=
  sol = solve(ode, ParsaniKetchesonDeconinck3S53(;thread = OrdinaryDiffEq.True());
                dt = 1.0,
                ode_default_options()..., callback=callbacks);
  =#
  
  #=
  sol = solve(ode, SSPRK33(;thread = OrdinaryDiffEq.True());
              dt = 1.0,
              ode_default_options()..., callback=callbacks);
  =#

  sol = solve(ode, RDPK3SpFSAL35(;thread = OrdinaryDiffEq.True());
              adaptive = false,
              dt = 1.0,
              ode_default_options()..., callback=callbacks);
  
#end

summary_callback() # print the timer summary

plot(sol)

pd = PlotData2D(sol)

V1 = pd.data[2]
V2 = pd.data[3]

B1 = pd.data[6]
B2 = pd.data[7]

using DelimitedFiles

# Export points
writedlm("x.csv", pd.x, ',')
writedlm("y.csv", pd.y, ',')

# Export Velocity field
writedlm("V1.csv", V1, ',')
writedlm("V2.csv", V2, ',')

# Export Magnetic field
writedlm("B1.csv", B1, ',')
writedlm("B2.csv", B2, ',')

#=
p1 = plot(pd["B1"])
p2 = heatmap(B1, aspect_ratio = :equal)

heatmap(B1[1:100, 1:1000], aspect_ratio = :equal)

plot(p1, p2, layout=(2, 1))
=#

plot(pd["rho"], c = :jet, title = "\$ ρ, t_f = 3.0 \$", 
           xticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]),
           yticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]))

plot(pd["p"], c = :jet, title = "\$ p, t = 1.0 \$",
     xticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]),
     yticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]))

plot(getmesh(pd), xlabel = "\$x\$", ylabel="\$y\$", title = "Mesh at \$t = 1.0\$",
           xticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]),
           yticks=([0, pi, 2pi], [0, "\$π\$", "\$2π\$"]))