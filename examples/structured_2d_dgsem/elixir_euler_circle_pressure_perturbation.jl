using OrdinaryDiffEq, Plots
using Trixi

equations = CompressibleEulerEquations2D(1.4)

boundary_conditions = boundary_condition_slip_wall

flux = Trixi.FluxHLL(min_max_speed_davis)
solver = DGSEM(polydeg=2, surface_flux=flux,
               volume_integral=VolumeIntegralFluxDifferencing(flux_ranocha))

r0 = 0.5 # inner radius
r1 = 9.0 # outer radius

function initial_condition_pressure_perturbation(x, t, equations::CompressibleEulerEquations2D)
  #xs = 1.5 # location of the initial disturbance on the x axis
  xs = 0.6 * r1 # location of the initial disturbance on the x axis
  w = 1/8 * r1/2.5 # half width
  #p = exp(-log(2) * ((x[1]-xs)^2 + x[2]^2)/w^2) + 1.0
  p = 2^(-((x[1]-xs)^2 + x[2]^2)/w^2) + 1.0
  
  v1 = 0.0
  v2 = 0.0
  rho = 1.0

  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_pressure_perturbation


f1(xi)  = SVector( r0 + 0.5 * (r1 - r0) * (xi + 1), 0.0) # right line
f2(xi)  = SVector(-r0 - 0.5 * (r1 - r0) * (xi + 1), 0.0) # left line
f3(eta) = SVector(r0 * cos(0.5 * pi * (eta + 1)), r0 * sin(0.5 * pi * (eta + 1))) # inner circle (Bottom line)
f4(eta) = SVector(r1 * cos(0.5 * pi * (eta + 1)), r1 * sin(0.5 * pi * (eta + 1))) # outer circle (Top line)

N_x = 192
#N_x = 48 # Better for plotting?
N_y = Int(N_x/1.5)
cells_per_dimension = (N_x, N_y)

N_max = max(N_x, N_y)
N_ref = 12 # Configuration used for optimization
CFL_Discretization = N_ref/N_max # Reduction required due to discretization

mesh = StructuredMesh(cells_per_dimension, (f1, f2, f3, f4), periodicity=false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

analysis_interval = 10000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,extra_analysis_errors=(:conservation_error,))

summary_callback = SummaryCallback()
alive_callback = AliveCallback(analysis_interval=analysis_interval)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback)

S_min = 4

Add_Levels = 0 # S_max = 4
#Add_Levels = 1 # S_max = 6
#Add_Levels = 2 # S_max = 8
#Add_Levels = 3 # S_max = 10
#Add_Levels = 4 # S_max = 12
#Add_Levels = 5 # S_max = 14
Add_Levels = 6 # S_max = 16

Stages = [4]

for i in 1:Add_Levels
  push!(Stages, 4 + 2*i)
end
reverse!(Stages) # Require descending order

b1 = 0.0
bS = 1 - b1
cEnd = 0.5/bS

dtRatios = Stages ./ Stages[1]

ode_algorithm = PERK_Multi(Stages, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_CEE_Structured/",
                           dtRatios,
                           bS, cEnd)
        
CFL_PERK = ((4 + 2*Add_Levels)/4)
#=
# 48 x 32
CFL_Stab = 1.01 # S_max = 4
#CFL_Stab = 1.03 # S_max = 6
#CFL_Stab = 1.04 # S_max = 8
#CFL_Stab = 1.04 # S_max = 10
#CFL_Stab = 0.97 # S_max = 12
#CFL_Stab = 0.98 # S_max = 14
#CFL_Stab = 0.95 # S_max = 16
=#

# 192 x 128
CFL_Stab = 0.95 # S_max = 4
#CFL_Stab = 0.97 # S_max = 6
#CFL_Stab = 0.97 # S_max = 8
#CFL_Stab = 0.94 # S_max = 10
#CFL_Stab = 0.94 # S_max = 12
#CFL_Stab = 0.94 # S_max = 14
CFL_Stab = 0.93 # S_max = 16


#=
ode_algorithm = PERK(S_min+2*Add_Levels, "/home/daniel/git/MA/EigenspectraGeneration/Spectra/2D_CEE_Structured/",
                     bS, cEnd)
=#

CFL = CFL_Stab * CFL_PERK * CFL_Discretization

# S = 4, p = 2, NumCells = 12
dt = 0.0830890595862001646 * CFL

#for i = 1:10
  sol = Trixi.solve(ode, ode_algorithm, dt = dt, save_everystep=false, callback=callbacks);
  #plot(sol)

  #=
  sol = solve(ode, SSPRK33(),
              dt=2.6e-3,
              save_everystep=false, callback=callbacks);
  =#              
#end

summary_callback() # print the timer summary

pd = PlotData2D(sol)
#plot(pd["p"], title = "\$p, t_f = 5.0\$", xlabel = "\$x\$", ylabel = "\$y \$")
plot(pd["p"], title = "\$p, t_0 = 0\$", xlabel = "\$x\$", ylabel = "\$y \$")
plot!(getmesh(pd))

plot(getmesh(pd), xlabel = "\$x\$", ylabel="\$y\$")

using Trixi2Vtk
trixi2vtk("out/PERK/solution_000000.h5")