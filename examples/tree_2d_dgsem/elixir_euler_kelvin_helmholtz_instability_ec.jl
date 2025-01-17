
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-1,+1]^2
  slope = 15
  B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
  rho = 0.5 + 0.75 * B
  v1 = 0.5 * (B - 1)
  v2 = 0.1 * sin(2 * pi * x[1])
  p = 1.0
  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability

flux = flux_ranocha

surface_flux = flux
volume_flux  = flux
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
Refinement = 7
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=Refinement,
                n_cells_max=100_000)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0) # For this discretization: Limit around 3.2
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[],
                                     # Note: entropy defaults to mathematical entropy
                                     analysis_integrals = (entropy,),
                                     #analysis_filename = "entropy_standard.dat",
                                     analysis_filename = "entropy_ER.dat",
                                     save_analysis = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback)

###############################################################################
# run the simulation

path = "/home/daniel/git/Paper_PERRK/Data/Kelvin_Helmholtz/"

ode_algorithm = Trixi.PairedExplicitRK3(7, path * "p3/")
ode_algorithm = Trixi.PairedExplicitRelaxationRK3(7, path * "p3/")

# p = 3, S = 7
dt = 3e-3

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary

pd = PlotData2D(sol)

plot(pd["rho"], title = "\$ρ, t_f = 3.0\$", xlabel = "\$x\$", ylabel = "\$y \$", c = :jet)
plot!(getmesh(pd), xlabel = "\$x\$", ylabel="\$y\$", title = "Mesh at \$t_f = 3.0\$")

