using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

function initial_cond_gauss(x, t, equation::LinearScalarAdvectionEquation1D)
    if t > 8
        t -= 8
    end
    # Store translated coordinate for easy use of exact solution
    x_trans = x - equation.advection_velocity * t

    scalar = exp(-(x_trans[1]^2))
    return SVector(scalar)
end

k = 5 # polynomial degree

# Entropy-conservative flux:
#num_flux = flux_central

# Diffusive fluxes
num_flux = flux_godunov
#num_flux = flux_lax_friedrichs

solver = DGSEM(polydeg = k, surface_flux = num_flux)

coordinates_min = -4.0
coordinates_max = 4.0
length = coordinates_max - coordinates_min

refinement_patches = ((type = "box", coordinates_min = (-1.0,),
                       coordinates_max = (1.0,)),)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                refinement_patches = refinement_patches,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_cond_gauss,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

t_end = 1.0
t_end = 1 * length + 1 # = 9
ode = semidiscretize(semi, (0.0, t_end))

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 100,
                                     extra_analysis_errors = (:conservation_error,))

callbacks = CallbackSet(summary_callback,
                        analysis_callback)

###############################################################################
# run the simulation

path = "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/"

dtRatios = [1, 0.5]
Stages = [16, 8]

ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)

cfl = 2.0^(-5)

dt = 0.2 * cfl
sol = Trixi.solve(ode, ode_alg,
                  dt = dt,
                  save_everystep = false, callback = callbacks);

using Plots

pd = PlotData1D(sol)

plot(getmesh(pd), label = "")

plot!(pd["scalar"], xlabel = "\$x\$", ylabel = "\$u\$",
      label = "Accumulating",
      linewidth = 3, color = RGB(246 / 256, 169 / 256, 0), # Orange
      guidefont = font("Computer Modern", 16), tickfont = font("Computer Modern", 14),
      legend = true)

plot!(pd["scalar"], title = "Flux Handling Strategies",
      label = "Standard",
      linewidth = 3, color = RGB(0, 84 / 256, 159 / 256), # Blue
      titlefont = font("Computer Modern", 18),
      legendfont = font("Computer Modern", 16),
      legend = :topright,
      xlims = (-2.5, 2.5))