using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

# Ratio of specific heats
gamma = 1.4

equations = CompressibleEulerEquations2D(gamma)

EdgeLength = 20.0

N_passes = 1
T_end = EdgeLength * N_passes
tspan = (0.0, T_end)

"""
    initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)

The classical isentropic vortex test case as presented in 
https://spectrum.library.concordia.ca/id/eprint/985444/1/Paired-explicit-Runge-Kutta-schemes-for-stiff-sy_2019_Journal-of-Computation.pdf
"""
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
    # Evaluate error after full domain traversion
    if t == T_end
        t = 0
    end

    # initial center of the vortex
    inicenter = SVector(0.0, 0.0)
    # strength of the vortex
    S = 13.5
    # Radius of vortex
    R = 1.5
    # Free-stream Mach 
    M = 0.4
    # base flow
    v1 = 1.0
    v2 = 1.0
    vel = SVector(v1, v2)

    cent = inicenter + vel * t      # advection of center
    cent = x - cent               # distance to centerpoint
    cent = SVector(cent[2], -cent[1])
    r2 = cent[1]^2 + cent[2]^2

    f = (1 - r2) / (2 * R^2)

    rho = (1 - (S * M / pi)^2 * (gamma - 1) * exp(2 * f) / 8)^(1 / (gamma - 1))

    du = S / (2 * π * R) * exp(f) # vel. perturbation
    vel = vel + du * cent
    v1, v2 = vel

    p = rho^gamma / (gamma * M^2)
    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_isentropic_vortex

surf_flux = flux_hllc # Better flux, allows much larger timesteps
PolyDeg = 3
solver = DGSEM(polydeg = PolyDeg, surface_flux = surf_flux)

coordinates_min = (-EdgeLength / 2, -EdgeLength / 2)
coordinates_max = (EdgeLength / 2, EdgeLength / 2)

Refinement = 5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = Refinement,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 10_000)

alive_callback = AliveCallback(alive_interval = 100)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

Stages = 10

CFL_Mesh = 2.0^(3 - Refinement)

if Stages == 10
  CFL = 0.96 * CFL_Mesh
  dt = 0.40838134765625
elseif Stages == 12
  CFL = 0.93 * CFL_Mesh
  dt = 0.528019866943359
elseif Stages == 14
  CFL = 0.9 * CFL_Mesh
  dt = 0.644441833496094
elseif Stages == 16
  CFL = 0.78 * CFL_Mesh
  dt = 0.758966064453125
elseif Stages == 18
  CFL = 0.68 * CFL_Mesh
  dt = 0.87565185546875
elseif Stages == 20
  CFL = 0.6 * CFL_Mesh
  dt = 0.986700439453125
else
  error("Unsupported number of stages")
end

path = "/home/daniel/git/DissDoc/Data/IsentropicVortex/IsentropicVortex/k3/p4/multiprec/"
ode_algorithm = Trixi.PairedExplicitRK4(Stages, path)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt * CFL, 
                  save_everystep = false, callback = callbacks);
