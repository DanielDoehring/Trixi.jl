using Trixi
using OrdinaryDiffEqLowStorageRK
using Krylov, SparseArrays

###############################################################################

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3, # 4
                n_cells_max = 80_000)

# Build pure diffusion (Laplace) operator
advection_velocity = (0, 0)
advection_eq = LinearScalarAdvectionEquation2D(advection_velocity)
diffusivity() = 1
diffusion_eq = LaplaceDiffusion2D(diffusivity(), advection_eq)

polydeg = 3
# The hyperbolic flux does not matter for this example since
# the hyperbolic part is zero.
solver_gravity = DGSEM(polydeg = polydeg, surface_flux = flux_central)

function initial_condition_jeans_instability(x, t,
                                             equations::LinearScalarAdvectionEquation2D)
    # gravity equation: -Δϕ = -4πGρ
    # Constants taken from the FLASH manual
    # https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node189.html#SECTION010131000000000000000
    rho0 = 1.5e7
    delta0 = 1e-3

    phi = rho0 * delta0 # constant background perturbation magnitude
    return phi
end

# `solver_parabolic = ViscousFormulationLocalDG()` strictly required for elliptic/diffusion-dominated problem
semi_gravity = SemidiscretizationHyperbolicParabolic(mesh,
                                                     (advection_eq, diffusion_eq),
                                                     initial_condition_jeans_instability,
                                                     solver_gravity;
                                                     solver_parabolic = ViscousFormulationLocalDG())

# Note that `linear_structure` does not access the `initial_condition`/steady-state solution
A_map, b = linear_structure(semi_gravity)
A = Matrix(A_map)
A_sparse = sparse(A)

du_gravity_ode = similar(b)

function calc_potential_derivative(phi_node, mesh::TreeMesh{2},
                                   equations::CompressibleEulerEquations2D,
                                   dg::DGSEM, cache, i, j, element)
    @unpack derivative_matrix = dg.basis

    phi_x = 0 # derivative of phi in x direction
    for ii in eachnode(dg)
        #phi_x = phi_x + derivative_matrix[i, ii] * phi_node
        phi_x += derivative_matrix[i, ii] * phi_node
    end

    phi_y = 0 # derivative of phi in y direction
    for jj in eachnode(dg)
        #phi_y = phi_y + derivative_matrix[j, jj] * phi_node
        phi_y += derivative_matrix[j, jj] * phi_node
    end
    
    return (phi_x, phi_y) .* cache.elements.inverse_jacobian[element]
end

rho0 = 1.5e7
G = 6.674e-8
atol = 1e-7
rtol = 1e-6

# TODO: Can I dispatch on `source_terms` to make this less hacky?
function Trixi.calc_sources!(du_euler, u_euler, t, source_terms,
                             equations::CompressibleEulerEquations2D, dg::DG, cache,
                             element_indices = eachelement(dg, cache))
    #grav_scale = -4.0 * pi * G
    grav_scale = -4.0 * pi * G # TODO minus yes-no?

    # Step 1: Update RHS of gravity solver
    #
    # Source term: Jeans instability OR coupling convergence test OR blast wave
    # put in gravity source term proportional to Euler density
    # OBS! subtract off the background density ρ_0 (spatial mean value)
    du_gravity = Trixi.wrap_array(du_gravity_ode, mesh, advection_eq, solver_gravity,
                                  cache)
    Trixi.@threaded for element in element_indices
        @views @. du_gravity[1, .., element] = grav_scale *
                                               (u_euler[1, .., element] - rho0)
    end

    # Step 2: Solve for gravitational potential ϕ

    #u_gravity_ode, stats = gmres(A_map, du_gravity_ode, atol = atol, rtol = rtol)
    u_gravity_ode = A_sparse \ du_gravity_ode

    u_gravity = Trixi.wrap_array(u_gravity_ode, mesh, advection_eq, solver_gravity,
                                 cache)

    # Step 3: Update Euler RHS with gravitational source terms
    Trixi.@threaded for element in element_indices
        for j in eachnode(dg), i in eachnode(dg)
            phi_node = u_gravity[1, i, j, element]
            (phi_x, phi_y) = calc_potential_derivative(phi_node, mesh, equations,
                                                       dg, cache, i, j, element)

            du_euler[2, i, j, element] -= u_euler[1, i, j, element] * phi_x
            du_euler[3, i, j, element] -= u_euler[1, i, j, element] * phi_y
            du_euler[4, i, j, element] -= (u_euler[2, i, j, element] * phi_x +
                                           u_euler[3, i, j, element] * phi_y)
        end
    end

    return nothing
end

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 5 / 3
equations_euler = CompressibleEulerEquations2D(gamma)

surf_flux_euler = FluxHLL(min_max_speed_naive) # As in paper
#surf_flux_euler = flux_hllc # Reduces oscillations on non-unfiorm grids
solver_euler = DGSEM(polydeg, surf_flux_euler)

function initial_condition_jeans_instability(x, t,
                                             equations::CompressibleEulerEquations2D)
    # Jeans gravitational instability test case
    # see Derigs et al. https://arxiv.org/abs/1605.03572; Sec. 4.6
    # OBS! this uses cgs (centimeter, gram, second) units
    # periodic boundaries
    # domain size [0,L]^2 depends on the wave number chosen for the perturbation
    # OBS! Be very careful here L must be chosen such that problem is periodic
    # typical final time is T = 5
    # gamma = 5/3
    dens0 = 1.5e7 # g/cm^3
    pres0 = 1.5e7 # dyn/cm^2
    delta0 = 1e-3
    # set wave vector values for perturbation (units 1/cm)
    # see FLASH manual: https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node189.html#SECTION010131000000000000000
    kx = 2.0 * pi / 0.5 # 2π/λ_x, λ_x = 0.5
    ky = 0.0   # 2π/λ_y, λ_y = 1e10
    k_dot_x = kx * x[1] + ky * x[2]
    # perturb density and pressure away from reference states ρ_0 and p_0
    dens = dens0 * (1.0 + delta0 * cos(k_dot_x))                 # g/cm^3
    pres = pres0 * (1.0 + equations.gamma * delta0 * cos(k_dot_x)) # dyn/cm^2
    # flow starts as stationary
    velx = 0.0 # cm/s
    vely = 0.0 # cm/s
    return prim2cons((dens, velx, vely, pres), equations)
end

dummy() = 42 # something different from `nothing`

semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler,
                                          initial_condition_jeans_instability,
                                          solver_euler, source_terms = dummy())

tspan = (0.0, 5.0)
ode = semidiscretize(semi_euler, tspan)

summary_callback = SummaryCallback()

stepsize_callback = StepsizeCallback(cfl = 0.5) # value as used in the paper

analysis_interval = 100
alive_callback = AliveCallback(alive_interval = 100)

analysis_callback = AnalysisCallback(semi_euler, interval = analysis_interval)

callbacks = CallbackSet(summary_callback, stepsize_callback,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
