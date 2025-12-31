using OrdinaryDiffEqLowStorageRK
using Trixi
using ForwardDiff

###############################################################################
# Semidiscretization of the quasi 1d compressible Euler equations
# See Chan et al.  https://doi.org/10.48550/arXiv.2307.12089 for details

equations = CompressibleEulerEquationsQuasi1D(1.4)

initial_condition = initial_condition_convergence_test

polydeg = 5 # governs in this case only the number of FV subcells per DG cell
basis = LobattoLegendreBasis(polydeg)
surface_flux = (flux_lax_friedrichs, flux_nonconservative_chan_etal)

@inline function Trixi.calcflux_fvO2!(fstar1_L, fstar1_R, u,
                                mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                nonconservative_terms::Trixi.True,
                                equations, volume_flux_fv, dg::DGSEM, element, cache,
                                x_interfaces, reconstruction_mode, slope_limiter)
    volume_flux, nonconservative_flux = volume_flux_fv
                        
    for i in 2:nnodes(dg) # We compute FV02 fluxes at the (nnodes(dg) - 1) subcell boundaries
        ## Obtain unlimited values in primitive variables ##

        # Note: If i - 2 = 0 we do not go to neighbor element, as one would do in a finite volume scheme.
        # Here, we keep it purely cell-local, thus overshoots between elements are not strictly ruled out,
        # **unless** `reconstruction_mode` is set to `reconstruction_O2_inner`
        u_ll = cons2prim(get_node_vars(u, equations, dg, max(1, i - 2), element),
                         equations)
        u_lr = cons2prim(get_node_vars(u, equations, dg, i - 1, element),
                         equations)
        u_rl = cons2prim(get_node_vars(u, equations, dg, i, element),
                         equations)
        # Note: If i + 1 > nnodes(dg) we do not go to neighbor element, as one would do in a finite volume scheme.
        # Here, we keep it purely cell-local, thus overshoots between elements are not strictly ruled out,
        # **unless** `reconstruction_mode` is set to `reconstruction_O2_inner`
        u_rr = cons2prim(get_node_vars(u, equations, dg, min(nnodes(dg), i + 1),
                                       element), equations)

        # Obtain unlimited nozzle area interface values
        a_lr = u_lr[4]
        a_rl = u_rl[4]

        ## Reconstruct values at interfaces with limiting ##
        u_l, u_r = reconstruction_mode(u_ll, u_lr, u_rl, u_rr,
                                       x_interfaces, i,
                                       slope_limiter, dg)

        # Overwrite reconstructed/limited nozzle area values
        u_l_fixed = SVector(u_l[1:3]..., a_lr)
        u_r_fixed = SVector(u_r[1:3]..., a_rl)

        ## Convert primitive variables back to conservative variables ##
        u_l_cons = prim2cons(u_l_fixed, equations)
        u_r_cons = prim2cons(u_r_fixed, equations)
        flux = volume_flux(u_l_cons, u_r_cons,
                              1, equations) # orientation 1: x direction

        # Compute nonconservative part
        # Note the factor 0.5 necessary for the nonconservative fluxes based on
        # the interpretation of global SBP operators coupled discontinuously via
        # central fluxes/SATs
        f1_L = flux + 0.5f0 * nonconservative_flux(u_l_cons, u_r_cons, 1, equations)
        f1_R = flux + 0.5f0 * nonconservative_flux(u_r_cons, u_l_cons, 1, equations)

        Trixi.set_node_vars!(fstar1_L, f1_L, equations, dg, i)
        Trixi.set_node_vars!(fstar1_R, f1_R, equations, dg, i)
    end

    return nothing
end

volume_integral = VolumeIntegralPureLGLFiniteVolumeO2(basis,
                                                      volume_flux_fv = surface_flux,
                                                      reconstruction_mode = reconstruction_O2_full,
                                                      slope_limiter = monotonized_central)
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:l2_error_primitive,
                                                              :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, ParsaniKetchesonDeconinck3S82();
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
