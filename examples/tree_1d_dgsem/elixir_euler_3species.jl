using Trixi
using OrdinaryDiffEqLowStorageRK

gamma_molec = 1.4
gamma_atom  = 5/3

# Universal gas constant in CGS
const R_univ_cgs = 8.314e7 # [g·cm²/(s²·mol·K)]

# Specific gas constants [cm²/(s²·K)]
const R_N2 = R_univ_cgs / M_N2_cgs
const R_O2 = R_univ_cgs / M_O2_cgs
const R_O  = R_univ_cgs / M_O_cgs

equations = CompressibleEulerMulticomponentEquations1D(gammas = (gamma_molec, gamma_molec, gamma_atom),
                                                       gas_constants = (R_N2, R_O2, R_O))

@inline function initial_condition(x, t, equations::CompressibleEulerMulticomponentEquations1D)
    RealT = eltype(x)

    dens_air = 1.255 * 1e-3 # [g/cm^3]

    rho_O = 1e-4 * dens_air # something nonzero but small
    rho_O2 = (0.22 - 0.5e-4) * dens_air # [g/cm^3]
    rho_N2 = (0.78 - 0.5e-4) * dens_air # [g/cm^3]

    v1 = 0

    p = 1 * 1.01325e6 # [dyn/cm^2] = 100 atm

    return prim2cons(SVector(v1, p, rho_N2, rho_O2, rho_O), equations)
end

# Molar masses [g/mol] — stay in CGS to match Gnoffo's C coefficients
const M_N2_cgs = 28.014 # g/mol
const M_O2_cgs = 31.998 # g/mol
const M_O_cgs  = 15.999 # g/mol

# Gnoffo 1989 Table 3: Kinetic Model of Park.
# Reaction 1: O2 + M <=> 2O + M
# C in [cm^3/mol/s], E in [K], n dimensionless
# Reaction 1: M = {N2, O2} (molecule partners)
const C1_O2 = 2.9e+23
const n1_O2 = -2
const E1_O2 = 5.975e+04 # Activation energy, [J]

# Reaction 2: M = {N, O} (atom partners)
const C2_O2 = 9.68e+22
const n2_O2 = -2
const E2_O2 = 5.975e+04 # Activation energy, [J]

# Equilibrium constant coefficients, Park curve fit
const B_O2 = (2.855, 0.998, -6.181, -0.023, -0.001)

# Forward reaction rate coefficient
@inline function kf_O2(T, C, n, E)
    return C * T^n * exp(-E / T) # [cm^3/mol/s]
end

# Equilibrium constant, see eq. (47)
@inline function K_c(T)
    Z = 10_000 / T
    return exp(B_O2[1] + B_O2[2]*log(Z) + B_O2[3]*Z + B_O2[4]*Z^2 + B_O2[5]*Z^3)
end

@inline function source_terms_o2_dissociation(u, x, t,
                                              equations::CompressibleEulerMulticomponentEquations1D)
    # Unpack — species order: N2=3, O2=4, O=5
    _, _, rho_N2, rho_O2, rho_O = u
    rho = density(u, equations)
    T   = temperature(u, equations)

    # Molar concentrations in CGS [mol/cm^3]
    C_N2 = rho_N2 / M_N2_cgs
    C_O2 = rho_O2 / M_O2_cgs
    C_O  = rho_O  / M_O_cgs

    # Third-body concentrations
    M_molec = C_N2 + C_O2 # molecule partners (N absent in reduced system)
    M_atom  = C_O         # atom partners (N absent)

    # Forward rates [cm^3/mol/s]
    k_f1 = kf_O2(T, C1_O2, n1_O2, E1_O2)
    k_f2 = kf_O2(T, C2_O2, n2_O2, E2_O2)

    # Backward reaction rate coefficients, see eq. (46)
    k_b1 = k_f1 / K_c(T)
    k_b2 = k_f2 / K_c(T)

    # Forward reaction rates, see eqs. (42) and (43)
    R_f = (k_f1 * M_molec + k_f2 * M_atom) * C_O2
    R_b = (k_b1 * M_molec + k_b2 * M_atom) * C_O^2

    # Net rate [mol/(cm^3·s)]
    r_dot = R_f - R_b

    # Mass production rates [g/(cm^3·s)]
    omega_O2 = -M_O2_cgs * r_dot
    omega_O  = 2 * M_O_cgs * r_dot

    return SVector(zero(rho_N2), zero(rho_N2), zero(rho_N2), omega_O2, omega_O)
end

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.8,
                                         alpha_min = 0.0,
                                         alpha_smooth = true,
                                         variable = pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0,)
coordinates_max = (1.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic,
                                    source_terms = source_terms_o2_dissociation
                                    )

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1e-4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000

analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        #stepsize_callback
                        )

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1e-11, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);