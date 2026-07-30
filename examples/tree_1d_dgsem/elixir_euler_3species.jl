using Trixi
using OrdinaryDiffEqLowStorageRK

gamma_molec = 1.4
gamma_atom  = 5/3

# Universal gas constant in CGS
const R_univ = 8.31446261815324 # [J / (mol K)]

# Molar masses [g/mol] — stay in CGS to match Gnoffo's C coefficients
const M_N2 = 28.014 * 1e-3 # kg/mol
const M_O2 = 31.998 * 1e-3 # kg/mol
const M_O  = 15.999 * 1e-3 # kg/mol

# Specific gas constants [m²/(s²·K)]
const R_N2 = R_univ / M_N2
const R_O2 = R_univ / M_O2
const R_O  = R_univ / M_O

equations = CompressibleEulerMulticomponentEquations1D(gammas = (gamma_molec, gamma_molec, gamma_atom),
                                                       gas_constants = (R_N2, R_O2, R_O))

@inline function initial_condition(x, t, equations::CompressibleEulerMulticomponentEquations1D)
    RealT = eltype(x)

    dens_air = 1.255 # [kg/m^3]

    rho_O = 1e-4 * dens_air # something nonzero but small
    rho_O2 = (0.22 - 0.5e-4) * dens_air # [kg/m^3]
    rho_N2 = (0.78 - 0.5e-4) * dens_air # [kg/m^3]

    v1 = 0

    p = 101325.0  # [Pa] = [N/m^2] = 1 atm
    p = 101325.0 * 10 # [Pa] = [N/m^2] = 1 atm

    return prim2cons(SVector(v1, p, rho_N2, rho_O2, rho_O), equations)
end

u = initial_condition(0.0, 0.0, equations)
temperature(u, equations) # ~ 280 K

# Gnoffo 1989 Table 3: Kinetic Model of Park.
# Reaction 1: O2 + M <=> 2O + M
# C in [cm^3/mol/s], E in [K], n dimensionless
# Reaction 1: M = {N2, O2} (molecule partners)
const C1_O2 = 2.9e+23
const n1_O2 = -2
const E1_O2_over_k = 5.975e+04 # Activation energy, [J]

# Reaction 2: M = {N, O} (atom partners)
const C2_O2 = 9.68e+22
const n2_O2 = -2
const E2_O2_over_k = 5.975e+04 # Activation energy, [J]

# Equilibrium constant coefficients, Park curve fit
const B_O2 = (2.855, 0.998, -6.181, -0.023, -0.001)

# Forward reaction rate coefficient
@inline function kf_O2(T, C, n, E_over_k)
    return clamp(C * T^n * exp(-E_over_k / T), 1e-9, 1e9) # [cm^3/mol/s]
end

# Equilibrium constant, see eq. (47)
@inline function K_c(T)
    Z = 10_000 / T
    return clamp(exp(B_O2[1] + B_O2[2]*log(Z) + B_O2[3]*Z + B_O2[4]*Z^2 + B_O2[5]*Z^3), 1e-9, 1e9) # [mol/cm^3]
end

@inline function source_terms_o2_dissociation(u, x, t,
                                              equations::CompressibleEulerMulticomponentEquations1D)
    # Unpack — species order: N2=3, O2=4, O=5
    _, _, rho_N2, rho_O2, rho_O = u
    rho = density(u, equations)
    T   = temperature(u, equations)

    # Molar concentrations in [mol/m^3]
    C_N2 = rho_N2 / M_N2
    C_O2 = rho_O2 / M_O2
    C_O  = rho_O  / M_O

    # Third-body concentrations
    M_molec = C_N2 + C_O2 # molecule partners (N absent in reduced system)
    M_atom  = C_O         # atom partners (N absent)

    # Forward rates [cm^3/mol/s]
    k_f1 = kf_O2(T, C1_O2, n1_O2, E1_O2_over_k)
    k_f2 = kf_O2(T, C2_O2, n2_O2, E2_O2_over_k)

    # Backward reaction rate coefficients, see eq. (46)
    k_b1 = k_f1 / K_c(T)
    k_b2 = k_f2 / K_c(T)

    # Forward reaction rates, see eqs. (42) and (43)
    R_f = 1000 * (k_f1 * M_molec + k_f2 * M_atom) * (1e-3 * C_O2)
    R_b = 1000 * (k_b1 * M_molec + k_b2 * M_atom) * (1e-3 * C_O)^2

    # Net rate [mol/(m^3·s)]
    r_dot = R_f - R_b

    # Mass production rates [kg/(m^3·s)]
    omega_O2 = -M_O2 * r_dot
    omega_O  = 2 * M_O * r_dot

    return SVector(zero(rho_N2), zero(rho_N2), zero(rho_N2), omega_O2, omega_O)
end

u = initial_condition(0.0, 0.0, equations)
source_terms_o2_dissociation(u, 0.0, 0.0, equations)

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

tspan = (0.0, 1e-3)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100_000

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
            dt = 1e-12, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);