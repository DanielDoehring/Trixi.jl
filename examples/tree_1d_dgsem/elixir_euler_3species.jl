# Universal gas constant in CGS
const R_univ_cgs = 8.314e7   # [erg/(mol·K)] = [g·cm²/(s²·mol·K)]

# Molar masses [g/mol] — stay in CGS to match Gnoffo's C coefficients
const M_N2_cgs = 28.014   # g/mol
const M_O2_cgs = 31.998   # g/mol
const M_O_cgs  = 15.999   # g/mol

# Specific gas constants [cm²/(s²·K)]
const R_N2 = R_univ_cgs / M_N2_cgs   # ≈ 2.9682e6
const R_O2 = R_univ_cgs / M_O2_cgs   # ≈ 2.5985e6
const R_O  = R_univ_cgs / M_O_cgs    # ≈ 5.1966e6

# Gnoffo 1989 Table B1: O2 + M <=> 2O + M
# C in [cm^3/mol/s], E in [K], n dimensionless
# Reaction 1: M = {N2, O2} (molecule partners)
const C1_O2 = 2.9e+23    # note: Gnoffo p.158
const n1_O2 = -2
const E1_O2 = 5.975e+04

# Reaction 2: M = {N, O} (atom partners)
const C2_O2 = 9.68e+22   # again, check against your copy of table B1
const n2_O2 = -2
const E2_O2 = 5.975e+04

# Equilibrium constant coefficients (Gnoffo Table B2, O2 <=> 2O)
# Kp fit: ln(Kp) = b1 + b2*z + b3*z^2 + b4*z^3 + b5*z^4, z = 10000/T
# Note: Gnoffo gives Kp in [atm^(change in moles)], need Kc for concentration-based rates
const b_O2 = (2.855, 0.998, -6.181, -0.023, -0.001)

@inline function kf_O2(T, C, n, E)
    return C * T^n * exp(-E / T)   # [cm^3/mol/s]
end

@inline function kb_O2(T)
    z = 10_000 / T
    return exp(b_O2[1] + b_O2[2]*ln(z) + b_O2[3]*z + b_O2[4]*z^2 + b_O2[5]*z^3)
end

@inline function source_terms_o2_dissociation(u, x, t,
                                               equations::CompressibleEulerMulticomponentEquations1D)
    # Unpack — species order: N2=3, O2=4, O=5
    rho_v1, rho_e, rho_N2, rho_O2, rho_O = u
    rho = density(u, equations)
    T   = temperature(u, equations)

    # Molar concentrations in CGS [mol/cm^3]
    C_N2 = rho_N2 / M_N2_cgs
    C_O2 = rho_O2 / M_O2_cgs
    C_O  = rho_O  / M_O_cgs

    # Third-body concentrations
    M_molec  = C_N2 + C_O2 # molecule partners (N absent in reduced system)
    M_atom = C_O           # atom partners (N absent)

    # Forward rates [cm^3/mol/s]
    kf1 = kf_O2(T, C1_O2, n1_O2, E1_O2)
    kf2 = kf_O2(T, C2_O2, n2_O2, E2_O2)

    kb1 = kf1 / kb_O2(T)
    kb2 = kf2 / kb_O2(T)

    # Net rate [mol/(cm^3·s)]
    r_dot = (kf1 * M_molec + kf2 * M_atom) * C_O2 -
            (kb1 * M_molec + kb2 * M_atom) * C_O^2

    # Mass production rates [g/(cm^3·s)]
    omega_O2 = -M_O2_cgs * r_dot
    omega_O  =  2 * M_O_cgs * r_dot

    return SVector(zero(rho_v1), zero(rho_e), zero(rho_N2), omega_O2, omega_O)
end