# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct CompressibleEulerMultiIonEquations1D{NVARS, NCOMP, RealT <: Real,
                                                    ElectronPressure,
                                                    ElectronTemperature} <:
               AbstractCompressibleEulerMultiIonEquations{1, NVARS, NCOMP}
    gammas::SVector{NCOMP, RealT} # Heat capacity ratios
    inv_gammas_minus_one::SVector{NCOMP, RealT} # = inv(gamma - 1)
    charge_to_mass::SVector{NCOMP, RealT} # Charge to mass ratios
    gas_constants::SVector{NCOMP, RealT} # Specific gas constants
    molar_masses::SVector{NCOMP, RealT} # Molar masses
    ion_ion_collision_constants::Array{RealT, 2} # Collision coefficients
    ion_electron_collision_constants::SVector{NCOMP, RealT} # Ion-electron collision constants
    electron_pressure::ElectronPressure # Electron pressure function
    electron_temperature::ElectronTemperature # Electron temperature function

    # Inner Constructor
    function CompressibleEulerMultiIonEquations1D{NVARS, NCOMP, RealT, ElectronPressure,
                                                  ElectronTemperature}(gammas::SVector{NCOMP,
                                                                                       RealT},
                                                                       charge_to_mass::SVector{NCOMP,
                                                                                               RealT},
                                                                       gas_constants::SVector{NCOMP,
                                                                                              RealT},
                                                                       molar_masses::SVector{NCOMP,
                                                                                             RealT},
                                                                       ion_ion_collision_constants::Array{RealT,
                                                                                                          2},
                                                                       ion_electron_collision_constants::SVector{NCOMP,
                                                                                                                 RealT},
                                                                       electron_pressure::ElectronPressure,
                                                                       electron_temperature::ElectronTemperature) where {
                                                                                                                         NVARS,
                                                                                                                         NCOMP,
                                                                                                                         RealT <:
                                                                                                                         Real,
                                                                                                                         ElectronPressure,
                                                                                                                         ElectronTemperature
                                                                                                                         }
        NCOMP >= 1 ||
            throw(DimensionMismatch("`gammas` and `charge_to_mass` must contain at least one value"))

        # Precompute inverse gamma - 1
        inv_gammas_minus_one = SVector{NCOMP, RealT}(inv.(gammas .- 1))

        new{NVARS, NCOMP, RealT, ElectronPressure, ElectronTemperature}(gammas,
                                                                        inv_gammas_minus_one,
                                                                        charge_to_mass,
                                                                        gas_constants,
                                                                        molar_masses,
                                                                        ion_ion_collision_constants,
                                                                        ion_electron_collision_constants,
                                                                        electron_pressure,
                                                                        electron_temperature)
    end
end

# Outer Constructor Delegating to Inner Constructor
function CompressibleEulerMultiIonEquations1D(; gammas, charge_to_mass,
                                              gas_constants = zero(SVector{length(gammas),
                                                                           eltype(gammas)}),
                                              molar_masses = zero(SVector{length(gammas),
                                                                          eltype(gammas)}),
                                              ion_ion_collision_constants = zeros(eltype(gammas),
                                                                                  length(gammas),
                                                                                  length(gammas)),
                                              ion_electron_collision_constants = zero(SVector{length(gammas),
                                                                                              eltype(gammas)}),
                                              electron_pressure = electron_pressure_zero,
                                              electron_temperature = electron_pressure_zero)
    # Promote input types
    _gammas = promote(gammas...)
    _charge_to_mass = promote(charge_to_mass...)
    _gas_constants = promote(gas_constants...)
    _molar_masses = promote(molar_masses...)
    _ion_electron_collision_constants = promote(ion_electron_collision_constants...)
    RealT = promote_type(eltype(_gammas), eltype(_charge_to_mass),
                         eltype(_gas_constants), eltype(_molar_masses),
                         eltype(ion_ion_collision_constants),
                         eltype(_ion_electron_collision_constants))
    __gammas = SVector(map(RealT, _gammas))
    __charge_to_mass = SVector(map(RealT, _charge_to_mass))
    __gas_constants = SVector(map(RealT, _gas_constants))
    __molar_masses = SVector(map(RealT, _molar_masses))
    __ion_ion_collision_constants = map(RealT, ion_ion_collision_constants)
    __ion_electron_collision_constants = SVector(map(RealT,
                                                     _ion_electron_collision_constants))

    NCOMP = length(_gammas)
    NVARS = 3 * NCOMP

    return CompressibleEulerMultiIonEquations1D{NVARS, NCOMP, RealT,
                                                typeof(electron_pressure),
                                                typeof(electron_temperature)}(__gammas,
                                                                              __charge_to_mass,
                                                                              __gas_constants,
                                                                              __molar_masses,
                                                                              __ion_ion_collision_constants,
                                                                              __ion_electron_collision_constants,
                                                                              electron_pressure,
                                                                              electron_temperature)
end

@inline function Base.real(::CompressibleEulerMultiIonEquations1D{NVARS, NCOMP, RealT}) where {
                                                                                               NVARS,
                                                                                               NCOMP,
                                                                                               RealT <:
                                                                                               Real
                                                                                               }
    RealT
end

function varnames(::typeof(cons2cons), equations::CompressibleEulerMultiIonEquations1D)
    cons = ()
    for i in eachcomponent(equations)
        cons = (cons...,
                tuple("rho_" * string(i), "rho_v1_" * string(i),
                      "rho_e_" * string(i))...)
    end

    return cons
end

function varnames(::typeof(cons2prim), equations::CompressibleEulerMultiIonEquations1D)
    prim = ()
    for i in eachcomponent(equations)
        prim = (prim...,
                tuple("rho_" * string(i), "v1_" * string(i), "p_" * string(i))...)
    end
    return prim
end

"""
    initial_condition_constant(x, t, equations::CompressibleEulerMultiIonEquations1D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t,
                                    equations::CompressibleEulerMultiIonEquations1D)
    cons = zero(MVector{nvariables(equations), eltype(x)})

    rho = 0.1
    rho_v1 = 1
    rho_e = 10

    for k in eachcomponent(equations)
        set_component!(cons, k, rho, rho_v1, rho_e, equations)
    end

    return SVector(cons)
end

"""
    initial_condition_convergence_test(x, t, equations::CompressibleEulerMultiIonEquations1D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t,
                                            equations::CompressibleEulerMultiIonEquations1D)
    RealT = eltype(x)
    cons = zero(MVector{nvariables(equations), RealT})

    c = 2
    A = convert(RealT, 0.1)
    L = 2
    f = 1.0f0 / L
    ω = 2 * convert(RealT, pi) * f
    ini = c + A * sin(ω * (x[1] - t))

    rho = ini
    rho_v1 = ini
    rho_e = ini^2

    for k in eachcomponent(equations)
        set_component!(cons, k, rho, rho_v1, rho_e, equations)
    end

    return SVector(cons)
end

"""
    source_terms_convergence_test(u, x, t, equations::CompressibleEulerMultiIonEquations1D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t,
                                               equations::CompressibleEulerMultiIonEquations1D)
    # Same settings as in `initial_condition`
    RealT = eltype(u)
    cons = zero(MVector{nvariables(equations), RealT})

    c = 2
    A = convert(RealT, 0.1)
    L = 2
    f = 1.0f0 / L
    ω = 2 * convert(RealT, pi) * f

    x1, = x

    si, co = sincos(ω * (x1 - t))

    for k in eachcomponent(equations)
        gamma = equations.gammas[k]
        rho = c + A * si
        rho_x = ω * A * co

        # Note that d/dt rho = -d/dx rho.
        # This yields du2 = du3 = d/dx p (derivative of pressure).
        # Other terms vanish because of v = 1.
        du1 = 0
        du2 = rho_x * (2 * rho - 0.5f0) * (gamma - 1)
        du3 = du2
        set_component!(cons, k, du1, du2, du3, equations)
    end

    return SVector(cons)
end

@doc raw"""
    source_terms_collision_ion_ion(u, x, t,
                                   equations::CompressibleEulerMultiIonEquations1D)

Compute the ion-ion collision source terms for the momentum and energy equations of each ion species as
```math
\begin{aligned}
  \vec{s}_{\rho_k \vec{v}_k} =&  \rho_k\sum_{k'}\bar{\nu}_{kk'}(\vec{v}_{k'} - \vec{v}_k),\\
  s_{E_k}  =& 
    3 \sum_{k'} \left(
    \bar{\nu}_{kk'} \frac{\rho_k M_1}{M_{k'} + M_k} R_1 (T_{k'} - T_k)
    \right) + 
    \sum_{k'} \left(
        \bar{\nu}_{kk'} \rho_k \frac{M_{k'}}{M_{k'} + M_k} \|\vec{v}_{k'} - \vec{v}_k\|^2
        \right)
        +
        \vec{v}_k \cdot \vec{s}_{\rho_k \vec{v}_k},
\end{aligned}
```
where ``M_k`` is the molar mass of ion species `k` provided in `equations.molar_masses`, 
``R_k`` is the specific gas constant of ion species `k` provided in `equations.gas_constants`, and
 ``\bar{\nu}_{kk'}`` is the effective collision frequency of species `k` with species `k'`, which is computed as
```math
\begin{aligned}
  \bar{\nu}_{kk'} = \bar{\nu}^1_{kk'} \tilde{B}_{kk'} \frac{\rho_{k'}}{T_{k k'}^{3/2}},
\end{aligned}
```
with the so-called reduced temperature ``T_{k k'}`` and the ion-ion collision constants ``\tilde{B}_{kk'}`` provided
in `equations.ion_electron_collision_constants` (see [`CompressibleEulerMultiIonEquations1D`](@ref)).

The additional coefficient ``\bar{\nu}^1_{kk'}`` is a non-dimensional drift correction factor proposed by Rambo and Denavit.

References:
- P. Rambo, J. Denavit, Interpenetration and ion separation in colliding plasmas, Physics of Plasmas 1 (1994) 4050–4060.
- Schunk, R. W., Nagy, A. F. (2000). Ionospheres: Physics, plasma physics, and chemistry. 
  Cambridge university press.
"""
@inline function source_terms_collision_ion_ion(u, x, t,
                                                equations::CompressibleEulerMultiIonEquations1D)
    s = zero(MVector{nvariables(equations), eltype(u)})
    @unpack gas_constants, molar_masses, ion_ion_collision_constants = equations

    prim = cons2prim(u, equations)

    for k in eachcomponent(equations)
        rho_k, v1_k, p_k = get_component(k, prim, equations)
        T_k = p_k / (rho_k * gas_constants[k])

        S_q1 = zero(eltype(u))
        S_E = zero(eltype(u))
        for l in eachcomponent(equations)
            # Do not compute collisions of an ion species with itself
            k == l && continue

            rho_l, v1_l, p_l = get_component(l, prim, equations)
            T_l = p_l / (rho_l * gas_constants[l])

            # Reduced temperature
            T_kl = (molar_masses[l] * T_k + molar_masses[k] * T_l) /
                   (molar_masses[k] + molar_masses[l])

            delta_v = (v1_l - v1_k)^2

            # Compute collision frequency without drifting correction
            v_kl = ion_ion_collision_constants[k, l] * rho_l / T_kl^(3 / 2)

            # Correct the collision frequency with the drifting effect
            z = delta_v / (p_l / rho_l + p_k / rho_k)
            v_kl /= (1 + (2 / (9 * pi))^(1 / 3) * z)^(3 / 2)

            S_q1 += rho_k * v_kl * (v1_l - v1_k)
            S_E += (3 * molar_masses[1] * gas_constants[1] * (T_l - T_k)
                    +
                    molar_masses[l] * delta_v) * v_kl * rho_k /
                   (molar_masses[k] + molar_masses[l])
        end

        S_E += v1_k * S_q1

        set_component!(s, k, 0, S_q1, S_E, equations)
    end
    return SVector{nvariables(equations), real(equations)}(s)
end

@doc raw"""
    source_terms_collision_ion_electron(u, x, t,
                                        equations::CompressibleEulerMultiIonEquations1D)

Compute the ion-electron collision source terms for the momentum and energy equations of each ion species. We assume ``v_e = v^+`` 
(no effect of currents on the electron velocity).

The collision sources read as
```math
\begin{aligned}
    \vec{s}_{\rho_k \vec{v}_k} =&  \rho_k \bar{\nu}_{ke} (\vec{v}_{e} - \vec{v}_k),
    \\
    s_{E_k}  =& 
    3  \left(
    \bar{\nu}_{ke} \frac{\rho_k M_{1}}{M_k} R_1 (T_{e} - T_k)
    \right) 
        +
        \vec{v}_k \cdot \vec{s}_{\rho_k \vec{v}_k},
\end{aligned}
```
where ``T_e`` is the electron temperature computed with the function `equations.electron_temperature`, 
``M_k`` is the molar mass of ion species `k` provided in `equations.molar_masses`, 
``R_k`` is the specific gas constant of ion species `k` provided in `equations.gas_constants`, and
``\bar{\nu}_{kk'}`` is the collision frequency of species `k` with the electrons, which is computed as
```math
\begin{aligned}
  \bar{\nu}_{ke} = \tilde{B}_{ke} \frac{e n_e}{T_e^{3/2}},
\end{aligned}
```
with the total electron charge ``e n_e`` (computed assuming quasi-neutrality), and the
ion-electron collision coefficient ``\tilde{B}_{ke}`` provided in `equations.ion_electron_collision_constants`,
which is scaled with the elementary charge (see [`CompressibleEulerMultiIonEquations1D`](@ref)).

References:
- P. Rambo, J. Denavit, Interpenetration and ion separation in colliding plasmas, Physics of Plasmas 1 (1994) 4050–4060.
- Schunk, R. W., Nagy, A. F. (2000). Ionospheres: Physics, plasma physics, and chemistry. 
  Cambridge university press.
"""
function source_terms_collision_ion_electron(u, x, t,
                                             equations::CompressibleEulerMultiIonEquations1D)
    s = zero(MVector{nvariables(equations), eltype(u)})
    @unpack gas_constants, molar_masses, ion_electron_collision_constants, electron_temperature = equations

    prim = cons2prim(u, equations)
    T_e = electron_temperature(u, equations)
    T_e32 = T_e^(3 / 2)
    v1_plus, vk1_plus = charge_averaged_velocities(u, equations)

    # Compute total electron charge
    total_electron_charge = zero(real(equations))
    for k in eachcomponent(equations)
        rho, _ = get_component(k, u, equations)
        total_electron_charge += rho * equations.charge_to_mass[k]
    end

    for k in eachcomponent(equations)
        rho_k, v1_k, p_k = get_component(k, prim, equations)
        T_k = p_k / (rho_k * gas_constants[k])

        # Compute effective collision frequency
        v_ke = ion_electron_collision_constants[k] * total_electron_charge / T_e32

        S_q1 = rho_k * v_ke * (v1_plus - v1_k)

        S_E = 3 * molar_masses[1] * gas_constants[1] * (T_e - T_k) * v_ke * rho_k /
              molar_masses[k]

        S_E += (v1_k * S_q1)

        set_component!(s, k, 0, S_q1, S_E, equations)
    end
    return SVector{nvariables(equations), real(equations)}(s)
end

"""
    flux(u, orientation::Integer, equations::CompressibleEulerMultiIonEquations1D)

Calculate the flux for the multiion system. The flux is calculated for each ion species separately.
"""
@inline function flux(u, orientation::Integer,
                      equations::CompressibleEulerMultiIonEquations1D)
    f = zero(MVector{nvariables(equations), eltype(u)})

    for k in eachcomponent(equations)
        rho, rho_v1, rho_e = get_component(k, u, equations)
        rho_inv = 1 / rho
        v1 = rho_v1 * rho_inv

        gamma = equations.gammas[k]
        p = (gamma - 1) * (rho_e - 0.5 * rho_v1 * v1)

        f1 = rho_v1
        f2 = rho_v1 * v1 + p
        f3 = (rho_e + p) * v1

        set_component!(f, k, f1, f2, f3, equations)
    end

    return SVector(f)
end

"""
    electron_pressure_zero(u, equations::CompressibleEulerMultiIonEquations1D)

Returns the value of zero for the electron pressure. Needed for consistency with the 
single-fluid compressible euler equations in the limit of one ion species.
"""
function electron_pressure_zero(u, equations::CompressibleEulerMultiIonEquations1D)
    return zero(u[1])
end

"""
    v1, vk1 = charge_averaged_velocities(u, equations::CompressibleEulerMultiIonEquations1D)


Compute the charge-averaged velocities (`v1`) and each ion species' contribution
to the charge-averaged velocities (`vk1`). The output variables `vk1` 
are `SVectors` of size `ncomponents(equations)`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
@inline function charge_averaged_velocities(u,
                                            equations::CompressibleEulerMultiIonEquations1D)
    total_electron_charge = zero(real(equations))

    vk1_plus = zero(MVector{ncomponents(equations), eltype(u)})

    for k in eachcomponent(equations)
        rho, rho_v1, _ = get_component(k, u, equations)

        total_electron_charge += rho * equations.charge_to_mass[k]
        vk1_plus[k] = rho_v1 * equations.charge_to_mass[k]
    end
    vk1_plus ./= total_electron_charge
    v1_plus = sum(vk1_plus)

    return v1_plus, SVector(vk1_plus)
end

"""
    get_component(k, u, equations::CompressibleEulerMultiIonEquations1D)

Get the variables of component (ion species) `k`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
@inline function get_component(k, u, equations::CompressibleEulerMultiIonEquations1D)
    return SVector(u[(k - 1) * 3 + 1],
                   u[(k - 1) * 3 + 2],
                   u[(k - 1) * 3 + 3])
end

"""
    set_component!(u, k, u1, u2, u3,
                   equations::CompressibleEulerMultiIonEquations1D)

Set the variables (`u1` to `u3`) of component (ion species) `k`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
@inline function set_component!(u, k, u1, u2, u3,
                                equations::CompressibleEulerMultiIonEquations1D)
    u[(k - 1) * 3 + 1] = u1
    u[(k - 1) * 3 + 2] = u2
    u[(k - 1) * 3 + 3] = u3

    return u
end

# Calculate estimates for maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::CompressibleEulerMultiIonEquations1D)

    # Calculate velocities
    v_mag_ll = zero(eltype(u_ll))
    v_mag_rr = zero(eltype(u_rr))
    c_ll = zero(eltype(u_ll))
    c_rr = zero(eltype(u_rr))

    for k in eachcomponent(equations)
        rho_ll, rho_v1_ll, rho_e_ll = get_component(k, u_ll, equations)
        rho_rr, rho_v1_rr, rho_e_rr = get_component(k, u_rr, equations)
        gamma = equations.gammas[k]

        # Calculate primitive variables and speed of sound
        v1_ll = rho_v1_ll / rho_ll
        p_ll = (gamma - 1) * (rho_e_ll - 0.5f0 * rho_ll * abs(v1_ll)^2)
        v_mag_ll = max(v_mag_ll, abs(v1_ll))
        c_ll = max(c_ll, sqrt(gamma * p_ll / rho_ll))

        v1_rr = rho_v1_rr / rho_rr
        p_rr = (gamma - 1) * (rho_e_rr - 0.5f0 * rho_rr * abs(v1_rr)^2)
        v_mag_rr = max(v_mag_rr, abs(v1_rr))
        c_rr = max(c_rr, sqrt(gamma * p_rr / rho_rr))
    end

    λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
end

@inline function max_abs_speeds(u,
                                equations::CompressibleEulerMultiIonEquations1D)
    prim = cons2prim(u, equations)
    v1_max = zero(eltype(u))
    for k in eachcomponent(equations)
        rho, v1, p = get_component(k, prim, equations)
        gamma = equations.gammas[k]
        c = sqrt(gamma * p / rho)
        v1_max = max(v1_max, abs(v1) + c)
    end
    return (v1_max,)
end

# Calculate estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::CompressibleEulerMultiIonEquations1D)
    prim_ll = cons2prim(u_ll, equations)
    prim_rr = cons2prim(u_rr, equations)

    λ_min = oftype(u_ll[1], Inf)
    λ_max = oftype(u_ll[1], -Inf)

    for k in eachcomponent(equations)
        rho_ll, v1_ll, p_ll = get_component(k, prim_ll, equations)
        rho_rr, v1_rr, p_rr = get_component(k, prim_rr, equations)
        gamma = equations.gammas[k]

        λ_min = min(λ_min, v1_ll - sqrt(gamma * p_ll / rho_ll))
        λ_max = max(λ_max, v1_rr + sqrt(gamma * p_rr / rho_rr))
    end

    #Assert that λ_min and λ_max are not Inf
    @assert !isinf(λ_min) "λ_min is Inf"
    @assert !isinf(λ_max) "λ_max is Inf"

    return λ_min, λ_max
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::CompressibleEulerMultiIonEquations1D)
    prim_ll = cons2prim(u_ll, equations)
    prim_rr = cons2prim(u_rr, equations)

    λ_min = oftype(u_ll[1], Inf)
    λ_max = oftype(u_ll[1], -Inf)

    for k in eachcomponent(equations)
        rho_ll, v1_ll, p_ll = get_component(k, prim_ll, equations)
        rho_rr, v1_rr, p_rr = get_component(k, prim_rr, equations)
        gamma = equations.gammas[k]

        c_ll = sqrt(gamma * p_ll / rho_ll)
        c_rr = sqrt(gamma * p_rr / rho_rr)

        λ_min = min(λ_min, min(v1_ll - c_ll, v1_rr - c_rr))
        λ_max = max(λ_max, max(v1_ll + c_ll, v1_rr + c_rr))
    end

    return λ_min, λ_max
end

@inline function cons2prim(u, equations::CompressibleEulerMultiIonEquations1D)
    prim = zero(MVector{nvariables(equations), eltype(u)})
    for k in eachcomponent(equations)
        rho, rho_v1, rho_e = get_component(k, u, equations)
        v1 = rho_v1 / rho
        p = (equations.gammas[k] - 1) * (rho_e - 0.5 * rho * v1^2)
        set_component!(prim, k, rho, v1, p, equations)
    end

    return SVector(prim)
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleEulerMultiIonEquations1D)
    w = zero(MVector{nvariables(equations), eltype(u)})

    for k in eachcomponent(equations)
        rho, rho_v1, rho_e = get_component(k, u, equations)
        gamma = equations.gammas[k]
        inv_gamma_minus_one = equations.inv_gammas_minus_one[k]
        v1 = rho_v1 / rho
        v_square = v1^2
        p = (gamma - 1) * (rho_e - 0.5 * rho * v_square)
        s = log(p) - gamma * log(rho)
        rho_p = rho / p

        w1 = (gamma - s) * inv_gamma_minus_one -
             0.5 * rho_p * v_square
        w2 = rho_p * v1
        w3 = -rho_p
        set_component!(w, k, w1, w2, w3, equations)
    end

    return SVector(w)
end

@inline function entropy2cons(w, equations::CompressibleEulerMultiIonEquations1D)
    cons = zero(MVector{nvariables(equations), eltype(w)})
    # See Hughes, Franca, Mallet (1986) A new finite element formulation for CFD
    # [DOI: 10.1016/0045-7825(86)90127-1](https://doi.org/10.1016/0045-7825(86)90127-1)

    # convert to entropy `-rho * s` used by Hughes, France, Mallet (1986)
    # instead of `-rho * s / (gamma - 1)`
    for k in eachcomponent(equations)
        w_k = get_component(k, w, equations)
        gamma = equations.gammas[k]
        inv_gamma_minus_one = equations.inv_gammas_minus_one[k]
        V1, V2, V5 = w_k .* (gamma - 1)

        # specific entropy, eq. (53)
        s = gamma - V1 + 0.5f0 * (V2^2) / V5

        # eq. (52)
        energy_internal = ((gamma - 1) / (-V5)^gamma)^(inv_gamma_minus_one) *
                          exp(-s * inv_gamma_minus_one)

        # eq. (51)
        rho = -V5 * energy_internal
        rho_v1 = V2 * energy_internal
        rho_e = (1 - 0.5f0 * (V2^2) / V5) * energy_internal
        set_component!(cons, k, rho, rho_v1, rho_e, equations)
    end
    return SVector(cons)
end

@inline function prim2cons(prim, equations::CompressibleEulerMultiIonEquations1D)
    cons = zero(MVector{nvariables(equations), eltype(prim)})

    for k in eachcomponent(equations)
        rho, v1, p = get_component(k, prim, equations)
        inv_gamma_minus_one = equations.inv_gammas_minus_one[k]
        rho_v1 = rho * v1
        rho_e = p * inv_gamma_minus_one + 0.5f0 * (rho_v1 * v1)
        set_component!(cons, k, rho, rho_v1, rho_e, equations)
    end

    return SVector(cons)
end

@inline function density(u, equations::CompressibleEulerMultiIonEquations1D)
    num_components = div(nvariables(equations), 3)
    rhos = zero(MVector{num_components, eltype(u)})

    for k in eachcomponent(equations)
        rho, _, _ = get_component(k, u, equations)
        rhos[k] = rho
    end

    return SVector(rhos)
end

@inline function velocity(u, equations::CompressibleEulerMultiIonEquations1D)
    num_components = div(nvariables(equations), 3)
    velocities = zero(MVector{num_components, eltype(u)})

    for k in eachcomponent(equations)
        rho, rho_v1, _ = get_component(k, u, equations)
        velocities[k] = rho_v1 / rho
    end

    return SVector(velocities)
end

@inline function pressure(u, equations::CompressibleEulerMultiIonEquations1D)
    num_components = div(nvariables(equations), 3)
    pressures = zero(MVector{num_components, eltype(u)})

    for k in eachcomponent(equations)
        rho, rho_v1, rho_e = get_component(k, u, equations)
        p = (equations.gammas[k] - 1) * (rho_e - 0.5f0 * (rho_v1^2) / rho)
        pressures[k] = p
    end
    return SVector(pressures)
end
end # @muladd
