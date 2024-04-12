# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Speed of sound is variable in space, but constant in time
# => One more variable
struct LinearizedEulerEquationsVarSoS2D{RealT <: Real} <:
       AbstractLinearizedEulerEquations{2, 5}
    v_mean_global::SVector{2, RealT}
    rho_mean_global::RealT
end

function LinearizedEulerEquationsVarSoS2D(v_mean_global::NTuple{2, <:Real},
                                          rho_mean_global::Real)
    if rho_mean_global < 0
        throw(ArgumentError("rho_mean_global must be non-negative"))
    end

    return LinearizedEulerEquationsVarSoS2D(SVector(v_mean_global), rho_mean_global)
end

function LinearizedEulerEquationsVarSoS2D(; v_mean_global::NTuple{2, <:Real},
                                          rho_mean_global::Real)
    return LinearizedEulerEquationsVarSoS2D(v_mean_global, rho_mean_global)
end

function varnames(::typeof(cons2cons), ::LinearizedEulerEquationsVarSoS2D)
    ("rho_prime", "v1_prime", "v2_prime", "p_prime", "c_mean")
end
function varnames(::typeof(cons2prim), ::LinearizedEulerEquationsVarSoS2D)
    ("rho_prime", "v1_prime", "v2_prime", "p_prime", "c_mean")
end

"""
    initial_condition_convergence_test(x, t, equations::LinearizedEulerEquationsVarSoS2D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equations::LinearizedEulerEquationsVarSoS2D)
    rho_prime = -cospi(2 * t) * (sinpi(2 * x[1]) + sinpi(2 * x[2]))
    v1_prime = sinpi(2 * t) * cospi(2 * x[1])
    v2_prime = sinpi(2 * t) * cospi(2 * x[2])
    p_prime = rho_prime
    c_mean = 1.0

    return SVector(rho_prime, v1_prime, v2_prime, p_prime, c_mean)
end

"""
    boundary_condition_wall(u_inner, orientation, direction, x, t, surface_flux_function,
                                equations::LinearizedEulerEquationsVarSoS2D)

Boundary conditions for a solid wall.
"""
function boundary_condition_wall(u_inner, orientation, direction, x, t,
                                 surface_flux_function,
                                 equations::LinearizedEulerEquationsVarSoS2D)
    # Boundary state is equal to the inner state except for the velocity. For boundaries
    # in the -x/+x direction, we multiply the velocity in the x direction by -1.
    # Similarly, for boundaries in the -y/+y direction, we multiply the velocity in the
    # y direction by -1
    if direction in (1, 2) # x direction
        u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4], u_inner[5])
    else # y direction
        u_boundary = SVector(u_inner[1], u_inner[2], -u_inner[3], u_inner[4], u_inner[5])
    end

    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end

    return flux
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::LinearizedEulerEquationsVarSoS2D)
    @unpack v_mean_global, rho_mean_global = equations

    rho_prime, v1_prime, v2_prime, p_prime, c = u
    if orientation == 1
        f1 = v_mean_global[1] * rho_prime + rho_mean_global * v1_prime
        f2 = v_mean_global[1] * v1_prime + p_prime / rho_mean_global
        f3 = v_mean_global[1] * v2_prime
        f4 = v_mean_global[1] * p_prime + c^2 * rho_mean_global * v1_prime
        f5 = 0.0
    else
        f1 = v_mean_global[2] * rho_prime + rho_mean_global * v2_prime
        f2 = v_mean_global[2] * v1_prime
        f3 = v_mean_global[2] * v2_prime + p_prime / rho_mean_global
        f4 = v_mean_global[2] * p_prime + c^2 * rho_mean_global * v2_prime
        f5 = 0.0
    end

    return SVector(f1, f2, f3, f4, f5)
end

# Calculate 1D flux for a single point
@inline function flux(u, normal_direction::AbstractVector,
                      equations::LinearizedEulerEquationsVarSoS2D)
    @unpack v_mean_global, rho_mean_global = equations
    rho_prime, v1_prime, v2_prime, p_prime, c = u

    v_mean_normal = v_mean_global[1] * normal_direction[1] +
                    v_mean_global[2] * normal_direction[2]
    v_prime_normal = v1_prime * normal_direction[1] + v2_prime * normal_direction[2]

    f1 = v_mean_normal * rho_prime + rho_mean_global * v_prime_normal
    f2 = v_mean_normal * v1_prime + normal_direction[1] * p_prime / rho_mean_global
    f3 = v_mean_normal * v2_prime + normal_direction[2] * p_prime / rho_mean_global
    f4 = v_mean_normal * p_prime + c^2 * rho_mean_global * v_prime_normal
    f5 = 0.0

    return SVector(f1, f2, f3, f4, f5)
end

@inline have_constant_speed(::LinearizedEulerEquationsVarSoS2D) = False()

@inline function max_abs_speeds(u, equations::LinearizedEulerEquationsVarSoS2D)
  @unpack v_mean_global = equations

  _, _, _, _, c = u

  return abs(v_mean_global[1]) + c, abs(v_mean_global[2]) + c
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedEulerEquationsVarSoS2D)
    @unpack v_mean_global = equations

    _, _, _, _, c_ll = u_ll
    _, _, _, _, c_rr = u_rr

    c_max = max(c_ll, c_rr)

    if orientation == 1
        return abs(v_mean_global[1]) + c_max
    else # orientation == 2
        return abs(v_mean_global[2]) + c_max
    end
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::LinearizedEulerEquationsVarSoS2D)
    @unpack v_mean_global = equations

    _, _, _, _, c_ll = u_ll
    _, _, _, _, c_rr = u_rr

    c_max = max(c_ll, c_rr)

    v_mean_normal = normal_direction[1] * v_mean_global[1] +
                    normal_direction[2] * v_mean_global[2]

    return abs(v_mean_normal) + c_max * norm(normal_direction)
end

# Calculate estimate for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedEulerEquationsVarSoS2D)
    min_max_speed_davis(u_ll, u_rr, orientation, equations)
end

@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::LinearizedEulerEquationsVarSoS2D)
    min_max_speed_davis(u_ll, u_rr, normal_direction, equations)
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::LinearizedEulerEquationsVarSoS2D)
    @unpack v_mean_global = equations

    _, _, _, _, c_ll = u_ll
    _, _, _, _, c_rr = u_rr

    c_max = max(c_ll, c_rr)

    λ_min = v_mean_global[orientation] - c_max
    λ_max = v_mean_global[orientation] + c_max

    return λ_min, λ_max
end

@inline function min_max_speed_davis(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::LinearizedEulerEquationsVarSoS2D)
    @unpack v_mean_global = equations

    norm_ = norm(normal_direction)

    v_normal = v_mean_global[1] * normal_direction[1] +
               v_mean_global[2] * normal_direction[2]

    _, _, _, _, c_ll = u_ll
    _, _, _, _, c_rr = u_rr

    c_max = max(c_ll, c_rr)

    # The v_normals are already scaled by the norm
    λ_min = v_normal - c_max * norm_
    λ_max = v_normal + c_max * norm_

    return λ_min, λ_max
end

# Convert conservative variables to primitive
@inline cons2prim(u, equations::LinearizedEulerEquationsVarSoS2D) = u
@inline cons2entropy(u, ::LinearizedEulerEquationsVarSoS2D) = u
end # muladd
