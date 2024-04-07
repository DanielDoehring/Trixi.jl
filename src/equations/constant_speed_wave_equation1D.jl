# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    ConstantSpeedWaveEquation1D

The wave equation
```math
\partial_t u - c^2 \partial_1 u = 0
```
in one space dimension with constant velocity `c`.
With the change of variables ``v \coloneqq c \partial_1 u`` and ``w \coloneqq - \partial_t u``
this equation can be transformed into c first order hyperbolic system
```math
\partial_t
\begin{pmatrix}
    v \\ w
\end{pmatrix}
+ \partial_1
\begin{pmatrix}
    0 & c \\ c & 0
\end{pmatrix}
= 
\begin{pmatrix}
0 \\ 0
\end{pmatrix}
.
"""
struct ConstantSpeedWaveEquation1D{RealT <: Real} <:
       AbstractLinearScalarAdvectionEquation{1, 2}
    propagation_speed::SVector{1, RealT}
end

function ConstantSpeedWaveEquation1D(c::Real)
    ConstantSpeedWaveEquation1D(SVector(c))
end

function varnames(::typeof(cons2cons), ::ConstantSpeedWaveEquation1D)
  ("v", "w")
end
function varnames(::typeof(cons2prim), ::ConstantSpeedWaveEquation1D)
  ("v", "w")
end

"""
    initial_condition_convergence_test(x, t, equations::ConstantSpeedWaveEquation1D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equations::ConstantSpeedWaveEquation1D)
    c = equations.propagation_speed[1]
    char_pos = t + x[1] / c

    # Solution: u = sin(2 pi * (t + x/c))

    # v = c u_x
    v = 2 * pi * cos(2 * pi * char_pos)
    # w = -u_t
    w = -2 * pi * cos(2 * pi * char_pos)

    return SVector(v, w)
end

# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::ConstantSpeedWaveEquation1D)

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer,
                      equation::ConstantSpeedWaveEquation1D)
    v, w = u
    c = equation.propagation_speed[orientation]
    return SVector(c * w, c * v)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Int,
                                     equation::ConstantSpeedWaveEquation1D)
    λ_max = abs(equation.propagation_speed[orientation])
end

@inline have_constant_speed(::ConstantSpeedWaveEquation1D) = True()

@inline function max_abs_speeds(equation::ConstantSpeedWaveEquation1D)
    return abs.(equation.propagation_speed)
end

@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::ConstantSpeedWaveEquation1D)
    min_max_speed_davis(u_ll, u_rr, orientation, equations)
end

@inline function min_max_speed_davis(u_ll, u_rr, orientation::Integer,
                                     equations::ConstantSpeedWaveEquation1D)
    c_abs = abs(equations.propagation_speed[orientation])

    λ_min = -c_abs
    λ_max = c_abs

    return λ_min, λ_max
end

# TODO: Something is wrong here
function compute_u(u_sol, mesh, equations::ConstantSpeedWaveEquation1D, dg,
                   cache)            
                  
    u = zeros(eltype(u_sol), nelements(dg, cache))
    c = equations.propagation_speed[1]

    @assert c != 0 "Propagation speed must be non-zero"
    @unpack weights = dg.basis

    for element in eachelement(dg, cache)
      for i in eachnode(dg)
          u[element] += 1/c * weights[i] * 
                        get_node_vars(u_sol, equations, dg, i, element)[1]
      end
      volume_jacobian_ = volume_jacobian(element, mesh, cache)
      u[element] *= volume_jacobian_
  end

  return u
end

# Convert conservative variables to primitive
@inline cons2prim(u, ::ConstantSpeedWaveEquation1D) = u
@inline cons2entropy(u, ::ConstantSpeedWaveEquation1D) = u
end # @muladd
