
# Compressible Navier-Stokes equations
abstract type AbstractCompressibleNavierStokesDiffusion{NDIMS, NVARS, GradientVariables} <:
              AbstractEquationsParabolic{NDIMS, NVARS, GradientVariables} end

# This enables "forwarded" accesses to e.g.`equations_parabolic.gamma` of the "underlying" `equations_hyperbolic`
# while keeping direct access to parabolic-specific fields like `mu` or `kappa`.
@inline function Base.getproperty(equations_parabolic::AbstractCompressibleNavierStokesDiffusion,
                                  field::Symbol)
    if field === :gamma || field === :inv_gamma_minus_one
        return getproperty(getfield(equations_parabolic, :equations_hyperbolic), field)
    else
        return getfield(equations_parabolic, field)
    end
end

# Provide property names for e.g. tab-completion by combining
# the names from the underlying hyperbolic equations with the fields of this parabolic part.
@inline function Base.propertynames(equations_parabolic::AbstractCompressibleNavierStokesDiffusion,
                                    private::Bool = false)
    names_hyp = (:gamma, :inv_gamma_minus_one)
    names_para = fieldnames(typeof(equations_parabolic))
    names_hyp_para = (names_hyp..., names_para...)

    return names_hyp_para
end

# TODO: can we generalize this to V(R)-MHD?
"""
    struct BoundaryConditionNavierStokesWall

Creates a wall-type boundary conditions for the compressible Navier-Stokes equations, see
[`CompressibleNavierStokesDiffusion1D`](@ref), [`CompressibleNavierStokesDiffusion2D`](@ref), and
[`CompressibleNavierStokesDiffusion3D`](@ref).
The fields `boundary_condition_velocity` and `boundary_condition_heat_flux` are intended
to be boundary condition types such as the [`NoSlip`](@ref) velocity boundary condition and the
[`Adiabatic`](@ref) or [`Isothermal`](@ref) heat boundary condition.
"""
struct BoundaryConditionNavierStokesWall{V, H}
    boundary_condition_velocity::V
    boundary_condition_heat_flux::H
end

"""
    struct NoSlip

Use to create a no-slip boundary condition with [`BoundaryConditionNavierStokesWall`](@ref).
The field `boundary_value_function` should be a function with signature
`boundary_value_function(x, t, equations)` and return a `SVector{NDIMS}`
whose entries are the velocity vector at a point `x` and time `t`.
"""
struct NoSlip{F}
    boundary_value_function::F # value of the velocity vector on the boundary
end

"""
    struct Slip

Creates a symmetric velocity boundary condition which eliminates any normal velocity gradients across the boundary, i.e.,
allows only the tangential velocity gradients to be non-zero.
When combined with the heat boundary condition [`Adiabatic`](@ref), this creates a truly symmetric boundary condition.
Any boundary on which this combined boundary condition is applied thus acts as a symmetry plane for the flow.
In contrast to the [`NoSlip`](@ref) boundary condition, `Slip` does not require a function to be supplied.

The (purely) hyperbolic equivalent boundary condition is [`boundary_condition_slip_wall`](@ref) which
permits only tangential velocities.

This boundary condition can also be employed as a reflective wall.

Note that in 1D this degenerates to the [`NoSlip`](@ref) boundary condition which must be used instead.

!!! note
    Currently this (velocity) boundary condition is only implemented for
    [`P4estMesh`](@ref) and [`GradientVariablesPrimitive`](@ref).
"""
struct Slip end

"""
    struct Isothermal

Used to create a no-slip boundary condition with [`BoundaryConditionNavierStokesWall`](@ref).
The field `boundary_value_function` should be a function with signature
`boundary_value_function(x, t, equations)` and return a scalar value for the
temperature at point `x` and time `t`.
"""
struct Isothermal{F}
    boundary_value_function::F # value of the temperature on the boundary
end

"""
    struct Adiabatic

Used to create a no-slip boundary condition with [`BoundaryConditionNavierStokesWall`](@ref).
The field `boundary_value_normal_flux_function` should be a function with signature
`boundary_value_normal_flux_function(x, t, equations)` and return a scalar value for the
normal heat flux at point `x` and time `t`.
"""
struct Adiabatic{F}
    boundary_value_normal_flux_function::F # scaled heat flux 1/T * kappa * dT/dn
end

"""
`GradientVariablesPrimitive` is a gradient variable type parameter for the [`CompressibleNavierStokesDiffusion1D`](@ref),
[`CompressibleNavierStokesDiffusion2D`](@ref), and [`CompressibleNavierStokesDiffusion3D`](@ref).
The other available gradient variable type parameter is [`GradientVariablesEntropy`](@ref).
By default, the gradient variables are set to be `GradientVariablesPrimitive`.
"""
struct GradientVariablesPrimitive end

"""
`GradientVariablesEntropy` is a gradient variable type parameter for the [`CompressibleNavierStokesDiffusion1D`](@ref),
[`CompressibleNavierStokesDiffusion2D`](@ref), and [`CompressibleNavierStokesDiffusion3D`](@ref).
The other available gradient variable type parameter is [`GradientVariablesPrimitive`](@ref).

Specifying `GradientVariablesEntropy` uses the entropy variable formulation from
- Hughes, Mallet, Franca (1986)
  A new finite element formulation for computational fluid dynamics: I. Symmetric forms of the
  compressible Euler and Navier-Stokes equations and the second law of thermodynamics.
  [https://doi.org/10.1016/0045-7825(86)90127-1](https://doi.org/10.1016/0045-7825(86)90127-1)

Under `GradientVariablesEntropy`, the Navier-Stokes discretization is provably entropy stable.
"""
struct GradientVariablesEntropy end

"""
    dynamic_viscosity(u, equations)

Wrapper for the dynamic viscosity that calls
`dynamic_viscosity(u, equations.mu, equations)`, which dispatches on the type of
`equations.mu`.
For constant `equations.mu`, i.e., `equations.mu` is of `Real`-type it is returned directly.
In all other cases, `equations.mu` is assumed to be a function with arguments
`u` and `equations` and is called with these arguments.
"""
dynamic_viscosity(u, equations) = dynamic_viscosity(u, equations.mu, equations)
dynamic_viscosity(u, mu::Real, equations) = mu
dynamic_viscosity(u, mu::T, equations) where {T} = mu(u, equations)

"""
    have_constant_diffusivity(::AbstractCompressibleNavierStokesDiffusion)

# Returns
- `False()`

Used in parabolic CFL condition computation (see [`StepsizeCallback`](@ref)) to indicate that the
diffusivity is not constant in space and that [`max_diffusivity`](@ref) needs to be computed
at every node in every element.

Also employed in [`linear_structure`](@ref) and [`linear_structure_parabolic`](@ref) to check
if the diffusion term is linear in the variables/constant.
"""
@inline have_constant_diffusivity(::AbstractCompressibleNavierStokesDiffusion) = False()

# Radiative-equilibrium no-slip wall BC for CompressibleNavierStokesDiffusion1D
#
# Physics: convective/conductive heat reaching the wall is balanced by
# grey-body radiative emission to a cold far field (neglecting incoming
# radiation):
#
#     k(T_w) * dT/dn|_w  =  eps * sigma * (T_w^4 - T_far_field^4)
#
# T_w is not prescribed -- it's the root of this nonlinear balance, solved
# locally (per boundary node) via Newton's method, using a one-sided estimate
# of dT/dn purely as the *iteration* surrogate. Trixi's own (more accurate,
# lifted) gradient machinery is what actually gets used in the real flux,
# once we hand back the converged T_w as the boundary value -- exactly the
# same structural slot the built-in `Isothermal` BC fills with a prescribed T.

"""
    RadiativeEquilibrium
 
Marker type analogous to `Trixi.Isothermal` / `Trixi.Adiabatic`, to be used as
the temperature-BC slot of `BoundaryConditionNavierStokesWall`:
 
    BoundaryConditionNavierStokesWall(NoSlip(velocity_fn),
                                       RadiativeEquilibrium(eps, sigma, dist_fn))
 
    # with a nonzero far-field/background temperature:
    BoundaryConditionNavierStokesWall(NoSlip(velocity_fn),
                                       RadiativeEquilibrium(eps, sigma, dist_fn;
                                                             T_far_field = 300.0))
 
`emissivity`, `stefan_boltzmann`, and `T_far_field` may each be `Real` or
callables `(x, t, equations) -> Real`, to allow spatially/temporally varying
values. The radiative balance solved is
 
    k(T_w) * (T_inner - T_w) / delta  =  eps * sigma * (T_w^4 - T_far_field^4)
 
`T_far_field = 0` recovers the "neglect far-field" case.
 
IMPORTANT -- what `boundary_node_distance` is and why it matters:
There is no mesh/basis object available inside a `BoundaryConditionNavierStokesWall`
call, so Trixi cannot hand you the LGL node spacing automatically. But this
function's job is NOT cosmetic: it supplies `delta`, the one-sided distance used
in a *local* finite-difference estimate of dT/dn,
 
    dT/dn|_w  ~=  (T_inner - T_w) / delta
 
which only exists to give the *Newton iteration inside this BC* a residual to
drive to zero. Trixi's own (BR1-lifted) gradient, used in the real flux that
actually enters the RHS, is computed completely separately and does NOT use
`delta` at all -- once T_w has converged, Trixi recomputes the gradient itself
from the returned boundary value, the same way it does for `Isothermal`.
 
However, because `delta` appears directly inside the iteration's residual,
the *converged* T_w does depend on what `delta` you supply -- a wrong delta
gives a different fixed point, not just a slower path to the same one. Get
delta right: for DGSEM with LGL nodes of degree `p` on an element of physical
length `L`,
 
    delta = (L / 2) * (nodes[2] - nodes[1])   # nodes from dg.basis.nodes
 
is the genuine physical distance from the boundary node to the nearest
interior node, and should be closed over from your basis/mesh object at BC
construction time, e.g.:
 
    nodes = dg.basis.nodes  # reference LGL nodes on [-1, 1]
    delta = dx / 2 * (nodes[2] - nodes[1])
    dist_fn(x, direction) = delta
"""
struct RadiativeEquilibrium{
    ConvectiveHeatTransferCoefficient <: Real,
    Emissivity <: Real,
                            Absorptivity <: Real,
                            TempFarfield <: Real,
                            StefanBoltzmannConst <: Real}
    convective_heat_transfer_coefficient::ConvectiveHeatTransferCoefficient
    emissivity::Emissivity
    temp_farfield::TempFarfield
    stefan_boltzmann_const::StefanBoltzmannConst
end

"""
    RadiativeEquilibrium(;

        emissivity = 1.0,
        T_far_field = 0.0f0, stefan_boltzmann = 5.670374419f-8)
"""
function RadiativeEquilibrium(;
                              emissivity = 1.0,
                              T_far_field = 0.0f0, stefan_boltzmann = 5.670374419f-8)
    return RadiativeEquilibrium{typeof(emissivity), typeof(absorptivity),
                              typeof(T_far_field), typeof(stefan_boltzmann)}(
        boundary_node_distance, emissivity, absorptivity, T_far_field, stefan_boltzmann)
end

@inline function solve_radiative_equilibrium_temperature(T_inner, rad_bc,
    equations)

    # TODO: Reconstruct h on-the fly from flux_inner?
    h = rad_bc.conv_heat_transfer_coefficient
    eps = rad_bc.emissivity
    sigma = rad_bc.stefan_boltzmann_const

    @unpack kappa = equations

    T_w = T_inner
    T_far4 = rad_bc.temp_farfield^4
 
    for _ in 1:max_iter
        q_cond = h * (T_inner - T_w)
        q_rad = eps * sigma * (T_w^4 - T_far4)
        q_diff = q_cond - q_rad
 
        dq_cond_dT = -h
        dq_rad_dT = 4 * eps * sigma * T_w^3
        dq_diff_dT = dq_cond_dT - dq_rad_dT
 
        dT = -q_diff / dq_diff_dT
        T_w += dT
        T_w = max(T_w, 1)
 
        if abs(dT) < tol * max(T_w, 1)
            break
        end
    end
 
    return T_w
end



include("compressible_navier_stokes_1d.jl")
include("compressible_navier_stokes_2d.jl")
include("compressible_navier_stokes_3d.jl")
