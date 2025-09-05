# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    PairedExplicitRK2Coupled(num_stages,
                          base_path_monomial_coeffs_1::AbstractString,
                          base_path_monomial_coeffs_2::AbstractString;
                          dt_opt = nothing, bS = 1.0, cS = 0.5)
"""
struct PairedExplicitRK2Coupled <:
       AbstractPairedExplicitRKCoupledSingle{2}
    num_stages::Int

    a_matrix_1::Matrix{Float64}
    a_matrix_2::Matrix{Float64}
    c::Vector{Float64}
    b1::Float64
    bS::Float64
    cS::Float64

    dt_opt::Union{Float64, Nothing}
end

# Constructor that reads the coefficients from a file
function PairedExplicitRK2Coupled(num_stages,
                                  base_path_monomial_coeffs_1::AbstractString,
                                  base_path_monomial_coeffs_2::AbstractString;
                                  dt_opt = nothing,
                                  bS = 1.0, cS = 0.5)
    @assert num_stages>=2 "PERK2 requires at least two stages"
    # If the user has the monomial coefficients, they also must have the optimal time step
    a_matrix_1, c = compute_PairedExplicitRK2_butcher_tableau(num_stages,
                                                              base_path_monomial_coeffs_1,
                                                              bS, cS)

    a_matrix_2, _ = compute_PairedExplicitRK2_butcher_tableau(num_stages,
                                                              base_path_monomial_coeffs_2,
                                                              bS, cS)

    return PairedExplicitRK2Coupled(num_stages, a_matrix_1, a_matrix_2, c, 1 - bS, bS,
                                    cS, dt_opt)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRK2CoupledIntegrator{RealT <: Real,
                                                  uType <: AbstractVector,
                                                  Params, Sol, F,
                                                  PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKCoupledSingleIntegrator{2}
    u::uType
    du::uType # In-place output of `f`
    u_tmp::uType # Used for building the argument to `f`
    t::RealT
    tdir::RealT # DIRection of time integration, i.e., if one marches forward or backward in time
    dt::RealT # current time step
    dtcache::RealT # manually set time step
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F # `rhs!` of the semidiscretization
    alg::PairedExplicitRK2Coupled
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # Additional PERK register
    k1::uType
end

function init(ode::ODEProblem, alg::PairedExplicitRK2Coupled;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    k1 = zero(u0) # Additional PERK register

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    integrator = PairedExplicitRK2CoupledIntegrator(u0, du, u_tmp,
                                                    t0, tdir, dt, zero(dt),
                                                    iter, ode.p,
                                                    (prob = ode,), ode.f,
                                                    alg,
                                                    PairedExplicitRKOptions(callback,
                                                                            ode.tspan;
                                                                            kwargs...),
                                                    false, true, false,
                                                    k1)

    initialize_callbacks!(callback, integrator)

    return integrator
end
end # @muladd
