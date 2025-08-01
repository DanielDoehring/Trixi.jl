# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    SplitPairedExplicitRK2(num_stages,
                          base_path_monomial_coeffs::AbstractString,
                          base_path_monomial_coeffs_para::AbstractString;
                          dt_opt = nothing, bS = 1.0, cS = 0.5)
"""
struct SplitPairedExplicitRK2 <:
       AbstractPairedExplicitRKSingle{2}
    num_stages::Int

    a_matrix::Matrix{Float64}
    a_matrix_para::Matrix{Float64}
    c::Vector{Float64}
    b1::Float64
    bS::Float64
    cS::Float64

    dt_opt::Union{Float64, Nothing}
end

# Constructor that reads the coefficients from a file
function SplitPairedExplicitRK2(num_stages,
                                base_path_monomial_coeffs::AbstractString,
                                base_path_monomial_coeffs_para::AbstractString;
                                dt_opt = nothing,
                                bS = 1.0, cS = 0.5)
    @assert num_stages>=2 "PERK2 requires at least two stages"
    # If the user has the monomial coefficients, they also must have the optimal time step
    a_matrix, c = compute_PairedExplicitRK2_butcher_tableau(num_stages,
                                                            base_path_monomial_coeffs,
                                                            bS, cS)

    a_matrix_para, _ = compute_PairedExplicitRK2_butcher_tableau(num_stages,
                                                                 base_path_monomial_coeffs_para,
                                                                 bS, cS)

    return SplitPairedExplicitRK2(num_stages, a_matrix, a_matrix_para, c, 1 - bS, bS,
                                  cS, dt_opt)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct SplitPairedExplicitRK2Integrator{RealT <: Real, uType,
                                                Params, Sol, F,
                                                PairedExplicitRKOptions} <:
               AbstractSplitPairedExplicitRKSingleIntegrator{2}
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    tdir::RealT # DIRection of time integration, i.e., if one marches forward or backward in time
    dt::RealT # current time step
    dtcache::RealT # manually set time step
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F # `rhs!` of the semidiscretization
    alg::SplitPairedExplicitRK2
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool

    k1::uType # Additional PERK register
    # For split (hyperbolic-parabolic) problems
    du_para::uType # Stores the parabolic part of the overall rhs!
    k1_para::uType # Additional PERK register for the parabolic part
end

function init(ode::ODEProblem, alg::SplitPairedExplicitRK2;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    k1 = zero(u0) # Additional PERK register
    du_para = zero(u0) # Stores the parabolic part of the overall rhs!
    k1_para = zero(u0) # Additional PERK register for the parabolic part

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    integrator = SplitPairedExplicitRK2Integrator(u0, du, u_tmp,
                                                  t0, tdir, dt, zero(dt),
                                                  iter, ode.p,
                                                  (prob = ode,), ode.f,
                                                  alg,
                                                  PairedExplicitRKOptions(callback,
                                                                          ode.tspan;
                                                                          kwargs...),
                                                  false, true, false,
                                                  k1, du_para, k1_para)

    # initialize callbacks
    if callback isa CallbackSet
        for cb in callback.continuous_callbacks
            throw(ArgumentError("Continuous callbacks are unsupported with paired explicit Runge-Kutta methods."))
        end
        for cb in callback.discrete_callbacks
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return integrator
end
end # @muladd
