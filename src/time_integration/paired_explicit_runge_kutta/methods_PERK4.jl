# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function compute_PairedExplicitRK4_butcher_tableau(num_stages,
                                                   base_path_a_coeffs::AbstractString;
                                                   c_const = 1.0) # Default value for best internal stability
    c = c_const * ones(num_stages) # Use same abscissae for free coefficients
    c[1] = 0.0

    cS3 = c_const
    c[num_stages - 3] = cS3
    c[num_stages - 2] = 0.479274057836310
    c[num_stages - 1] = sqrt(3) / 6 + 0.5
    c[num_stages] = -sqrt(3) / 6 + 0.5

    num_coeffs_max = num_stages - 5

    a_matrix = zeros(num_coeffs_max, 2)
    a_matrix[:, 1] = c[3:(num_stages - 3)]

    path_a_coeffs = joinpath(base_path_a_coeffs,
                             "a_" * string(num_stages) * ".txt")

    @assert isfile(path_a_coeffs) "Couldn't find file $path_a_coeffs"
    a_coeffs = readdlm(path_a_coeffs, Float64)
    num_a_coeffs = size(a_coeffs, 1)

    @assert num_a_coeffs == num_coeffs_max
    if num_coeffs_max > 0
        a_matrix[:, 1] -= a_coeffs
        a_matrix[:, 2] = a_coeffs
    end

    # Constant/non-optimized part of the Butcher matrix
    a_matrix_constant = [0.479274057836310-0.114851811257441 / cS3 0.114851811257441/cS3
                         0.1397682537005989 0.648906880894214
                         0.1830127018922191 0.028312163512968]

    return a_matrix, a_matrix_constant, c
end

@doc raw"""
    PairedExplicitRK4(num_stages, base_path_a_coeffs::AbstractString, dt_opt = nothing;
                      c_const = 1.0f0)

    Parameters:
    - `num_stages` (`Int`): Number of stages in the paired explicit Runge-Kutta (P-ERK) method.
    - `base_path_a_coeffs` (`AbstractString`): Path to a file containing some coefficients in the A-matrix in 
      the Butcher tableau of the Runge Kutta method.
      The matrix should be stored in a text file at `joinpath(base_path_a_coeffs, "a_$(num_stages).txt")` and separated by line breaks.
    - `dt_opt` (`Float64`, optional): Optimal time step size for the simulation setup. Can be `nothing` if it is unknown. 
       In this case the optimal CFL number cannot be computed and the [`StepsizeCallback`](@ref) cannot be used.
    - `c_const` (`Float64`, optional): Value of abscissae $c$ in the Butcher tableau for the optimized coefficients. 
       Default is 1.0.

The following structures and methods provide an implementation of
the fourth-order paired explicit Runge-Kutta (P-ERK) method
optimized for a certain simulation setup (PDE, IC & BC, Riemann Solver, DG Solver).
The method has been proposed in 
- D. Doehring, L. Christmann, M. Schlottke-Lakemper, G. J. Gassner and M. Torrilhon (2024).
  Fourth-Order Paired-Explicit Runge-Kutta Methods
  [DOI:10.48550/arXiv.2408.05470](https://doi.org/10.48550/arXiv.2408.05470)
"""
mutable struct PairedExplicitRK4 <: AbstractPairedExplicitRKSingle
    const num_stages::Int # S

    a_matrix::Matrix{Float64}
    # This part of the Butcher array matrix A is constant for all PERK methods, i.e., 
    # regardless of the optimized coefficients.
    a_matrix_constant::Matrix{Float64}
    c::Vector{Float64}
    dt_opt::Union{Float64, Nothing}
end # struct PairedExplicitRK4

# Constructor for previously computed A Coeffs
function PairedExplicitRK4(num_stages, base_path_a_coeffs::AbstractString,
                           dt_opt = nothing;
                           c_const = 1.0f0)
    a_matrix, a_matrix_constant, c = compute_PairedExplicitRK4_butcher_tableau(num_stages,
                                                                               base_path_a_coeffs;
                                                                               c_const)

    return PairedExplicitRK4(num_stages, a_matrix, a_matrix_constant, c, dt_opt)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct PairedExplicitRK4Integrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                           PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKSingleIntegrator
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    tdir::Real
    dt::RealT # current time step
    dtcache::RealT # manually set time step
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F
    alg::Alg # This is our own class written above; Abbreviation for ALGorithm
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # PairedExplicitRK stages:
    k1::uType
    k_higher::uType
end

function init(ode::ODEProblem, alg::PairedExplicitRK4;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    # PairedExplicitRK stages
    k1 = zero(u0)
    k_higher = zero(u0)

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    integrator = PairedExplicitRK4Integrator(u0, du, u_tmp, t0, tdir, dt, dt, iter,
                                             ode.p,
                                             (prob = ode,), ode.f, alg,
                                             PairedExplicitRKOptions(callback,
                                                                     ode.tspan;
                                                                     kwargs...),
                                             false, true, false,
                                             k1, k_higher)

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

@inline function last_three_stages!(integrator, alg, p)
    for stage in 1:2
        @threaded for u_ind in eachindex(integrator.u)
            integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                      alg.a_matrix_constant[stage, 1] *
                                      integrator.k1[u_ind] +
                                      alg.a_matrix_constant[stage, 2] *
                                      integrator.k_higher[u_ind]
        end

        integrator.f(integrator.du, integrator.u_tmp, p,
                     integrator.t +
                     alg.c[alg.num_stages - 3 + stage] * integrator.dt)

        @threaded for u_ind in eachindex(integrator.du)
            integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
        end
    end

    # Last stage
    @threaded for i in eachindex(integrator.du)
        integrator.u_tmp[i] = integrator.u[i] +
                              alg.a_matrix_constant[3, 1] * integrator.k1[i] +
                              alg.a_matrix_constant[3, 2] * integrator.k_higher[i]
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[alg.num_stages] * integrator.dt)

    @threaded for u_ind in eachindex(integrator.u)
        # Note that 'k_higher' carries the values of K_{S-1}
        # and that we construct 'K_S' "in-place" from 'integrator.du'
        integrator.u[u_ind] += 0.5 * (integrator.k_higher[u_ind] +
                                integrator.du[u_ind] * integrator.dt)
    end
end

function step!(integrator::PairedExplicitRK4Integrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    modify_dt_for_tstops!(integrator)

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end ||
       isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
    end

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
        k1!(integrator, prob.p, alg.c)

        # k2
        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[2] * integrator.dt)

        @threaded for i in eachindex(integrator.du)
            integrator.k_higher[i] = integrator.du[i] * integrator.dt
        end

        # Higher stages until "constant" stages
        for stage in 3:(alg.num_stages - 3)
            # Construct current state
            @threaded for i in eachindex(integrator.du)
                integrator.u_tmp[i] = integrator.u[i] +
                                      alg.a_matrix[stage - 2, 1] *
                                      integrator.k1[i] +
                                      alg.a_matrix[stage - 2, 2] *
                                      integrator.k_higher[i]
            end

            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                         integrator.t + alg.c[stage] * integrator.dt)

            @threaded for i in eachindex(integrator.du)
                integrator.k_higher[i] = integrator.du[i] * integrator.dt
            end
        end

        last_three_stages!(integrator, alg, prob.p)
    end # PairedExplicitRK step timer

    integrator.iter += 1
    integrator.t += integrator.dt

    # handle callbacks
    if callbacks isa CallbackSet
        for cb in callbacks.discrete_callbacks
            if cb.condition(integrator.u, integrator.t, integrator)
                cb.affect!(integrator)
            end
        end
    end

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
        @warn "Interrupted. Larger maxiters is needed."
        terminate!(integrator)
    end
end
end # @muladd
