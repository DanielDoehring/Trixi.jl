# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
using DelimitedFiles: readdlm

@muladd begin
#! format: noindent

# Initialize Butcher array abscissae c for PairedExplicitRK3 based on SSPRK33 base method
function compute_c_coeff_SSP33(num_stages, c_s2)
    c = zeros(num_stages)

    # Last timesteps as for SSPRK33
    c[num_stages] = 0.5
    c[num_stages - 1] = 1

    # Linear increasing timestep for remainder
    for i in 2:(num_stages - 2)
        c[i] = c_s2 * (i - 1) / (num_stages - 3)
    end

    return c
end

function PairedExplicitRK3_butcher_tableau_objective_function(a_unknown, num_stages,
                                                              num_stage_evals,
                                                              monomial_coeffs, c_s2)
    c_ts = compute_c_coeff_SSP33(num_stages, c_s2) # ts = timestep

    # Equality Constraint array that ensures that the stability polynomial computed from 
    # the to-be-constructed Butcher-Tableau matches the monomial coefficients of the 
    # optimized stability polynomial.
    c_eq = zeros(num_stage_evals - 2) # Add equality constraint that c_s2 is equal to 1
    # Both terms should be present
    for i in 1:(num_stage_evals - 4)
        term1 = a_unknown[num_stage_evals - 1]
        term2 = a_unknown[num_stage_evals]
        for j in 1:i
            term1 *= a_unknown[num_stage_evals - 1 - j]
            term2 *= a_unknown[num_stage_evals - j]
        end
        term1 *= c_ts[num_stages - 2 - i] * 1 / 6
        term2 *= c_ts[num_stages - 1 - i] * 4 / 6

        c_eq[i] = monomial_coeffs[i] - (term1 + term2)
    end

    # Highest coefficient: Only one term present
    i = num_stage_evals - 3
    term2 = a_unknown[num_stage_evals]
    for j in 1:i
        term2 *= a_unknown[num_stage_evals - j]
    end
    term2 *= c_ts[num_stages - 1 - i] * 4 / 6

    c_eq[i] = monomial_coeffs[i] - term2
    c_eq[num_stage_evals - 2] = 1.0 - 4 * a_unknown[num_stage_evals] -
                                a_unknown[num_stage_evals - 1]

    return c_eq
end

function compute_PairedExplicitRK3_butcher_tableau(num_stages, tspan,
                                                   eig_vals::Vector{ComplexF64};
                                                   verbose = false, c_s2)
    # Initialize array of c
    c = compute_c_coeff_SSP33(num_stages, c_s2)

    # Initialize the array of our solution
    a_unknown = zeros(num_stages)

    # Special case of e = 3
    if num_stages == 3
        a_unknown = [0, c[2], 0.25]
    else
        # Calculate coefficients of the stability polynomial in monomial form
        consistency_order = 3
        dtmax = tspan[2] - tspan[1]
        dteps = 1e-9

        num_eig_vals, eig_vals = filter_eig_vals(eig_vals; verbose)

        monomial_coeffs, dt_opt = bisect_stability_polynomial(consistency_order,
                                                              num_eig_vals, num_stages,
                                                              dtmax,
                                                              dteps,
                                                              eig_vals; verbose)
        monomial_coeffs = undo_normalization!(monomial_coeffs, consistency_order,
                                              num_stages)

        # Call function that use the library NLsolve to solve the nonlinear system
        a_unknown = solve_a_unknown(num_stages, monomial_coeffs, c_s2, c; verbose)
    end

    # For debugging purpose
    println("a_unknown")
    println(a_unknown[3:end])

    a_matrix = zeros(num_stages - 2, 2)
    a_matrix[:, 1] = c[3:end]
    a_matrix[:, 1] -= a_unknown[3:end]
    a_matrix[:, 2] = a_unknown[3:end]

    return a_matrix, c, dt_opt
end

function compute_PairedExplicitRK3_butcher_tableau(num_stages,
                                                   base_path_monomial_coeffs::AbstractString,
                                                   c_s2)

    # Initialize array of c
    c = compute_c_coeff_SSP33(num_stages, c_s2)

    # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    coeffs_max = num_stages - 2

    a_matrix = zeros(coeffs_max, 2)
    a_matrix[:, 1] = c[3:end]

    # TODO: update this to work with CI mktempdir()
    path_monomial_coeffs = base_path_monomial_coeffs * "a_" * string(num_stages) * "_" *
                           string(num_stages) * ".txt"
    @assert isfile(path_monomial_coeffs) "Couldn't find file"
    A = readdlm(path_monomial_coeffs, Float64)
    num_monomial_coeffs = size(A, 1)

    @assert num_monomial_coeffs == coeffs_max
    a_matrix[:, 1] -= A
    a_matrix[:, 2] = A

    println("A matrix: ")
    display(a_matrix)
    println()

    return a_matrix, c
end

"""
    PairedExplicitRK3()
The following structures and methods provide a implementation of
the third-order paired explicit Runge-Kutta method
optimized for a certain simulation setup (PDE, IC & BC, Riemann Solver, DG Solver).
The original paper is
- Nasab, Vermeire (2022)
Third-order Paired Explicit Runge-Kutta schemes for stiff systems of equations
[DOI: 10.1016/j.jcp.2022.111470](https://doi.org/10.1016/j.jcp.2022.111470)
While the changes to SSPRK33 base-scheme are described in 
- Doehring, Schlottke-Lakemper, Gassner, Torrilhon (2024)
Multirate Time-Integration based on Dynamic ODE Partitioning through Adaptively Refined Meshes for Compressible Fluid Dynamics
[Arxiv: 10.48550/arXiv.2403.05144](https://doi.org/10.48550/arXiv.2403.05144)
"""
mutable struct PairedExplicitRK3 <: AbstractPairedExplicitRKSingle
    const num_stages::Int

    a_matrix::Matrix{Float64}
    c::Vector{Float64}
    dt_opt::Float64
end # struct PairedExplicitRK3

# Constructor for previously computed A Coeffs
function PairedExplicitRK3(num_stages, base_path_monomial_coeffs::AbstractString,
                           dt_opt;
                           c_s2 = 1.0)
    a_matrix, c = compute_PairedExplicitRK3_butcher_tableau(num_stages,
                                                            base_path_monomial_coeffs;
                                                            c_s2)

    return PairedExplicitRK3(num_stages, a_matrix, c, dt_opt)
end

# Constructor that computes Butcher matrix A coefficients from a semidiscretization
function PairedExplicitRK3(num_stages, tspan, semi::AbstractSemidiscretization;
                           verbose = false, c_s2 = 1.0)
    eig_vals = eigvals(jacobian_ad_forward(semi))

    return PairedExplicitRK3(num_stages, tspan, eig_vals; verbose, c_s2)
end

# Constructor that calculates the coefficients with polynomial optimizer from a list of eigenvalues
function PairedExplicitRK3(num_stages, tspan, eig_vals::Vector{ComplexF64};
                           verbose = false, c_s2 = 1.0)
    a_matrix, c, dt_opt = compute_PairedExplicitRK3_butcher_tableau(num_stages,
                                                                    tspan,
                                                                    eig_vals;
                                                                    verbose, c_s2)
    return PairedExplicitRK3(num_stages, a_matrix, c, dt_opt)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRK3Integrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                           PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKSingleIntegrator
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F
    alg::Alg # This is our own class written above; Abbreviation for ALGorithm
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    # PairedExplicitRK stages:
    k1::uType
    k_higher::uType
    k_s1::uType # Required for custom third order version of PairedExplicitRK3
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::PairedExplicitRK3;
               dt, callback = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    # PairedExplicitRK stages
    k1 = zero(u0)
    k_higher = zero(u0)
    k_s1 = zero(u0)

    t0 = first(ode.tspan)
    iter = 0

    integrator = PairedExplicitRK3Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter,
                                             ode.p,
                                             (prob = ode,), ode.f, alg,
                                             PairedExplicitRKOptions(callback,
                                                                     ode.tspan;
                                                                     kwargs...),
                                             false,
                                             k1, k_higher, k_s1)

    # initialize callbacks
    if callback isa CallbackSet
        for cb in callback.continuous_callbacks
            error("unsupported")
        end
        for cb in callback.discrete_callbacks
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    elseif !isnothing(callback)
        error("unsupported")
    end

    solve!(integrator)
end

function solve!(integrator::PairedExplicitRK3Integrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    integrator.finalstep = false

    #@trixi_timeit timer() "main loop" while !integrator.finalstep
    while !integrator.finalstep
        if isnan(integrator.dt)
            error("time step size `dt` is NaN")
        end

        # if the next iteration would push the simulation beyond the end time, set dt accordingly
        if integrator.t + integrator.dt > t_end ||
           isapprox(integrator.t + integrator.dt, t_end)
            integrator.dt = t_end - integrator.t
            terminate!(integrator)
        end

        @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
            # k1
            integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
            @threaded for i in eachindex(integrator.du)
                integrator.k1[i] = integrator.du[i] * integrator.dt
            end

            # Construct current state
            @threaded for i in eachindex(integrator.du)
                integrator.u_tmp[i] = integrator.u[i] + alg.c[2] * integrator.k1[i]
            end
            # k2
            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                         integrator.t + alg.c[2] * integrator.dt)

            @threaded for i in eachindex(integrator.du)
                integrator.k_higher[i] = integrator.du[i] * integrator.dt
            end

            if alg.num_stages == 3
                @threaded for i in eachindex(integrator.du)
                    integrator.k_s1[i] = integrator.k_higher[i]
                end
            end

            # Higher stages
            for stage in 3:(alg.num_stages)
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

                # IDEA: Stop for loop at num_stages -1 to avoid if (maybe more performant?)
                if stage == alg.num_stages - 1
                    @threaded for i in eachindex(integrator.du)
                        integrator.k_s1[i] = integrator.k_higher[i]
                    end
                end
            end

            @threaded for i in eachindex(integrator.u)
                # "Own" PairedExplicitRK based on SSPRK33
                integrator.u[i] += (integrator.k1[i] + integrator.k_s1[i] +
                                    4.0 * integrator.k_higher[i]) / 6.0
            end
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
    end # "main loop" timer

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PairedExplicitRK3Integrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)
    resize!(integrator.k_s1, new_size)
end
end # @muladd
