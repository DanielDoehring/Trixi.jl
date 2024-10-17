# Package extension for adding Convex-based features to Trixi.jl
module TrixiConvexClarabelExt

# Required for coefficient optimization in P-ERK scheme integrators
if isdefined(Base, :get_extension)
    using Convex: MOI, solve!, Variable, minimize, evaluate, dot
    using Clarabel: Optimizer
else
    # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
    using ..Convex: MOI, solve!, Variable, minimize, evaluate
    using ..Clarabel: Optimizer
end

# Use functions that are to be extended and additional symbols that are not exported
using Trixi: Trixi, @muladd

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Compute new version of stability polynomial of the embedded scheme for paired explicit Runge-Kutta 
# up to specified consistency order(p = 2), including contributions from free coefficients for higher
# orders, and return the maximum absolute value
function embedded_scheme_stability_polynomials!(pnoms,
                                                num_stages_embedded,
                                                num_stage_evals_embedded,
                                                normalized_powered_eigvals_scaled,
                                                a, b, c)

    # Construct a full b coefficient vector #TODO: is there a way to not do this and just use b directly?
    b_coeff = [
        1 - sum(b),
        zeros(Float64, num_stages_embedded - num_stage_evals_embedded)...,
        b...,
        0
    ]
    num_eig_vals = length(pnoms)

    # Initialize with 1 + z
    for i in 1:num_eig_vals
        pnoms[i] = 1.0 + normalized_powered_eigvals_scaled[i, 1]
    end
    # z^2: b^T * c
    #pnoms += dot(b, c) * normalized_powered_eigvals_scaled[:, 2]
    pnoms += 0.5 * normalized_powered_eigvals_scaled[:, 2]

    # Contribution from free coefficients
    for i in 3:num_stage_evals_embedded
        sum = 0.0
        for j in (i + num_stages_embedded - num_stage_evals_embedded):num_stages_embedded
            prod = 1.0
            for k in (3 + j - i):j
                prod *= a[k]
            end
            sum += prod * b_coeff[j] * c[j - i + 2]
        end
        pnoms += sum * normalized_powered_eigvals_scaled[:, i]
    end

    # For optimization only the maximum is relevant
    return maximum(abs(pnoms))
end

function Trixi.solve_b_butcher_coeffs_unknown(num_eig_vals, eig_vals,
                                              num_stages, num_stage_evals,
                                              num_stages_embedded,
                                              num_stage_evals_embedded,
                                              a_unknown, c, dtmax, dteps)
    dtmin = 0.0
    dt = -1.0
    abs_p = -1.0

    # Construct a full a coefficient vector
    a = zeros(num_stages)
    num_a_unknown = length(a_unknown)

    for i in 1:num_a_unknown
        a[num_stages - i + 1] = a_unknown[num_a_unknown - i + 1]
    end

    # Construct stability polynomial for each eigenvalue
    pnoms = ones(Complex{Float64}, num_eig_vals, 1)

    # There are e - 2 free variables for the stability polynomial of the embedded scheme
    b = Variable(num_stage_evals - 2)

    normalized_powered_eigvals = zeros(Complex{Float64}, num_eig_vals, num_stage_evals-1)

    for j in 1:num_stage_evals-1
        #fac_j = factorial(j)
        for i in 1:num_eig_vals
            #normalized_powered_eigvals[i, j] = eig_vals[i]^j / fac_j
            # Try first without factorial normalization
            normalized_powered_eigvals[i, j] = eig_vals[i]^j
        end
    end

    normalized_powered_eigvals_scaled = similar(normalized_powered_eigvals)

    # Bisection on timestep
    while dtmax - dtmin > dteps
        dt = 0.5 * (dtmax + dtmin)

        # Compute stability polynomial for current timestep
        for k in 1:num_stage_evals-1
            dt_k = dt^k
            for i in 1:num_eig_vals
                normalized_powered_eigvals_scaled[i, k] = dt_k *
                                                          normalized_powered_eigvals[i,
                                                                                     k]
            end
        end

        # Second-order constraint
        # Since c[1] is always 0 we can ignore the contribution of b[1] and only account for the ones from other non-zero entries of b
        constraints = [b >= 0,
            2 * dot(b, c[(num_stages - num_stage_evals + 2):(num_stages - 1)]) == 1.0]

        # Use last optimal values for b in (potentially) next iteration
        problem = minimize(embedded_scheme_stability_polynomials!(pnoms,
                                                                  num_stages_embedded,
                                                                  num_stage_evals_embedded,
                                                                  normalized_powered_eigvals_scaled,
                                                                  a, b, c), constraints)

        #=                                               
        solve!(problem,
               # Parameters taken from default values for EiCOS
               MOI.OptimizerWithAttributes(Optimizer, "b" => 0.99,
                                           "delta" => 2e-7,
                                           "feastol" => 1e-9,
                                           "abstol" => 1e-9,
                                           "reltol" => 1e-9,
                                           "feastol_inacc" => 1e-4,
                                           "abstol_inacc" => 5e-5,
                                           "reltol_inacc" => 5e-5,
                                           "nitref" => 9,
                                           "maxit" => 100,
                                           "verbose" => 3); silent_solver = true)
        =#

        
        solve!(problem,
               MOI.OptimizerWithAttributes(Optimizer,
                                           "tol_feas" => 1e-4);
               silent_solver = true)
        
        print("dt: ", dt, "\n")
        print("Problem value: ", problem.optval, "\n")
        abs_p = problem.optval

        if abs_p < 1
            dtmin = dt
        else
            dtmax = dt
        end
    end

    b_opt = evaluate(b)

    # Catch case S = 3 (only one opt. variable)
    if isa(b_opt, Number)
        b_opt = [b_opt]
    end

    return b_opt, dt
end
end # @muladd

end # module TrixiConvexClarabelExt
