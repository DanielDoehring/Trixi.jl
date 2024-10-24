# Package extension for adding NLsolve-based features to Trixi.jl
module TrixiNLsolveExt

# Required for finding coefficients in Butcher tableau in the third order of 
# P-ERK scheme integrators
if isdefined(Base, :get_extension)
    using NLsolve: nlsolve
else
    # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
    using ..NLsolve: nlsolve
end

# Use other necessary libraries for the dot product in finding b_unknown
using LinearAlgebra: dot

# We use a random initialization of the nonlinear solver.
# To make the tests reproducible, we need to seed the RNG
using StableRNGs: StableRNG, rand

# Use functions that are to be extended and additional symbols that are not exported
using Trixi: Trixi, compute_c_coeffs, @muladd

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Compute residuals for nonlinear equations to match a stability polynomial with given coefficients,
# in order to find A-matrix in the Butcher-Tableau
function PairedExplicitRK3_butcher_tableau_objective_function!(c_eq, a_unknown,
                                                               num_stages,
                                                               num_stage_evals,
                                                               monomial_coeffs,
                                                               cS2)
    c_ts = compute_c_coeffs(num_stages, cS2) # ts = timestep
    # For explicit methods, a_{1,1} = 0 and a_{2,1} = c_2 (Butcher's condition)
    a_coeff = [0, c_ts[2], a_unknown...]
    # Equality constraint array that ensures that the stability polynomial computed from 
    # the to-be-constructed Butcher-Tableau matches the monomial coefficients of the 
    # optimized stability polynomial.
    # For details, see Chapter 4.3, Proposition 3.2, Equation (3.3) from 
    # Hairer, Wanner: Solving Ordinary Differential Equations 2
    # DOI: 10.1007/978-3-662-09947-6

    # Lower-order terms: Two summands present
    for i in 1:(num_stage_evals - 4)
        term1 = a_coeff[num_stage_evals - 1]
        term2 = a_coeff[num_stage_evals]
        for j in 1:i
            term1 *= a_coeff[num_stage_evals - 1 - j]
            term2 *= a_coeff[num_stage_evals - j]
        end
        term1 *= c_ts[num_stages - 2 - i] * 1 / 6 # 1 / 6 = b_{S-1}
        term2 *= c_ts[num_stages - 1 - i] * 2 / 3 # 2 / 3 = b_S

        c_eq[i] = monomial_coeffs[i] - (term1 + term2)
    end

    # Highest coefficient: Only one term present
    i = num_stage_evals - 3
    term2 = a_coeff[num_stage_evals]
    for j in 1:i
        term2 *= a_coeff[num_stage_evals - j]
    end
    term2 *= c_ts[num_stages - 1 - i] * 2 / 3 # 2 / 3 = b_S

    c_eq[i] = monomial_coeffs[i] - term2
    # Third-order consistency condition (Cf. eq. (27) from https://doi.org/10.1016/j.jcp.2022.111470
    c_eq[num_stage_evals - 2] = 1 - 4 * a_coeff[num_stage_evals] -
                                a_coeff[num_stage_evals - 1]
end

function EmbeddedPairedExplicitRK3_butcher_tableau_objective_function!(b_eq, x,
                                                                       num_stages,
                                                                       num_stage_evals,
                                                                       embedded_monomial_coeffs,
                                                                       c, a)

    # Construct a full b coefficient vector
    b_coeff = [
        1 - sum(x),
        zeros(Float64, num_stages - num_stage_evals)...,
        x...,
        0
    ]

    println("Length of b_coeff")
    println(length(b_coeff)) # Debugging

    b_eq_count = 0

    #TODO: check the logic of this loop cos b is getting absurdly large
    for i in 3:num_stage_evals-1
        sum = 0.0
        fac_i = factorial(i)
        for j in (i + num_stages - num_stage_evals):num_stages-1
            prod = 1.0
            for k in (3 + j - i):j
                prod *= a[k]
            end
            sum += prod * b_coeff[j] * c[j - i + 2] 
        end
        b_eq[i-2] = embedded_monomial_coeffs[i-2] - sum * fac_i
        b_eq_count += 1
    end

    # TODO: there is a more efficient way to compute the dot product for sure...
    b_eq[num_stage_evals-2] = 0.5 - dot(b_coeff, c)
    b_eq_count += 1

    println("b_eq_count: ", b_eq_count) # Debugging
end

# Find the values of the a_{i, i-1} in the Butcher tableau matrix A by solving a system of
# non-linear equations that arise from the relation of the stability polynomial to the Butcher tableau.
# For details, see Proposition 3.2, Equation (3.3) from 
# Hairer, Wanner: Solving Ordinary Differential Equations 2
function Trixi.solve_a_butcher_coeffs_unknown!(a_unknown, num_stages, num_stage_evals,
                                               monomial_coeffs,
                                               c_s2, c;
                                               verbose, max_iter = 100000)

    # Define the objective_function
    function objective_function!(c_eq, x)
        return PairedExplicitRK3_butcher_tableau_objective_function!(c_eq, x,
                                                                     num_stages,
                                                                     num_stage_evals,
                                                                     monomial_coeffs,
                                                                     c_s2)
    end

    # RealT is determined as the type of the first element in monomial_coeffs to ensure type consistency
    RealT = typeof(monomial_coeffs[1])

    # To ensure consistency and reproducibility of results across runs, we use 
    # a seeded random initial guess.
    rng = StableRNG(555)

    # Flag for criteria going beyond satisfaction of non-linear equations
    is_sol_valid = false

    for _ in 1:max_iter
        # Due to the nature of the nonlinear solver, different initial guesses can lead to 
        # small numerical differences in the solution.

        x0 = convert(RealT, 0.1) .* rand(rng, RealT, num_stage_evals - 2)

        sol = nlsolve(objective_function!, x0, method = :trust_region,
                      ftol = 4.0e-16, # Enforce objective up to machine precision
                      iterations = 10^4, xtol = 1.0e-13, autodiff = :forward)

        a_unknown = sol.zero # Retrieve solution (root = zero)

        # Check if the values a[i, i-1] >= 0.0 (which stem from the nonlinear solver) 
        # and also c[i] - a[i, i-1] >= 0.0 since all coefficients should be non-negative
        # to avoid downwinding of numerical fluxes.
        is_sol_valid = all(x -> !isnan(x) && x >= 0, a_unknown) &&
                       all(!isnan(c[i] -
                                  a_unknown[i - num_stages + num_stage_evals - 2]) &&
                           c[i] - a_unknown[i - num_stages + num_stage_evals - 2] >= 0
                           for i in eachindex(c)
                           if i > num_stages - num_stage_evals + 2)

        if is_sol_valid
            return a_unknown
        else
            if verbose
                println("Solution invalid. Restart the process of solving non-linear system of equations again.")
            end
        end
    end

    error("Maximum number of iterations ($max_iter) reached. Cannot find valid sets of coefficients.")
end

function Trixi.solve_b_butcher_coeffs_unknown!(b_unknown, num_stages, num_stage_evals, embedded_monomial_coeffs, c, a_unknown;
    verbose, max_iter = 100000)

    verbose = true

    # Construct a full a coefficient vector
    a = zeros(num_stages)
    num_a_unknown = length(a_unknown)

    for i in 1:num_a_unknown
        a[num_stages - i + 1] = a_unknown[num_a_unknown - i + 1]
    end

    # Define the objective_function
    function embedded_scheme_objective_function!(b_eq, x)
        return EmbeddedPairedExplicitRK3_butcher_tableau_objective_function!(b_eq, x,
                                                                     num_stages,
                                                                     num_stage_evals,
                                                                     embedded_monomial_coeffs,
                                                                     c, a)
    end

    # RealT is determined as the type of the first element in monomial_coeffs to ensure type consistency
    RealT = typeof(embedded_monomial_coeffs[1])

    # To ensure consistency and reproducibility of results across runs, we use 
    # a seeded random initial guess.
    rng = StableRNG(55555)

    # Due to the nature of the nonlinear solver, different initial guesses can lead to 
    # small numerical differences in the solution.

    for _ in 1:max_iter

        # There is e-2 free variables of b of the embedded scheme
        x0 = convert(RealT, 0.1) .* rand(rng, RealT, num_stage_evals - 2)

        sol = nlsolve(embedded_scheme_objective_function!, x0, method = :trust_region,
                    ftol = 4.0e-16, # Enforce objective up to machine precision
                    iterations = 10^4, xtol = 1.0e-13, autodiff = :forward)

        b_unknown = sol.zero # Retrieve solution (root = zero)

        is_sol_valid = all(x -> !isnan(x) && x >= 0, b_unknown) && (sum(b_unknown) <= 1.0)

        return b_unknown

    end
end
end # @muladd

end # module TrixiNLsolveExt
