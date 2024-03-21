using LinearAlgebra: eigvals
using ECOS
using Convex
const MOI = Convex.MOI

function read_in_eig_vals(path_to_eval_file)
    # Declare and set to some value
    num_eig_vals = -1
    open(path_to_eval_file, "r") do eval_file
        num_eig_vals = countlines(eval_file)
    end
    eig_vals = Array{Complex{Float64}}(undef, num_eig_vals)
    line_index = 0
    open(path_to_eval_file, "r") do eval_file
        # Read till end of file
        while !eof(eval_file)
            # Read a new / next line for every iteration          
            line_content = readline(eval_file)
            eig_vals[line_index + 1] = parse(Complex{Float64}, line_content)
            line_index += 1
        end
    end

    return num_eig_vals, eig_vals
end

function filter_eigvals(eig_vals, threshold)
    filtered_eigvals_counter = 0
    filtered_eig_vals = Complex{Float64}[]

    for eig_val in eig_vals
        if abs(eig_val) < threshold
            filtered_eigvals_counter += 1
        else
            push!(filtered_eig_vals, eig_val)
        end
    end

    println("$filtered_eigvals_counter eigenvalue(s) are not passed on because they are in magnitude smaller than $threshold \n")

    return length(filtered_eig_vals), filtered_eig_vals
end

function polynoms(cons_order, num_stage_evals, num_eig_vals,
                  normalized_powered_eigvals_scaled, pnoms, gamma::Variable)
    for i in 1:num_eig_vals
        pnoms[i] = 1.0
    end

    for k in 1:cons_order
        for i in 1:num_eig_vals
            pnoms[i] += normalized_powered_eigvals_scaled[i, k]
        end
    end

    for k in (cons_order + 1):num_stage_evals
        pnoms += gamma[k - cons_order] * normalized_powered_eigvals_scaled[:, k]
    end

    return maximum(abs(pnoms))
end

function bisection(cons_order, num_eig_vals, num_stage_evals, dt_max, dt_eps, eig_vals)
    dt_min = 0.0
    dt = -1.0
    abs_p = -1.0

    pnoms = ones(Complex{Float64}, num_eig_vals, 1)

    # Init datastructure for results
    gamma = Variable(num_stage_evals - cons_order)

    normalized_powered_eigvals = zeros(Complex{Float64}, num_eig_vals, num_stage_evals)

    for j in 1:num_stage_evals
        fac_j = factorial(j)
        for i in 1:num_eig_vals
            normalized_powered_eigvals[i, j] = eig_vals[i]^j / fac_j
        end
    end

    normalized_powered_eigvals_scaled = zeros(Complex{Float64}, num_eig_vals,
                                              num_stage_evals)

    println("Start optimization of stability polynomial \n")

    while dt_max - dt_min > dt_eps
        dt = 0.5 * (dt_max + dt_min)

        for k in 1:num_stage_evals
            dt_k = dt^k
            for i in 1:num_eig_vals
                normalized_powered_eigvals_scaled[i, k] = dt_k *
                                                          normalized_powered_eigvals[i, k]
            end
        end

        # Use last optimal values for gamma in (potentially) next iteration
        problem = minimize(polynoms(cons_order, num_stage_evals, num_eig_vals,
                                    normalized_powered_eigvals_scaled, pnoms, gamma))

        Convex.solve!(problem,
                      # Parameters taken from default values for EiCOS
                      MOI.OptimizerWithAttributes(ECOS.Optimizer, "gamma" => 0.99,
                                                  "delta" => 2e-7,
                                                  #"eps" => 1e13, # 1e13
                                                  "feastol" => 1e-9, # 1e-9
                                                  "abstol" => 1e-9, # 1e-9
                                                  "reltol" => 1e-9, # 1e-9
                                                  "feastol_inacc" => 1e-4,
                                                  "abstol_inacc" => 5e-5,
                                                  "reltol_inacc" => 5e-5,
                                                  "nitref" => 9,
                                                  "maxit" => 100,
                                                  "verbose" => 3); silent_solver = true)

        abs_p = problem.optval

        println("MaxAbsP: ", abs_p, "\ndt: ", dt, "\n")

        if abs_p < 1.0
            dt_min = dt
        else
            dt_max = dt
        end
    end

    println("Concluded stability polynomial optimization \n")

    return evaluate(gamma), dt
end

function undo_normalization!(cons_order, num_stage_evals, gamma_opt)
    for k in (cons_order + 1):num_stage_evals
        gamma_opt[k - cons_order] = gamma_opt[k - cons_order] / factorial(k)
    end
    return gamma_opt
end