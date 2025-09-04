# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Version with DIFFERENT number of stages for hyperbolic and parabolic part
struct PairedExplicitRelaxationRK4SplitMulti{RelaxationSolver} <:
       AbstractPairedExplicitRKSplitMulti{4}
    PERK4SplitMulti::PairedExplicitRK4SplitMulti
    relaxation_solver::RelaxationSolver
end

function PairedExplicitRelaxationRK4SplitMulti(stages::Vector{Int64},
                                               stages_para::Vector{Int64},
                                               base_path_mon_coeffs::AbstractString,
                                               base_path_mon_coeffs_para::AbstractString,
                                               dt_ratios, dt_ratios_para;
                                               cS3 = 1.0f0,
                                               relaxation_solver = RelaxationSolverNewton())
    return PairedExplicitRelaxationRK4SplitMulti{typeof(relaxation_solver)}(PairedExplicitRK4SplitMulti(stages,
                                                                                                        stages_para,
                                                                                                        base_path_mon_coeffs,
                                                                                                        base_path_mon_coeffs_para,
                                                                                                        dt_ratios,
                                                                                                        dt_ratios_para;
                                                                                                        cS3 = cS3),
                                                                            relaxation_solver)
end

# Version with DIFFERENT number of stages and partitioning(!) for hyperbolic and parabolic part
mutable struct PairedExplicitRelaxationRK4SplitMultiIntegrator{RealT <: Real, uType <: AbstractVector,
                                                               Params, Sol, F,
                                                               PairedExplicitRKOptions,
                                                               RelaxationSolver} <:
               AbstractPairedExplicitRelaxationRKSplitMultiIntegrator{4}
    u::uType
    du::uType # In-place output of `f`
    u_tmp::uType # Used for building the argument to `f`
    t::RealT
    tdir::RealT # DIRection of time integration, i.e., if one marches forward or backward in time
    dt::RealT # current time step
    dtcache::RealT # Used for euler-acoustic coupling
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F # `rhs!` of the semidiscretization
    alg::PairedExplicitRK4SplitMulti
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # Additional PERK register
    k1::uType
    # For split (hyperbolic-parabolic) problems
    du_para::uType # Stores the parabolic part of the overall rhs!
    k1_para::uType # Additional PERK register for the parabolic part

    # Variables managing level-dependent integration
    level_info_elements::Vector{Vector{Int64}}
    level_info_elements_acc::Vector{Vector{Int64}}

    level_info_interfaces_acc::Vector{Vector{Int64}}
    level_info_mpi_interfaces_acc::Vector{Vector{Int64}}

    level_info_boundaries_acc::Vector{Vector{Int64}}

    level_info_mortars_acc::Vector{Vector{Int64}}
    level_info_mpi_mortars_acc::Vector{Vector{Int64}}

    level_info_u::Vector{Vector{Int64}}

    coarsest_lvl::Int64
    n_levels::Int64

    # For parabolic part
    level_info_elements_para::Vector{Vector{Int64}}
    level_info_elements_para_acc::Vector{Vector{Int64}}

    level_info_interfaces_para_acc::Vector{Vector{Int64}}
    level_info_mpi_interfaces_para_acc::Vector{Vector{Int64}}

    level_info_boundaries_para_acc::Vector{Vector{Int64}}

    level_info_mortars_para_acc::Vector{Vector{Int64}}
    level_info_mpi_mortars_para_acc::Vector{Vector{Int64}}

    level_info_u_para::Vector{Vector{Int64}}

    coarsest_lvl_para::Int64
    n_levels_para::Int64

    # Entropy Relaxation additions
    gamma::RealT # Relaxation parameter
    S_old::RealT # Entropy of previous iterate
    relaxation_solver::RelaxationSolver
end

function init(ode::ODEProblem, alg::PairedExplicitRelaxationRK4SplitMulti;
              dt, callback = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    k1 = zero(u0) # Additional PERK register
    du_para = zero(u0) # Stores the parabolic part of the overall rhs!
    k1_para = zero(u0) # Additional PERK register for the parabolic part

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    ### Set datastructures for handling of level-dependent integration ###
    semi = ode.p
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

    n_levels, n_levels_para = get_n_levels(mesh, alg.PERK4SplitMulti)

    level_info_elements = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

    # MPI additions
    level_info_mpi_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]
    level_info_mpi_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

    # For parabolic part
    level_info_elements_para = [Vector{Int64}() for _ in 1:n_levels_para]
    level_info_elements_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]

    level_info_interfaces_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]

    level_info_boundaries_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]

    level_info_mortars_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]

    level_info_mpi_interfaces_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]
    level_info_mpi_mortars_para_acc = [Vector{Int64}() for _ in 1:n_levels_para]

    # For entropy relaxation
    gamma = one(eltype(u0))
    u_wrap = wrap_array(u0, semi)
    S_old = integrate(entropy, u_wrap, mesh, equations, dg, cache)

    if !mpi_isparallel()
        partition_variables!(level_info_elements,
                             level_info_elements_acc,
                             level_info_interfaces_acc,
                             level_info_boundaries_acc,
                             level_info_mortars_acc,
                             n_levels, semi,
                             alg.PERK4SplitMulti)

        # Partition parabolic helper variables
        partition_variables!(level_info_elements_para,
                             level_info_elements_para_acc,
                             level_info_interfaces_para_acc,
                             level_info_boundaries_para_acc,
                             level_info_mortars_para_acc,
                             n_levels_para, semi,
                             alg.PERK4SplitMulti;
                             quadratic_scaling = true)
    else
        if mesh isa ParallelP4estMesh
            # Get cell distribution for standard partitioning
            global_first_quadrant = unsafe_wrap(Array,
                                                unsafe_load(mesh.p4est).global_first_quadrant,
                                                mpi_nranks() + 1)
            # Need to copy `global_first_quadrant` to different variable as the former will change 
            # due to the call to `partition!`
            old_global_first_quadrant = copy(global_first_quadrant)

            # Get (global) element distribution to accordingly balance the solver
            partition_variables!(level_info_elements, n_levels,
                                 semi, alg.PERK4SplitMulti)

            # Balance such that each rank has the same number of RHS calls                                    
            balance_p4est_perk!(mesh, dg, cache, level_info_elements,
                                alg.PERK4SplitMulti.stages)
            # Actual move of elements across ranks
            rebalance_solver!(u0, mesh, equations, dg, cache, old_global_first_quadrant)
            reinitialize_boundaries!(semi.boundary_conditions, cache) # Needs to be called after `rebalance_solver!`

            # Reset `level_info_elements` after rebalancing
            level_info_elements = [Vector{Int64}() for _ in 1:n_levels]

            # Resize ODE vectors
            n_new = length(u0)
            resize!(du, n_new)
            resize!(u_tmp, n_new)
            resize!(k1, n_new)
        end
        partition_variables!(level_info_elements,
                             level_info_elements_acc,
                             level_info_interfaces_acc,
                             level_info_boundaries_acc,
                             level_info_mortars_acc,
                             # MPI additions
                             level_info_mpi_interfaces_acc,
                             level_info_mpi_mortars_acc,
                             n_levels, semi,
                             alg.PERK4SplitMulti)
    end

    for i in 1:n_levels
        println("Number Elements integrated with level $i: ",
                length(level_info_elements[i]))
    end
    for i in 1:n_levels_para
        println("Number Elements integrated with level $i (parabolic): ",
                length(level_info_elements_para[i]))
    end

    # Set (initial) distribution of DG nodal values
    level_info_u = [Vector{Int64}() for _ in 1:n_levels]
    partition_u!(level_info_u, level_info_elements,
                 n_levels, u0, semi)

    # For parabolic part
    level_info_u_para = [Vector{Int64}() for _ in 1:n_levels_para]
    partition_u!(level_info_u_para, level_info_elements_para,
                 n_levels_para, u0, semi)

    ### Done with setting up for handling of level-dependent integration ###
    du_para = zero(u0)
    integrator = PairedExplicitRelaxationRK4SplitMultiIntegrator(u0, du, u_tmp,
                                                                 t0, tdir,
                                                                 dt, zero(dt),
                                                                 iter, semi,
                                                                 (prob = ode,),
                                                                 ode.f,
                                                                 alg.PERK4SplitMulti,
                                                                 PairedExplicitRKOptions(callback,
                                                                                         ode.tspan;
                                                                                         kwargs...),
                                                                 false, true, false,
                                                                 k1, du_para, k1_para,
                                                                 level_info_elements,
                                                                 level_info_elements_acc,
                                                                 level_info_interfaces_acc,
                                                                 level_info_mpi_interfaces_acc,
                                                                 level_info_boundaries_acc,
                                                                 level_info_mortars_acc,
                                                                 level_info_mpi_mortars_acc,
                                                                 level_info_u,
                                                                 -1, n_levels,
                                                                 level_info_elements_para,
                                                                 level_info_elements_para_acc,
                                                                 level_info_interfaces_para_acc,
                                                                 level_info_mpi_interfaces_para_acc,
                                                                 level_info_boundaries_para_acc,
                                                                 level_info_mortars_acc,
                                                                 level_info_mpi_mortars_para_acc,
                                                                 level_info_u_para,
                                                                 -1, n_levels_para,
                                                                 gamma, S_old,
                                                                 alg.relaxation_solver)

    initialize_callbacks!(callback, integrator)

    return integrator
end

@inline function PERK4_kS2_to_kS!(integrator::AbstractPairedExplicitRelaxationRKSplitIntegrator{4},
                                  p, alg)
    mesh, equations, dg, cache = mesh_equations_solver_cache(p)

    for stage in 1:2
        a1_dt = alg.a_matrix_constant[1, stage] * integrator.dt
        a2_dt = alg.a_matrix_constant[2, stage] * integrator.dt
        @threaded for i in eachindex(integrator.u)
            integrator.u_tmp[i] = integrator.u[i] +
                                  a1_dt * integrator.k1[i] + # `k1` contains already parabolic part
                                  a2_dt * (integrator.du[i] + integrator.du_para[i])
        end

        # Hyperbolic part
        integrator.f.f2(integrator.du, integrator.u_tmp, p,
                        integrator.t +
                        alg.c[alg.num_stages - 3 + stage] * integrator.dt)

        # Parabolic part
        integrator.f.f1(integrator.du_para, integrator.u_tmp, p,
                        integrator.t +
                        alg.c[alg.num_stages - 3 + stage] * integrator.dt)
    end

    # Accumulate hyperbolic and parabolic contributions into `du` for entropy computation
    @threaded for i in eachindex(integrator.du)
        integrator.du[i] = integrator.du[i] + integrator.du_para[i] # Faster than broadcasted version (with .=)
    end

    du_wrap = wrap_array(integrator.du, p)
    u_tmp_wrap = wrap_array(integrator.u_tmp, p)
    # Entropy change due to S-1 stage
    b_dt = 0.5 * integrator.dt # 0.5 = b_{S-1}
    dS = b_dt *
         integrate_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

    # Last stage
    a1_dt = alg.a_matrix_constant[1, 3] * integrator.dt
    a2_dt = alg.a_matrix_constant[2, 3] * integrator.dt
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              a1_dt * integrator.k1[i] +
                              a2_dt * integrator.du[i]
    end

    @threaded for i in eachindex(integrator.u)
        # Store K_{S-1} in `k1`
        integrator.k1[i] = integrator.du[i] # Already added hyperbolic and parabolic part
    end

    # Hyperbolic part
    integrator.f.f2(integrator.du, integrator.u_tmp, p,
                    integrator.t + alg.c[alg.num_stages] * integrator.dt)

    # Parabolic part
    integrator.f.f1(integrator.du_para, integrator.u_tmp, p,
                    integrator.t + alg.c[alg.num_stages] * integrator.dt)

    # Accumulate hyperbolic and parabolic contributions into `du` for entropy computation
    @threaded for i in eachindex(integrator.du)
        integrator.du[i] = integrator.du[i] + integrator.du_para[i] # Faster than broadcasted version (with .=)
    end

    # Entropy change due to last (i = S) stage
    dS += b_dt * # 0.5 = b_{S}
          integrate_w_dot_stage(du_wrap, u_tmp_wrap, mesh, equations, dg, cache)

    # Note: We re-use `du` for the "direction"
    # Note: For efficiency, we multiply the direction with dt already here!
    @threaded for i in eachindex(integrator.u)
        integrator.du[i] = b_dt * (integrator.k1[i] + integrator.du[i])
    end

    u_wrap = wrap_array(integrator.u, integrator.p)
    @trixi_timeit timer() "Relaxation solver" relaxation_solver!(integrator,
                                                                 u_tmp_wrap, u_wrap,
                                                                 du_wrap, dS,
                                                                 mesh, equations,
                                                                 dg, cache,
                                                                 integrator.relaxation_solver)

    integrator.iter += 1
    update_t_relaxation!(integrator)

    # Do relaxed update
    @threaded for i in eachindex(integrator.u)
        # Note: We re-use `du` for the "direction"
        #integrator.u[i] += integrator.gamma * integrator.du[i]
        # Try optimize for `@muladd`: avoid `+=`
        integrator.u[i] = integrator.u[i] + integrator.gamma * integrator.du[i]
    end

    return nothing
end

function step!(integrator::AbstractPairedExplicitRelaxationRKSplitIntegrator{4})
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    #modify_dt_for_tstops!(integrator)

    limit_dt!(integrator, t_end)

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
        PERK_k1!(integrator, prob.p)
        PERK_k2!(integrator, prob.p, alg)

        # Higher stages until shared stages with constant Butcher tableau
        for stage in 3:(alg.num_stages - 3)
            PERK_ki!(integrator, prob.p, alg, stage)
        end

        # Accumulate hyperbolic and parabolic contributions into `k1` for 
        # last three stages with shared Butcher coefficients
        @threaded for i in eachindex(integrator.k1)
            # Try to optimize for `@muladd`: avoid `+=`
            integrator.k1[i] = integrator.k1[i] + integrator.k1_para[i]
        end

        PERK4_kS2_to_kS!(integrator, prob.p, alg)
    end

    @trixi_timeit timer() "Step-Callbacks" handle_callbacks!(callbacks, integrator)

    check_max_iter!(integrator)

    return nothing
end
end # @muladd
