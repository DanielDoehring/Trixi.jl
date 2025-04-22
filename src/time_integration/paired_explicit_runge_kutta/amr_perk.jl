# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Custom implementation for PERK integrator
function (amr_callback::AMRCallback)(integrator::Union{AbstractPairedExplicitRKMultiIntegrator,
                                                       AbstractPairedExplicitRelaxationRKMultiIntegrator};
                                     kwargs...)
    u_ode = integrator.u
    semi = integrator.p

    @trixi_timeit timer() "AMR-PERK" begin
        has_changed = amr_callback(u_ode, semi,
                                   integrator.t, integrator.iter; kwargs...)

        if has_changed
            u_modified!(integrator, true)

            ### PERK additions ###
            @trixi_timeit timer() "PERK stage identifiers update" begin
                mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

                integrator.n_levels = get_n_levels(mesh, integrator.alg)
                n_dims = ndims(mesh) # Spatial dimension

                # TODO: Can you avoid complete re-assignment, i.e., only update/remove changed elements?
                # Re-initialize storage for level-wise information
                if integrator.n_levels != length(integrator.level_info_elements_acc)
                    integrator.level_info_elements = [Vector{Int64}()
                                                      for _ in 1:(integrator.n_levels)]
                    integrator.level_info_elements_acc = [Vector{Int64}()
                                                          for _ in 1:(integrator.n_levels)]

                    integrator.level_info_interfaces_acc = [Vector{Int64}()
                                                            for _ in 1:(integrator.n_levels)]
                    integrator.level_info_mpi_interfaces_acc = [Vector{Int64}()
                                                                for _ in 1:(integrator.n_levels)]

                    integrator.level_info_boundaries_acc = [Vector{Int64}()
                                                            for _ in 1:(integrator.n_levels)]
                    # For efficient treatment of boundaries we need additional datastructures
                    integrator.level_info_boundaries_orientation_acc = [[Vector{Int64}()
                                                                         for _ in 1:(2 * n_dims)]
                    # Need here n_levels, otherwise this is not Vector{Vector{Int64}} but Vector{Vector{Vector{Int64}}
                                                                        for _ in 1:(integrator.n_levels)]
                    integrator.level_info_mortars_acc = [Vector{Int64}()
                                                         for _ in 1:(integrator.n_levels)]
                    integrator.level_info_mpi_mortars_acc = [Vector{Int64}()
                                                             for _ in 1:(integrator.n_levels)]

                    integrator.level_u_indices_elements = [Vector{Int64}()
                                                           for _ in 1:(integrator.n_levels)]
                else # Just empty datastructures
                    for level in 1:(integrator.n_levels)
                        empty!(integrator.level_info_elements[level])
                        empty!(integrator.level_info_elements_acc[level])

                        empty!(integrator.level_info_interfaces_acc[level])
                        empty!(integrator.level_info_mpi_interfaces_acc[level])

                        empty!(integrator.level_info_boundaries_acc[level])
                        for dim in 1:(2 * n_dims)
                            empty!(integrator.level_info_boundaries_orientation_acc[level][dim])
                        end

                        empty!(integrator.level_info_mortars_acc[level])
                        empty!(integrator.level_info_mpi_mortars_acc[level])

                        empty!(integrator.level_u_indices_elements[level])
                    end
                end

                if !mpi_isparallel()
                    partition_variables!(integrator.level_info_elements,
                                         integrator.level_info_elements_acc,
                                         integrator.level_info_interfaces_acc,
                                         integrator.level_info_boundaries_acc,
                                         integrator.level_info_boundaries_orientation_acc,
                                         integrator.level_info_mortars_acc,
                                         integrator.n_levels, n_dims, mesh, dg,
                                         cache, integrator.alg)
                else
                    partition_variables!(integrator.level_info_elements,
                                         integrator.level_info_elements_acc,
                                         integrator.level_info_interfaces_acc,
                                         integrator.level_info_boundaries_acc,
                                         integrator.level_info_boundaries_orientation_acc,
                                         integrator.level_info_mortars_acc,
                                         # MPI additions
                                         integrator.level_info_mpi_interfaces_acc,
                                         integrator.level_info_mpi_mortars_acc,
                                         integrator.n_levels, n_dims, mesh, dg,
                                         cache, integrator.alg)
                end

                partition_u!(integrator.level_u_indices_elements,
                             integrator.level_info_elements,
                             integrator.n_levels,
                             u_ode, mesh, equations, dg, cache)

                # For AMR: Counting RHS evals
                #=
                for level = 1:length(integrator.level_info_elements)
                    integrator.RHSCalls += amr_callback.interval * integrator.alg.stages[level] * 
                                           length(integrator.level_info_elements[level])
                end
                =#

                resize!(integrator, length(u_ode)) # `resize!` integrator after PERK partitioning data structures
            end # "PERK stage identifiers update" timing
        end # if has_changed
    end # "AMR" timing

    return has_changed
end
end # @muladd
