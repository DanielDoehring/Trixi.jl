# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
Trixi.@muladd begin
#! format: noindent

# Compute the L2 projection matrix for projecting polynomials 
# from a higher degree to a lower degree using Gauss-Legendre quadrature.
#
# Arguments
# - `nodes_high`: GLL/LGL nodes of the higher-degree polynomial
# - `nodes_low`: GLL/LGL nodes of the lower-degree polynomial
# - `::Val{:gauss}`: Use Gauss-Legendre quadrature (accuracy 2N - 1)
# - `RealT`: Type of the output matrix (default: Float64)
#
# Returns
# The projection matrix such that multiplying with it projects 
# a higher degree Lagrange interpolation/solution polynomial
# to a lower degree Lagrange interpolation/solution polynomial.
function polynomial_l2projection_matrix(nodes_high, nodes_low, ::Val{:gauss},
                                        RealT = Float64)
    n_high = length(nodes_high)
    n_low = length(nodes_low)

    # Get Gauss-Legendre nodes and weights for quadrature
    # Use enough nodes to exactly integrate polynomials of degree n_high + n_low - 1
    n_quad = div(n_high + n_low + 1, 2)
    gauss_nodes, gauss_weights = gauss_nodes_weights(n_quad, RealT)

    # Get barycentric weights for interpolation
    wbary_high = barycentric_weights(nodes_high)
    wbary_low = barycentric_weights(nodes_low)

    # Weights for the low-degree mass matrix (diagonal for Gauss-Lobatto)
    weights_low = gauss_lobatto_nodes_weights(n_low, RealT)[2]

    # Build projection matrix
    projection_matrix = zeros(RealT, n_low, n_high)

    for q in 1:n_quad
        # Evaluate low-degree basis functions at Gauss quadrature point
        poly_low = lagrange_interpolating_polynomials(gauss_nodes[q], nodes_low,
                                                      wbary_low)

        # Evaluate high-degree basis functions at Gauss quadrature point
        poly_high = lagrange_interpolating_polynomials(gauss_nodes[q], nodes_high,
                                                       wbary_high)

        for i in 1:n_low, j in 1:n_high
            # Build integral using Gauss quadrature
            projection_matrix[i, j] += poly_low[i] * poly_high[j] * gauss_weights[q] /
                                       weights_low[i]
        end
    end

    return projection_matrix
end

# Need to overload this due to breaking change after restart files have been constructed
function Trixi.varnames(::typeof(cons2cons), ::CompressibleEulerEquations3D)
    return ("rho", "rho_v1", "rho_v2", "rho_v3", "rho_e")
end

function load_restart_file(semi::Trixi.AbstractSemidiscretization,
                                 restart_file, interpolate_high2low)
    load_restart_file(Trixi.mesh_equations_solver_cache(semi)...,
                      restart_file, interpolate_high2low)
end

# Load solution variables from restart file and 
# convert them to a different polynomial degree.
# Version for non-MPI-parallel run
function convert_restart_file_polydeg!(u, file, polydeg_file,
                                       mesh, equations, dg::DGSEM, cache,
                                       nnodes_file, conversion_matrix)
    all_variables = zeros(eltype(u),
                          (nvariables(equations),
                           ntuple(_ -> nnodes_file, ndims(mesh))...,
                           nelements(dg, cache)))
    for v in eachvariable(equations)
        all_variables[v, .., :] = read(file["variables_$v"])
    end

    # NOTE: Transform to reference element currently not needed, as geometry does not change.
    #=
    if mesh isa P4estMesh || mesh isa T8codeMesh || mesh isa UnstructuredMesh2D
        # Reconstruct basis of the solver used in the restart file
        basis_file = LobattoLegendreBasis(polydeg_file)
        # Reconstruct elements of the solver used in the restart file
        elements_file = init_elements(mesh, equations, basis_file, eltype(u))
        inverse_jacobian_file = elements_file.inverse_jacobian

        # Multiply file Jacobian prior to interpolation/projection,
        # i.e., move to reference element
        for element_id in eachelement(dg, cache)
            for v in eachvariable(equations)
                all_variables[v, .., element_id] ./= inverse_jacobian_file[..,
                                                                           element_id]
            end
        end
    end
    =#

    # Perform interpolation/projection to new polynomial degree
    for element in eachelement(dg, cache)
        u[.., element] = multiply_dimensionwise(conversion_matrix,
                                                all_variables[.., element])
    end

    # NOTE: Transform back to physical element currently not needed, as geometry does not change.
    #=
    if mesh isa P4estMesh || mesh isa T8codeMesh || mesh isa UnstructuredMesh2D
        # Apply Jacobian of the new solver to the coefficients
        inverse_jacobian = cache.elements.inverse_jacobian

        # Divide interpolated/projected coefficients by the current inverse Jacobian
        # to move back to physical space
        for element_id in eachelement(dg, cache)
            for v in eachvariable(equations)
                u[v, .., element_id] .*= inverse_jacobian[.., element_id]
            end
        end
    end
    =#

    return nothing
end

function load_restart_file(mesh::Union{Trixi.TreeMeshSerial, StructuredMesh,
                                             UnstructuredMesh2D, Trixi.P4estMeshSerial,
                                             Trixi.T8codeMeshSerial},
                                 equations, dg::DG, cache,
                                 restart_file, interpolate_high2low)

    # allocate memory
    u_ode = Trixi.allocate_coefficients(mesh, equations, dg, cache)
    u = Trixi.wrap_array_native(u_ode, mesh, equations, dg, cache)

    Trixi.h5open(restart_file, "r") do file
        # Read attributes to perform some sanity checks
        if read(Trixi.attributes(file)["ndims"]) != ndims(mesh)
            error("restart mismatch: ndims differs from value in restart file")
        end
        if read(Trixi.attributes(file)["equations"]) != Trixi.get_name(equations)
            error("restart mismatch: equations differ from value in restart file")
        end
        if read(Trixi.attributes(file)["n_elements"]) != nelements(dg, cache)
            error("restart mismatch: number of elements in solver differs from value in restart file")
        end
        for v in eachvariable(equations)
            # Check if variable name matches
            var = file["variables_$v"]
            if (name = read(Trixi.attributes(var)["name"])) !=
               Trixi.varnames(cons2cons, equations)[v]
                error("mismatch: variables_$v should be '$(Trixi.varnames(cons2cons, equations)[v])', but found '$name'")
            end
        end

        ### Read variable data ###
        polydeg_file = read(Trixi.attributes(file)["polydeg"])
        if polydeg_file != Trixi.polydeg(dg) # Conversion is necessary
            nnodes_file = polydeg_file + 1
            nodes_file = Trixi.gauss_lobatto_nodes_weights(nnodes_file)[1]

            nodes_solver = Trixi.gauss_lobatto_nodes_weights(nnodes(dg))[1]

            if (polydeg_file < Trixi.polydeg(dg)) || interpolate_high2low # Interpolation from lower to higher
                conversion_matrix = polynomial_interpolation_matrix(nodes_file,
                                                                    nodes_solver)
            else # Projection from higher to lower
                conversion_matrix = polynomial_l2projection_matrix(nodes_file,
                                                                   nodes_solver,
                                                                   Val(:gauss))
            end
            convert_restart_file_polydeg!(u, file, polydeg_file,
                                          mesh, equations, dg, cache,
                                          nnodes_file, conversion_matrix)
        else # Read in variables separately
            for v in eachvariable(equations)
                u[v, .., :] = read(file["variables_$v"])
            end
        end
    end

    return u_ode
end

"""
    semidiscretize(semi::AbstractSemidiscretization, tspan,
                   restart_file::AbstractString;
                   interpolate_high2low = true,
                   jac_prototype::Union{AbstractMatrix, Nothing} = nothing,
                   colorvec::Union{AbstractVector, Nothing} = nothing)

Wrap the semidiscretization `semi` as an ODE problem in the time interval `tspan`
that can be passed to `solve` from the [SciML ecosystem](https://diffeq.sciml.ai/latest/).

The initial condition etc. is taken from the `restart_file`.

Optional keyword arguments:
- `interpolate_high2low` applies only to the case when a simulation is restarted with a 
  lower polynomial degree than the one used in the original simulation.
  In that case, the solution is either interpolated (default) from the higher-degree polynomial
  to the lower-degree polynomial.
  This preserves the values at the cell interfaces, thus a formerly continuous solution
  is still continuous after the interpolation.
  For `interpolate_high2low = false`, the solution is projected with minimal L2-error onto the
  lower-degree polynomial.
  This results in overall smaller L2-errors, but does not preserve continuity at the cell interfaces.
- `jac_prototype`: Expected to come from [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl).
  Specifies the sparsity structure of the Jacobian to enable e.g. efficient implicit time stepping.
- `colorvec`: Expected to come from [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl).
  Allows for even faster Jacobian computation. Not necessarily required when `jac_prototype` is given.
"""
function Trixi.semidiscretize(semi::Trixi.AbstractSemidiscretization, tspan,
                              restart_file::AbstractString;
                              jac_prototype::Union{AbstractMatrix, Nothing} = nothing,
                              colorvec::Union{AbstractVector, Nothing} = nothing,
                              interpolate_high2low = true,
                              reset_threads = true)
    # Optionally reset Polyester.jl threads. See
    # https://github.com/trixi-framework/Trixi.jl/issues/1583
    # https://github.com/JuliaSIMD/Polyester.jl/issues/30
    if reset_threads
        Trixi.Polyester.reset_threads!()
    end

    # Load initial condition from restart file
    u0_ode = load_restart_file(semi, restart_file, interpolate_high2low)

    # TODO: MPI, do we want to synchronize loading and print debug statements, e.g. using
    #       mpi_isparallel() && MPI.Barrier(mpi_comm())
    #       See https://github.com/trixi-framework/Trixi.jl/issues/328
    iip = true # is-inplace, i.e., we modify a vector when calling rhs!
    specialize = Trixi.SciMLBase.FullSpecialize # specialize on rhs! and parameters (semi)

    # Check if Jacobian prototype is provided for sparse Jacobian
    if jac_prototype !== nothing
        # Convert `jac_prototype` to real type, as seen here:
        # https://docs.sciml.ai/DiffEqDocs/stable/tutorials/advanced_ode_example/#Declaring-a-Sparse-Jacobian-with-Automatic-Sparsity-Detection
        ode = Trixi.SciMLBase.ODEFunction(rhs!,
                                    jac_prototype = convert.(eltype(u0_ode),
                                                             jac_prototype),
                                    colorvec = colorvec) # coloring vector is optional

        return Trixi.ODEProblem{iip, specialize}(ode, u0_ode, tspan, semi)
    else
        # We could also construct an `ODEFunction` explicitly without the Jacobian here,
        # but we stick to the lean direct in-place function `rhs!` and
        # let OrdinaryDiffEq.jl handle the rest
        return Trixi.ODEProblem{iip, specialize}(Trixi.rhs!, u0_ode, tspan, semi)
    end
end

end # @muladd
