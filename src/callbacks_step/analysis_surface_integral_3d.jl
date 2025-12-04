# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LiftCoefficientPressure3D(aoa, rho_inf, u_inf, a_inf)

Compute the lift coefficient
```math
C_{L,p} \coloneqq \frac{\oint_{\partial \Omega} p \boldsymbol n \cdot \psi_L \, \mathrm{d} S}
                        {0.5 \rho_{\infty} U_{\infty}^2 A_{\infty}}
```
based on the pressure distribution along a boundary.
In 3D, the freestream-normal unit vector ``\psi_L`` is given by
```math
\psi_L \coloneqq \begin{pmatrix} -\sin(\alpha) \\ \cos(\alpha) \\ 0 \end{pmatrix}
```
where ``\alpha`` is the angle of attack.
This employs the convention that the wing is oriented such that the streamwise flow is in 
x-direction, the angle of attack rotates the flow into the y-direction, and that wing extends spanwise in the z-direction.

Supposed to be used in conjunction with [`AnalysisSurfaceIntegral`](@ref)
which stores the the to-be-computed variables (for instance `LiftCoefficientPressure3D`) 
and boundary information.

- `aoa::Real`: Angle of attack in radians (for airfoils etc.)
- `rho_inf::Real`: Free-stream density
- `u_inf::Real`: Free-stream velocity
- `a_inf::Real`: Reference area of geometry (e.g. projected wing surface)
"""
function LiftCoefficientPressure3D(aoa, rho_inf, u_inf, a_inf)
    # `psi_lift` is the normal unit vector to the freestream direction.
    # Note: The choice of the normal vector `psi_lift = (-sin(aoa), cos(aoa), 0)`
    # leads to positive lift coefficients for positive angles of attack for airfoils.
    # One could also use `psi_lift = (sin(aoa), -cos(aoa), 0)` which results in the same
    # value, but with the opposite sign.
    psi_lift = (-sin(aoa), cos(aoa), zero(aoa))
    return LiftCoefficientPressure(ForceState(psi_lift, rho_inf, u_inf, a_inf))
end

@doc raw"""
    DragCoefficientPressure3D(aoa, rho_inf, u_inf, a_inf)

Compute the drag coefficient
```math
C_{D,p} \coloneqq \frac{\oint_{\partial \Omega} p \boldsymbol n \cdot \psi_D \, \mathrm{d} S}
                        {0.5 \rho_{\infty} U_{\infty}^2 A_{\infty}}
```
based on the pressure distribution along a boundary.
In 3D, the freestream-tangent unit vector ``\psi_D`` is given by
```math
\psi_D \coloneqq \begin{pmatrix} \cos(\alpha) \\ \sin(\alpha) \\ 0 \end{pmatrix}
```
where ``\alpha`` is the angle of attack.
This employs the convention that the wing is oriented such that the streamwise flow is in 
x-direction, the angle of attack rotates the flow into the y-direction, and that wing extends spanwise in the z-direction.

Supposed to be used in conjunction with [`AnalysisSurfaceIntegral`](@ref)
which stores the the to-be-computed variables (for instance `DragCoefficientPressure3D`) 
and boundary information.

- `aoa::Real`: Angle of attack in radians (for airfoils etc.)
- `rho_inf::Real`: Free-stream density
- `u_inf::Real`: Free-stream velocity
- `a_inf::Real`: Reference area of geometry (e.g. projected wing surface)
"""
function DragCoefficientPressure3D(aoa, rho_inf, u_inf, a_inf)
    # `psi_drag` is the unit vector tangent to the freestream direction
    psi_drag = (cos(aoa), sin(aoa), zero(aoa))
    return DragCoefficientPressure(ForceState(psi_drag, rho_inf, u_inf, a_inf))
end

@doc raw"""
    DragCoefficientShearStress3D(aoa, rho_inf, u_inf, l_inf)

Compute the drag coefficient
```math
C_{D,f} \coloneqq \frac{\oint_{\partial \Omega} \boldsymbol \tau_w \cdot \psi_D \, \mathrm{d} S}
                        {0.5 \rho_{\infty} U_{\infty}^2 L_{\infty}}
```
based on the wall shear stress vector ``\tau_w`` along a boundary.
In 3D, the freestream-tangent unit vector ``\psi_D`` is given by
```math
\psi_D \coloneqq \begin{pmatrix} \cos(\alpha) \\ \sin(\alpha) \\ 0 \end{pmatrix}
```
where ``\alpha`` is the angle of attack.
Supposed to be used in conjunction with [`AnalysisSurfaceIntegral`](@ref)
which stores the the to-be-computed variables (for instance `DragCoefficientShearStress3D`) 
and boundary information.

- `aoa::Real`: Angle of attack in radians (for airfoils etc.)
- `rho_inf::Real`: Free-stream density
- `u_inf::Real`: Free-stream velocity
- `l_inf::Real`: Reference length of geometry (e.g. airfoil chord length)
"""
function DragCoefficientShearStress3D(aoa, rho_inf, u_inf, l_inf)
    # `psi_drag` is the unit vector tangent to the freestream direction
    psi_drag = (cos(aoa), sin(aoa), zero(aoa))
    return DragCoefficientShearStress(ForceState(psi_drag, rho_inf, u_inf, l_inf))
end

# Compute the three components of the 2D symmetric viscous stress tensor
# (tau_11, tau_12, tau_22) based on the gradients of the velocity field.
# This is required for drag and lift coefficients based on shear stress,
# as well as for the non-integrated quantities such as
# skin friction coefficient (to be added).
function viscous_stress_tensor(u, equations_parabolic,
                               gradients_1, gradients_2, gradients_3)
    _, dv1dx, dv2dx, dv3dx, _ = convert_derivative_to_primitive(u, gradients_1,
                                                                equations_parabolic)
    _, dv1dy, dv2dy, dv3dy, _ = convert_derivative_to_primitive(u, gradients_2,
                                                                equations_parabolic)
    _, dv1dz, dv2dz, dv3dz, _ = convert_derivative_to_primitive(u, gradients_3,
                                                                equations_parabolic)

    # Components of viscous stress tensor

    # Diagonal parts
    # (4 * (v1)_x / 3 - 2 * ((v2)_y + (v3)_z)) / 3)
    tau_11 = (4 * dv1dx - 2 * (dv2dy + dv3dz)) / 3
    # (4 * (v2)_y / 3 - 2 * ((v1)_x + (v3)_z) / 3)
    tau_22 = (4 * dv2dy - 2 * (dv1dx + dv3dz)) / 3
    # (4 * (v3)_z / 3 - 2 * ((v1)_x + (v2)_y) / 3)
    tau_33 = (4 * dv3dz - 2 * (dv1dx + dv2dy)) / 3

    # Off diagonal parts, exploit that stress tensor is symmetric
    # ((v1)_y + (v2)_x)
    tau_12 = dv1dy + dv2dx # = tau_21
    # ((v1)_z + (v3)_x)
    tau_13 = dv1dz + dv3dx # = tau_31
    # ((v2)_z + (v3)_y)
    tau_23 = dv2dz + dv3dy # = tau_32

    mu = dynamic_viscosity(u, equations_parabolic)

    return mu .* (tau_11, tau_12, tau_13,
                  tau_22, tau_23,
                  tau_33)
end

# 2D viscous stress vector based on contracting the viscous stress tensor
# with the normalized `normal_direction` vector.
function viscous_stress_vector(u, normal_direction, equations_parabolic,
                               gradients_1, gradients_2, gradients_3)
    #  Normalize normal direction, should point *into* the fluid => *(-1)
    n_normal = -normal_direction / norm(normal_direction)

    tau_11, tau_12, tau_13,
    tau_22, tau_23,
    tau_33 = viscous_stress_tensor(u, equations_parabolic,
                                   gradients_1, gradients_2, gradients_3)

    # Viscous stress vector: Stress tensor * normal vector
    viscous_stress_vector_1 = tau_11 * n_normal[1] +
                              tau_12 * n_normal[2] +
                              tau_13 * n_normal[3]

    viscous_stress_vector_2 = tau_12 * n_normal[1] +
                              tau_22 * n_normal[2] +
                              tau_23 * n_normal[3]

    viscous_stress_vector_3 = tau_13 * n_normal[1] +
                              tau_23 * n_normal[2] +
                              tau_33 * n_normal[3]

    return (viscous_stress_vector_1, viscous_stress_vector_2, viscous_stress_vector_3)
end

function (drag_coefficient::DragCoefficientShearStress{RealT, 3})(u, normal_direction,
                                                                  x, t,
                                                                  equations_parabolic,
                                                                  gradients_1,
                                                                  gradients_2,
                                                                  gradients_3) where {RealT <:
                                                                                      Real}
    visc_stress_vector = viscous_stress_vector(u, normal_direction, equations_parabolic,
                                               gradients_1, gradients_2, gradients_3)
    @unpack psi, rho_inf, u_inf, l_inf = drag_coefficient.force_state
    return (visc_stress_vector[1] * psi[1] +
            visc_stress_vector[2] * psi[2] +
            visc_stress_vector[3] * psi[3]) /
           (0.5f0 * rho_inf * u_inf^2 * l_inf)
end

# 3D version of the `analyze` function for `AnalysisSurfaceIntegral`, i.e., 
# `LiftCoefficientPressure` and `DragCoefficientPressure`.
function analyze(surface_variable::AnalysisSurfaceIntegral, du, u, t,
                 mesh::P4estMesh{3},
                 equations, dg::DGSEM, cache, semi)
    @unpack boundaries = cache
    @unpack node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis

    @unpack variable, boundary_symbols = surface_variable
    @unpack boundary_symbol_indices = semi.boundary_conditions
    boundary_indices = get_boundary_indices(boundary_symbols, boundary_symbol_indices)

    # Restore boundary values for parabolic equations
    # which overwrite the solution boundary values with the gradients
    if semi isa SemidiscretizationHyperbolicParabolic
        prolong2boundaries!(cache, u, mesh, equations, dg)
    end

    surface_integral = zero(eltype(u))
    index_range = eachnode(dg)
    for boundary in boundary_indices
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_3d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_3d(node_indices[2], index_range)
        k_node_start, k_node_step = index_to_start_step_3d(node_indices[3], index_range)

        # In 3D, boundaries are surfaces => `node_index1`, `node_index2`
        for node_index1 in index_range
            # Reset node indices
            i_node = i_node_start
            j_node = j_node_start
            k_node = k_node_start
            for node_index2 in index_range
                u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg,
                                             node_index1, node_index2, boundary)
                # Extract normal direction at nodes which points from the elements outwards,
                # i.e., *into* the structure.
                normal_direction = get_normal_direction(direction,
                                                        contravariant_vectors,
                                                        i_node, j_node, k_node, element)

                # Coordinates at a boundary node
                x = get_node_coords(node_coordinates, equations, dg,
                                    i_node, j_node, k_node, element)

                # L2 norm of normal direction (contravariant_vector) is the surface element
                dS = weights[node_index1] * weights[node_index2] *
                     norm(normal_direction)

                # Integral over entire boundary surface. Note, it is assumed that the
                # `normal_direction` is normalized to be a normal vector within the
                # function `variable` and the division of the normal scaling factor
                # `norm(normal_direction)` is then accounted for with the `dS` quantity.
                surface_integral += variable(u_node, normal_direction, x, t,
                                             equations) * dS

                i_node += i_node_step
                j_node += j_node_step
                k_node += k_node_step
            end
        end
    end
    return surface_integral
end

# 3D version of the `analyze` function for `AnalysisSurfaceIntegral` of viscous, i.e.,
# variables that require gradients of the solution variables.
# These are for parabolic equations readily available.
# Examples is `DragCoefficientShearStress`.
function analyze(surface_variable::AnalysisSurfaceIntegral{Variable}, du, u, t,
                 mesh::P4estMesh{3},
                 equations, equations_parabolic,
                 dg::DGSEM, cache, semi,
                 cache_parabolic) where {Variable <: VariableViscous}
    @unpack boundaries = cache
    @unpack node_coordinates, contravariant_vectors = cache.elements
    @unpack weights = dg.basis

    @unpack variable, boundary_symbols = surface_variable
    @unpack boundary_symbol_indices = semi.boundary_conditions
    boundary_indices = get_boundary_indices(boundary_symbols, boundary_symbol_indices)

    # Restore boundary values for parabolic equations
    # which overwrite the solution boundary values with the gradients
    prolong2boundaries!(cache, u, mesh, equations, dg)

    # Additions for parabolic
    @unpack viscous_container = cache_parabolic
    @unpack gradients = viscous_container

    gradients_x, gradients_y, gradients_z = gradients

    surface_integral = zero(eltype(u))
    index_range = eachnode(dg)
    for boundary in boundary_indices
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)
        k_node_start, k_node_step = index_to_start_step_2d(node_indices[3], index_range)

        i_node = i_node_start
        j_node = j_node_start
        k_node = k_node_start

        # In 3D, boundaries are surfaces => `node_index1`, `node_index2`
        for node_index1 in index_range
            # Reset node indices
            i_node = i_node_start
            j_node = j_node_start
            k_node = k_node_start
            for node_index2 in index_range
                u_node = Trixi.get_node_vars(cache.boundaries.u, equations, dg,
                                             node_index1, node_index2, boundary)
                # Extract normal direction at nodes which points from the elements outwards,
                # i.e., *into* the structure.
                normal_direction = get_normal_direction(direction,
                                                        contravariant_vectors,
                                                        i_node, j_node, k_node, element)

                # Coordinates at a boundary node
                x = get_node_coords(node_coordinates, equations, dg,
                                    i_node, j_node, k_node, element)

                # L2 norm of normal direction (contravariant_vector) is the surface element
                dS = weights[node_index1] * weights[node_index2] *
                     norm(normal_direction)

                gradients_1 = get_node_vars(gradients_x, equations_parabolic, dg,
                                            i_node, j_node, k_node, element)
                gradients_2 = get_node_vars(gradients_y, equations_parabolic, dg,
                                            i_node, j_node, k_node, element)
                gradients_3 = get_node_vars(gradients_z, equations_parabolic, dg,
                                            i_node, j_node, k_node, element)

                # Integral over whole boundary surface. Note, it is assumed that the
                # `normal_direction` is normalized to be a normal vector within the
                # function `variable` and the division of the normal scaling factor
                # `norm(normal_direction)` is then accounted for with the `dS` quantity.
                surface_integral += variable(u_node, normal_direction, x, t,
                                             equations_parabolic,
                                             gradients_1, gradients_2, gradients_3) * dS

                i_node += i_node_step
                j_node += j_node_step
                k_node += k_node_step
            end
        end
    end
    return surface_integral
end
end # muladd
