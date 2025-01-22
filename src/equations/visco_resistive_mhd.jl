"""
    struct BoundaryConditionVRMHDWall
"""
struct BoundaryConditionVRMHDWall{V, H, B}
    boundary_condition_velocity::V
    boundary_condition_heat_flux::H
    boundary_condition_magnetic::B
end

"""
    struct Isomagnetic
"""
struct Isomagnetic{F}
    boundary_value_function::F # value of the magnetic field on the boundary
end
