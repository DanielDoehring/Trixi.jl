using HOHQMesh, GLMakie

# Create a new HOHQMesh model project.
cylinder_flow = newProject("Cylinder_symmetric", "out")

# Reset polynomial order of the mesh model curves and output format.
# The "ABAQUS" mesh file format is needed for the adaptive mesh
# capability of Trixi.jl.
setPolynomialOrder!(cylinder_flow, 3)
setMeshFileFormat!(cylinder_flow, "ABAQUS")

# A background grid is required for the mesh generation. In this example we lay a
# background grid of Cartesian boxes with size 2.0
base_size = 2.0
addBackgroundGrid!(cylinder_flow, [base_size, base_size, 0.0])

# Create all the outer boundary curves and add them to the mesh project.
# Note: (1) Curve names are those that will be present in the mesh file
#       (2) Boundaries named "symmetry" are those over which reflection occurs

x_min = -5.0
x_max = 60.0
y_min = 0.0
y_max = 10.0

radius = 0.5

left = newEndPointsLineCurve("Left", [x_min, y_max, 0.0], [x_min, y_min, 0.0])

bottom1 = newEndPointsLineCurve(":symmetry", [x_min, y_min, 0.0], [-radius, y_min, 0.0])

cylinder = newCircularArcCurve("Circle",        # curve name
                               [0.0, y_min, 0.0], # circle center
                               radius,             # circle radius
                               180.0,           # start angle
                               0.0,             # end angle
                               "degrees")       # angle units

bottom2 = newEndPointsLineCurve(":symmetry", [radius, y_min, 0.0], [x_max, y_min, 0.0])

right = newEndPointsLineCurve("Right", [x_max, y_min, 0.0], [x_max, y_max, 0.0])

top = newEndPointsLineCurve("Top", [x_max, y_max, 0.0], [x_min, y_max, 0.0])

# Outer boundary curve chain is created to have counter-clockwise
# orientation, as required by HOHQMesh generator
addCurveToOuterBoundary!(cylinder_flow, left)
addCurveToOuterBoundary!(cylinder_flow, bottom1)
addCurveToOuterBoundary!(cylinder_flow, cylinder)
addCurveToOuterBoundary!(cylinder_flow, bottom2)
addCurveToOuterBoundary!(cylinder_flow, right)
addCurveToOuterBoundary!(cylinder_flow, top)

# Add a refinement line for the wake region.
wake_region_mesh_size = 0.5
wake_region_width = 4.0 # 5.0
ref_wake_region = newRefinementLine("wake_region", "smooth",
                                    [0.0, 0.0, 0.0], [40.0, 0.0, 0.0],
                                    wake_region_mesh_size, wake_region_width)

addRefinementRegion!(cylinder_flow, ref_wake_region)

# Smaller refinement for farfield wake region
wake_region_far_mesh_size = 1.0
wake_region_far_width = 2.5
ref_wake_region_far = newRefinementLine("wake_region", "smooth",
                                        [40.0, 0.0, 0.0], [x_max, 0.0, 0.0],
                                        wake_region_far_mesh_size, wake_region_far_width)

addRefinementRegion!(cylinder_flow, ref_wake_region_far)

# Add refinement around cylinder
cylinder_region_mesh_size = 0.25
cylinder_region_radius = 3.0 # 4.0
refine_circle = newRefinementCenter("cylinder", "smooth", [0.0, 0.0, 0.0],
                                    cylinder_region_mesh_size, cylinder_region_radius)

addRefinementRegion!(cylinder_flow, refine_circle)

# Visualize the model, refinement region and background grid
# prior to meshing.
plotProject!(cylinder_flow, MODEL + REFINEMENTS + GRID)

# Generate the mesh. Saves the mesh file to the directory "out".
generate_mesh(cylinder_flow)
