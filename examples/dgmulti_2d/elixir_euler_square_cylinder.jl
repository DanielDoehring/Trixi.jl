using OrdinaryDiffEqLowStorageRK
using Trixi
using StartUpDG

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_constant

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

polydeg = 3
basis = DGMultiBasis(Tri(), polydeg, approximation_type = SBP())

dg = DGMulti(basis,
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

mesh_file = Trixi.download("https://raw.githubusercontent.com/jlchan/StartUpDG.jl/refs/heads/main/test/testset_Gmsh_meshes/squareCylinder2D.msh",
                           joinpath(@__DIR__, "squareCylinder2D.mesh"))
VXY, EToV = read_Gmsh_2D(mesh_file)
md = MeshData(VXY, EToV, basis)

boundary_names = [:Wall, :Inflow, :Outflow]
#boundary_names = ["Wall", "Inflow", "Outflow"]
#boundary_names = ["1", "2", "3"]
mesh = DGMultiMesh(md, boundary_names)

bc_dir = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; Wall = bc_dir,
                       Inflow = bc_dir,
                       Outflow = bc_dir)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 0.15)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(); abstol = 1.0e-6, reltol = 1.0e-6,
            ode_default_options()..., callback = callbacks);
