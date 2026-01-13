module TestExamples1DEulerMultiion

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Compressible Euler Multi-Ion" begin
#! format: noindent

@trixi_testset "elixir_eulermultiion_collisions.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermultiion_collisions.jl"),
                        l2=[
                            9.741166526712686e-18, 0.05955342093372054,
                            0.01971848560074372,
                            4.130462730494001e-17, 0.05955342093372052,
                            0.01738507026976905
                        ],
                        linf=[
                            1.3877787807814457e-17, 0.05955342093372055,
                            0.019718485600743726,
                            1.1102230246251565e-16, 0.059553420933720534,
                            0.017385070269769053
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end
end

end # module
