module TestExamples2DEulerMultiion

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_2d_dgsem")

@testset "Compressible Euler Multi-Ion" begin
#! format: noindent

@trixi_testset "elixir_eulermultiion_collisions.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermultiion_collisions.jl"),
                        l2=[
                            1.260701577954499e-17, 0.0595534208484378,
                            1.2870890904597232e-20, 0.019718485574500753,
                            7.144325530681256e-17, 0.05955342084843781,
                            1.263367389039714e-19, 0.017385070243529387
                        ],
                        linf=[
                            2.7755575615628914e-17, 0.059553420848437816,
                            3.95684742991817e-20, 0.019718485574500757,
                            2.220446049250313e-16, 0.05955342084843784,
                            3.8872432701939665e-19, 0.017385070243529394
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
