using MPI

mpiexec() do cmd
    
    run(`$cmd -n 2 $(Base.julia_cmd()) --threads=1 --project=@. -e 
    'include("examples/tree_2d_dgsem/elixir_euler_vortex_PERK4.jl")'`)
    
    #=
    run(`$cmd -n 2 $(Base.julia_cmd()) --threads=1 --project=@. -e 
    'include("examples/tree_2d_dgsem/elixir_euleracoustics_co-rotating_vortex_pair_PERK4.jl")'`)
    =#
end