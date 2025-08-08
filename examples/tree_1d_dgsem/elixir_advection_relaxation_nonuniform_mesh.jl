using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

k = 3 # polynomial degree

# Entropy-conservative flux:
#num_flux = flux_central

# Diffusive fluxes
num_flux = flux_godunov
#num_flux = flux_lax_friedrichs

solver = DGSEM(polydeg = k, surface_flux = num_flux)

coordinates_min = -4.0
coordinates_max = 4.0
length = coordinates_max - coordinates_min

refinement_patches = ((type = "box", coordinates_min = (-1.0,),
                       coordinates_max = (1.0,)),)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                refinement_patches = refinement_patches,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_gauss,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

t_end = 1.0
t_end = length + 1
ode = semidiscretize(semi, (0.0, t_end))

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 1,
                                     extra_analysis_errors = (:conservation_error,),
                                     extra_analysis_integrals = (Trixi.entropy_math,),
                                     analysis_filename = "entropy_ER.dat",
                                     #analysis_filename = "entropy_standard.dat",
                                     save_analysis = true)
cfl = 3.5 # [16, 8]
#cfl = 5.5 # [32, 16]

# Employed only for finding the roughly stable timestep
stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        #analysis_callback
                        #stepsize_callback
                        )

###############################################################################
# run the simulation

path = "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/"
#path = "/home/daniel/git/MA/EigenspectraGeneration/1D_Adv/Joint/"

dtRatios = [1, 0.5]
Stages = [16, 8]

relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 10, root_tol = 1e-14, gamma_tol = 1e-15)

ode_alg = Trixi.PairedExplicitRelaxationRK2Multi(Stages, path, dtRatios;
                                                 relaxation_solver = relaxation_solver)

#ode_alg = Trixi.PairedExplicitRK2Multi(Stages, path, dtRatios)

#ode_alg = Trixi.PairedExplicitRelaxationRK2(16, path; relaxation_solver = relaxation_solver)
#ode_alg = Trixi.PairedExplicitRK2(16, path)

dt = 0.2
sol = Trixi.solve(ode, ode_alg,
                  dt = dt,
                  save_everystep = false, callback = callbacks);

###############################################################################

# Compute node values for scatter plot in physical domain

using LinearAlgebra

function gauss_lobatto_nodes(cell_min, cell_max)
    # Gauss-Lobatto nodes in reference coordinates [-1, 1]
    ref_nodes = [-1, -1 / sqrt(5), 1 / sqrt(5), 1]

    # Map reference nodes to physical coordinates
    cell_center = (cell_max + cell_min) / 2
    cell_half_width = (cell_max - cell_min) / 2
    physical_nodes = cell_center .+ cell_half_width .* ref_nodes

    return physical_nodes
end

# For plotting of the overlapping DG nodes
function gauss_lobatto_nodes_inward(cell_min, cell_max)
    inward = 0.2
    ref_nodes = [-1 + inward, -1 / sqrt(5), 1 / sqrt(5), 1 - inward]

    # Map reference nodes to physical coordinates
    cell_center = (cell_max + cell_min) / 2
    cell_half_width = (cell_max - cell_min) / 2
    physical_nodes = cell_center .+ cell_half_width .* ref_nodes

    return physical_nodes
end

function compute_all_nodes()
    all_nodes = []
    cell_min = -4.0

    # First 6 cells of size 0.5
    for i in 1:6
        cell_max = cell_min + 0.5
        #nodes = gauss_lobatto_nodes(cell_min, cell_max)
        nodes = gauss_lobatto_nodes_inward(cell_min, cell_max)
        append!(all_nodes, nodes)
        cell_min = cell_max
    end

    # Next 8 cells of size 0.25
    for i in 1:8
        cell_max = cell_min + 0.25
        #nodes = gauss_lobatto_nodes(cell_min, cell_max)
        nodes = gauss_lobatto_nodes_inward(cell_min, cell_max)
        append!(all_nodes, nodes)
        cell_min = cell_max
    end

    # Last 6 cells of size 0.5
    for i in 1:6
        cell_max = cell_min + 0.5
        #nodes = gauss_lobatto_nodes(cell_min, cell_max)
        nodes = gauss_lobatto_nodes_inward(cell_min, cell_max)
        append!(all_nodes, nodes)
        cell_min = cell_max
    end

    return all_nodes
end

# Compute nodes for all cells
all_nodes = compute_all_nodes()

function gauss_lobatto_nodes_inward(cell_min, cell_max)
    inward = 0.2
    ref_nodes = [-1 + inward, -1 / sqrt(5), 1 / sqrt(5), 1 - inward]

    # Map reference nodes to physical coordinates
    cell_center = (cell_max + cell_min) / 2
    cell_half_width = (cell_max - cell_min) / 2
    physical_nodes = cell_center .+ cell_half_width .* ref_nodes

    return physical_nodes
end

function compute_all_nodes()
    all_nodes = []
    cell_min = -4.0

    # First 6 cells of size 0.5
    for i in 1:6
        cell_max = cell_min + 0.5
        nodes = gauss_lobatto_nodes(cell_min, cell_max)
        #nodes = gauss_lobatto_nodes_inward(cell_min, cell_max)

        append!(all_nodes, nodes)
        cell_min = cell_max
    end

    # Next 8 cells of size 0.25
    for i in 1:8
        cell_max = cell_min + 0.25
        nodes = gauss_lobatto_nodes(cell_min, cell_max)
        #nodes = gauss_lobatto_nodes_inward(cell_min, cell_max)
        
        append!(all_nodes, nodes)
        cell_min = cell_max
    end

    # Last 6 cells of size 0.5
    for i in 1:6
        cell_max = cell_min + 0.5
        nodes = gauss_lobatto_nodes(cell_min, cell_max)
        #nodes = gauss_lobatto_nodes_inward(cell_min, cell_max)

        append!(all_nodes, nodes)
        cell_min = cell_max
    end

    return all_nodes
end

# Compute nodes for all cells
all_nodes = compute_all_nodes()

using DelimitedFiles
#writedlm("Inward_GLL_nodes.txt", all_nodes)
writedlm("GLL_nodes.txt", all_nodes)

#writedlm("u_Relaxation.txt", sol.u[end])
#writedlm("u_Standard.txt", sol.u[end])

minimum(sol.u[end]) # Check for negative values/positivity violation

###############################################################################

# Compute fully discrete matrices
using LinearAlgebra

A_map, _ = linear_structure(semi)
A = Matrix(A_map)
N = size(A, 1)

Stages = [16, 8]
NumStagesMax = 16
b1 = 0
bS = 1
cS = 0.5
AMatrices, c, ActiveLevels, _, _ = Trixi.compute_PairedExplicitRK2Multi_butcher_tableau(Stages,
                                                                                        NumStagesMax,
                                                                                        path,
                                                                                        bS,
                                                                                        cS)

# Build P-ERK linear operator (matrix)
AFine = copy(A)
I_Fine = Matrix(1.0 * I, N, N)

N_cells_coarse = 12

# Outer sections: Set to zero
for i in 1:Int(N_cells_coarse / 2 * (k + 1))
    AFine[i, :] = zeros(N)
    I_Fine[i, :] = zeros(N)
end

for i in (N - Int(N_cells_coarse / 2) * (k + 1) + 1):N
    AFine[i, :] = zeros(N)
    I_Fine[i, :] = zeros(N)
end
count(x -> x == 1.0, I_Fine)

I_Coarse = I - I_Fine
count(x -> x == 1.0, I_Coarse)

K1 = dt * A
K_higher = copy(K1)
if ActiveLevels[2] == 1
    global K_higher = dt * AFine * (I + c[2] * K1)
else
    global K_higher = dt * A * (I + c[2] * K1)
end

for i in 3:NumStagesMax
    K_temp = I + AMatrices[1, 1, i - 2] * I_Fine * K1 +
             AMatrices[1, 2, i - 2] * I_Fine * K_higher +
             AMatrices[2, 1, i - 2] * I_Coarse * K1 +
             AMatrices[2, 2, i - 2] * I_Coarse * K_higher

    if ActiveLevels[i] == 1
        K_higher = dt * AFine * K_temp
    else
        K_higher = dt * A * K_temp
    end
end
K_Perk = I + b1 * K1 + bS * K_higher

row_sums = sum(K_Perk, dims = 2) # sanity check, should all be 1

# Check for linear stability
K_PERK_EigVals = eigvals(K_Perk)
# Complex conjugate eigenvalues have same modulus
K_PERK_EigVals = K_PERK_EigVals[imag(K_PERK_EigVals) .>= 0]
writedlm("K_PERK_EigVals.txt", K_PERK_EigVals)
spectral_radius = maximum(abs.(K_PERK_EigVals))

minimum(K_Perk)
maximum(K_Perk)

# First steps' relaxation parameter gamma
gamma = 0.7721910021820937
K_Perk_Relaxation = I + gamma * (b1 * K1 + bS * K_higher)

row_sums = sum(K_Perk_Relaxation, dims = 2) # sanity check, should all be 1

# Check for linear stability
K_PERK_EigVals = eigvals(K_Perk_Relaxation)
# Complex conjugate eigenvalues have same modulus
K_PERK_EigVals = K_PERK_EigVals[imag(K_PERK_EigVals) .>= 0]
writedlm("K_PERK_Relaxation_1_EigVals.txt", K_PERK_EigVals)
spectral_radius = maximum(abs.(K_PERK_EigVals))

minimum(K_Perk_Relaxation)
maximum(K_Perk_Relaxation)
