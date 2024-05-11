
using OrdinaryDiffEq
using Trixi
using LinearAlgebra, SparseArrays
using DelimitedFiles, Plots

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
k = 3
solver = DGSEM(polydeg = k, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0
coordinates_max = 1.0

Ref_lvl = 7
N_cells_uni = 2^Ref_lvl
N_cells_coarse = N_cells_uni / 2

mesh_uni = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = Ref_lvl,
                    n_cells_max = 30_000)

refinement_patches = ((type = "box", coordinates_min = (-0.5), coordinates_max = (0.5)),)
mesh_ref = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = Ref_lvl,
                    refinement_patches = refinement_patches,
                    n_cells_max = 30_000)

#=                    
# Semidiscretization for uniform mesh for spectrum generation                    
semi_uni = SemidiscretizationHyperbolic(mesh_uni, equations, initial_condition_convergence_test, solver)

J = jacobian_ad_forward(semi_uni)
Eigenvalues = eigvals(J)

# Complex conjugate eigenvalues have same modulus
Eigenvalues = Eigenvalues[imag(Eigenvalues) .>= 0]

# Sometimes due to numerical issues some eigenvalues have positive real part, which is erronous (for hyperbolic eqs)
Eigenvalues = Eigenvalues[real(Eigenvalues) .< 0]

EigValsReal = real(Eigenvalues)
EigValsImag = imag(Eigenvalues)

scatter(EigValsReal, EigValsImag, label = "Spectrum Uniform")

EigValFile = "EigenvalueList.txt"
ofstream = open(EigValFile, "w")
for i in eachindex(Eigenvalues)
  realstring = string(EigValsReal[i])
  write(ofstream, realstring)

  write(ofstream, "+")

  imstring = string(EigValsImag[i])
  write(ofstream, imstring)
  write(ofstream, "i") # Cpp uses "I" for the imaginary unit
  if i != length(Eigenvalues)
    write(ofstream, "\n")
  end
end
close(ofstream)
=#

semi_ref = SemidiscretizationHyperbolic(mesh_ref, equations, initial_condition_convergence_test, solver)

A_map, _ = linear_structure(semi_ref)
A = Matrix(A_map)
N = size(A, 1)

#=
# Double-check matrix ordering by printing nonzero entries
filtered_A = [row[row .!= 0] for row in eachrow(A)]

open("A_NZ.txt", "w") do io
    for row in filtered_A
        writedlm(io, row', ',')
    end
end
=#

Stages = [16, 10]
path = "/home/daniel/git/Paper_PERK4/PlotScripts/LinearStability/"
NumStagesMax = 16
AMatrices, AMatrix, c, ActiveLevels, _, _ = Trixi.ComputePERK4_Multi_ButcherTableau(Stages, NumStagesMax, path)

ActiveLevels

CFL = 0.99
dt = 0.105336966330572 / 2^(Ref_lvl - 4) * CFL


# Build P-ERK linear operator (matrix)
AFine = copy(A)
I_Fine = Matrix(1.0*I, N, N)

# Outer sections: Set to zero
for i = 1:Int(N_cells_coarse/2 * (k+1))
  AFine[i, :] = zeros(N)
  I_Fine[i, :] = zeros(N)
end

for i = (N - Int(N_cells_coarse/2) * (k+1) + 1):N
  AFine[i, :] = zeros(N)
  I_Fine[i, :] = zeros(N)
end
count(x->x==1.0, I_Fine)

I_Coarse = I - I_Fine
count(x->x==1.0, I_Coarse)

K1 = dt * A
K_higher = copy(K1)
if length(ActiveLevels[2]) == 1
  K_higher = dt * AFine * (I + c[2]*K1)
else
  K_higher = dt * A * (I + c[2]*K1)
end

for stage = 3:(NumStagesMax - 3)
  K_temp = I + AMatrices[stage - 2, 1, 1] * I_Fine * K1 + 
               AMatrices[stage - 2, 2, 1] * I_Fine * K_higher + 
               AMatrices[stage - 2, 1, 2] * I_Coarse * K1 + 
               AMatrices[stage - 2, 2, 2] * I_Coarse * K_higher
                 
  if length(ActiveLevels[stage]) == 1
    K_higher = dt * AFine * K_temp
  else
    K_higher = dt * A * K_temp
  end
end

KS1 = Matrix(undef, N, N)
for stage = 1:3
  K_temp = I + AMatrix[stage, 1] * K1 + 
               AMatrix[stage, 2] * K_higher

  K_higher = dt * A * K_temp

  if stage == 2
    KS1 = K_higher
  end
end

K_Perk = I + 0.5 * (KS1 + K_higher)


K_PERK_EigVals = eigvals(K_Perk)
# Complex conjugate eigenvalues have same modulus
K_PERK_EigVals = K_PERK_EigVals[imag(K_PERK_EigVals) .>= 0]
spectral_radius = maximum(abs.(K_PERK_EigVals))

scatter(real(K_PERK_EigVals), imag(K_PERK_EigVals), label = "Spectrum PERK")
writedlm("K_PERK4_EigVals.txt", K_PERK_EigVals)


###############################################################################
# ODE solvers, callbacks etc.

#t_span = (0.0, 100.0) # For long stability check
t_span = (0.0, dt) # For comparison of implementations
ode = semidiscretize(semi_ref, t_span);

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi_ref, interval = 100)

callbacks = CallbackSet(summary_callback, analysis_callback)

###############################################################################
# run the simulation

ode_algorithm = PERK4_Multi(Stages, path, [42.0, 42.0])

sol = Trixi.solve(ode, ode_algorithm,
                  dt = dt,
                  save_everystep=false, callback=callbacks);

u_Trixi = sol.u[end]
u_PERK = K_Perk * sol.u[1]

norm(u_Trixi - u_PERK, Inf)
norm(u_Trixi - u_PERK, 2)

plot(sol)
plot!(getmesh(PlotData1D(sol)))

# Print the timer summary
summary_callback()