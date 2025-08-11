using Trixi

# Functionality for automatic sparsity detection
using SparseDiffTools, Symbolics

###############################################################################################
### Overloads to construct the `LobattoLegendreBasis` with `Real` type (supertype of `Num`) ###

# Required for setting up the Lobatto-Legendre basis for abstract `Real` type.
# Constructing the Lobatto-Legendre basis with `Real` instead of `Num` is 
# significantly easier as we do not have to care about e.g. if-clauses.
# As a consequence, we need to provide some overloads hinting towards the intended behavior.

const float_type = Float64 # Actual floating point type for the simulation

# Newton tolerance for finding LGL nodes & weights
Trixi.eps(::Type{Real}) = Base.eps(float_type)
# There are some places where `one(RealT)` or `zero(uEltype)` is called where `RealT` or `uEltype` is `Real`.
# This returns an `Int64`, i.e., `1` or `0`, respectively which gives errors when a floating-point alike type is expected.
Trixi.one(::Type{Real}) = Base.one(float_type)
Trixi.zero(::Type{Real}) = Base.zero(float_type)

module RealMatMulOverload

# Multiplying two Matrix{Real}s gives a Matrix{Any}.
# This causes problems when instantiating the Legendre basis, which calls
# `calc_{forward,reverse}_{upper, lower}` which in turn uses the matrix multiplication
# which is overloaded here in construction of the interpolation/projection operators 
# required for mortars.
function Base.:*(A::Matrix{Real}, B::Matrix{Real})::Matrix{Real}
    m, n = size(A, 1), size(B, 2)
    kA = size(A, 2)
    kB = size(B, 1)
    @assert kA==kB "Matrix dimensions must match for multiplication"

    C = Matrix{Real}(undef, m, n)
    for i in 1:m, j in 1:n
        #acc::Real = zero(promote_type(typeof(A[i,1]), typeof(B[1,j])))
        acc = zero(Real)
        for k in 1:kA
            acc += A[i, k] * B[k, j]
        end
        C[i, j] = acc
    end
    return C
end
end

import .RealMatMulOverload

###############################################################################################

###############################################################################
# semidiscretization of the linear advection diffusion equation

advection_velocity = 0.05
equations = LinearScalarAdvectionEquation1D(advection_velocity)
diffusivity() = 0.05
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

# `RealT = Real` requires fewer overloads than the more explicit `RealT = Num` from Symbolics
# `solver_real` is used for computing the Jacobian sparsity pattern
solver_real = DGSEM(polydeg = 5, surface_flux = flux_lax_friedrichs, RealT = Real)
# `solver_float` is  used for the subsequent simulation
solver_float = DGSEM(polydeg = 5, surface_flux = flux_lax_friedrichs)

coordinates_min = -convert(Float64, pi)
coordinates_max = convert(Float64, pi)

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000)

function x_trans_periodic(x, domain_length = SVector(oftype(x[1], 2 * pi)),
                          center = SVector(oftype(x[1], 0)))
    x_normalized = x .- center
    x_shifted = x_normalized .% domain_length
    x_offset = ((x_shifted .< -0.5f0 * domain_length) -
                (x_shifted .> 0.5f0 * domain_length)) .*
               domain_length
    return center + x_shifted + x_offset
end

# Define initial condition
function initial_condition_diffusive_convergence_test(x, t,
                                                      equation::LinearScalarAdvectionEquation1D)
    # Store translated coordinate for easy use of exact solution
    x_trans = x_trans_periodic(x - equation.advection_velocity * t)

    nu = diffusivity()
    c = 0
    A = 1
    omega = 1
    scalar = c + A * sin(omega * sum(x_trans)) * exp(-nu * omega^2 * t)
    return SVector(scalar)
end
initial_condition = initial_condition_diffusive_convergence_test

# `semi_real` is used for computing the Jacobian sparsity pattern
semi_real = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                  initial_condition,
                                                  solver_real; solver_parabolic = ViscousFormulationLocalDG())

# `semi_float` is used for the subsequent simulation
semi_float = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                                  initial_condition,
                                                  solver_float; solver_parabolic = ViscousFormulationLocalDG())

###############################################################################################
### Compute the Jacobian of the stage function with SparseDiffTools ###

const t0 = 0.0
const t_end = 10.0

const t_span = (t0, t_end)

ode = semidiscretize(semi_float, t_span)
u0_ode = ode.u0
du_ode = similar(u0_ode)

###############################################################################################
### Compute the Jacobian with SparseDiffTools ###

sd = SymbolicsSparsityDetection()
ad_type = AutoFiniteDiff()
sparse_adtype = AutoSparse(ad_type)

# For the operator split employed here, the implicit equation is the parabolic part only
# The sparsity pattern of the residual function is identical with the one from the rhs
rhs_para = (du_ode, u0_ode) -> Trixi.rhs_parabolic!(du_ode, u0_ode, semi_real, t0)

sparse_cache = sparse_jacobian_cache(sparse_adtype, sd, rhs_para, du_ode, u0_ode)

jac_prototype = sparse_cache.jac_prototype
colorvec = sparse_cache.coloring.colorvec

# Revert overrides from above for the actual simulation - 
# not strictly necessary, but good practice
Trixi.eps(x::Type{Real}) = Base.eps(x)
Trixi.one(x::Type{Real}) = Base.one(x)
Trixi.zero(x::Type{Real}) = Base.zero(x)

###############################################################################
# ODE solvers, callbacks etc.

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi_float, interval = 100,
                                     extra_analysis_errors = (:conservation_error,))

callbacks = CallbackSet(summary_callback, analysis_callback)

###############################################################################
# run the simulation

# NOTE: If weird error messages show up (Intel mkl one) update packages, seemes to solve the thing

#ode_alg = Trixi.IMEX_LobattoIIIAp2_Heun()

ode_alg = Trixi.IMEX_Midpoint_Midpoint()

dt = 2.0 / (2^0) # Time step size

integrator = Trixi.init(ode, ode_alg; dt = dt, callback = callbacks,
                        jac_prototype = jac_prototype, colorvec = colorvec);
sol = Trixi.solve!(integrator);

sol = Trixi.solve(ode, ode_alg, dt = dt,
                  save_everystep = false, callback = callbacks);
