abstract type AbstractViscoResistiveMhdDiffusion{NDIMS, NVARS, GradientVariables} <:
              AbstractEquationsParabolic{NDIMS, NVARS, GradientVariables} end

# This enables "forwarded" accesses to e.g.`equations_parabolic.gamma` of the "underlying" `equations_hyperbolic`
# while keeping direct access to parabolic-specific fields like `mu` or `kappa`.
@inline function Base.getproperty(equations_parabolic::AbstractViscoResistiveMhdDiffusion,
                                  field::Symbol)
    if field === :gamma || field === :inv_gamma_minus_one
        return getproperty(getfield(equations_parabolic, :equations_hyperbolic), field)
    else
        return getfield(equations_parabolic, field)
    end
end

# Provide property names for e.g. tab-completion by combining
# the names from the underlying hyperbolic equations with the fields of this parabolic part.
@inline function Base.propertynames(equations_parabolic::AbstractViscoResistiveMhdDiffusion,
                                    private::Bool = false)
    names_hyp = (:gamma, :inv_gamma_minus_one)
    names_para = fieldnames(typeof(equations_parabolic))
    names_hyp_para = (names_hyp..., names_para...)

    return names_hyp_para
end

include("visco_resistive_mhd_3d.jl")
