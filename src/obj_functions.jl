struct bfgsObjFunc{pType<:AbstractParams}
	params::pType
	g_cash::G_Cash
	psamples::Vector{Vector{NTuple{3, Float64}}}
	γ::Float64
end

bfgsObjFunc(params::AbstractParams, rdisp::ReducedDispersion, Nₚ, γ) = bfgsObjFunc(params, G_Cash(params), generate_psamples(rdisp, Nₚ), γ)

function (fg!::bfgsObjFunc)(F,G, bs)
	ΔF, ∂F = precompute_step!(fg!.g_cash, bs, fg!.params, fg!.psamples, fg!.γ)
	if !(G==nothing)
		G.=∂F
	end
	if !(F==nothing)
		return ΔF
	end
end

function construct_objective(params::AbstractParams, rdisp::ReducedDispersion, Nₚ, γ, bs0=seed_sn(params, rdisp, Nₚ))
    @assert get_length(params)==length(bs0)
    fg! = bfgsObjFunc(params, rdisp, Nₚ, γ)
    return OnceDifferentiable(only_fg!(fg!), bs0), bs0
end


function construct_objective(params::AbstractParams, psamples::Vector{Vector{NTuple{3, Float64}}}, γ, bs0)
    @assert get_length(params)==length(bs0)
	fg! = bfgsObjFunc(params, G_Cash(params), psamples, γ)
    return OnceDifferentiable(only_fg!(fg!), bs0)
end
