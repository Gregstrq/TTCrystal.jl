struct bfgsObjFunc{pType<:AbstractParams}
	params::pType
	g_cash::G_Cash
	psamples::Vector{Vector{NTuple{3, Float64}}}
end

bfgsObjFunc(params::AbstractParams, rdisp::ReducedDispersion, Nₚ) = bfgsObjFunc(params, G_Cash(params), generate_psamples(rdisp, Nₚ))

function (fg!::bfgsObjFunc)(F,G, bs)
	ΔF, ∂F = precompute_step!(fg!.g_cash, bs, fg!.params, fg!.psamples)
	if !(G==nothing)
		G.=∂F
	end
	if !(F==nothing)
		return ΔF
	end
end

function construct_objective(params::AbstractParams, rdisp::ReducedDispersion, Nₚ, bs0=seed_sn(params, rdisp, Nₚ))
    @assert get_length(params)==length(bs0)
    fg! = bfgsObjFunc(params, rdisp, Nₚ)
    return OnceDifferentiable(only_fg!(fg!), b0), b0
end


