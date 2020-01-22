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


