const TupType = Tuple{Float64, Float64, Float64, Float128, Float128}

struct bfgsObjFunc{pType<:AbstractParams}
	params::pType
	g_cash::G_Cash
    psamples::DArray{TupType, 1, Vector{TupType}}
end

bfgsObjFunc(params::AbstractParams, rdisp::ReducedDispersion, Nₚ, γ) = bfgsObjFunc(params, G_Cash(params), generate_psamples(rdisp, Nₚ), γ)

function (fg!::bfgsObjFunc)(F,G, bs)
	if !(G==nothing)
		ΔF, ∂F = precompute_step!(fg!.g_cash, bs, fg!.params, fg!.psamples)
		G.=∂F
		if !(F==nothing)
			return ΔF
		end
	else
		if !(F==nothing)
			return precompute_step(bs, fg!.params, fg!.psamples)
		end
	end
end



function construct_objective(params::AbstractParams, psamples::DArray{TupType, 1, Vector{TupType}}, bs0)
    @assert get_length(params)==length(bs0)
	fg! = bfgsObjFunc(params, G_Cash(params), psamples)
    return OnceDifferentiable(only_fg!(fg!), bs0)
end
