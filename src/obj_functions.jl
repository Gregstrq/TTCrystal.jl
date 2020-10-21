const TupType = Tuple{Float64, Float64, Float64, Float128, Float128}

struct bfgsObjFunc{pType<:AbstractParams}
	params::pType
	g_cash::G_Cash
    psamples::DArray{TupType, 1, Vector{TupType}}
end

struct bfgsObjFunc_repul{pType<:AbstractParams, GT<:AbstractRepulsion}
	params::pType
	g_cash::G_Cash
    psamples::DArray{TupType, 1, Vector{TupType}}
    grep::GT
end


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

function (fg!::bfgsObjFunc_repul)(F,G, bs)
	if !(G==nothing)
		ΔF, ∂F = precompute_step!(fg!.g_cash, bs, fg!.params, fg!.psamples)
        f, ∂f = compute_repulsion!(fg!.grep, bs)
        ΔF += f
        ∂F[1:length(∂f)] .+= ∂f
		G.=∂F
		if !(F==nothing)
			return ΔF
		end
	else
		if !(F==nothing)
            return precompute_step(bs, fg!.params, fg!.psamples) + compute_repulsion(fg!.grep, bs)
		end
	end
end



function construct_objective(params::AbstractParams, psamples::DArray{TupType, 1, Vector{TupType}}, bs0)
    @assert get_length(params)==length(bs0)
	fg! = bfgsObjFunc(params, G_Cash(params), psamples)
    return OnceDifferentiable(only_fg!(fg!), bs0)
end

function construct_objective(params::AbstractParams, psamples::DArray{TupType, 1, Vector{TupType}}, bs0, ω₀::Float64)
    @assert get_length(params)==length(bs0)
    fg! = bfgsObjFunc_repul(params, G_Cash(params), psamples, GaugeRepulsion2(params, ω₀))
    return OnceDifferentiable(only_fg!(fg!), bs0)
end
function construct_objective(params::AbstractParams, psamples::DArray{TupType, 1, Vector{TupType}}, bs0, reptyp::AbstractRepulsionType)
    @assert get_length(params)==length(bs0)
    fg! = bfgsObjFunc_repul(params, G_Cash(params), psamples, construct_repulsion(reptyp, params))
    return OnceDifferentiable(only_fg!(fg!), bs0)
end
