
function newton_step!(bs, ∂Fs, ∂²Fs, params::AbstractParams, psamples)
	dbs = -∂²Fs[1]\∂Fs[1]
	bs .+= dbs
	return precompute_step!(∂Fs, ∂²Fs, bs, params, psamples)..., norm(dbs)
end

get_is(params::ParamsNoB1_pinned) = [i for i=1:get_length(params) if (i!=1)&&(i!=params.i_pinned)]

function get_is(params::ParamsB1_pinned)
	N = params.N
	i_pinned = params.i_pinned
	blacklist = [1, i_pinned, N+div(N,4)+1, N+div(N,4)+i_pinned]
	return [i for i=1:get_length(params) if !(i in blacklist)]
end

function newton_step!(bs, ∂Fs, ∂²Fs, params::T, psamples) where {T<:Union{ParamsB1_pinned, ParamsNoB1_pinned}}
	is = get_is(params) 
	∂Fv = @view ∂Fs[1][is]
	∂²Fv = @view ∂²Fs[1][is,is]
	dbs = fill!(similar(bs), 0.0)
	dbsv = @view dbs[is]
	dbsv .= -∂²Fv\∂Fv
	bs .+= dbs
	return precompute_step!(∂Fs, ∂²Fs, bs, params, psamples)..., norm(dbs)
end

mutable struct Optimizer{OptAlgType<:AbstractOptAlg, paramsType<:AbstractParams, cashType<:AbstractSharedCash, LSType, T}
    alg::OptAlgType
    bs::Vector{Float64}
    ∂bs::Vector{Float64}
    params::paramsType
    psamples::Vector{Vector{NTuple{3,Float64}}}
    ΔF::Float64
    iter::Int64
    max_iter::Int64
    ϵ::Float64
    ϵ₀::Float64
    f_cash::cashType
    ls::LSType
    algCash::T
    pinned_idxs::Vector{Int64}
    free_idxs::Vector{Float64}

    function Optimizer(alg::OptAlgType, bs, params::paramsType, psamples, max_iter, ϵ₀, f_cash::cashType, ls::LSType, algCash::T) where {OptAlgType<:AbstractOptAlg, paramsType<:AbstractParams, cashType<:AbstractSharedCash, LSType, T}
        new{OptAlgType, paramsType, cashType, LSType, T}(alg, bs, similar(bs), params, psamples, 0.0, 0, max_iter, 0.0, ϵ₀, f_Cash, ls, algCash, get_pinned_idxs(params), get_free_idxs(params))
    end
end

function Optimizer(alg::BFGSOptAlg, bs, params, psamples, max_iter=60, ϵ₀=1e-12)
    f_cash = generate_cash(alg, params)
    ls = MoreThuente()
    N′′ = length(get_free_idxs(params))
    algCash = Matrix{Float64}(I, N′′, N′′)
    return Optimizer(alg, bs, params, psamples, max_iter, ϵ₀, f_cash, ls, algCash)
end

function precompute_step!(O::Optimizer{BFGSOptAlg})
    ΔF, ∂F = precompute_step!(O.f_cash, O.bs, O.params, O.psamples)
    O.ΔF = ΔF
    project_onto(O.∂bs) = 
end
