abstract type AbstractParams end
abstract type AbstractParamsB1 <: AbstractParams end
abstract type AbstractParamsNoB1 <: AbstractParams end

struct ParamsB1 <: AbstractParamsB1
    N::Int64
    m::Int64
    β::Float64
    u::Float64
    u₁::Float64
    ParamsB1(N::Int64, m::Int64, β::Float64, U0::Float64, Ut0::Float64) = new(N, m, β, 1/U0, 1/Ut0)
end

struct ParamsB1_pinned <: AbstractParamsB1
    N::Int64
    m::Int64
    β::Float64
    u::Float64
    u₁::Float64
	i_pinned::Int64
	ParamsB1_pinned(N::Int64, m::Int64, β::Float64, U0::Float64, Ut0::Float64) = new(N, m, β, 1/U0, 1/Ut0, div(N,2)+1)
end

struct ParamsNoB1 <: AbstractParamsNoB1
    N::Int64
    m::Int64
    β::Float64
    u::Float64
    ParamsNoB1(N::Int64, m::Int64, β::Float64, U0::Float64) = new(N, m, β, 1/U0)
end

struct ParamsNoB1_pinned <: AbstractParamsNoB1
    N::Int64
    m::Int64
    β::Float64
    u::Float64
	i_pinned::Int64
    ParamsNoB1_pinned(N::Int64, m::Int64, β::Float64, U0::Float64) = new(N, m, β, 1/U0, div(N,2)+1)
end


function generate_shared_cash(params::AbstractParamsB1)
	np = nprocs()
	N = params.N
	∂Fs = Vector{SharedVector{Float64}}()
	∂²Fs = Vector{SharedMatrix{Float64}}()
	for i in Base.OneTo(np)
		push!(∂Fs, SharedVector{Float64}(2N))
		push!(∂²Fs, SharedMatrix{Float64}(2N, 2N))
	end
	return ∂Fs, ∂²Fs
end

function generate_shared_cash(params::AbstractParamsNoB1)
	np = nprocs()
	N = params.N
	∂Fs = Vector{SharedVector{Float64}}()
	∂²Fs = Vector{SharedMatrix{Float64}}()
	for i in Base.OneTo(np)
		push!(∂Fs, SharedVector{Float64}(N))
		push!(∂²Fs, SharedMatrix{Float64}(N, N))
	end
	return ∂Fs, ∂²Fs
end


struct ReducedDispersion
    α::Float64
    P::Float64
	μ::Float64
    Λ::Float64
end

wfunc(y::Float64) = y==0.0 ? 0.0 : y^2*log((1+sqrt(1-y^6))/abs(y)^3)

(rdisp::ReducedDispersion)(y::Float64, dy::Float64) = (-rdisp.μ, rdisp.P + rdisp.α*rdisp.Λ*y^3, wfunc(y)*6dy*rdisp.Λ/(2π)^2)::NTuple{3,Float64}

function get_psamples(rdisp::ReducedDispersion, Nₚ)
    psamples_raw = rdisp.(range(-1.0,1.0; length = Nₚ), 2.0/(Nₚ-1))
	psamples_raw[1] = (psamples_raw[1][1], psamples_raw[1][2], 0.5*psamples_raw[1][3])
	psamples_raw[end] = (psamples_raw[end][1], psamples_raw[end][2], 0.5*psamples_raw[end][3])
	return psamples_raw
end

function separate_psamples(psamples_raw::Vector{NTuple{3,Float64}})
	np = nprocs()
	Nₚ = length(psamples_raw)
	chunk_size = div(Nₚ, np)
	psamples = Vector{Vector{NTuple{3,Float64}}}(undef, np)
	for cid = 0:(np-1)
		psamples[cid+1] = psamples_raw[(chunk_size*cid+1):min(Nₚ, chunk_size*(cid+1))]
	end
	return psamples
end

function generate_psamples(rdisp::ReducedDispersion, Nₚ)
	return separate_psamples(get_psamples(rdisp, Nₚ))
end
