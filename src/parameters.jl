abstract type AbstractParams end
abstract type AbstractParamsB1 <: AbstractParams end
abstract type AbstractParamsNoB1 <: AbstractParams end

abstract type AbstractDispersion end
abstract type AbstractSharedCash end

struct ParamsB1 <: AbstractParamsB1
    N::Int64
    m::Int64
    β::Float64
    u::Float64
    u₁::Float64
end

struct ParamsB1_pinned <: AbstractParamsB1
    N::Int64
    m::Int64
    β::Float64
    u::Float64
    u₁::Float64
	i_pinned::Int64
	ParamsB1_pinned(N::Int64, m::Int64, β::Float64, u::Float64, u₁::Float64) = new(N, m, β, u, u₁, div(N,2)+1)
end

struct ParamsB1_pinned² <: AbstractParamsB1
    N::Int64
    m::Int64
    β::Float64
    u::Float64
    u₁::Float64
	i_pinned::Int64
	ParamsB1_pinned²(N::Int64, m::Int64, β::Float64, u::Float64, u₁::Float64) = new(N, m, β, u, u₁, div(N,2)+1)
end

struct ParamsNoB1 <: AbstractParamsNoB1
    N::Int64
    m::Int64
    β::Float64
    u::Float64
end

struct ParamsNoB1_pinned <: AbstractParamsNoB1
    N::Int64
    m::Int64
    β::Float64
    u::Float64
	i_pinned::Int64
    ParamsNoB1_pinned(N::Int64, m::Int64, β::Float64, u::Float64) = new(N, m, β, u, div(N,2)+1)
end

function get_u₀(β, psamples_raw)
    s = 0.0
	for (εₚ⁺, εₚ⁻, wₚ) in psamples_raw
        κₚ = sqrt(εₚ⁻^2 + 1.0)
        s += wₚ*(sign(εₚ⁺+κₚ) + sign(κₚ-εₚ⁺))/κₚ
    end
    return s/4.0
end

for Params_constr in Symbol.(subtypes(AbstractParamsB1))
	@eval begin
		function $Params_constr(β::Float64, m::Int64, Δτ::Float64, a::Float64, psamples_raw::Vector{NTuple{3, Float64}})
			N = 4*(div(ceil(Int64, β/(m*Δτ)), 4) + 1)
			u₀ = get_u₀(β, psamples_raw)
			u₁ = u₀*a
			return $Params_constr(N, m, β, u₀, u₁)
		end
	end
end

for Params_constr in Symbol.(subtypes(AbstractParamsNoB1))
	@eval begin
		function $Params_constr(β::Float64, m::Int64, Δτ::Float64, psamples_raw::Vector{NTuple{3, Float64}})
			N = 4*(div(ceil(Int64, β/(m*Δτ)), 4) + 1)
			u₀ = get_u₀(β, psamples_raw)
			return $Params_constr(N, m, β, u₀)
		end
	end
end

get_pinned_idxs(params::Union{ParamsNoB1, ParamsB1}) = Int64[]
get_pinned_idxs(params::Union{ParamsB1_pinned, ParamsNoB1_pinned}) = [1, params.i_pinned]
function get_pinned_idxs(params::Union{ParamsB1_pinned²})
    N, i_pinned = params.N, params.i_pinned
    return [1, i_pinned, N+div(N,4)+1, N+div(N,4)+i_pinned]
end

get_free_idxs(params::AbstractParams) = [i for i=Base.OneTo(get_length(params)) if !(i in get_pinned_idxs(params))]

@inline pin_bs!(bs::AbstractVector, params::AbstractParams) = pin_bs!(bs, get_pinned_idxs(params))
@inline pin_bs!(bs::AbstractVector, params::Union{ParamsB1, ParamsNoB1}) = bs
function pin_bs!(bs::AbstractVector{T}, pinned_idxs::Vector{Int64}) where {T}
    for i in pinned_idxs
        bs[i] = zero(T)
    end
    return bs
end

@inline project_onto(a, params::AbstractParams) = project_onto(a, get_free_idxs(params))
@inline project_onto(a, params::Union{ParamsB1, ParamsNoB1}) = a
@inline project_onto(a::AbstractVector, free_idxs::Vector{Int64}) = view(a, free_idxs)
@inline project_onto(a::AbstractMatrix, free_idxs::Vector{Int64}) = view(a, free_idxs)

################################

abstract type AbstractOptAlg end

struct NewtonOptAlg <: AbstractOptAlg end
struct BFGSOptAlg <: AbstractOptAlg end

generate_cash(::NewtonOptAlg, params::AbstractParams) = GH_Cash(params)
generate_cash(::BFGSOptAlg, params::AbstractParams) = G_Cash(params)

get_length(params::AbstractParamsB1) = 2params.N
get_length(params::AbstractParamsNoB1) = params.N

struct GH_Cash <: AbstractSharedCash
	∂Fs::Vector{SharedVector{Float64}}
	∂²Fs::Vector{SharedMatrix{Float64}}
	function GH_Cash(params::AbstractParams)
		np = nprocs()
		N′ = get_length(params)
		∂Fs = Vector{SharedVector{Float64}}()
		∂²Fs = Vector{SharedMatrix{Float64}}()
        ∂Fs = [@fetchfrom pid SharedArrays.SharedVector{Float64}(N′; pids = procs()) for pid=procs()]
        ∂²Fs = [@fetchfrom pid SharedArrays.SharedMatrix{Float64}(N′, N′; pids = procs()) for pid=procs()]
		new(∂Fs, ∂²Fs)
	end
end

mutable struct G_Cash <: AbstractSharedCash
	∂Fs::Vector{SharedVector{Float64}}
	∂Us::DArray{SMatrix{2,2,Complex{Float128}}, 1, Vector{SMatrix{2,2,Complex{Float128}}}}
	U_cashes::DArray{SMatrix{2,2,Complex{Float128}}, 1, Vector{SMatrix{2,2,Complex{Float128}}}}
	function G_Cash(params::AbstractParams)
		np = nprocs()
		N = params.N
		N′ = get_length(params)
        ∂Fs = [SharedVector{Float64}(N′) for pid=procs()]
		∂Us = DArray([@spawnat i LinearAlgebra.ones(StaticArrays.SMatrix{2,2, Complex{Float128}}, N′) for i=procs()])
		U_cashes = DArray([@spawnat i LinearAlgebra.ones(StaticArrays.SMatrix{2,2, Complex{Float128}}, N+1) for i=procs()])
		g_cash = new(∂Fs, ∂Us, U_cashes)
		finalizer(free_darrays, g_cash)
		return g_cash
	end
end

function free_darrays(g_cash::G_Cash)
	close(g_cash.∂Us)
	close(g_cash.U_cashes)
end

################################

struct ReducedDispersion <: AbstractDispersion
    α::Float64
    P::Float64
	μ::Float64
    Λ::Float64
end

wfunc(y::Float64) = y==0.0 ? 0.0 : y^2*log((1+sqrt(1-y^6))/abs(y)^3)
wfunc_gauss(y::Real) = log((1+sqrt(1-y^2))/abs(y))

(rdisp::ReducedDispersion)(y::Float64, dy::Float64) = (-rdisp.μ, rdisp.P + rdisp.α*rdisp.Λ*y^3, wfunc(y)*6dy*rdisp.Λ/(2π)^2)::NTuple{3,Float64}

#function get_psamples(rdisp::ReducedDispersion, Nₚ)
#    psamples_raw = rdisp.(range(-1.0,1.0; length = Nₚ), 2.0/(Nₚ-1))
#	psamples_raw[1] = (psamples_raw[1][1], psamples_raw[1][2], 0.5*psamples_raw[1][3])
#	psamples_raw[end] = (psamples_raw[end][1], psamples_raw[end][2], 0.5*psamples_raw[end][3])
#	return psamples_raw
#end

function get_psamples(rdisp::ReducedDispersion, Nₚ)
	temp_quad(f, _a, _b; kwargs...) = quadgk(f, -1.0, 0.0, 1.0; kwargs...)
	ys, ws = gauss(wfunc_gauss, Nₚ, -1.0, 1.0; quad=temp_quad)
	return [(-rdisp.μ, rdisp.P + rdisp.Λ*ys[i], 2*rdisp.Λ*ws[i]/(rdisp.α*(2pi)^2)) for i in eachindex(ws)]
end

function separate_psamples(psamples_raw::Vector{NTuple{3,Float64}})
	np = nprocs()
	Nₚ = length(psamples_raw)
	chunk_size, num_extra = divrem(Nₚ, np)
	psamples = Vector{Vector{NTuple{3,Float64}}}(undef, np)
	for cid = 0:(num_extra-1)
		psamples[cid+1] = psamples_raw[((chunk_size+1)*cid+1):(chunk_size+1)*(cid+1)]
	end
	n₀ = (chunk_size+1)*num_extra
	for cid = 0:(np-num_extra-1)
		psamples[num_extra+cid+1] = psamples_raw[(n₀+chunk_size*cid+1):min(Nₚ, n₀+chunk_size*(cid+1))]
	end
	return psamples
end

function generate_psamples(rdisp::ReducedDispersion, Nₚ)
	return separate_psamples(get_psamples(rdisp, Nₚ))
end
