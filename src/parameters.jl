abstract type AbstractParams{T<:Union{Int64,Nothing}} end
abstract type AbstractParamsB1{T<:Union{Int64,Nothing}} <: AbstractParams{T} end
abstract type AbstractParamsNoB1{T<:Union{Int64,Nothing}} <: AbstractParams{T} end

abstract type AbstractDispersion end
abstract type AbstractSharedCash end

struct ParamsB1{T<:Union{Int64, Nothing}} <: AbstractParamsB1{T}
    N::Int64
    m::T
    W::Float64
    u::Float64
    u₁::Float64
    Δτ::Float64
	ParamsB1(N, m::T, W, u, u₁) where T<:Union{Int64, Nothing} = new{T}(N, m, W, u, u₁, W/N)
end

struct ParamsB1_pinned{T<:Union{Int64, Nothing}} <: AbstractParamsB1{T}
    N::Int64
    m::T
    W::Float64
    u::Float64
    u₁::Float64
    Δτ::Float64
	i_pinned::Int64
	ParamsB1_pinned(N::Int64, m::T, W::Float64, u::Float64, u₁::Float64) where T<:Union{Int64, Nothing} = new{T}(N, m, W, u, u₁, W/N, div(N,2)+1)
end

struct ParamsB1_pinned²{T<:Union{Int64, Nothing}} <: AbstractParamsB1{T}
    N::Int64
    m::T
    W::Float64
    u::Float64
    u₁::Float64
    Δτ::Float64
	i_pinned::Int64
	ParamsB1_pinned²(N::Int64, m::T, W::Float64, u::Float64, u₁::Float64) where T<:Union{Int64, Nothing} = new{T}(N, m, W, u, u₁, W/N, div(N,2)+1)
end

struct ParamsNoB1{T<:Union{Int64, Nothing}} <: AbstractParamsNoB1{T}
    N::Int64
    m::T
    W::Float64
    u::Float64
    Δτ::Float64
	ParamsNoB1(N::Int64, m::T, W::Float64, u::Float64) where T<:Union{Int64, Nothing} = new{T}(N, m, W, u, W/N)
end

struct ParamsNoB1_pinned{T<:Union{Int64, Nothing}} <: AbstractParamsNoB1{T}
    N::Int64
    m::T
    W::Float64
    u::Float64
    Δτ::Float64
	i_pinned::Int64
	ParamsNoB1_pinned(N::Int64, m::T, W::Float64, u::Float64) where T<:Union{Int64, Nothing} = new{T}(N, m, W, u, W/N, div(N,2)+1)
end

function get_u₀(β::AbstractFloat, psamples_raw::Vector{NTuple{3, Float64}})
    s = 0.0
	for (εₚ⁺, εₚ⁻, wₚ) in psamples_raw
        κₚ = sqrt(εₚ⁻^2 + 1.0)
        s += wₚ*(tanh(β*(εₚ⁺+κₚ)/2) + tanh(β*(κₚ-εₚ⁺)/2))/κₚ
    end
    return s/4.0
end
function get_u₀(psamples_raw)
    s = 0.0
	for (εₚ⁺, εₚ⁻, wₚ) in psamples_raw
        κₚ = sqrt(εₚ⁻^2 + 1.0)
        s += wₚ/κₚ
    end
    return s/2.0
end
function get_u₀(psamples_raw::Vector{NTuple{3, Float64}}, γ::AbstractFloat)
    s = 0.0
	for (εₚ⁺, εₚ⁻, wₚ) in psamples_raw
        κₚ = sqrt(εₚ⁻^2 + γ^2)
        s += wₚ/κₚ
    end
    return s/2.0
end
get_u₀(W, m::Int64, psamples_raw) = get_u₀(W*m, psamples_raw)
get_u₀(W, m::Nothing, psamples_raw) = get_u₀(psamples_raw)

for Params_constr in (:ParamsB1, :ParamsB1_pinned, :ParamsB1_pinned²)
	@eval begin
		function $Params_constr(N::Int64, m::Union{Int64, Nothing}, W::Float64, a::Float64, psamples_raw::Vector{NTuple{3, Float64}})
			u₀ = get_u₀(W, nothing, psamples_raw)
			u₁ = u₀*a
			return $Params_constr(N + 1 - rem(N, 2), m, W, u₀, u₁)
		end
	end
end

for Params_constr in (:ParamsNoB1, :ParamsNoB1_pinned)
	@eval begin
		function $Params_constr(N::Int64, m::Union{Int64, Nothing}, W::Float64, psamples_raw::Vector{NTuple{3, Float64}})
			u₀ = get_u₀(W, nothing, psamples_raw)
			return $Params_constr(N + 1 - rem(N, 2), m, W, u₀)
		end
	end
end

########

kw(W) = find_zero(k -> (W-4K(k^2)), (0.0, 1.0))

get_length(params::AbstractParamsB1) = 2params.N
get_length(params::AbstractParamsNoB1) = params.N

get_τs(params::AbstractParams) = collect(0:params.N-1)*params.Δτ

bs_init_func(τs, k::AbstractFloat) = k*Jacobi.sn.(τs, k^2)
bs_init_func(τs, k::Nothing) = tanh.(τs)
function get_bs(params::AbstractParams)
    W = params.W
    if W < 40.0
        return get_bs(params, kw(W))
    end
    τs = get_τs(params)
    try
        k = kw(W)
        return _fill_bs(τs, W,  k)
    catch e
        return _fill_bs(τs, W,  nothing)
    end
end
function get_bs(params::AbstractParams, k)
    bs = zeros(get_length(params))
    bs[1:params.N] .= bs_init_func(get_τs(params), k)
    pin_bs!(bs, params)
    return bs
end
function _fill_bs(τs, W, k)
    N = length(τs)
    @assert N>10
    Nhalf = div(N, 2) + 1
    Nquarter = div(Nhalf, 2) + 1
    bs = similar(τs)
    bs[1:Nquarter] = bs_init_func(τs[1:Nquarter], k)
    bs[Nquarter+1:Nhalf] .= bs_init_func(W/2 .- τs[Nquarter+1:Nhalf], k)
    bs[end:-1:Nhalf+1] .= -bs[2:N-Nhalf+1]
    return bs
end

########

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

mutable struct G_Cash2 <: AbstractSharedCash
    ΔFss::DArray{Float64, 1, Vector{Float64}}
    ∂Fss::DArray{Vector{Float64}, 1, Vector{Vector{Float64}}}
	∂Us::DArray{SMatrix{2,2,Complex{Float128}}, 1, Vector{SMatrix{2,2,Complex{Float128}}}}
	U_cashes::DArray{SMatrix{2,2,Complex{Float128}}, 1, Vector{SMatrix{2,2,Complex{Float128}}}}
    function G_Cash2(params::AbstractParams, psamples)
		np = nprocs()
		N = params.N
		N′ = get_length(params)
        ΔFss = DArray([@spawnat i LinearAlgebra.zeros(Float64, length(localpart(psamples))) for i=procs()])
        ∂Fss = DArray([@spawnat i [LinearAlgebra.zeros(Float64, N′) for j=1:length(localpart(psamples))] for i=procs()])
		∂Us = DArray([@spawnat i LinearAlgebra.ones(StaticArrays.SMatrix{2,2, Complex{Float128}}, N′) for i=procs()])
		U_cashes = DArray([@spawnat i LinearAlgebra.ones(StaticArrays.SMatrix{2,2, Complex{Float128}}, N+1) for i=procs()])
		g_cash = new(ΔFss, ∂Fss, ∂Us, U_cashes)
		finalizer(free_darrays, g_cash)
		return g_cash
    end
end

function free_darrays(g_cash::G_Cash2)
	close(g_cash.ΔFss)
	close(g_cash.∂Fss)
	close(g_cash.∂Us)
	close(g_cash.U_cashes)
end

export G_Cash2


################################

struct ReducedDispersion <: AbstractDispersion
    α::Float64
    P::Float64
	μ::Float64
    Λ::Float64
end


wfunc(y::Float64) = y==0.0 ? 0.0 : y^2*log((1+sqrt(1-y^6))/abs(y)^3)
@inline wfunc_gauss(y::Real) = y==0.0 ? 0.0 : log((1+sqrt(1-y^2))/abs(y))
@inline wfunc_supplementary(y::Real) = -1/y*wfunc_gauss(y)

function get_quadrature(rdisp::ReducedDispersion, rtol::Float64 = 1e-6, limit::Int64 = 200)
	α, P, μ, Λ = rdisp.α, rdisp.P, rdisp.μ, rdisp.Λ
	f(x) = 1/x*wfunc_gauss((x-P)/Λ)
	if abs(P)<Λ
		I, E, segs = quadgk_cauchy(f, P-Λ, P+Λ, 0.0, P; rtol=rtol, limit=limit)
	else
		I, E, segs = quadgk_custom(f, P-Λ, P+Λ, P; rtol = rtol, limit=limit)
	end
	x₁, w₁, x₂, w₂ = construct_quadrature(segs)
    w₂ .*= wfunc_gauss.((x₂.-P)./Λ)./(α*2*π^2)
	return x₂, w₂
end

function get_psamples(rdisp::ReducedDispersion, rtol::Float64 = 1e-6, limit::Int64 = 200)
	ε⁻s, ws = get_quadrature(rdisp, rtol, limit)
	μ = rdisp.μ
	return [(μ, ε⁻s[i], ws[i]) for i in eachindex(ws)]
end

#########################################################

(rdisp::ReducedDispersion)(y::Float64, dy::Float64) = (-rdisp.μ, rdisp.P + rdisp.Λ*y^3, wfunc(y)*6dy*rdisp.Λ/(rdisp.α*(2π)^2))::NTuple{3,Float64}
apply_rdisp(rdisp::ReducedDispersion, y::Float64, w::Float64) = (-rdisp.μ, rdisp.P + rdisp.Λ*y^3, 6w*rdisp.Λ/(rdisp.α*(2π)^2))::NTuple{3,Float64}

function get_psamples_new(rdisp::ReducedDispersion, Nₚ)
	ys, ws = gauss(wfunc, Nₚ, -1.0, 1.0)
	return [apply_rdisp(rdisp, ys[i], ws[i]) for i = 1:Nₚ] 
end

function get_psamples_old(rdisp::ReducedDispersion, Nₚ)
    psamples_raw = rdisp.(range(-1.0,1.0; length = Nₚ), 2.0/(Nₚ-1))
	psamples_raw[1] = (psamples_raw[1][1], psamples_raw[1][2], 0.5*psamples_raw[1][3])
	psamples_raw[end] = (psamples_raw[end][1], psamples_raw[end][2], 0.5*psamples_raw[end][3])
	return psamples_raw
end

function get_psamples(rdisp::ReducedDispersion, Nₚ)
	temp_quad(f, _a, _b; kwargs...) = quadgk(f, -1.0, 0.0, 1.0; kwargs...)
	ys, ws = gauss(wfunc_gauss, Nₚ, -1.0, 1.0; quad=temp_quad)
	return [(-rdisp.μ, rdisp.P + rdisp.Λ*ys[i], 2*rdisp.Λ*ws[i]/(rdisp.α*(2pi)^2)) for i in eachindex(ws)]
end

#########################################################

widen(psamples_raw, W, m) = widen(psamples_raw, W, m, one(Float128))
function widen(psamples_raw::Vector{NTuple{3, Float64}}, W::Float64, m::Int64, γ::Float128)
	β′ = Float128(W)*m
	psamples_widened = Vector{Tuple{Float64, Float64, Float64, Float128, Float128}}()
	for psample in psamples_raw
		εₚ⁺′ = Float128(psample[1])
		κₚ⁰ = sqrt(γ^2 + psample[2]^2)
        a = exp(-β′*(κₚ⁰ - εₚ⁺′)) + exp(-β′*(κₚ⁰ + εₚ⁺′))
        b = log(one(Float128) + exp(-2β′*κₚ⁰) + a)
		push!(psamples_widened, (psample..., a, b))
	end
	return psamples_widened
end

function widen(psamples_raw::Vector{NTuple{3, Float64}}, W::Float64, m::Nothing, γ::Float128)
	psamples_widened = Vector{Tuple{Float64, Float64, Float64, Float128, Float128}}()
	for psample in psamples_raw
		εₚ⁺′ = Float128(psample[1])
		κₚ⁰ = sqrt(γ^2 + psample[2]^2)
		a = W*(εₚ⁺′ - κₚ⁰)
		b = max(a, zero(Float128))
		push!(psamples_widened, (psample..., a, b))
	end
	return psamples_widened
end


function separate_psamples_old(psamples_raw::Vector{Tuple{Float64,Float64,Float64, Float128, Float128}})
	np = nprocs()
	Nₚ = length(psamples_raw)
	chunk_size, num_extra = divrem(Nₚ, np)
	psamples = Vector{Vector{eltype(psamples_raw)}}(undef, np)
	for cid = 0:(num_extra-1)
		psamples[cid+1] = psamples_raw[((chunk_size+1)*cid+1):(chunk_size+1)*(cid+1)]
	end
	n₀ = (chunk_size+1)*num_extra
	for cid = 0:(np-num_extra-1)
		psamples[num_extra+cid+1] = psamples_raw[(n₀+chunk_size*cid+1):min(Nₚ, n₀+chunk_size*(cid+1))]
	end
	return psamples
end

separate_psamples(psamples_raw::AbstractVector) = distribute(psamples_raw; procs = procs())

function generate_psamples(rdisp::ReducedDispersion, Nₚ)
	return separate_psamples(get_psamples(rdisp, Nₚ))
end

struct Dispersion <:AbstractDispersion
	α::Float64
	α′::Float64
	μ::Float64
	Λ::Float64
	P::Float64
end
Dispersion(α, α′, μ, Λ) = Dispersion(α, α′, μ, Λ, 0.0)

wfunc_φ(φ) = 1/sqrt(1-φ^2)

macro unpack(var, args...)
	exs = [:($arg = $(esc(var)).$arg) for arg in args]
    return Expr(:block, exs...)
end

function get_psamples_bad(disp::Dispersion, Nₜ, Nᵩ)
	#@unpack disp α α′ μ Λ P
	α, α′, μ, Λ, P = disp.α, disp.α′, disp.μ, disp.Λ, disp.P
	φs, wᵩs = gauss(wfunc_φ, Nᵩ, -1.0, 1.0)
	ts, wₜs = gauss(Nₜ, 0.0, 1.0)
	c1 = 4*Λ/((α + α′)*(2π)^2) 
	c2 = (α - α′)/(α + α′)*Λ
	psamples_raw = Vector{NTuple{3, Float64}}()
	for iₜ = 1:length(ts)
		t = ts[iₜ]
		wₜ = wₜs[iₜ]
		εₚ⁺ = c2*t - μ
		for iᵩ = 1:length(φs)
			wₚ = c1*wₜ*wᵩs[iᵩ]
			εₚ⁻ = Λ*t*φs[iᵩ] + P
			push!(psamples_raw, (εₚ⁺, εₚ⁻, wₚ))
		end
	end
	return psamples_raw
end

function get_psamples_simple(disp::Dispersion, Nₜ, Nᵩ)
	#@unpack disp α α′ μ Λ P
	α, α′, μ, Λ, P = disp.α, disp.α′, disp.μ, disp.Λ, disp.P
	φs, wᵩs = [(i-0.5)*π/Nᵩ for i=1:Nᵩ], fill!(zeros(Nᵩ), π/Nᵩ)
	ts, wₜs = gauss(Nₜ, 0.0, 1.0)
	c1 = 4*Λ/((α + α′)*(2π)^2) 
	c2 = (α - α′)/(α + α′)*Λ
	psamples_raw = Vector{NTuple{3, Float64}}()
	for iₜ = 1:Nₜ
		t = ts[iₜ]
		wₜ = wₜs[iₜ]
		εₚ⁺ = c2*t - μ
		for iᵩ = 1:Nᵩ
			wₚ = c1*wₜ*wᵩs[iᵩ]
			εₚ⁻ = Λ*t*cos(φs[iᵩ]) + P
			push!(psamples_raw, (εₚ⁺, εₚ⁻, wₚ))
		end
	end
	return psamples_raw
end

get_psamples(disp::Dispersion, Nₜ, Nᵩ) = get_psamples_simple(disp, Nₜ, Nᵩ)

###############

function get_static_energy(a₂, rdisp, int_rtol::Real=1e-6, limits::Int64 = 200)
	psamples_raw = get_psamples(rdisp, int_rtol, limits)
    u′ = get_u₀(psamples_raw)
    u′′ = u′*(1+a₂)
    g = find_zero(x -> (u′′ - get_u₀(psamples_raw, x)), (1e-20, 1e20)) 
    s = 0.0
	for (εₚ⁺, εₚ⁻, wₚ) in psamples_raw
        s += wₚ*(abs(εₚ⁻) - sqrt(εₚ⁻^2 + g^2))
    end
    s += g^2*u′′
end

export get_static_energy

θₕ(x) = x>0.0 ? 1.0 : 0.0

function get_u₀_quad(rdisp, γ²)
    α, P, μₐ, Λ = rdisp.α, rdisp.P, abs(rdisp.μ), rdisp.Λ
    f(x) = θₕ(sqrt(x^2 + γ²) - μₐ)*wfunc_gauss((x-P)/Λ)/(sqrt(x^2 + γ²)*4α*π^2)
    return quadgk_custom(f, P-Λ, P, P+Λ; rtol = 1e-6)[1]
end
function get_u₀_quad(rdisp, γ², β)
    α, P, μₐ, Λ = rdisp.α, rdisp.P, abs(rdisp.μ), rdisp.Λ
	temp_func(x, μₐ, γ², β) = 0.5*(tanh(0.5β*(μₐ + sqrt(x^2+γ²))) + tanh(0.5β*(sqrt(x^2 + γ²) - μₐ)))
	f(x) = θₕ(sqrt(x^2 + γ²) - μₐ)*wfunc_gauss((x-P)/Λ)*temp_func(x, μₐ, γ², β)/(sqrt(x^2 + γ²)*4α*π^2)
    return quadgk_custom(f, P-Λ, P, P+Λ; rtol = 1e-6)[1]
end
function calc_se_quad(rdisp, γ², u)
    α, P, μₐ, Λ = rdisp.α, rdisp.P, abs(rdisp.μ), rdisp.Λ
    f(x) = wfunc_gauss((x-P)/Λ)*(max(μₐ, abs(x)) - max(μₐ, sqrt(x^2 + γ²)))/(2α*π^2)
    return quadgk(f, P-Λ, P, P+Λ)[1] + u*γ²
end
function calc_se_quad(rdisp, γ², u, β)
	β′ = BigFloat(β)
    α, P, μₐ, Λ = rdisp.α, rdisp.P, abs(rdisp.μ), rdisp.Λ
	energy_func(x, μₐ, γ², β′) = Float64(log((cosh(β′*μₐ) + cosh(β′*x))/(cosh(β′*μₐ) + cosh(β′*sqrt(x^2 + γ²))))/β′)
    f(x) = wfunc_gauss((x-P)/Λ)*energy_func(x, μₐ, γ², β′)/(2α*π^2)
    return quadgk(f, P-Λ, P, P+Λ)[1] + u*γ²
end
function get_static_energy_quad(a₂, rdisp)
    u = get_u₀_quad(rdisp, 1.0)
    u′ = u*(1+a₂)
    γ² = find_zero(x -> (u′ - get_u₀_quad(rdisp, x)), (1e-30, 1e30))
    return calc_se_quad(rdisp, γ², u′), γ²
end

get_static_energy_quad(a₂, rdisp, β::Nothing) = get_static_energy_quad(a₂, rdisp)
function get_static_energy_quad(a₂, rdisp, β::AbstractFloat)
	u = get_u₀_quad(rdisp, 1.0)
	u′ = u*(1+a₂)
	γ² = find_zero(x -> (u′ - get_u₀_quad(rdisp, x, β)), (1e-40, 1e40))
	return calc_se_quad(rdisp, γ², u′, β), γ²
end

function get_u₀_quad0(rdisp, β)
    α, P, μₐ, Λ = rdisp.α, rdisp.P, abs(rdisp.μ), rdisp.Λ
	temp_func(x, μₐ, β) = 0.5*(tanh(0.5β*(μₐ + abs(x))) + tanh(0.5β*(abs(x) - μₐ)))
	f(x) = θₕ(abs(x) - μₐ)*wfunc_gauss((x-P)/Λ)*temp_func(x, μₐ, β)/(abs(x)*4α*π^2)
    return quadgk_custom(f, P-Λ, P, P+Λ; rtol = 1e-6)[1]
end
function get_cutoff_β(a₂, rdisp)
	u = get_u₀_quad(rdisp, 1.0)
	u′ = u*(1+a₂)
	β = find_zero(x -> (u′ - get_u₀_quad0(rdisp, x)), (1e-20, 1000.0))
	return β
end
export get_static_energy_quad, get_cutoff_β
