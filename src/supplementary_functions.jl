function get_U0(γ, β, rdisp, Nₚ)
    psamples_raw = get_psamples(rdisp, Nₚ)
    s = 0.0
    for psample in psamples_raw
        εₚ⁺, εₚ⁻, wₚ = psample
        κₚ = sqrt(εₚ⁻^2 + γ^2)
        s += wₚ*(tanh(β*0.5*(εₚ⁺+κₚ)) + tanh(β*0.5*(κₚ-εₚ⁺)))/κₚ
    end
    return 4.0/s
end

check_gamma(γ::Real, params::AbstractParams, rdisp::ReducedDispersion, Nₚ) = check_gamma(γ, params, get_samples(rdisp, Nₚ))
check_gamma(γ::Real, params::AbstractParams, psamples_raw::Vector{NTuple{3, Float64}}) = check_gamma(γ, params.β, params.u, psamples_raw)

function check_gamma(γ::Real, β::Real, u::Real, psamples_raw::Vector{NTuple{3, Float64}})
    s = 0.0
    for psample in psamples_raw
        εₚ⁺, εₚ⁻, wₚ = psample
        κₚ = sqrt(εₚ⁻^2 + γ^2)
        s += wₚ*(tanh(β*0.5*(εₚ⁺+κₚ)) + tanh(β*0.5*(κₚ-εₚ⁺)))/κₚ
    end
    return 0.25*s-u
end

function get_gamma(params::AbstractParams, rdisp::ReducedDispersion, Nₚ, range)
    psamples_raw = get_psamples(rdisp, Nₚ)
    return find_zero((γ)->check_gamma(γ, params, psamples_raw), range, Bisection())
end

function get_gamma(params::AbstractParams, psamples_raw::Vector{NTuple{3, Float64}}, range)
    return find_zero((γ)->check_gamma(γ, params.β, params.u, psamples_raw), range, Bisection())
end

function get_gamma(β::Real, u::Real, psamples_raw::Vector{NTuple{3, Float64}}, range)
    return find_zero((γ)->check_gamma(γ, β, u, psamples_raw), range, Bisection())
end

calculate_sce_rhs(k, params::AbstractParams, rdisp::ReducedDispersion, Nₚ::Int64) = calculate_sce_rhs(k, params, get_psamples(rdisp, Nₚ))


function calculate_sce_rhs(k::T, params::AbstractParams, psamples_raw::AbstractVector{NTuple{3, Float64}}) where {T<:Number}
    β, m, U0 = params.β, params.m, 1/params.u
    k² = k^2
    γ = 4*K(k²)*m/β
    kk = 4k/(1+k)^2
    s = zero(T)
    @fastmath for psample in psamples_raw
        εₚ⁺, εₚ⁻, wₚ = psample
        nₚ = γ^2*k/(εₚ⁻^2+0.25*γ^2*(1+k)^2)
		κₚ = sqrt((εₚ⁻^2+0.25*γ^2*(1-k)^2)/(1.0+0.25*(γ/εₚ⁻)^2*(1+k)^2))*Pi(nₚ, pi/2, kk)/K(kk)
        s += wₚ*(tanh(β*0.5*(εₚ⁺+κₚ)) + tanh(β*0.5*(κₚ-εₚ⁺)))*abs(εₚ⁻)/sqrt((εₚ⁻^2+0.25*γ^2*(1+k)^2)*(εₚ⁻^2+0.25*γ^2*(1-k)^2))
    end
    return 1.0 - 0.25*U0*s
end

function solve_sce_nob1(params::AbstractParams, psamples_raw::Vector{NTuple{3, Float64}}, range = (0.5, 0.99999))
    k = find_zero((x)->calculate_sce_rhs(x, params, psamples_raw), range, Bisection())
    γ = 4*K(k^2)*params.m/params.β
    return k, γ
end


function _seed_sn(params::AbstractParams, psamples_raw::Vector{NTuple{3, Float64}}, range = (0.5, 0.99999))
    β, m, N, N′ = params.β, params.m, params.N, get_length(params)
    k, γ = solve_sce_nob1(params, psamples_raw, range)
    dτ = β/(N*m) 
    τs = [dτ*(i-1) for i = 1:N]
    sns = zeros(N′)
    sns[1:N] .= k*γ*Jacobi.sn.(γ*τs, k^2)
    pin_bs!(sns, params)
    return sns
end

function _seed_sn_asymp(params::AbstractParams, psamples_raw::Vector{NTuple{3, Float64}}, range = (0.001, 1000.0))
    β, m, N, N′ = params.β, params.m, params.N, get_length(params)
    γ = get_gamma(params, psamples_raw, range)
    dτ = β/(N*m) 
    τs = [dτ*(i-1) for i = 1:N]
    sns = zeros(N′)
    N1 = div(N, 4)
    N2 = div(N, 2)
    sns[1:N1] .= γ.*tanh.(γ.*τs[1:N1])
    sns[(N1+1):N2] .= γ.*tanh.(γ.*(τs[N2+1].-τs[(N1+1):N2]))
    sns[(N2+1):N] .= -sns[1:N2]
    pin_bs!(sns, params)
    return sns
end

function get_static_fmin(β::Real, u::Real, psamples_raw::Vector{NTuple{3, Float64}}, range = (0.001, 1000.0))
	γ = get_gamma(β, u, psamples_raw, range)
	γ′ = Float128(γ)
	β′ = Float128(β)
	s = β*u*γ^2
	for (εₚ⁺, εₚ⁻, wₚ) in psamples_raw
		εₚ⁻′ = Float128(εₚ⁻)
		εₚ⁺′ = Float128(εₚ⁺)
		κₚ′ = sqrt(γ′^2 + εₚ⁻′^2)
		s -= wₚ*Float64(log((cosh(β′*κₚ′) + cosh(β′*εₚ⁺′))/(cosh(β′*εₚ⁻′) + cosh(β′*εₚ⁺′))))
	end
	return s
end

seed_sn(params::AbstractParams, rdisp::ReducedDispersion, Nₚ::Int64) = seed_sn(params, get_psamples(rdisp, Nₚ))
function seed_sn(params::AbstractParams, psamples_raw::Vector{NTuple{3, Float64}})
    try
        _seed_sn(params, psamples_raw)
    catch e
        _seed_sn_asymp(params, psamples_raw)
    end
end

function get_τs(params::AbstractParams)
    β, m, N = params.β, params.m, params.N
    dτ = β/(N*m) 
    return [dτ*(i-1) for i = 1:N]
end
function get_τs(β::Real, m::Int64, N::Int64)
    dτ = β/(N*m) 
    return [dτ*(i-1) for i = 1:N]
end

show_sn(params::AbstractParams, rdisp::ReducedDispersion, Nₚ, range = (0.5, 0.99999)) = get_τs(params), seed_sn(params, rdisp, Nₚ, range)

################################

seed_const(c, params::TTCrystal.AbstractParamsNoB1) = fill!(Vector{Float64}(undef, params.N),c)
function seed_const(c1, c2, params::AbstractParamsB1)
    N = params.N
    bs = Vector{Float64}(undef, 2N)
    bs[1:N] .= c1
    bs[(N+1):2N] .= c2
    return bs
end

################################

isB1(params::AbstractParamsB1) = Val(true)
isB1(params::AbstractParamsNoB1) = Val(false)

function show_bs(res::Optim.MultivariateOptimizationResults, ::Val{true})
    bs = Optim.minimizer(res)
    N = div(length(bs), 2)
    b0 = bs[1:N]
    b1 = bs[(N+1):2N]
    plot(b0)
    plot!(b1)
end
function show_bs(res::Optim.MultivariateOptimizationResults, ::Val{false})
    bs = Optim.minimizer(res)
    plot(bs)
end

show_bs(res, params::AbstractParams) = show_bs(res, isB1(params))

function check_derivative(bs)
    N = div(length(bs), 2)
    b0 = bs[1:N]
    b1 = bs[(N+1):2N]
    b_der = similar(b0)
    b_der[1] = b0[2]-b0[end]
    b_der[end] = b0[1]-b0[end-1]
    for i=2:(N-1)
        b_der[i] = b0[i+1] - b0[i-1]
    end
    #b_der .*= -sum(abs2, b1)/sum(abs2, b_der)
	b_der .*= b1[div(N,2)+1]/b_der[div(N,2)+1]
    plot(b_der)
    plot!(b1)
end
check_derivative(res::Optim.MultivariateOptimizationResults) = check_derivative(Optim.minimizer(res))
