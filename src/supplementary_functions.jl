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

check_gamma(γ, params, rdisp, Nₚ) = check_gamma(γ, params, get_samples(rdisp, Nₚ))
function check_gamma(γ, params::AbstractParams, psamples_raw)
    β = params.β
    U0 = 1/params.u
    s = 0.0
    for psample in psamples_raw
        εₚ⁺, εₚ⁻, wₚ = psample
        κₚ = sqrt(εₚ⁻^2 + γ^2)
        s += wₚ*(tanh(β*0.5*(εₚ⁺+κₚ)) + tanh(β*0.5*(κₚ-εₚ⁺)))/κₚ
    end
    return U0*0.25*s-1.0
end

function get_gamma(params, rdisp, Nₚ, range = (0.001, 1000.0))
    psamples_raw = get_psamples(rdisp, Nₚ)
    return find_zero((γ)->check_gamma(γ, params, psamples_raw), range, Bisection())
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
        κₚ = abs(εₚ⁻)*sqrt((εₚ⁻^2+0.25*γ^2*(1-k)^2)/(εₚ⁻^2+0.25*γ^2*(1+k)^2))*Pi(nₚ, pi/2, kk)/K(kk)
        s += wₚ*(tanh(β*0.5*(εₚ⁺+κₚ)) + tanh(β*0.5*(κₚ-εₚ⁺)))*abs(εₚ⁻)/sqrt((εₚ⁻^2+0.25*γ^2*(1+k)^2)*(εₚ⁻^2+0.25*γ^2*(1-k)^2))
    end
    return 1.0 - 0.25*U0*s
end

function solve_sce_nob1(params::AbstractParams, rdisp::ReducedDispersion, Nₚ, range = (0.5, 0.99999))
    psamples_raw = get_psamples(rdisp, Nₚ)
    k = find_zero((x)->calculate_sce_rhs(x, params, psamples_raw), range, Bisection())
    γ = 4*K(k^2)*params.m/params.β
    return k, γ
end


function _seed_sn(params::AbstractParams, rdisp::ReducedDispersion, Nₚ, range = (0.5, 0.99999))
    β, m, N, N′ = params.β, params.m, params.N, get_length(params)
    k, γ = solve_sce_nob1(params, rdisp, Nₚ, range)
    dτ = β/(N*m) 
    τs = [dτ*(i-1) for i = 1:N]
    sns = zeros(N′)
    sns[1:N] .= k*γ*Jacobi.sn.(γ*τs, k^2)
    pin_bs!(sns, params)
    return sns
end

function _seed_sn_asymp(params::AbstractParams, rdisp::ReducedDispersion, Nₚ, range = (0.001, 1000.0))
    β, m, N, N′ = params.β, params.m, params.N, get_length(params)
    γ = get_gamma(params, rdisp, Nₚ, range)
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

function seed_sn(params::AbstractParams, rdisp::ReducedDispersion, Nₚ)
    try
        _seed_sn(params, rdisp, Nₚ)
    catch e
        _seed_sn_asymp(params, rdisp, Nₚ)
    end
end

function get_τs(params::AbstractParams)
    β, m, N = params.β, params.m, params.N
    dτ = β/(N*m) 
    return [dτ*(i-1) for i = 1:N]
end

show_sn(params::AbstractParams, rdisp::ReducedDispersion, Nₚ, range = (0.5, 0.99999)) = get_τs(params), seed_sn(params, rdisp, Nₚ, range)
