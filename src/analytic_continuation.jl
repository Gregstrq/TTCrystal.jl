mutable struct S_Cash <: AbstractSharedCash
	Uxs::DArray{SMatrix{2,2,Complex{Float128}}, 1, Vector{SMatrix{2,2,Complex{Float128}}}}
	Uys::DArray{SMatrix{2,2,Complex{Float128}}, 1, Vector{SMatrix{2,2,Complex{Float128}}}}
	Uzs::DArray{SMatrix{2,2,Complex{Float128}}, 1, Vector{SMatrix{2,2,Complex{Float128}}}}
	U_cashes::DArray{SMatrix{2,2,Complex{Float128}}, 1, Vector{SMatrix{2,2,Complex{Float128}}}}
    Sps::Vector{SharedMatrix{SVector{3,Complex{Float64}}}}
    function S_Cash(N, psamples)
        Uxs = DArray([@spawnat i LinearAlgebra.ones(StaticArrays.SMatrix{2,2, Complex{Float128}}, N) for i=procs()])
        Uys = DArray([@spawnat i LinearAlgebra.ones(StaticArrays.SMatrix{2,2, Complex{Float128}}, N) for i=procs()])
        Uzs = DArray([@spawnat i LinearAlgebra.ones(StaticArrays.SMatrix{2,2, Complex{Float128}}, N) for i=procs()])
		U_cashes = DArray([@spawnat i LinearAlgebra.ones(StaticArrays.SMatrix{2,2, Complex{Float128}}, N+1) for i=procs()])
        Sps = [@fetchfrom pid SharedArrays.SharedMatrix{SVector{3,Complex{Float64}}}(N, length(localpart(psamples)); pids = procs()) for pid=procs()]
        s_cash = new(Uxs, Uys, Uzs, U_cashes, Sps)
        finalizer(free_darrays, s_cash)
    end
end


struct RealFFTOperator{T1, T2}
    ker_dft::Vector{Complex{Float64}}
    ys::Vector{Complex{Float64}}
    rp::T1
    irp::T2
    N::Int
    function RealFFTOperator(ker_dft::Vector{Complex{Float64}})
        N = length(ker_dft)
        ys = similar(ker_dft)
        rp = plan_fft(ker_dft)
        irp = plan_ifft(ker_dft)
        return new{typeof(rp), typeof(irp)}(ker_dft, ys, rp, irp, N)
    end
end

function free_darrays(s_cash::S_Cash)
    close(s_cash.Uxs)
    close(s_cash.Uys)
    close(s_cash.Uzs)
    close(s_cash.U_cashes)
end


function process_configuration(bs, params::ParamsNoB1, psamples)
    N = params.N
    s_cash = S_Cash(N, psamples)
    Sps = calculate_initial_conditions!(s_cash, bs, params, psamples)
    return Sps
end

function calculate_initial_conditions!(s_cash::S_Cash, bs, params::ParamsNoB1, psamples)
    Uxs, Uys, Uzs, U_cashes, Sps = s_cash.Uxs, s_cash.Uys, s_cash.Uzs, s_cash.U_cashes, s_cash.Sps
    @sync begin
        for wid in true_workers()
            @spawnat wid chunk_initial_conditions!(localpart(Uxs), localpart(Uys), localpart(Uzs), localpart(U_cashes), Sps[wid], bs, params, localpart(psamples))
        end
        @spawnat 1 chunk_initial_conditions!(localpart(Uxs), localpart(Uys), localpart(Uzs), localpart(U_cashes), Sps[1], bs, params, localpart(psamples))
    end
    return hcat([Sps[i] for i in eachindex(Sps)]...)
end

function chunk_initial_conditions!(Ux::T, Uy::T, Uz::T, U_cash::T, Sp::SharedMatrix{SVector{3,Complex{Float64}}}, bs, params, psamples) where T<:AbstractVector{SMatrix{2,2,Complex{Float128}}}
    for i in eachindex(psamples)
        Sp_view = view(Sp, :, i)
        p_initial_conditions!(Ux, Uy, Uz, U_cash, Sp_view, bs, params, psamples[i]...)
    end
end

function p_initial_conditions!(Ux::T, Uy::T, Uz::T, U_cash::T, Sp_view, bs, params::ParamsNoB1{Int64}, εₚ⁺, εₚ⁻, wₚ, a, b) where T<:AbstractVector{SMatrix{2,2,Complex{Float128}}}
    N = params.N
    U = compute_Us!(Ux, Uy, Uz, U_cash, bs, εₚ⁻, params.Δτ)
	λ₁, λ₂, S, S⁻¹ = custom_eigen(U)
	λ₁, λ₂, S, S⁻¹ = custom_eigen(U)
    λ₁ᵐ⁻¹ = λ₁^(m-1)
    λ₂ᵐ⁻¹ = λ₂^(m-1)
    Fₚ = a + λ₁^m + λ₂^m
	Fₚ⁻¹ = 1/Fₚ
	for i = 1:N
        Uxⁱₛ = S⁻¹*Ux[i]*S
        Uyⁱₛ = S⁻¹*Uy[i]*S
        Uzⁱₛ = S⁻¹*Uz[i]*S
        Sp_view[i] = SA[(λ₁ᵐ⁻¹*Uxⁱₛ[1,1]+λ₂ᵐ⁻¹*Uxⁱₛ[2,2]);
                        (λ₁ᵐ⁻¹*Uyⁱₛ[1,1]+λ₂ᵐ⁻¹*Uyⁱₛ[2,2]);
                        (λ₁ᵐ⁻¹*Uzⁱₛ[1,1]+λ₂ᵐ⁻¹*Uzⁱₛ[2,2])]*Fₚ⁻¹
	end
end


function p_initial_conditions!(Ux::T, Uy::T, Uz::T, U_cash::T, Sp_view, bs, params::ParamsNoB1{Nothing}, εₚ⁺, εₚ⁻, wₚ, a, b) where T<:AbstractVector{SMatrix{2,2,Complex{Float128}}}
    N = params.N
    U = compute_Us!(Ux, Uy, Uz, U_cash, bs, εₚ⁻, params.Δτ)
	λ₁, λ₂, S, S⁻¹ = custom_eigen(U)
	λ₁, λ₂, S, S⁻¹ = custom_eigen(U)
    Fₚ⁰ = log(abs(λ₁))
    if Fₚ⁰>=a
		Fₚ⁻¹ = 1/λ₁
        for i = 1:N
            Uxⁱₛ = S⁻¹*Ux[i]*S
            Uyⁱₛ = S⁻¹*Uy[i]*S
            Uzⁱₛ = S⁻¹*Uz[i]*S
            Sp_view[i] = SA[Uxⁱₛ[1,1];
                            Uyⁱₛ[1,1];
                            Uzⁱₛ[1,1]]*Fₚ⁻¹
        end
    end
end

const Σx = SMatrix{2,2,Float128}(0.0,1.0,1.0,0.0)
const Σy = SMatrix{2,2,Complex{Float128}}(0.0,1.0im,-1.0im,0.0)
const Σz = SMatrix{2,2,Float128}(1.0,0.0,0.0,-1.0)

function compute_Us!(Ux::T, Uy::T, Uz::T, U_cash::T, bs, εₚ⁻, Δτ) where T<:AbstractVector{SMatrix{2,2,Complex{Float128}}}
    N = length(bs)
    compute_U_cash!(U_cash, bs, εₚ⁻, Δτ)
	Mb = one(SMatrix{2,2, Complex{Float128}})
	for i = 1:N
		Ux[i] = U_cash[i+1]*Σx*Mb
		Uy[i] = U_cash[i+1]*Σy*Mb
		Uz[i] = U_cash[i+1]*Σz*Mb
		Mb = Br(bs, i, εₚ⁻, Δτ)*Mb
	end
	return U_cash[1]
end

#################################

K₀(τ, ω₀, W) = 0.5(exp(-ω₀*abs(τ)) + exp(ω₀*(abs(τ) - W)))/(1 - exp(-ω₀*W))/ω₀
K₀′(τ, ω₀, W) = -0.5sign(τ)*(exp(-ω₀*abs(τ)) - exp(ω₀*(abs(τ) - W)))/(1 - exp(-ω₀*W))

function initial_state(i::Int, bs, Sps, psamples_raw, τs, Δτ, W, ω₀, u₀⁻¹, a₂)
    εₚ⁻s = [psample[2] for psample in psamples_raw]
    wₚs = [psample[3]*0.5 for psample in psamples_raw]
    Sp₀ = [elt[2] for elt in Sps*wₚs]
    Spi = Sps[i,:]
    Sp = [getindex.(Spi, 1), getindex.(Spi, 2), getindex.(Spi, 3)]
    ω̃₀ = ω₀/sqrt(1+a₂)
    #####
    b₀ = bs[i] + 0im
    K̃′ = dot(K₀′.(τs[i].-τs, ω̃₀, W), bs)*Δτ
    b₀′ = 2sum(wₚs.*εₚ⁻s.*Sp[1])*u₀⁻¹ - im*a₂*ω̃₀^2*K̃′
    return sTimeRHS(ω₀^2, u₀⁻¹, u₀⁻¹/(1+a₂)*ω₀^2, εₚ⁻s, wₚs), ArrayPartition([b₀, b₀′], Sp...)
end

function initial_state(i::Int, bs, Sps, psamples_raw, τs, Δτ, W, u₀⁻¹)
    εₚ⁻s = [psample[2] for psample in psamples_raw]
    wₚs = [psample[3]*0.5 for psample in psamples_raw]
    Sp₀ = Sps*wₚs
    Sp = Sps[i,:]
    #print(bs[i] - u₀⁻¹*Sp₀[i][2], "\n")
    return sTimeRHS_simple(u₀⁻¹, εₚ⁻s, wₚs), ArrayPartition([bs[i]+0.0im], Sp)
end

function initial_state′(i::Int, bs, Sps, psamples_raw, τs, Δτ, W, u₀⁻¹)
    εₚ⁻s = [psample[2] for psample in psamples_raw]
    wₚs = [psample[3]*0.5 for psample in psamples_raw]
    Sp₀ = Sps*wₚs
    Spi = Sps[i,:]
    Sp = [getindex.(Spi, 1), getindex.(Spi, 2), getindex.(Spi, 3)]
    print(bs[i] - u₀⁻¹*Sp₀[i][2], "\n")
    return sTimeRHS_simple′(u₀⁻¹, εₚ⁻s, wₚs), ArrayPartition([bs[i]+0.0im], Sp...)
end

function initial_state(bs::AbstractVector, Sps, psamples_raw, τs, Δτ, W, ω₀, u₀⁻¹, a₂)
    εₚ⁻s = [psample[2] for psample in psamples_raw]
    wₚs = [psample[3]*0.5 for psample in psamples_raw]
    Sp₀ = [elt[2] for elt in Sps*wₚs]
    Kc = similar(bs)
    Ks = similar(bs, Complex{eltype(bs)})
    @inbounds for i in eachindex(bs)
        Kc[i] = dot(K₀.(τs[i].-τs, ω₀, W), Sp₀)*0.5ω₀^2*Δτ*u₀⁻¹*a₂/(1+a₂)
        Ks[i] = dot(K₀′.(τs[i].-τs, ω₀, W), Sp₀)*0.5ω₀^2*Δτ*u₀⁻¹*a₂/(1+a₂)*im
    end
    return fTimeRHS(Kc, Ks, ω₀, u₀⁻¹, εₚ⁻s, wₚs), ArrayPartition(bs.+0.0im, deepcopy(Sps))
end

function initial_state2(bs::AbstractVector, Sps, psamples_raw, τs, Δτ, W, ω₀, u₀⁻¹, a₂)
    εₚ⁻s = [psample[2] for psample in psamples_raw]
    wₚs = [psample[3]*0.5 for psample in psamples_raw]
    Sp₀ = [elt[2] for elt in Sps*wₚs]
    K = construct_FFT_operator(length(bs), Δτ, W, ω₀, u₀⁻¹, a₂)
    return fTimeRHS2(K, ω₀, u₀⁻¹, εₚ⁻s, wₚs), ArrayPartition(bs.+0.0im, deepcopy(Sps))
end

struct sTimeRHS
    ω₀²::Float64
    u₀⁻¹::Float64
    ũ₀⁻¹::Float64
    εₚ⁻s::Vector{Float64}
    wₚs::Vector{Float64}
    b_cache1::Vector{Complex{Float64}}
    b_cache2::Vector{Complex{Float64}}
    sTimeRHS(ω₀², u₀⁻¹, ũ₀⁻¹, εₚ⁻s, wₚs) = new(ω₀², u₀⁻¹, ũ₀⁻¹, εₚ⁻s, wₚs, zeros(Complex{Float64}, Threads.nthreads()), zeros(Complex{Float64}, Threads.nthreads()))
end

struct fTimeRHS
    Kc::Vector{Float64}
    Ks::Vector{Complex{Float64}}
    ω₀::Float64
    u₀⁻¹::Float64
    εₚ⁻s::Vector{Float64}
    wₚs::Vector{Float64}
    bs_cache::Matrix{Complex{Float64}}
    function fTimeRHS(Kc, Ks, ω₀, u₀⁻¹, εₚ⁻s, wₚs)
        bs_cache = Matrix{Complex{Float64}}(undef, length(Kc), Threads.nthreads())
        return new(Kc, Ks, ω₀, u₀⁻¹, εₚ⁻s, wₚs, bs_cache)
    end
end

struct fTimeRHS2{T1, T2}
    K::RealFFTOperator{T1,T2}
    ω₀::Float64
    u₀⁻¹::Float64
    εₚ⁻s::Vector{Float64}
    wₚs::Vector{Float64}
    bs_cache::Matrix{Complex{Float64}}
    function fTimeRHS2(K::RealFFTOperator{T1,T2}, ω₀, u₀⁻¹, εₚ⁻s, wₚs) where {T1,T2}
        bs_cache = Matrix{Complex{Float64}}(undef, K.N, Threads.nthreads())
        return new{T1,T2}(K, ω₀, u₀⁻¹, εₚ⁻s, wₚs, bs_cache)
    end
end

mutable struct sTimeRHS_simple
    u₀⁻¹::Float64
    εₚ⁻s::Vector{Float64}
    wₚs::Vector{Float64}
    b_cache::Vector{Complex{Float64}}
    sTimeRHS_simple(u₀⁻¹, εₚ⁻s, wₚs) = new(u₀⁻¹, εₚ⁻s, wₚs, Vector{Complex{Float64}}(undef, Threads.nthreads()))
end

struct sTimeRHS_simple′
    u₀⁻¹::Float64
    εₚ⁻s::Vector{Float64}
    wₚs::Vector{Float64}
    b_cache::Vector{Complex{Float64}}
    sTimeRHS_simple′(u₀⁻¹, εₚ⁻s, wₚs) = new(u₀⁻¹, εₚ⁻s, wₚs, Vector{Complex{Float64}}(undef, Threads.nthreads()))
end

function (rhs::sTimeRHS)(du::T, u::T, p, t) where {T <: ArrayPartition} 
    b = u.x[1]
    Sp_x = u.x[2]
    Sp_y = u.x[3]
    Sp_z = u.x[4]
    db = du.x[1]
    dSp_x = du.x[2]
    dSp_y = du.x[3]
    dSp_z = du.x[4]
    b_cache1 = rhs.b_cache1
    b_cache2 = rhs.b_cache2
    wₚs = rhs.wₚs
    εₚ⁻s = rhs.εₚ⁻s
    ω₀² = rhs.ω₀²
    u₀⁻¹ = rhs.u₀⁻¹
    ũ₀⁻¹ = rhs.ũ₀⁻¹
    ##################
    fill!(b_cache1, 0.0)
    fill!(b_cache2, 0.0)
    Threads.@threads for i = eachindex(Sp_x)
        dSp_x[i] = -2(b[1]*Sp_z[i] + εₚ⁻s[i]*Sp_y[i])
        dSp_y[i] = 2εₚ⁻s[i]*Sp_x[i]
        dSp_z[i] = 2b[1]*Sp_x[i]
        b_cache1[Threads.threadid()] -= 4wₚs[i]*εₚ⁻s[i]*(b[1]*Sp_z[i] + εₚ⁻s[i]*Sp_y[i])
        b_cache2[Threads.threadid()] += wₚs[i]*Sp_y[i]
    end
    db[1] = b[2]
    db[2] = u₀⁻¹*sum(b_cache1) + ũ₀⁻¹*sum(b_cache2) - ω₀²*b[1]
end

function (rhs::fTimeRHS)(du::T, u::T, p, t) where {T<:ArrayPartition}
    bs = u.x[1]
    dbs = du.x[1]
    Sps = u.x[2]
    dSps = du.x[2]
    bs_cache = rhs.bs_cache
    fill!(bs_cache, 0.0)
    Threads.@threads for j = 1:size(Sps, 2)
        for i = 1:size(Sps, 1)
            Bₚ = SVector{3,Complex{Float64}}(0.0, -bs[i], rhs.εₚ⁻s[j])
            dSps[i, j] = 2*cross_product(Bₚ, Sps[i, j])
            bs_cache[i, Threads.threadid()] += rhs.wₚs[j]*dSps[i, j][2]
        end
    end
    @views dbs .= bs_cache[:, 1]
    for i = 2:size(bs_cache, 2)
        @views dbs .+= bs_cache[:, i]
    end
    dbs .*= rhs.u₀⁻¹
    @. dbs += rhs.Kc*sin(rhs.ω₀*t) - rhs.Ks*cos(rhs.ω₀*t)
end

function (rhs::fTimeRHS2)(du::T, u::T, p, t) where {T<:ArrayPartition}
    bs = u.x[1]
    dbs = du.x[1]
    Sps = u.x[2]
    dSps = du.x[2]
    bs_cache = rhs.bs_cache
    fill!(bs_cache, 0.0)
    Threads.@threads for j = 1:size(Sps,2)
        for i = 1:size(Sps, 1)
            Bₚ = SVector{3,Complex{Float64}}(0.0, -bs[i], rhs.εₚ⁻s[j])
            dSps[i, j] = 2*cross_product(Bₚ, Sps[i, j])
            bs_cache[i, Threads.threadid()] += rhs.wₚs[j]*dSps[i, j][2]
        end
    end
    for j = 2:size(bs_cache, 2)
        @views bs_cache[:,1] .+= bs_cache[:, j]
    end
    mul!(dbs, rhs.K, view(bs_cache, :, 1))
    @views dbs  .+= rhs.u₀⁻¹*bs_cache[:,1]
end

function (rhs::sTimeRHS_simple)(du::T, u::T, p, t) where {T<:ArrayPartition{Complex{Float64},Tuple{Array{Complex{Float64},1},Array{SArray{Tuple{3},Complex{Float64},1,3},1}}}}
    b = u[1]
    Sp = u.x[2]
    dSp = du.x[2]
    b_cache = rhs.b_cache
    fill!(b_cache, 0.0)
    Threads.@threads for i = 1:length(Sp)
    #for i = 1:length(Sp)
    Bₚ = SVector{3,Complex{Float64}}(0.0, -b, rhs.εₚ⁻s[i])
        dSp[i] = 2*cross_product(Bₚ, Sp[i])
        b_cache[Threads.threadid()] += rhs.wₚs[i]*dSp[i][2]
    end
    du[1] = rhs.u₀⁻¹*sum(b_cache)
end

function (rhs::sTimeRHS_simple′)(du::T, u::T, p, t) where {T<:ArrayPartition}
    b = u[1]
    Sp_x = u.x[2]
    Sp_y = u.x[3]
    Sp_z = u.x[4]
    dSp_x = du.x[2]
    dSp_y = du.x[3]
    dSp_z = du.x[4]
    b_cache = rhs.b_cache
    fill!(b_cache, 0.0)
    Threads.@threads for i = eachindex(Sp_x)
    #for i = 1:length(Sp)
        dSp_x[i] = -2(b[1]*Sp_z[i] + rhs.εₚ⁻s[i]*Sp_y[i])
        dSp_y[i] = 2rhs.εₚ⁻s[i]*Sp_x[i]
        dSp_z[i] = 2b[1]*Sp_x[i]
        b_cache[Threads.threadid()] += rhs.wₚs[i]*dSp_y[i]
    end
    du[1] = rhs.u₀⁻¹*sum(b_cache)
end

function cross_product(v1::T, v2::T) where {T<:SVector{3, Complex{Float64}}}
    return SVector{3,Complex{Float64}}(v1[2]*v2[3] - v1[3]*v2[2], v1[3]*v2[1] - v1[1]*v2[3], v1[1]*v2[2] - v1[2]*v2[1])
end

export process_configuration, initial_state

import Base.isnan

isnan(v::StaticArray) = any(isnan, v)


function LinearAlgebra.mul!(y::AbstractVector{Complex{Float64}}, O::RealFFTOperator, x::AbstractVector{Complex{Float64}})
    @assert length(y) == length(x) == O.N
    mul!(O.ys, O.rp, x)
    O.ys .*= O.ker_dft
    mul!(y, O.irp, O.ys)
end

function construct_FFT_operator(N, Δτ, W, ω₀, u₀⁻¹, a₂)
    ker_dft = zeros(Complex{Float64}, N)
    ker_dft .= full_kernel_dft.(collect(0:N-1), ω₀*W, N)*0.5ω₀*Δτ*u₀⁻¹*a₂/(1+a₂)
    return RealFFTOperator(ker_dft)
end


function retarded_integral!(sin_integral, cos_integral, bₜs, ts, Δt, ω̃₀)
    sin_integral[1] = 0
    cos_integral[1] = 0
    st = sin(ω̃₀*Δt)
    ct = cos(ω̃₀*Δt)
    Δtₕ = Δt/2
    for i = 2:length(sin_integral)
        sin_integral[i] = sin_integral[i-1]*ct + cos_integral[i-1]*st + st*bₜs[i-1]*Δtₕ
        cos_integral[i] = cos_integral[i-1]*ct - sin_integral[i-1]*st + (bₜs[i] + ct*bₜs[i-1])*Δtₕ
    end
    return sin_integral
end
function retarded_integral(bₜs, ts, Δt, ω̃₀)
    sin_integral = similar(bₜs)
    cos_integral = similar(bₜs)
    return retarded_integral!(sin_integral, cos_integral, bₜs, ts, Δt, ω̃₀)
end

function ie_rhs!(b′ₜs, Sp, sin_integral, cos_integral, b_cache, bₜs, ts, εₚ⁻s, wₚs, K, K′, Δt, ω̃₀, u₀⁻¹, a₂)
    Sp = deepcopy(Sp)
    retarded_integral!(sin_integral, cos_integral, bₜs, ts, Δt, ω̃₀)
    a′ = a₂*ω̃₀
    a′′ = a₂*ω̃₀^2
    b′ₜs[1] = sum(wₚs.*getindex.(Sp, 2))*u₀⁻¹
    for i = 2:length(b′ₜs)
        fill!(b_cache, 0)
        Threads.@threads for j = eachindex(Sp)
            Bₚ = SVector{3,Complex{Float64}}(0.0, -bₜs[i-1], εₚ⁻s[j])
            Sp[j] = Sp[j] + 2Δt*cross_product(Bₚ, Sp[j])
            b_cache[Threads.threadid()] += wₚs[j]*Sp[j][2]
        end
        b′ₜs[i] = sum(b_cache)*u₀⁻¹
    end
    b′ₜs .-= (sin_integral .+ (K′*im).*sin.(ω̃₀.*ts)).*a′ .+ cos.(ω̃₀.*ts).*(a′′*K)
    return b′ₜs, bₜs
end

function setup_ie_params(sol, i::Int, bs, Sps, psamples_raw, τs, Δτ, W, ω₀, u₀⁻¹, a₂)
    ts = sol.t
    bₜs = map(first, sol.u)
    Δt = ts[end]/(length(ts)-1)
    εₚ⁻s = [psample[2] for psample in psamples_raw]
    wₚs = [psample[3]*0.5 for psample in psamples_raw]
    Sp = Sps[i, :]
    ω̃₀ = ω₀/sqrt(1+a₂)
    K = dot(K₀.(τs[i].-τs, ω̃₀, W), bs)*Δτ
    K′ = dot(K₀′.(τs[i].-τs, ω̃₀, W), bs)*Δτ
    sin_integral = similar(bₜs)
    cos_integral = similar(bₜs)
    b′ₜs = similar(bₜs)
    b_cache = similar(bₜs, Threads.nthreads())
    return b′ₜs, Sp, sin_integral, cos_integral, b_cache, bₜs, ts, εₚ⁻s, wₚs, K, K′, Δt, ω̃₀, u₀⁻¹, a₂
end

export retarded_integral!, retarded_integral, ie_rhs!, setup_ie_params

