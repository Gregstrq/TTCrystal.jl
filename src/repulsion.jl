abstract type AbstractRepulsion end
abstract type AbstractRepulsionType end

kernel_dft(r::Int64, W::Float64, N::Int64) = r==0 ? 0.25W : 0.25W*exp(1im*π*(r/N - 0.5))/sin(π*r/N)

kernel_func(τ′, τ′′, W, c) = (0.25W - 0.5*abs(τ′-τ′′) - W*(0.5-τ′/W)*(0.5-τ′′/W))*c

struct GaugeRepulsion <: AbstractRepulsion
    fs::Vector{Float64}
    M::Matrix{Float64}
    N::Int64
    function GaugeRepulsion(params::AbstractParams, ω₀::Float64)
        W, N, u, Δτ = params.W, params.N, params.u, params.Δτ
        c = 2u*ω₀^2*Δτ^2
        fs = zeros(Float64, N)
        M = zeros(Float64, N, N)
        τs = Δτ*collect(0:N-1)
        for j = 1:N
            for i = 1:N
                M[i,j] = kernel_func(τs[i], τs[j], W, c)
            end
        end
        #print(c, "\n")
        new(fs, M, N)
    end
end

struct GaugeRepulsion2 <: AbstractRepulsion
	fs::Vector{Float64}
	ηs::Vector{Float64}
	N::Int64
	c::Float64
	function GaugeRepulsion2(params::AbstractParams, ω₀::Float64)
        W, N, u, Δτ = params.W, params.N, params.u, params.Δτ
        c = 2u*ω₀^2*Δτ^3
        #print(c, "\n")
        fs = zeros(Float64, N)
		ηs = zeros(Float64, N)
		new(fs, ηs, N, c)
	end
end

function compute_repulsion!(grep::GaugeRepulsion, bs::AbstractVector)
    fs, M, N = grep.fs, grep.M, grep.N
    b0 = view(bs, 1:N)
    mul!(fs, M, b0)
    return 0.5*dot(fs,b0), fs
end

function compute_repulsion!(grep::GaugeRepulsion2, bs::AbstractVector)
	fs, ηs, N, c = grep.fs, grep.ηs, grep.N, grep.c
	##
	ηs[1] = N*bs[1]
	@inbounds for i = 1:N-1
		ηs[1] += i*bs[i+1]
	end
	ηs[1] /= N
	for i = 2:N
		ηs[i] = ηs[i-1] + bs[i]
	end
	##
	fs[N] = ηs[N]*c
	for i = N-1:-1:2
		fs[i] = fs[i+1] + c*ηs[i]
	end
	##
	return 0.5*dot(ηs, ηs)*c, fs
end

export GaugeRepulsion, GaugeRepulsion2, compute_repulsion!

full_kernel_dft(r::Int64, α′W::Float64, N::Int64) = sinh(α′W/N)/(cosh(α′W/N) - cos(2π*r/N))

full_kernel_func(τ′, τ′′, α′, W) = cosh(α′*(0.5W - abs(τ′ - τ′′)))/sinh(0.5W*α′)
full_kernel_func2(τ′, τ′′, α′, W) = (exp(-α′*abs(τ′ - τ′′)) + exp(-α′*(W - abs(τ′ - τ′′))))/(1-exp(-α′*W))


struct FullRepulsionFFT{TP1, TP2} <: AbstractRepulsion
    W::Float64
    N::Int64
    Nhalf::Int64
    c::Float64
    ker_dft::Vector{Complex{Float64}}
    rp::TP1
    irp::TP2
    ys::Vector{Complex{Float64}}
    fs::Vector{Float64}
    function FullRepulsionFFT(params::AbstractParams, α₀::Float64, a₂::Float64)
        W, N, u, Δτ = params.W, params.N, params.u, params.Δτ
        α′ = α₀/sqrt(1 + a₂)
        c = u*a₂*α′*Δτ^2 
        Nhalf = div(N,2)+1 
        ts = 0.5.-collect(0:N-1)*Δτ
        ker_dft = zeros(Complex{Float64}, Nhalf)
        ker_dft .= full_kernel_dft.(collect(0:Nhalf-1), α′*W, N)
        ys = zeros(Complex{Float64}, Nhalf)
        fs = zeros(Float64, N)
        rp = plan_rfft(ts)
        irp = plan_irfft(ker_dft, N)
        new{typeof(rp), typeof(irp)}(W, N, Nhalf, c, ker_dft, rp, irp, ys, fs)
    end
end

function compute_repulsion!(grep::FullRepulsionFFT, bs::AbstractVector)
    N, W, c = grep.N, grep.W, grep.c
    fs = grep.fs
    b0 = view(bs, 1:N)
    mul!(grep.ys, grep.rp, b0)
    grep.ys .*= grep.ker_dft
    mul!(fs, grep.irp, grep.ys)
    #fs .= (fs .- ts.*(W*av)).*2uω₀²*Δτ^2
    fs .= fs.*c
    return 0.5*dot(fs, b0), fs
end

struct FullRepulsion <: AbstractRepulsion
    fs::Vector{Float64}
    M::Matrix{Float64}
    N::Int64
    function FullRepulsion(params::AbstractParams, α₀::Float64, a₂::Float64)
        W, N, u, Δτ = params.W, params.N, params.u, params.Δτ
        α′ = α₀/sqrt(1 + a₂)
        c = u*a₂*α′*Δτ^2 
        fs = zeros(Float64, N)
        M = zeros(Float64, N, N)
        τs = Δτ*collect(0:N-1)
        for j = 1:N
            for i = 1:N
                M[i,j] = full_kernel_func2(τs[i], τs[j], α′, W)*c
            end
        end
        print(c, "\n")
        new(fs, M, N)
    end
end

function compute_repulsion!(grep::FullRepulsion, bs::AbstractVector)
    fs, M, N = grep.fs, grep.M, grep.N
    b0 = view(bs, 1:N)
    mul!(fs, M, b0)
    return 0.5*dot(fs,b0), fs
end

export FullRepulsionFFT, FullRepulsion

struct Gauge <: AbstractRepulsionType
    ω₀::Float64
end
struct Gauge2 <: AbstractRepulsionType
    ω₀::Float64
end
struct FullFFT <: AbstractRepulsionType
    α₀::Float64
    a₂::Float64
end
struct Full <: AbstractRepulsionType
    α₀::Float64
    a₂::Float64
end

construct_repulsion(reptyp::Gauge, params) = GaugeRepulsion(params, reptyp.ω₀)
construct_repulsion(reptyp::Gauge2, params) = GaugeRepulsion2(params, reptyp.ω₀)
construct_repulsion(reptyp::FullFFT, params) = FullRepulsionFFT(params, reptyp.α₀, reptyp.a₂)
construct_repulsion(reptyp::Full, params) = FullRepulsion(params, reptyp.α₀, reptyp.a₂)

export Gauge, Gauge2, FullFFT, Full, construct_repulsion
