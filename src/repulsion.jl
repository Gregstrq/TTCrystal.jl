abstract type AbstractRepulsion end

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
