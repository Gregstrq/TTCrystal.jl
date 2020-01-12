function save_params!(g, params::ParamsB1)
	g["N"] = params.N
	g["m"] = params.m
	g["β"] = params.β
	g["u"] = params.u
	g["u₁"] = params.u₁
end
function save_params!(g, params::ParamsB1_pinned)
	g["N"] = params.N
	g["m"] = params.m
	g["β"] = params.β
	g["u"] = params.u
	g["u₁"] = params.u₁
	g["i_pinned"] = params.i_pinned
end
function save_params!(g, params::ParamsNoB1)
	g["N"] = params.N
	g["m"] = params.m
	g["β"] = params.β
	g["u"] = params.u
end
function save_params!(g, params::ParamsNoB1_pinned)
	g["N"] = params.N
	g["m"] = params.m
	g["β"] = params.β
	g["u"] = params.u
	g["i_pinned"] = params.i_pinned
end

function save_dispersion!(g, rdisp::ReducedDispersion)
	g["α"] = rdisp.α
	g["P"] = rdisp.P
	g["μ"] = rdisp.μ
	g["Λ"] = rdisp.Λ
end

isB1(params::AbstractParamsB1) = true
isB1(params::AbstractParamsNoB1) = false

struct Saver
	save_file::String
	log_file::String
	function Saver(filestring::AbstractString, params, rdisp::ReducedDispersion, Nₚ, max_iter)
		save_file = filestring * ".h5"
		init_file(save_file, params, rdisp, Nₚ, max_iter)
		log_file = filestring * ".log"
		new(save_file, log_file)
	end
end
@inline Saver(fname::Tuple{AbstractString, AbstractString}, args...) = (mkpath(fname[1]); Saver(fname[1]*fname[2], args...))

function init_file(save_file::AbstractString, params::AbstractParams, rdisp::ReducedDispersion, Nₚ::Int64, max_iter::Int64)
	max_iter = max_iter + 1
	h5open(save_file, "w") do file
        g = g_create(file, "parameters")
        g1 = g_create(g, "general")
        g2 = g_create(g, "dispersion")
        save_params!(g1, params)
        save_dispersion!(g2, rdisp)
        g["Nₚ"] = Nₚ
        g["isB1"] = isB1(params)
		#####
		N′ = get_length(params)
		file["bss"] = zeros(max_iter, N′)
		file["∂F"] = zeros(N′)
		file["∂²F"] = zeros(N′, N′)
		file["ΔFs"] = zeros(max_iter)
		file["ϵs"] = zeros(max_iter)
	end
end


function save_data(saver_o::Saver, bs, ΔF, ∂F, ∂²F, ϵ, params, rdisp, Nₚ, iter)
	iter = iter+1
	h5open(saver_o.save_file, "r+") do file
		file["bss"][iter, :] = bs
		file["∂F"][:] = ∂F
		file["∂²F"][:,:] = ∂²F
		file["ΔFs"][iter] = ΔF
		file["ϵs"][iter] = ϵ
	end
end
function log_data(saver_o::Saver, i, t, ϵ)
	io = open(saver_o.log_file, "a")
	print(io, @sprintf("%6d %12.3E %12.2E\n", i, t, ϵ))
	close(io)
end
function output(saver_o, bs, ΔF, ∂F, ∂²F, ϵ, t, i, params, rdisp, Nₚ, iter)
	save_data(saver_o, bs, ΔF, ∂F, ∂²F, ϵ, params, rdisp, Nₚ, iter)
	log_data(saver_o, i, t, ϵ)
end


function seek_minimum(params::AbstractParams, rdisp::ReducedDispersion, seed_func, Nₚ, fname, ϵ₀::Float64 = 1e-10, max_iter = 30)
	print(max_iter, "\n")
	saver_o = Saver(fname, params, rdisp, Nₚ, max_iter)
	bs = seed_func(params)
	∂Fs, ∂²Fs = generate_shared_cash(params)
	psamples = generate_psamples(rdisp, Nₚ)
	ΔF = 0.0
	ϵ = 0.0
	t = @elapsed data = precompute_newton_step!(∂Fs, ∂²Fs, bs, params, psamples)
    print(t, " ", ϵ, " ",  0, "\n")
	output(saver_o, bs, data..., 0.0, t, 0, params, rdisp, Nₚ, 0)
	for i in Base.OneTo(max_iter)
		t1 = @elapsed data1 = newton_step!(bs, ∂Fs, ∂²Fs, params, psamples)
        print(t1, " ", data1[4], " ", i, "\n")
		output(saver_o, bs, data1..., t1, i, params, rdisp, Nₚ, i)
		ΔF, ϵ = data1[1], data1[4]
		if ϵ < ϵ₀
			return bs, data1...
		end
	end
	return bs, ΔF, ∂Fs[1], ∂²Fs[1], ϵ
end

function compute_ΔFs(bs::AbstractVector{Float64}, params::AbstractParams, rdisp::ReducedDispersion, Nₚ::Int64)
	psamples = generate_psamples(rdisp, Nₚ)
	ΔFs_chunked = [SharedVector{Float64}(length(psamples[wid])) for wid in eachindex(psamples)]
    @sync begin
        for wid in true_workers()
			@spawnat wid process_ΔF_chunk!(ΔFs_chunked[wid], bs, params, psamples[wid])
        end
		if nworkers()>1
			@spawnat 1 process_ΔF_chunk!(ΔFs_chunked[1], bs, params, psamples[1])
		end
    end
	return vcat(ΔFs_chunked...)
end


function compute_ΔFs_sequential(bs::AbstractVector{Float64}, params::AbstractParams, rdisp::ReducedDispersion, Nₚ::Int64)
	psamples = get_psamples(rdisp, Nₚ)
	ΔFs = Vector{Float64}(undef, length(psamples))
    process_ΔF_chunk!(ΔFs, bs, params, psamples)
	return ΔFs
end

sum_free_energy(bs::AbstractVector{Float64}, ΔFs::AbstractVector{Float64}, params::AbstractParams) = final_part(bs, params) - sum(ΔFs)

function final_part(bs, params::AbstractParamsB1)
	N, m, β, u, u₁ = params.N, params.m, params.β, params.u, params.u₁
	mΔτ = β/N
	return mΔτ*(u*sum(abs2, bs[1:N]) + u₁*sum(abs2, bs[(N+1):2N]))
end

function final_part(bs, params::AbstractParamsNoB1)
	N, m, β, u = params.N, params.m, params.β, params.u
	mΔτ = β/N
	return mΔτ*u*sum(abs2, bs)
end

compute_free_energy(bs::AbstractVector{Float64}, params::AbstractParams, rdisp::ReducedDispersion, Nₚ::Int64) = sum_free_energy(bs, compute_ΔFs_sequential(bs, params, rdisp, Nₚ), params)
