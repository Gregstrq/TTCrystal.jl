function save_params!(g, params::ParamsB1)
	g["N"] = params.N
	g["m"] = params.m
	g["β"] = params.β
	g["u"] = params.u
	g["u₁"] = params.u₁
end
function save_params!(g, params::ParamsNoB1)
	g["N"] = params.N
	g["m"] = params.m
	g["β"] = params.β
	g["u"] = params.u
end

function save_dispersion!(g, rdisp::ReducedDispersion)
	g["α"] = rdisp.α
	g["P"] = rdisp.P
	g["μ"] = rdisp.μ
	g["Λ"] = rdisp.Λ
end

isB1(params::ParamsB1) = true
isB1(params::ParamsNoB1) = false

struct Saver
	save_file::String
	log_file::String
	function Saver(filestring::AbstractString, params, rdisp::ReducedDispersion, Nₚ)
		save_file = filestring * ".h5"
		log_file = filestring * ".log"
		new(save_file, log_file)
	end
end
@inline Saver(fname::Tuple{AbstractString, AbstractString}, args...) = (mkpath(fname[1]); Saver(fname[1]*fname[2], args...))


function save_data(saver_o::Saver, bs, ΔF, ∂F, ∂²F, ϵ, params, rdisp, Nₚ)
	h5open(saver_o.save_file, "w") do file
        g = g_create(file, "parameters")
        g1 = g_create(g, "general")
        g2 = g_create(g, "dispersion")
        save_params!(g1, params)
        save_dispersion!(g2, rdisp)
        g["Nₚ"] = Nₚ
        g["isB1"] = isB1(params)
		file["bs"] = bs
		file["∂F"] = ∂F
		file["∂²F"] = ∂²F
		file["ΔF"] = ΔF
		file["ϵ"] = ϵ
	end
end
function log_data(saver_o::Saver, i, t, ϵ)
	io = open(saver_o.log_file, "a")
	print(io, @sprintf("%6d %12.3E %12.2E", i, t, ϵ))
	close(io)
end
function output(saver_o, bs, ΔF, ∂F, ∂²F, ϵ, t, i, params, rdisp, Nₚ)
	save_data(saver_o, bs, ΔF, ∂F, ∂²F, ϵ, params, rdisp, Nₚ)
	log_data(saver_o, i, t, ϵ)
end


function seek_minimum(params::AbstractParams, rdisp::ReducedDispersion, seed_func, Nₚ, fname, ϵ₀::Float64 = 1e-10, max_iter = 30)
	saver_o = Saver(fname, params, rdisp, Nₚ)
	bs = seed_func(params)
	∂Fs, ∂²Fs = generate_shared_cash(params)
	psamples = generate_psamples(rdisp, Nₚ)
	ΔF = 0.0
	ϵ = 0.0
	t = @elapsed data = precompute_newton_step!(∂Fs, ∂²Fs, bs, params, psamples)
    print(t, " ", ϵ, " ",  0, "\n")
	output(saver_o, bs, data..., 0.0, t, 0, params, rdisp, Nₚ)
	for i in Base.OneTo(max_iter)
		t1 = @elapsed data1 = newton_step!(bs, ∂Fs, ∂²Fs, params, psamples)
        print(t1, " ", data1[4], " ", i, "\n")
		output(saver_o, bs, data1..., t1, i, params, rdisp, Nₚ)
		ΔF, ϵ = data1[1], data1[4]
		if ϵ < ϵ₀
			return bs, data1...
		end
	end
	return bs, ΔF, ∂Fs[1], ∂²Fs[1], ϵ
end
