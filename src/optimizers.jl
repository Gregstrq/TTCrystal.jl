mutable struct Saver
	dirname::AbstractString
    iter::Int64
	Saver(dirname::AbstractString = pwd()) = (mkpath(dirname); new(dirname, 1))
end

macro stash!(group, args...)
    exs = [:($(esc(group))[$(string(arg))] = $(esc(arg))) for arg in args]
    return Expr(:block, exs...)
end


function save_data(saver_o, P, Λ, static_fmin, m_opt, f_min, bs_opt, τs)
	jldopen("$(saver_o.dirname)/dset_$(saver_o.iter).jld2", "w") do file
        @stash!(file, P, Λ, static_fmin, m_opt, f_min, bs_opt, τs)
    end
    saver_o.iter += 1
end

function save_data(saver_o, P, Λ, μ, α, Nₚ, β, Δτ, a, opt, static_f_min, m_opt, f_min, bs_opt, τs)
    jldopen("$(saver_o.dirname)/dset_$(saver_o.iter).jld2", "w") do file
        @stash!(file, P, Λ, μ, α, Nₚ, β, Δτ, a, opt, static_f_min, m_opt, f_min, bs_opt, τs)
    end
    saver_o.iter += 1
end


function process_stacktrace(st)
	str = "\t[1]\t"*string(st[1])
	for i = 2:length(st)
		str = str * "\n\t[$i]\t$(string(st[i]))"
	end
	return str
end

#########################################################

const IRange{T} = Union{AbstractVector{T}, T}

function get_optimum(W::Float64, m::Union{Int64, Nothing}, N::Int64, a::Float64, reptyp::AbstractRepulsionType, rdisp::ReducedDispersion, opt::Optim.Options, int_rtol::Real=1e-6, limits::Int64 = 200)
	t′ = -time()
	psamples_raw = get_psamples(rdisp, int_rtol, limits)
	psamples = widen(psamples_raw, W, m)|>separate_psamples
	Nₚ = length(psamples_raw)
	@info "Number of ε⁻ points is: $Nₚ.\n\n"
    params = ParamsB1(N, m, W, a, psamples_raw)
    params0 = ParamsNoB1(params.N, params.m, params.W, params.u)
    ####
    f_nm = precompute_step(zeros(params.N), params0, psamples)
    @info "Free energy of normal metal: $(f_nm).\n\n"
	####
    τs = get_τs(params0)
    bs0 = get_bs(params0)
	d0 = construct_objective(params0, psamples, bs0, reptyp)
	results0 = optimize(d0, bs0, LBFGS(m=120, linesearch = LineSearches.MoreThuente()), opt)
	f_seed = Optim.minimum(results0)
	bs_seed = Optim.minimizer(results0)
    #bs_seed′ = zeros(get_length(params))
    #bs_seed′[1:params.N] .= bs_seed
	#t′ += time()
	@info "Free energy of without b1 is $f_seed. Calculation took $t′ s.\n\n"
	####
	#t′ = -time()
	#d = construct_objective(params, psamples, bs_seed′, ω₀)
	#results = optimize(d, bs_seed′, LBFGS(m=120, linesearch = LineSearches.MoreThuente()), opt)
	#f_final = Optim.minimum(results)
	#bs_final = Optim.minimizer(results)
	#t′ += time()
	#@info "Free energy with b1 is $f_final. Calculation took $t′ s.\n\n"
    return f_nm, f_seed, bs_seed, τs
end

function km_walkthrough_repul(W_range::AbstractVector, m_range::AbstractVector, N::Int64, a::Float64, reptyp_range::IRange{T}, rdisp::ReducedDispersion, saver_o::Saver, opt::Optim.Options, int_rtol::Float64 = 1e-7, limits::Int64 = 200, i₀::Int64 = 1) where {T<:AbstractRepulsionType}
	saver_o.iter = i₀
	tups = vec([tup for tup in product(W_range, m_range, reptyp_range)])[i₀:end]
    N_tot = length(tups)
    t0 = time()
    for i in eachindex(tups)
        t1 = time()
        W, m, reptyp = tups[i]
		@info "I am currently dealing with $i-th tuple of (k, m, reptyp), which is (k, m, reptyp) = ($W, $(trm(m)), $reptyp).\n\n"
		f_nm, f_seed, bs_seed, τs = get_optimum(W, m, N, a, reptyp, rdisp, opt, int_rtol, limits)
		jldopen("$(saver_o.dirname)/dset_$(saver_o.iter).jld2", "w") do file
			@stash!(file, W, m, N, a, reptyp, rdisp, opt, int_rtol, limits, f_nm, f_seed, bs_seed, τs)
		end
		saver_o.iter += 1
        t2 = time()
		@info "I am $(t2-t0) s into the computation.\n Finished $i-th run out of $(N_tot) for (k, m, reptyp) = ($W, $(trm(m)), $reptyp). This run took $(t2-t1) s.\n\n\n\n"
    end
end


trm(m) = m
trm(::Nothing) = "Inf"

export km_walkthrough_repul

extract_number(str::AbstractString) = parse(Int, match(r"\d+", str).match)
function get_files_in_dir(dir::AbstractString = pwd())
	curdir = pwd()
	files = readdir(dir)
	cd(dir)
	files = files |> files -> filter(str -> !isnothing(match(r"jld2", str)), files) |> files -> sort!(files; lt = (str1, str2)->isless(extract_number(str1), extract_number(str2)))
	cd(curdir)
	return map(file -> "$dir/$file", files)
end
