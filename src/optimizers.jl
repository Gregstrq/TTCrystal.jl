struct LocalExtremum
	val::Float64
	bs::Vector{Float64}
end

for op in (Symbol(==), :<, :>, :<=, :>=)
	@eval begin
		$op(a::T, b::T) where {T<:LocalExtremum} = $op(a.val, b.val)
	end
end

function FibonacciSearch(func)
	f1 = 1
	x = func(1)
	f2 = 2
	y = func(2)
	if y>=x
		return x, 1
	end
	f3 = 3
	z = func(3)
	while (z<y)
		f3, f2, f1 = (f3+f2, f3, f2)
		z ,y, x = func(f3), z, y
	end
	f′ = f2
	while (f′=f1+(f3-f2))!=f2
		v′ = func(f′)
		if f′>f2
			if v′>=y
				f3, f2, f1 = f′, f2, f1
				z, y, x = v′, y, x
			else
				f3, f2, f1 = f3, f′, f2
				z, y, x = z, v′, y
			end
		else
			if v′>y
				f3, f2, f1 = f3, f2, f′
				z, y, x = z, y, v′
			else
				f3, f2, f1 = f2, f′, f1
				z, y, x = y, v′, x
			end
		end
	end
	return y, f2
end

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

function get_optimum_fixed_m(m::Int64, β::Float64, Δτ::Float64, u, u₁, psamples, psamples_raw, opt::Optim.Options, ΔF₀)
	t′ = -time()
    @info "Performing calculation for m = $m.\n"
	N = 4*(div(ceil(Int64, β/(m*Δτ)), 4) + 1)
	params = ParamsB1B3(N, m, β, u, u₁)
	bs0 = seed_sn(params, psamples_raw)
	d= construct_objective(params, psamples, ΔF₀, bs0)
	results = optimize(d, bs0, LBFGS(m=120, linesearch = MoreThuente()), opt)
    #results = optimize(d, bs0, ConjugateGradient(), opt)
	val = Optim.minimum(results)
	bs = Optim.minimizer(results)
	t′ += time()
	@info "This calculation took $t′ s.\nCorresponding free energy is: $val.\n\n" 
	return LocalExtremum(val, bs)
end

function get_optimum_fixed_mi_nob1(m::Int64, β::Float64, Δτ::Float64, u, psamples, psamples_raw, opt::Optim.Options, ΔF₀)
	t′ = -time()
    @info "No B1. Performing calculation for m = $m.\n"
	N = 4*(div(ceil(Int64, β/(m*Δτ)), 4) + 1)
	params = ParamsNoB1(N, m, β, u)
	bs0 = seed_sn(params, psamples_raw)
	d= construct_objective(params, psamples, ΔF₀, bs0)
	results = optimize(d, bs0, LBFGS(m=120, linesearch = MoreThuente()), opt)
    #results = optimize(d, bs0, ConjugateGradient(), opt)
	val = Optim.minimum(results)
	bs = Optim.minimizer(results)
	t′ += time()
	@info "This calculation took $t′ s.\nCorresponding free energy is: $val.\n\n" 
	return LocalExtremum(val, bs)
end

function get_optimum(P::Float64, Λ::Float64, μ::Float64, α::Float64, Nₚ::Int64, β::Float64, Δτ::Float64, a::Float64, opt::Optim.Options)
	rdisp = ReducedDispersion(α, P, μ, Λ)
	psamples_raw = get_psamples(rdisp, Nₚ)
	u = get_u₀(β, psamples_raw)
	u₁ = u*a
    γ, static_fmin = get_static_fmin(β, u, psamples_raw)
	psamples = separate_psamples(widen(psamples_raw, Float128(β), γ))
	ΔF₀ = β*u*γ^2
	@info "Free energy of static configuration for this tuple is: $static_fmin.\n\n"
	get_opt_for_m = (m)->get_optimum_fixed_m(m, β, Δτ, u, u₁, psamples, psamples_raw, opt, ΔF₀)
	optimum, m_opt = FibonacciSearch(get_opt_for_m)
	N_opt = 4*(div(ceil(Int64, β/(m_opt*Δτ)), 4) + 1)
    return static_fmin, m_opt, optimum.val, optimum.bs, get_τs(β, m_opt, N_opt)
end

function params_walkthrough(P_range::AbstractRange, Λ_range::AbstractRange, saver_o::Saver, μ::Float64, α::Float64, Nₚ::Int64, β::Float64, Δτ::Float64, a::Float64, opt::Optim.Options)
	tups = reshape([tup for tup in product(P_range, Λ_range)], length(P_range)*length(Λ_range))
	sort!(tups; by=(tup)->sum(abs2, tup))
    f_mins = similar(tups, Union{Float64, Missing})
    static_f_mins = similar(tups, Union{Float64, Missing})
    fill!(f_mins, missing)
    fill!(static_f_mins, missing)
    N_tot = length(tups)
    t0 = time()
    for i in eachindex(tups)
        t1 = time()
        (P, Λ) = tups[i]
        @info "I am currently dealing with $i-th tuple of (P, Λ), which is (P, Λ) = ($P, $Λ).\n\n"
        try
		    static_f_min, m_opt, f_min, bs_opt, τs = get_optimum(P, Λ, μ, α, Nₚ, β, Δτ, a, opt)
			save_data(saver_o, P, Λ, static_f_min, m_opt, f_min, bs_opt, τs)
			static_f_mins[i] = static_f_min
			f_mins[i] = f_min
        catch ex
			stacktrace_string = catch_backtrace() |> stacktrace |> process_stacktrace
			@error "This Error occured for $i-th tuple of (P, Λ), which is (P, Λ) = ($P, $Λ).\n $ex\n$stacktrace_string\n\n"
			static_f_min, m_opt, f_min, bs_opt, τs = missing, missing, missing, missing, missing
			save_data(saver_o, P, Λ, static_f_min, m_opt, f_min, bs_opt, τs)
			static_f_mins[i] = missing
			f_mins[i] = missing
        end
        t2 = time()
        @info "I am $(t2-t0) s into the computation.\n Finished $i-th run out of $(N_tot) for (P, Λ) = ($P, $Λ). This run took $(t2-t1) s.\n\n\n\n"
	end
    return tups, static_fmins, fmins
end

function process_stacktrace(st)
	str = "\t[1]\t"*string(st[1])
	for i = 2:length(st)
		str = str * "\n\t[$i]\t$(string(st[i]))"
	end
	return str
end
