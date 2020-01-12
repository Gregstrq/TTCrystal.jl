function process_chunk!(∂F::SharedArray{Float64}, ∂²F::SharedArray{Float64}, bs, params::AbstractParams, psamples::AbstractVector{NTuple{3, Float64}})
	fill!(∂F, 0.0)
	fill!(∂²F, 0.0)
	∂Fₚ = similar(∂F)
	∂²Fₚ = similar(∂²F)
	∂U = similar(∂F, SMatrix{2,2, Complex{Float64}})
	∂²U = similar(∂²F, SMatrix{2,2, Complex{Float64}})
	U_cash = Vector{SMatrix{2,2, Complex{Float64}}}(undef, params.N+1)
	#Um_cash = OffsetVector(Vector{SMatrix{2,2, Complex{Float64}}}(undef, params.m+1), 0:params.m)
	ΔF = zero(Float64)
	for (εₚ⁺, εₚ⁻, wₚ) in psamples
		ΔF += wₚ*compute_full_span!(∂Fₚ, ∂²Fₚ, ∂U, ∂²U, U_cash, bs, params, εₚ⁺, εₚ⁻)
		∂F .+= wₚ.*∂Fₚ
		∂²F .+= wₚ.*∂²Fₚ
	end
	return ΔF
end

function process_ΔF_chunk!(ΔFs_chunk, bs, params::AbstractParams, psamples::AbstractVector{NTuple{3, Float64}})
	for i in eachindex(psamples)
		(εₚ⁺, εₚ⁻, wₚ) = psamples[i]
		ΔFs_chunk[i] = wₚ*compute_ΔFₚ!(bs, params, εₚ⁺, εₚ⁻)
	end
end

true_workers() = nworkers()>1 ? workers() : Int64[]


function precompute_newton_step!(∂Fs, ∂²Fs, bs, params::AbstractParams, psamples)
	fs = Vector{Future}(undef, nprocs())
    @sync begin
        for wid in true_workers()
			fs[wid] = @spawnat wid process_chunk!(∂Fs[wid], ∂²Fs[wid], bs, params, psamples[wid])
        end
		if nworkers()>1
			fs[1] = @spawnat 1 process_chunk!(∂Fs[1], ∂²Fs[1], bs, params, psamples[1])
		end
    end
	ΔF = sum(fetch.(fs))
    for wid in true_workers()
		∂Fs[1] .+= ∂Fs[wid]
		∂²Fs[1] .+= ∂²Fs[wid]
	end
	@everywhere GC.gc()
	return finalize!(∂Fs[1], ∂²Fs[1], ΔF, bs, params)
end

function finalize!(∂F, ∂²F, ΔF, bs, params::AbstractParamsB1)
	N, m, β, u, u₁ = params.N, params.m, params.β, params.u, params.u₁
	mΔτ = β/N
	ΔF = mΔτ*(u*sum(abs2, bs[1:N]) + u₁*sum(abs2, bs[(N+1):2N])) - ΔF
	∂F .*= -1.0
	∂²F .*= -1.0
	for i = 1:N
		∂F[i] += 2mΔτ*u*bs[i]
		∂²F[i,i] += 2mΔτ*u
	end
	for i = (N+1):2N
		∂F[i] += 2mΔτ*u₁*bs[i]
		∂²F[i,i] += 2mΔτ*u₁
	end
	return ΔF, ∂F, ∂²F
end

function finalize!(∂F, ∂²F, ΔF, bs, params::AbstractParamsNoB1)
	N, m, β, u = params.N, params.m, params.β, params.u
	mΔτ = β/N
	ΔF = mΔτ*u*sum(abs2, bs) - ΔF
	∂F .*= -1.0
	∂²F .*= -1.0
	for i = 1:N
		∂F[i] += 2mΔτ*u*bs[i]
		∂²F[i,i] += 2mΔτ*u
	end
	return ΔF, ∂F, ∂²F
end

function newton_step!(bs, ∂Fs, ∂²Fs, params::AbstractParams, psamples)
	dbs = -∂²Fs[1]\∂Fs[1]
	bs .+= dbs
	return precompute_newton_step!(∂Fs, ∂²Fs, bs, params, psamples)..., norm(dbs)
end

get_is(params::ParamsNoB1_pinned) = [i for i=1:get_length(params) if (i!=1)&&(i!=params.i_pinned)]

function get_is(params::ParamsB1_pinned)
	N = params.N
	i_pinned = params.i_pinned
	blacklist = [1, i_pinned, N+div(N,4)+1, N+div(N,4)+i_pinned]
	return [i for i=1:get_length(params) if !(i in blacklist)]
end

function newton_step!(bs, ∂Fs, ∂²Fs, params::T, psamples) where {T<:Union{ParamsB1_pinned, ParamsNoB1_pinned}}
	is = get_is(params) 
	∂Fv = @view ∂Fs[1][is]
	∂²Fv = @view ∂²Fs[1][is,is]
	dbs = fill!(similar(bs), 0.0)
	dbsv = @view dbs[is]
	dbsv .= -∂²Fv\∂Fv
	bs .+= dbs
	return precompute_newton_step!(∂Fs, ∂²Fs, bs, params, psamples)..., norm(dbs)
end

