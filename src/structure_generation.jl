############################

function compute_ordered_exp(bs::T, i::Int64, j::Int64, εₚ⁻::Float64, Δτ::Float64) where T<:Tuple{AbstractVector{Float64}, Vararg{AbstractVector{Float64}}}
	M = one(SMatrix{2,2, Complex{Float128}})
	for i′ = i:j
		M = B(bs, i′, εₚ⁻, Δτ)*M
	end
	return M
end

function compute_ordered_exp(bs::T, εₚ⁻::Float64, Δτ::Float64) where T<:Tuple{AbstractVector{Float64}, Vararg{AbstractVector{Float64}}}
	N = length(bs[1])
	M = one(SMatrix{2,2, Complex{Float128}})
	for i = N:-1:1
		M = M*B(bs, i, εₚ⁻, Δτ)
	end
	return M
end

function compute_U_cash!(U_cash, bs::T, εₚ⁻, Δτ) where T<:Tuple{AbstractVector{Float64}, Vararg{AbstractVector{Float64}}}
	N = length(bs[1])
	M = one(SMatrix{2,2, Complex{Float128}})
	U_cash[N+1] = M
	for i′ = N:-1:1
		M = M*B(bs, i′, εₚ⁻, Δτ)
		U_cash[i′] = M
	end
end

############################

function compute_single_period!(∂U::AbstractVector{SMatrix{2,2, Complex{Float128}}}, ∂²U::AbstractMatrix{SMatrix{2,2, Complex{Float128}}}, U_cash::AbstractVector{SMatrix{2,2, Complex{Float128}}}, bs::NTuple{2, AbstractVector{Float64}}, εₚ⁻, Δτ)
	N = length(bs[1])
	∂₀U = @view ∂U[1:N]
	∂₁U = @view ∂U[(N+1):2N]
	∂²₀₀U = @view ∂²U[1:N, 1:N]
	∂²₁₁U = @view ∂²U[(N+1):2N, (N+1):2N]
	∂²₀₁U = @view ∂²U[1:N, (N+1):2N]
	∂²₁₀U = @view ∂²U[(N+1):2N, 1:N]
	compute_U_cash!(U_cash, bs, εₚ⁻, Δτ)
	Mb = one(SMatrix{2,2, Complex{Float128}})
	for i = 1:N
		∂²₀₀U[i,i] = U_cash[i+1]*B′′₀₀(bs, i, εₚ⁻, Δτ)*Mb
		∂²₀₁U[i,i] = U_cash[i+1]*B′′₀₁(bs, i, εₚ⁻, Δτ)*Mb
		∂²₁₀U[i,i] = ∂²₀₁U[i,i]
		∂²₁₁U[i,i] = U_cash[i+1]*B′′₁₁(bs, i, εₚ⁻, Δτ)*Mb
		Mbm₀ = B′₀(bs, i, εₚ⁻, Δτ)*Mb
		Mbm₁ = B′₁(bs, i, εₚ⁻, Δτ)*Mb
		∂₀U[i] = U_cash[i+1]*Mbm₀
		∂₁U[i] = U_cash[i+1]*Mbm₁
		for j = i+1:N
			Me₀ = U_cash[j+1]*B′₀(bs, j, εₚ⁻, Δτ)
			Me₁ = U_cash[j+1]*B′₁(bs, j, εₚ⁻, Δτ)
			∂²₀₀U[j, i] = Me₀*Mbm₀
			∂²₀₀U[i, j] = ∂²₀₀U[j, i]
			∂²₁₁U[j, i] = Me₁*Mbm₁
			∂²₁₁U[i, j] = ∂²₁₁U[j, i]
			∂²₀₁U[j, i] = Me₀*Mbm₁
			∂²₀₁U[i, j] = Me₁*Mbm₀
			∂²₁₀U[i, j] = ∂²₀₁U[j, i]
			∂²₁₀U[j, i] = ∂²₀₁U[i, j]
			Bⱼ = B(bs, j, εₚ⁻, Δτ)
			Mbm₀ = Bⱼ*Mbm₀
			Mbm₁ = Bⱼ*Mbm₁
		end
		Mb = B(bs, i, εₚ⁻, Δτ)*Mb
	end
	return U_cash[1]
end

function compute_single_period!(∂U::AbstractVector{SMatrix{2,2, Complex{Float128}}}, ∂²U::AbstractMatrix{SMatrix{2,2, Complex{Float128}}}, U_cash::AbstractVector{SMatrix{2,2, Complex{Float128}}}, bs::Tuple{AbstractVector{Float64}}, εₚ⁻, Δτ)
	N = length(bs[1])
	compute_U_cash!(U_cash, bs, εₚ⁻, Δτ)
	Mb = one(SMatrix{2,2, Complex{Float128}})
	for i = 1:N
		∂²U[i,i] = U_cash[i+1]*B′′₀₀(bs, i, εₚ⁻, Δτ)*Mb
		Mbm₀ = B′₀(bs, i, εₚ⁻, Δτ)*Mb
		∂U[i] = U_cash[i+1]*Mbm₀
		for j = i+1:N
			∂²U[j, i] = U_cash[j+1]*B′₀(bs, j, εₚ⁻, Δτ)*Mbm₀
			∂²U[i, j] = ∂²U[j, i]
			Mbm₀ = B(bs, j, εₚ⁻, Δτ)*Mbm₀
		end
		Mb = B(bs, i, εₚ⁻, Δτ)*Mb
	end
	return U_cash[1]
end

function compute_single_period!(∂U::AbstractVector{SMatrix{2,2, Complex{Float128}}}, U_cash::AbstractVector{SMatrix{2,2, Complex{Float128}}}, bs::NTuple{3, AbstractVector{Float64}}, εₚ⁻, Δτ)
	N = length(bs[1])
	∂₀U = @view ∂U[1:N]
	∂₁U = @view ∂U[(N+1):2N]
	∂₃U = @view ∂U[(2N+1):3N]
	compute_U_cash!(U_cash, bs, εₚ⁻, Δτ)
	Mb = one(SMatrix{2,2, Complex{Float128}})
	for i = 1:N
		∂₀U[i] = U_cash[i+1]*B′₀(bs, i, εₚ⁻, Δτ)*Mb
		∂₁U[i] = U_cash[i+1]*B′₁(bs, i, εₚ⁻, Δτ)*Mb
		∂₃U[i] = U_cash[i+1]*B′₃(bs, i, εₚ⁻, Δτ)*Mb
		Mb = B(bs, i, εₚ⁻, Δτ)*Mb
	end
	return U_cash[1]
end

function compute_single_period!(∂U::AbstractVector{SMatrix{2,2, Complex{Float128}}}, U_cash::AbstractVector{SMatrix{2,2, Complex{Float128}}}, bs::NTuple{2, AbstractVector{Float64}}, εₚ⁻, Δτ)
	N = length(bs[1])
	∂₀U = @view ∂U[1:N]
	∂₁U = @view ∂U[(N+1):2N]
	compute_U_cash!(U_cash, bs, εₚ⁻, Δτ)
	Mb = one(SMatrix{2,2, Complex{Float128}})
	for i = 1:N
		∂₀U[i] = U_cash[i+1]*B′₀(bs, i, εₚ⁻, Δτ)*Mb
		∂₁U[i] = U_cash[i+1]*B′₁(bs, i, εₚ⁻, Δτ)*Mb
		Mb = B(bs, i, εₚ⁻, Δτ)*Mb
	end
	return U_cash[1]
end

function compute_single_period!(∂U::AbstractVector{SMatrix{2,2, Complex{Float128}}}, U_cash::AbstractVector{SMatrix{2,2, Complex{Float128}}}, bs::Tuple{AbstractVector{Float64}}, εₚ⁻, Δτ)
	N = length(bs[1])
	compute_U_cash!(U_cash, bs, εₚ⁻, Δτ)
	Mb = one(SMatrix{2,2, Complex{Float128}})
	for i = 1:N
		∂U[i] = U_cash[i+1]*B′₀(bs, i, εₚ⁻, Δτ)*Mb
		Mb = B(bs, i, εₚ⁻, Δτ)*Mb
	end
	return U_cash[1]
end

############################

process_bs(bs, params::AbstractParamsNoB1) = (bs,)
process_bs(bs, params::AbstractParamsB1) = (view(bs, 1:params.N), view(bs,(params.N+1):2params.N))
process_bs(bs, params::AbstractParamsB1B3) = (view(bs, 1:params.N), view(bs, (params.N+1):2params.N), view(bs, (2params.N+1):3params.N))

function custom_eigen(U::SMatrix{2,2, Complex{T}}) where {T<:Real}
	eigvals, eigvecs = eigen(Matrix(U))
	if abs(eigvals[1])>=abs(eigvals[2])
		λ₁, λ₂ = real(eigvals[1]), real(eigvals[2])
		S = SMatrix{2,2}(eigvecs)
		S⁻¹ = inv(S)
	else
		λ₁, λ₂ = real(eigvals[2]), real(eigvals[1])
		S = SMatrix{2,2}([eigvecs[1,2] eigvecs[1,1]; eigvecs[2,2] eigvecs[2,1]])
		S⁻¹ = inv(S)
	end
	#print(λ₁, "\n", λ₂, "\n", S, "\n", S⁻¹, "\n\n")
	return λ₁, λ₂, S, S⁻¹
end


function compute_full_span!(∂Fₚ::AbstractVector{Float64}, ∂²Fₚ::AbstractMatrix{Float64}, ∂U::AbstractVector{SMatrix{2,2, Complex{Float128}}}, ∂²U::AbstractMatrix{SMatrix{2,2, Complex{Float128}}}, U_cash::AbstractVector{SMatrix{2,2, Complex{Float128}}}, bs::Vector{Float64}, params::AbstractParams, εₚ⁺, εₚ⁻)
	N₀, N, m, β = length(bs), params.N, params.m, params.β
	U = compute_single_period!(∂U, ∂²U, U_cash, process_bs(bs, params), εₚ⁻, β/(N*m))
	λ₁, λ₂, S, S⁻¹ = custom_eigen(U)
	mFₚ⁻¹ = m/(2*cosh(β*εₚ⁺)/λ₁^m + 1.0 + (λ₂/λ₁)^m)
	ll = 0.0
	for k=0:m-2
		ll = ll*(λ₂/λ₁) + 1.0
	end
	for i = 1:N₀
		∂Uⁱₛ = S⁻¹*∂U[i]*S
		∂Fₚ[i] = mFₚ⁻¹*(1/λ₁)*real(∂Uⁱₛ[1,1]+(λ₂/λ₁)^(m-1)*∂Uⁱₛ[2,2])
	end
	for j = 1:N₀
		∂Uʲₛ = S⁻¹*∂U[j]*S
		for i = 1:N₀
			∂²Uⁱʲₛ = S⁻¹*∂²U[i,j]*S
			∂Uⁱₛ = S⁻¹*∂U[i]*S
			Δf = (1/λ₁)*real(∂²Uⁱʲₛ[1,1] + (λ₂/λ₁)^(m-1)*∂²Uⁱʲₛ[2,2])
			if m>1
				Δf += (1/λ₁^2)*real((m-1)*(∂Uⁱₛ[1,1]*∂Uʲₛ[1,1] + (λ₂/λ₁)^(m-2)*∂Uⁱₛ[2,2]*∂Uʲₛ[2,2]) + ll*(∂Uⁱₛ[1,2]*∂Uʲₛ[2,1] + ∂Uⁱₛ[2,1]*∂Uʲₛ[1,2]))
			end
			∂²Fₚ[i,j] = mFₚ⁻¹*Δf-∂Fₚ[i]*∂Fₚ[j]
		end
	end
	β′, λ₁′, λ₂′ = Float128(β), Float128(λ₁), Float128(λ₂)
	return Float64(log((cosh(β′*εₚ⁺) + 0.5*(λ₁′^m + λ₂′^m))/(cosh(β′*εₚ⁺)+cosh(β′*εₚ⁻))))
end

function compute_full_span!(∂Fₚ::AbstractVector{Float64}, ∂U::AbstractVector{SMatrix{2,2, Complex{Float128}}}, U_cash::AbstractVector{SMatrix{2,2, Complex{Float128}}}, bs::AbstractVector{Float64}, params::AbstractParams, εₚ⁺, εₚ⁻, wₚ, a, b)
	N₀, N, m = length(bs), params.N, params.m
	U = compute_single_period!(∂U, U_cash, process_bs(bs, params), εₚ⁻, params.Δτ)
	λ₁, λ₂, S, S⁻¹ = custom_eigen(U)
	Fₚ⁻¹ = 1/(2*a/λ₁^m + 1.0 + (λ₂/λ₁)^m)
	for i = 1:N₀
		∂Uⁱₛ = S⁻¹*∂U[i]*S
		∂Fₚ[i] += wₚ*Fₚ⁻¹*(1/λ₁)*real(∂Uⁱₛ[1,1]+(λ₂/λ₁)^(m-1)*∂Uⁱₛ[2,2])
	end
    return Float64(wₚ*(log(abs(a + 0.5*(λ₁^m + λ₂^m)))-b)/m)
end

function compute_full_span(bs::Vector{Float64}, params::AbstractParams, εₚ⁺, εₚ⁻, wₚ, a, b)
	m, N = params.m, params.N
	U = compute_ordered_exp(process_bs(bs, params), εₚ⁻, params.Δτ)
	λ₁, λ₂, S, S⁻¹ = custom_eigen(U)
    return Float64(wₚ*(log(abs(a + 0.5*(λ₁^m + λ₂^m)))-b)/m)
end

############################

function process_chunk!(∂F::AbstractVector{Float64}, ∂U::AbstractVector{SMatrix{2,2, Complex{Float128}}}, U_cash::AbstractVector{SMatrix{2,2, Complex{Float128}}}, bs, params::AbstractParams, psamples::AbstractVector{Tuple{Float64, Float64, Float64, Float128, Float128}})
	fill!(∂F, 0.0)
	ΔF = zero(Float64)
	for psample in psamples
		ΔF += compute_full_span!(∂F, ∂U, U_cash, bs, params, psample...)
	end
	return ΔF
end


function process_chunk2!(∂F::AbstractVector{SharedVector{Float64}}, ∂U::AbstractVector{SMatrix{2,2, Complex{Float128}}}, U_cash::AbstractVector{SMatrix{2,2, Complex{Float128}}}, bs, params::AbstractParams, psamples::AbstractVector{Tuple{Float64, Float64, Float64, Float128, Float128}})
	ΔF = zero(Float64)
    for i in eachindex(psamples)
        psample = psamples[i]
        fill!(∂F[i], 0.0)
        ΔF += compute_full_span!(∂F[i], ∂U, U_cash, bs, params, psample...)
	end
	return ΔF
end

function process_chunk(bs::AbstractVector{Float64}, params::AbstractParams, psamples::AbstractVector{Tuple{Float64, Float64, Float64, Float128, Float128}})
	ΔF = zero(Float64)
	for psample in psamples
		ΔF += compute_full_span(bs, params, psample...)
	end
	return ΔF
end


true_workers() = nworkers()>1 ? workers() : Int64[]




function precompute_step!(G_storage::G_Cash, bs, params::AbstractParams, psamples)
	∂Fs, ∂Us, U_cashes = G_storage.∂Fs, G_storage.∂Us, G_storage.U_cashes
	fs = Vector{Future}(undef, nprocs())
    @sync begin
        for wid in true_workers()
            fs[wid] = @spawnat wid process_chunk!(∂Fs[wid], localpart(∂Us), localpart(U_cashes), bs, params, localpart(psamples))
        end
        fs[1] = @spawnat 1 process_chunk!(∂Fs[1], localpart(∂Us), localpart(U_cashes), bs, params, localpart(psamples))
    end
	ΔF = sum(fetch.(fs))
	#print(typeof(∂Fs[2]), " ", length(∂Fs[2]), "\n")
    for wid in true_workers()
		∂Fs[1] .+= ∂Fs[wid]
	end
	return finalize!(∂Fs[1], ΔF, bs, params)
end

function compute_grad_components!(G_storage::G_Cash2, bs, params::AbstractParams, psamples)
	∂Fs, ∂Us, U_cashes = G_storage.∂Fs, G_storage.∂Us, G_storage.U_cashes
    ranges = [@fetchfrom wid DistributedArrays.localindices(psamples)[1] for wid in procs()]
    show(ranges)
    @sync begin
        for wid in true_workers()
            @spawnat wid process_chunk2!(view(∂Fs, ranges[wid]), localpart(∂Us), localpart(U_cashes), bs, params, localpart(psamples))
        end
        @spawnat 1 process_chunk2!(view(∂Fs, ranges[1]), localpart(∂Us), localpart(U_cashes), bs, params, localpart(psamples))
    end
	return ∂Fs
end

function precompute_step(bs::AbstractVector{Float64}, params::AbstractParams, psamples)
	fs = Vector{Future}(undef, nprocs())
    @sync begin
        for wid in true_workers()
			fs[wid] = @spawnat wid process_chunk(bs, params, localpart(psamples))
        end
		fs[1] = @spawnat 1 process_chunk(bs, params, localpart(psamples))
    end
	ΔF = sum(fetch.(fs))
   	return finalize(ΔF, bs, params)
end



function finalize!(∂F, ΔF, bs, params::AbstractParamsB1)
	N, Δτ, u, u₁ = params.N, params.Δτ, params.u, params.u₁
    ΔF = Δτ*(u*sum(x->(abs2(x)-1), bs[1:N]) + u₁*sum(abs2, bs[(N+1):2N])) - ΔF
	∂F .*= -1.0
	for i = 1:N
		∂F[i] += 2Δτ*u*bs[i]
	end
	for i = (N+1):2N
		∂F[i] += 2Δτ*u₁*bs[i]
	end
    return ΔF, pin_bs!(∂F, params)
end

function finalize(ΔF, bs, params::AbstractParamsB1)
	N, Δτ, u, u₁ = params.N, params.Δτ, params.u, params.u₁
	return Δτ*(u*sum(x->(abs2(x)-1), bs[1:N]) + u₁*sum(abs2, bs[(N+1):2N])) - ΔF
end

function finalize!(∂F, ΔF, bs, params::AbstractParamsB1B3)
	N, Δτ, u, u₁, u₃ = params.N, params.Δτ, params.u, params.u₁, params.u₃
    ΔF = Δτ*(u*sum(x->(abs2(x)-1), view(bs, 1:N)) + u₁*sum(abs2, view(bs, (N+1):2N)) + u₃*sum(abs2, view(bs, (2N+1):3N))) - ΔF
	∂F .*= -1.0
	for i = 1:N
		∂F[i] += 2Δτ*u*bs[i]
	end
	for i = (N+1):2N
		∂F[i] += 2Δτ*u₁*bs[i]
	end
    for i = (2N+1):3N
		∂F[i] += 2Δτ*u₃*bs[i]
	end
    return ΔF, pin_bs!(∂F, params)
end

function finalize(ΔF, bs, params::AbstractParamsB1B3)
	N, Δτ, u, u₁, u₃ = params.N, params.Δτ, params.u, params.u₁, params.u₃
    return Δτ*(u*sum(x->(abs2(x)-1), view(bs, 1:N)) + u₃*sum(abs2, view(bs, (2N+1):3N)) + u₁*sum(abs2, view(bs, (N+1):2N))) - ΔF
end


function finalize!(∂F, ΔF, bs, params::AbstractParamsNoB1)
	N, Δτ, u = params.N, params.Δτ, params.u
	ΔF = Δτ*u*sum(x->(abs2(x)-1), bs) - ΔF
	∂F .*= -1.0
	for i = 1:N
		∂F[i] += 2Δτ*u*bs[i]
	end
    return ΔF, pin_bs!(∂F, params)
end

function finalize(ΔF, bs, params::AbstractParamsNoB1)
	N, Δτ, u = params.N, params.Δτ, params.u
	return Δτ*u*sum(x->(abs2(x)-1), bs) - ΔF
end
