function compute_ordered_exp(b_vec, b₁_vec, i, j, εₚ⁻, Δτ)
	M = SA[1.0+0.0im 0.0; 0.0 1.0+0.0im]
	for i′ = i:j
		M = B(b_vec, b₁_vec, i′, εₚ⁻, Δτ)*M
	end
	return M
end

function compute_ordered_exp(b_vec, i, j, εₚ⁻, Δτ)
	M = SA[1.0+0.0im 0.0; 0.0 1.0+0.0im]
	for i′ = i:j
		M = B(b_vec, i′, εₚ⁻, Δτ)*M
	end
	return M
end

function compute_U_cash!(U_cash, b_vec, b₁_vec, εₚ⁻, Δτ)
	N = length(b_vec)
	M = SA[1.0+0.0im 0.0; 0.0 1.0+0.0im]
	U_cash[N+1] = M
	for i′ = N:-1:1
		M = M*B(b_vec, b₁_vec, i′, εₚ⁻, Δτ)
		U_cash[i′] = M
	end
end

function compute_U_cash!(U_cash, b_vec, εₚ⁻, Δτ)
	N = length(b_vec)
	M = SA[1.0+0.0im 0.0; 0.0 1.0+0.0im]
	U_cash[N+1] = M
	for i′ = N:-1:1
		M = M*B(b_vec, i′, εₚ⁻, Δτ)
		U_cash[i′] = M
	end
end

function compute_single_period!(∂U, ∂²U, b_vec::AbstractVector{Float64}, b₁_vec::AbstractVector{Float64}, εₚ⁻, Δτ)
	N = length(b_vec)
	∂₀U = @view ∂U[1:N]
	∂₁U = @view ∂U[(N+1):2N]
	∂²₀₀U = @view ∂²U[1:N, 1:N]
	∂²₁₁U = @view ∂²U[(N+1):2N, (N+1):2N]
	∂²₀₁U = @view ∂²U[1:N, (N+1):2N]
	∂²₁₀U = @view ∂²U[(N+1):2N, 1:N]
	N = length(b_vec)
	U = compute_ordered_exp(b_vec, b₁_vec, 1, N, εₚ⁻, Δτ)
	for i = 1:N
		Mb = compute_ordered_exp(b_vec, b₁_vec, 1, i-1, εₚ⁻, Δτ)
		Me = compute_ordered_exp(b_vec, b₁_vec, i+1, N, εₚ⁻, Δτ)
		∂²₀₀U[i,i] = Me*B′′₀₀(b_vec, b₁_vec, i, εₚ⁻, Δτ)*Mb
		∂²₀₁U[i,i] = Me*B′′₀₁(b_vec, b₁_vec, i, εₚ⁻, Δτ)*Mb
		∂²₁₁U[i,i] = Me*B′′₁₁(b_vec, b₁_vec, i, εₚ⁻, Δτ)*Mb
		Mb₀ = B′₀(b_vec, b₁_vec, i, εₚ⁻, Δτ)*Mb
		Mb₁ = B′₁(b_vec, b₁_vec, i, εₚ⁻, Δτ)*Mb
		∂₀U[i] = Me*Mb₀
		∂₁U[i] = Me*Mb₁
		for j = i+1:N
			Mm = compute_ordered_exp(b_vec, b₁_vec, i+1, j-1, εₚ⁻, Δτ)
			Mbm₀ = Mm*Mb₀
			Mbm₁ = Mm*Mb₁
			Me′ = compute_ordered_exp(b_vec, b₁_vec, j+1, N, εₚ⁻, Δτ)
			Me₀ = Me′*B′₀(b_vec, b₁_vec, j, εₚ⁻, Δτ)
			Me₁ = Me′*B′₁(b_vec, b₁_vec, j, εₚ⁻, Δτ)
			∂²₀₀U[j, i] = Me₀*Mbm₀
			∂²₀₀U[i, j] = ∂²₀₀U[j, i]
			∂²₁₁U[j, i] = Me₁*Mbm₁
			∂²₁₁U[i, j] = ∂²₁₁U[j, i]
			∂²₀₁U[j, i] = Me₀*Mbm₁
			∂²₀₁U[i, j] = Me₁*Mbm₀
			∂²₁₀U[i, j] = ∂²₀₁U[j, i]
			∂²₁₀U[j, i] = ∂²₀₁U[i, j]
		end
	end
	return U
end

function compute_single_period!(∂U, ∂²U, b_vec::AbstractVector{Float64}, εₚ⁻, Δτ)
	N = length(b_vec)
	U = compute_ordered_exp(b_vec, 1, N, εₚ⁻, Δτ)
	for i = 1:N
		Mb = compute_ordered_exp(b_vec, 1, i-1, εₚ⁻, Δτ)
		Me = compute_ordered_exp(b_vec, i+1, N, εₚ⁻, Δτ)
		∂²U[i,i] = Me*B′′₀₀(b_vec, i, εₚ⁻, Δτ)*Mb
		Mb₀ = B′₀(b_vec, i, εₚ⁻, Δτ)*Mb
		∂U[i] = Me*Mb₀
		for j = i+1:N
			Mm = compute_ordered_exp(b_vec, i+1, j-1, εₚ⁻, Δτ)
			Mbm₀ = Mm*Mb₀
			Me′ = compute_ordered_exp(b_vec, j+1, N, εₚ⁻, Δτ)
			Me₀ = Me′*B′₀(b_vec, j, εₚ⁻, Δτ)
			∂²U[j, i] = Me₀*Mbm₀
			∂²U[i, j] = ∂²U[j, i]
		end
	end
	return U
end


function compute_single_period!(∂U, ∂²U, U_cash, b_vec::AbstractVector{Float64}, b₁_vec::AbstractVector{Float64}, εₚ⁻, Δτ)
	N = length(b_vec)
	∂₀U = @view ∂U[1:N]
	∂₁U = @view ∂U[(N+1):2N]
	∂²₀₀U = @view ∂²U[1:N, 1:N]
	∂²₁₁U = @view ∂²U[(N+1):2N, (N+1):2N]
	∂²₀₁U = @view ∂²U[1:N, (N+1):2N]
	∂²₁₀U = @view ∂²U[(N+1):2N, 1:N]
	compute_U_cash!(U_cash, b_vec, b₁_vec, εₚ⁻, Δτ)
	Mb = SA[1.0+0.0im 0.0; 0.0 1.0+0.0im]
	for i = 1:N
		∂²₀₀U[i,i] = U_cash[i+1]*B′′₀₀(b_vec, b₁_vec, i, εₚ⁻, Δτ)*Mb
		∂²₀₁U[i,i] = U_cash[i+1]*B′′₀₁(b_vec, b₁_vec, i, εₚ⁻, Δτ)*Mb
		∂²₁₀U[i,i] = ∂²₀₁U[i,i]
		∂²₁₁U[i,i] = U_cash[i+1]*B′′₁₁(b_vec, b₁_vec, i, εₚ⁻, Δτ)*Mb
		Mbm₀ = B′₀(b_vec, b₁_vec, i, εₚ⁻, Δτ)*Mb
		Mbm₁ = B′₁(b_vec, b₁_vec, i, εₚ⁻, Δτ)*Mb
		∂₀U[i] = U_cash[i+1]*Mbm₀
		∂₁U[i] = U_cash[i+1]*Mbm₁
		for j = i+1:N
			Me₀ = U_cash[j+1]*B′₀(b_vec, b₁_vec, j, εₚ⁻, Δτ)
			Me₁ = U_cash[j+1]*B′₁(b_vec, b₁_vec, j, εₚ⁻, Δτ)
			∂²₀₀U[j, i] = Me₀*Mbm₀
			∂²₀₀U[i, j] = ∂²₀₀U[j, i]
			∂²₁₁U[j, i] = Me₁*Mbm₁
			∂²₁₁U[i, j] = ∂²₁₁U[j, i]
			∂²₀₁U[j, i] = Me₀*Mbm₁
			∂²₀₁U[i, j] = Me₁*Mbm₀
			∂²₁₀U[i, j] = ∂²₀₁U[j, i]
			∂²₁₀U[j, i] = ∂²₀₁U[i, j]
			Bⱼ = B(b_vec, b₁_vec, j, εₚ⁻, Δτ)
			Mbm₀ = Bⱼ*Mbm₀
			Mbm₁ = Bⱼ*Mbm₁
		end
		Mb = B(b_vec, b₁_vec, i, εₚ⁻, Δτ)*Mb
	end
	return U_cash[1]
end

function compute_single_period!(∂U, ∂²U, U_cash, b_vec::AbstractVector, εₚ⁻, Δτ)
	N = length(b_vec)
	compute_U_cash!(U_cash, b_vec, εₚ⁻, Δτ)
	Mb = SA[1.0+0.0im 0.0; 0.0 1.0+0.0im]
	for i = 1:N
		∂²U[i,i] = U_cash[i+1]*B′′₀₀(b_vec, i, εₚ⁻, Δτ)*Mb
		Mbm₀ = B′₀(b_vec, i, εₚ⁻, Δτ)*Mb
		∂U[i] = U_cash[i+1]*Mbm₀
		for j = i+1:N
			∂²U[j, i] = U_cash[j+1]*B′₀(b_vec, j, εₚ⁻, Δτ)*Mbm₀
			∂²U[i, j] = ∂²U[j, i]
			Mbm₀ = B(b_vec, j, εₚ⁻, Δτ)*Mbm₀
		end
		Mb = B(b_vec, i, εₚ⁻, Δτ)*Mb
	end
	return U_cash[1]
end

process_bs(bs, params::AbstractParamsNoB1) = (bs,)
process_bs(bs, params::AbstractParamsB1) = (bs[1:params.N], bs[(params.N+1):2params.N])

function custom_eigen(U::SMatrix{2,2, Complex{Float64}})
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
	return λ₁, λ₂, S, S⁻¹
end

function compute_full_span!(∂Fₚ, ∂²Fₚ, ∂U, ∂²U, U_cash, Um_cash::AbstractVector{SMatrix{2,2, Complex{Float64}}}, bs, params::AbstractParams, εₚ⁺, εₚ⁻)
	N₀, N, m, β = length(bs), params.N, params.m, params.β
	Um_cash[0] = SA[1.0+0.0im 0.0; 0.0 1.0+0.0im]
	Um_cash[1] = compute_single_period!(∂U, ∂²U, U_cash, process_bs(bs, params)..., εₚ⁻, β/(N*m))
	for k = 2:m
		Um_cash[k] = Um_cash[1]^k
	end
	mFₚ⁻¹ = m*1.0/(2*cosh(β*εₚ⁺) + real(tr(Um_cash[m])))
	for i = 1:N₀
		∂Fₚ[i] = mFₚ⁻¹*real(tr(Um_cash[m-1]*∂U[i]))
	end
	for j = 1:N₀
		for i = 1:N₀
			Δf = real(tr(Um_cash[m-1]*∂²U[i,j]))
			for k = 2:m
				Δf += real(tr(Um_cash[m-k]*∂U[i]*Um_cash[k-2]*∂U[j]))
			end
			∂²Fₚ[i,j] = mFₚ⁻¹*Δf-∂Fₚ[i]*∂Fₚ[j]
		end
	end
	return log((cosh(β*εₚ⁺) + 0.5*real(tr(Um_cash[m])))/(cosh(β*εₚ⁺)+cosh(β*εₚ⁻)))
end


function compute_full_span!(∂Fₚ, ∂²Fₚ, ∂U, ∂²U, U_cash, bs::Vector{Float64}, params::AbstractParams, εₚ⁺, εₚ⁻)
	N₀, N, m, β = length(bs), params.N, params.m, params.β
	U = compute_single_period!(∂U, ∂²U, U_cash, process_bs(bs, params)..., εₚ⁻, β/(N*m))
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
	return log((cosh(β*εₚ⁺) + 0.5*(λ₁^m + λ₂^m))/(cosh(β*εₚ⁺)+cosh(β*εₚ⁻)))
end
