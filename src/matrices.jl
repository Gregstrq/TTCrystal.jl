##############
# Special functions for construction of matrices
##############

cosh_sqrt(x²) = x²>=0 ? cosh(sqrt(x²)) : cos(sqrt(-x²))

function sinch_sqrt(x²::T) where {T}
    if x²>0
        t = sqrt(x²)
        return sinh(t)/t
    elseif x²<0
        t = sqrt(-x²)
        return sin(t)/t
    else
        return one(T)
    end
end

function lSinch_sqrt(x²::Real)
    if x²>2.5e-7
        t = sqrt(x²)
        return (t*cosh(t)-sinh(t))/t^3
    elseif x²<-2.5e-7
        t = sqrt(-x²)
        return (sin(t)-t*cos(t))/t^3
    else
        return lSinch_sqrt_taylor(x²)
    end
end
lSinch_sqrt_taylor(x²::T) where {T<:Real} = lSinch_sqrt_taylor(x², eps(one(T)))
function lSinch_sqrt_taylor(x²::Real, ε::Real)
    a = 1/3.0
    s = a
    k::Int64 = 0
    while abs(a *= x²*(1.0+1.0/(k+1))/((2k+4)*(2k+5)))>ε
        s += a
        k += 1
    end
    return s
end

function llSinch_sqrt(x²::Real)
    if x²>4e-6
        t = sqrt(x²)
        return (sinh(t)*(t^2+3)-3*t*cosh(t))/t^5
    elseif x²<-4e-6
        t = sqrt(-x²)
        return (sin(t)*(3-t^2)-3*t*cos(t))/t^5
    else
        return llSinch_sqrt_taylor(x²)
    end
end
function llSinch_sqrt_taylor(x²::Real, ε::Real)
    a = 1/15.0
    s = a
    k::Int64 = 0
    while abs(a *= x²*(1.0+2.0/(k+1))/((2k+6)*(2k+7)))>ε
        s += a
        k += 1
    end
    return s
end
llSinch_sqrt_taylor(x²::T) where {T<:Real} = llSinch_sqrt_taylor(x², eps(one(T)))


##############
# Routines for construction of matrices
##############

const im128 = Complex{Float128}(im)
const one128 = one(Complex{Float128})

@inline M(εₚ⁻, b, b₁) = SA[-εₚ⁻ im128*(b₁-b); im128*(b₁+b) εₚ⁻]

B(εₚ⁻, b, b₁, κₚ²Δτ², Δτ) = one(SMatrix{2,2, Complex{Float128}})*cosh_sqrt(κₚ²Δτ²) + M(εₚ⁻, b, b₁)*sinch_sqrt(κₚ²Δτ²)*Δτ

B⁻¹(εₚ⁻, b, b₁, κₚ²Δτ², Δτ) = one(SMatrix{2,2, Complex{Float128}})*cosh_sqrt(κₚ²Δτ²) - M(εₚ⁻, b, b₁)*sinch_sqrt(κₚ²Δτ²)*Δτ

B′₀(εₚ⁻, b, b₁, κₚ²Δτ², Δτ) = SA[b*Δτ -im128; +im128 b*Δτ]*sinch_sqrt(κₚ²Δτ²)*Δτ + M(εₚ⁻, b, b₁)*lSinch_sqrt(κₚ²Δτ²)*b*Δτ^3

B′ₑ(εₚ⁻, b, b₁, κₚ²Δτ², Δτ) = SA[(εₚ⁻*Δτ-one128) 0.0; 0.0 (εₚ⁻*Δτ+one128)]*sinch_sqrt(κₚ²Δτ²)*Δτ + M(εₚ⁻, b, b₁)*lSinch_sqrt(κₚ²Δτ²)*εₚ⁻*Δτ^3

B′₃(εₚ⁻, b, b₁, κₚ²Δτ², Δτ) = SA[(one128-εₚ⁻*Δτ) 0.0; 0.0 -(εₚ⁻*Δτ+one128)]*sinch_sqrt(κₚ²Δτ²)*Δτ - M(εₚ⁻, b, b₁)*lSinch_sqrt(κₚ²Δτ²)*εₚ⁻*Δτ^3

B′₁(εₚ⁻, b, b₁, κₚ²Δτ², Δτ) = SA[-b₁*Δτ im128; im128 -b₁*Δτ]*sinch_sqrt(κₚ²Δτ²)*Δτ - M(εₚ⁻, b, b₁)*lSinch_sqrt(κₚ²Δτ²)*b₁*Δτ^3

B′′₀₀(εₚ⁻, b, b₁, κₚ²Δτ², Δτ) = one(SMatrix{2,2, Complex{Float128}})*sinch_sqrt(κₚ²Δτ²)*Δτ^2 + SA[(-εₚ⁻+b^2*Δτ) im128*(b₁-3b); im128*(b₁+3b) (εₚ⁻+b^2*Δτ)]*lSinch_sqrt(κₚ²Δτ²)*Δτ^3 + M(εₚ⁻, b, b₁)*llSinch_sqrt(κₚ²Δτ²)*b^2*Δτ^5

B′′₁₁(εₚ⁻, b, b₁, κₚ²Δτ², Δτ) = -one(SMatrix{2,2, Complex{Float128}})*sinch_sqrt(κₚ²Δτ²)*Δτ^2 + SA[(εₚ⁻+b₁^2*Δτ) im128*(b-3b₁); -im128*(b+3b₁) (-εₚ⁻+b₁^2*Δτ)]*lSinch_sqrt(κₚ²Δτ²)*Δτ^3 + M(εₚ⁻, b, b₁)*llSinch_sqrt(κₚ²Δτ²)*b₁^2*Δτ^5

B′′₀₁(εₚ⁻, b, b₁, κₚ²Δτ², Δτ) = SA[-b*b₁*Δτ im128*(b+b₁); im128*(b+b₁) -b*b₁*Δτ]*lSinch_sqrt(κₚ²Δτ²)*Δτ^3 - M(εₚ⁻, b, b₁)*llSinch_sqrt(κₚ²Δτ²)*b*b₁*Δτ^5

for B_func in (:B, :B⁻¹, :B′₀, :B′₁, :B′ₑ, :B′₃, :B′′₀₀, :B′′₀₁, :B′′₁₁)
    @eval begin
        function $B_func(bs::NTuple{2, AbstractVector{TB}}, i::Int, εₚ⁻, Δτ) where {TB<:Real}
			b = bs[1][i]
			b₁ = bs[2][i]
			return $B_func(εₚ⁻, b, b₁, (εₚ⁻^2+b^2-b₁^2)*Δτ^2, Δτ)
		end

        function $B_func(bs::Tuple{AbstractVector{TB}}, i::Int, εₚ⁻, Δτ) where {TB<:Real}
            b = bs[1][i]
            return $B_func(εₚ⁻, b, zero(TB), (εₚ⁻^2+b^2)*Δτ^2, Δτ)
        end

        $(Symbol(B_func,:r))(bs::Tuple{AbstractVector{TB}, Vararg{AbstractVector{TB}}}, i::Int, εₚ⁻, Δτ, γ::Real) where {TB<:Real} = $B_func(bs, i, εₚ⁻, Δτ)*exp(-sqrt(εₚ⁻^2 + γ^2)*Δτ)
        @inline $(Symbol(B_func,:r))(bs::Tuple{AbstractVector{TB}, Vararg{AbstractVector{TB}}}, i::Int, εₚ⁻, Δτ) where {TB<:Real} = $(Symbol(B_func, :r))(bs, i, εₚ⁻, Δτ, one(TB))
		export $B_func, $(Symbol(B_func, :r))
    end
end
