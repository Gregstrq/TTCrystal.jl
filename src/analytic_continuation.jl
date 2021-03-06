mutable struct S_Cash <: AbstractSharedCash
	Uxs::DArray{SMatrix{2,2,Complex{Float128}}, 1, Vector{SMatrix{2,2,Complex{Float128}}}}
	Uys::DArray{SMatrix{2,2,Complex{Float128}}, 1, Vector{SMatrix{2,2,Complex{Float128}}}}
	Uzs::DArray{SMatrix{2,2,Complex{Float128}}, 1, Vector{SMatrix{2,2,Complex{Float128}}}}
	U_cashes::DArray{SMatrix{2,2,Complex{Float128}}, 1, Vector{SMatrix{2,2,Complex{Float128}}}}
    Sps::Vector{SharedMatrix{SVector{3,Complex{Float64}}}}
    function S_Cash(N, psamples)
        Uxs = DArray([@spawnat i LinearAlgebra.ones(StaticArrays.SMatrix{2,2, Complex{Float128}}, N) for i=procs()])
        Uys = DArray([@spawnat i LinearAlgebra.ones(StaticArrays.SMatrix{2,2, Complex{Float128}}, N) for i=procs()])
        Uzs = DArray([@spawnat i LinearAlgebra.ones(StaticArrays.SMatrix{2,2, Complex{Float128}}, N) for i=procs()])
		U_cashes = DArray([@spawnat i LinearAlgebra.ones(StaticArrays.SMatrix{2,2, Complex{Float128}}, N+1) for i=procs()])
        Sps = [@fetchfrom pid SharedArrays.SharedMatrix{SVector{3,Complex{Float64}}}(N, length(localpart(psamples)); pids = procs()) for pid=procs()]
        s_cash = new(Uxs, Uys, Uzs, U_cashes, Sps)
        finalizer(free_darrays, s_cash)
    end
end


struct RealFFTOperator{T1, T2}
    ker_dft::Vector{Complex{Float64}}
    ys::Vector{Complex{Float64}}
    rp::T1
    irp::T2
    N::Int
    function RealFFTOperator(ker_dft::Vector{Complex{Float64}})
        N = length(ker_dft)
        ys = similar(ker_dft)
        rp = plan_fft(ker_dft)
        irp = plan_ifft(ker_dft)
        return new{typeof(rp), typeof(irp)}(ker_dft, ys, rp, irp, N)
    end
end

function free_darrays(s_cash::S_Cash)
    close(s_cash.Uxs)
    close(s_cash.Uys)
    close(s_cash.Uzs)
    close(s_cash.U_cashes)
end


function process_configuration(bs, params::ParamsNoB1, psamples)
    N = params.N
    s_cash = S_Cash(N, psamples)
    Sps = calculate_initial_conditions!(s_cash, bs, params, psamples)
    return Sps
end

function calculate_initial_conditions!(s_cash::S_Cash, bs, params::ParamsNoB1, psamples)
    Uxs, Uys, Uzs, U_cashes, Sps = s_cash.Uxs, s_cash.Uys, s_cash.Uzs, s_cash.U_cashes, s_cash.Sps
    @sync begin
        for wid in true_workers()
            @spawnat wid chunk_initial_conditions!(localpart(Uxs), localpart(Uys), localpart(Uzs), localpart(U_cashes), Sps[wid], bs, params, localpart(psamples))
        end
        @spawnat 1 chunk_initial_conditions!(localpart(Uxs), localpart(Uys), localpart(Uzs), localpart(U_cashes), Sps[1], bs, params, localpart(psamples))
    end
    return hcat([Sps[i] for i in eachindex(Sps)]...)
end

function chunk_initial_conditions!(Ux::T, Uy::T, Uz::T, U_cash::T, Sp::SharedMatrix{SVector{3,Complex{Float64}}}, bs, params, psamples) where T<:AbstractVector{SMatrix{2,2,Complex{Float128}}}
    for i in eachindex(psamples)
        Sp_view = view(Sp, :, i)
        p_initial_conditions!(Ux, Uy, Uz, U_cash, Sp_view, bs, params, psamples[i]...)
    end
end

function p_initial_conditions!(Ux::T, Uy::T, Uz::T, U_cash::T, Sp_view, bs, params::ParamsNoB1{Int64}, ????????, ????????, w???, a, b) where T<:AbstractVector{SMatrix{2,2,Complex{Float128}}}
    N = params.N
    U = compute_Us!(Ux, Uy, Uz, U_cash, bs, ????????, params.????)
	?????, ?????, S, S????? = custom_eigen(U)
	?????, ?????, S, S????? = custom_eigen(U)
    ????????????? = ?????^(m-1)
    ????????????? = ?????^(m-1)
    F??? = a + ?????^m + ?????^m
	F???????? = 1/F???
	for i = 1:N
        Ux?????? = S?????*Ux[i]*S
        Uy?????? = S?????*Uy[i]*S
        Uz?????? = S?????*Uz[i]*S
        Sp_view[i] = SA[(?????????????*Ux??????[1,1]+?????????????*Ux??????[2,2]);
                        (?????????????*Uy??????[1,1]+?????????????*Uy??????[2,2]);
                        (?????????????*Uz??????[1,1]+?????????????*Uz??????[2,2])]*F????????
	end
end


function p_initial_conditions!(Ux::T, Uy::T, Uz::T, U_cash::T, Sp_view, bs, params::ParamsNoB1{Nothing}, ????????, ????????, w???, a, b) where T<:AbstractVector{SMatrix{2,2,Complex{Float128}}}
    N = params.N
    U = compute_Us!(Ux, Uy, Uz, U_cash, bs, ????????, params.????)
	?????, ?????, S, S????? = custom_eigen(U)
	?????, ?????, S, S????? = custom_eigen(U)
    F?????? = log(abs(?????))
    if F??????>=a
		F???????? = 1/?????
        for i = 1:N
            Ux?????? = S?????*Ux[i]*S
            Uy?????? = S?????*Uy[i]*S
            Uz?????? = S?????*Uz[i]*S
            Sp_view[i] = SA[Ux??????[1,1];
                            Uy??????[1,1];
                            Uz??????[1,1]]*F????????
        end
    end
end

const ??x = SMatrix{2,2,Float128}(0.0,1.0,1.0,0.0)
const ??y = SMatrix{2,2,Complex{Float128}}(0.0,1.0im,-1.0im,0.0)
const ??z = SMatrix{2,2,Float128}(1.0,0.0,0.0,-1.0)

function compute_Us!(Ux::T, Uy::T, Uz::T, U_cash::T, bs, ????????, ????) where T<:AbstractVector{SMatrix{2,2,Complex{Float128}}}
    N = length(bs)
    compute_U_cash!(U_cash, bs, ????????, ????)
	Mb = one(SMatrix{2,2, Complex{Float128}})
	for i = 1:N
		Ux[i] = U_cash[i+1]*??x*Mb
		Uy[i] = U_cash[i+1]*??y*Mb
		Uz[i] = U_cash[i+1]*??z*Mb
		Mb = Br(bs, i, ????????, ????)*Mb
	end
	return U_cash[1]
end

#################################

K???(??, ?????, W) = 0.5(exp(-?????*abs(??)) + exp(?????*(abs(??) - W)))/(1 - exp(-?????*W))/?????
K??????(??, ?????, W) = -0.5sign(??)*(exp(-?????*abs(??)) - exp(?????*(abs(??) - W)))/(1 - exp(-?????*W))

function initial_state(i::Int, bs, Sps, psamples_raw, ??s, ????, W, ?????, u????????, a???)
    ????????s = [psample[2] for psample in psamples_raw]
    w???s = [psample[3]*0.5 for psample in psamples_raw]
    Sp??? = [elt[2] for elt in Sps*w???s]
    Spi = Sps[i,:]
    Sp = [getindex.(Spi, 1), getindex.(Spi, 2), getindex.(Spi, 3)]
    ??????? = ?????/sqrt(1+a???)
    #####
    b??? = bs[i] + 0im
    K????? = dot(K??????.(??s[i].-??s, ???????, W), bs)*????
    b?????? = 2sum(w???s.*????????s.*Sp[1])*u???????? - im*a???*???????^2*K?????
    return sTimeRHS(?????^2, u????????, u????????/(1+a???)*?????^2, ????????s, w???s), ArrayPartition([b???, b??????], Sp...)
end

function initial_state(i::Int, bs, Sps, psamples_raw, ??s, ????, W, u????????)
    ????????s = [psample[2] for psample in psamples_raw]
    w???s = [psample[3]*0.5 for psample in psamples_raw]
    Sp??? = Sps*w???s
    Sp = Sps[i,:]
    #print(bs[i] - u????????*Sp???[i][2], "\n")
    return sTimeRHS_simple(u????????, ????????s, w???s), ArrayPartition([bs[i]+0.0im], Sp)
end

function initial_state???(i::Int, bs, Sps, psamples_raw, ??s, ????, W, u????????)
    ????????s = [psample[2] for psample in psamples_raw]
    w???s = [psample[3]*0.5 for psample in psamples_raw]
    Sp??? = Sps*w???s
    Spi = Sps[i,:]
    Sp = [getindex.(Spi, 1), getindex.(Spi, 2), getindex.(Spi, 3)]
    print(bs[i] - u????????*Sp???[i][2], "\n")
    return sTimeRHS_simple???(u????????, ????????s, w???s), ArrayPartition([bs[i]+0.0im], Sp...)
end

function initial_state(bs::AbstractVector, Sps, psamples_raw, ??s, ????, W, ?????, u????????, a???)
    ????????s = [psample[2] for psample in psamples_raw]
    w???s = [psample[3]*0.5 for psample in psamples_raw]
    Sp??? = [elt[2] for elt in Sps*w???s]
    Kc = similar(bs)
    Ks = similar(bs, Complex{eltype(bs)})
    @inbounds for i in eachindex(bs)
        Kc[i] = dot(K???.(??s[i].-??s, ?????, W), Sp???)*0.5?????^2*????*u????????*a???/(1+a???)
        Ks[i] = dot(K??????.(??s[i].-??s, ?????, W), Sp???)*0.5?????^2*????*u????????*a???/(1+a???)*im
    end
    return fTimeRHS(Kc, Ks, ?????, u????????, ????????s, w???s), ArrayPartition(bs.+0.0im, deepcopy(Sps))
end

function initial_state2(bs::AbstractVector, Sps, psamples_raw, ??s, ????, W, ?????, u????????, a???)
    ????????s = [psample[2] for psample in psamples_raw]
    w???s = [psample[3]*0.5 for psample in psamples_raw]
    Sp??? = [elt[2] for elt in Sps*w???s]
    K = construct_FFT_operator(length(bs), ????, W, ?????, u????????, a???)
    return fTimeRHS2(K, ?????, u????????, ????????s, w???s), ArrayPartition(bs.+0.0im, deepcopy(Sps))
end

struct sTimeRHS
    ???????::Float64
    u????????::Float64
    u??????????::Float64
    ????????s::Vector{Float64}
    w???s::Vector{Float64}
    b_cache1::Vector{Complex{Float64}}
    b_cache2::Vector{Complex{Float64}}
    sTimeRHS(???????, u????????, u??????????, ????????s, w???s) = new(???????, u????????, u??????????, ????????s, w???s, zeros(Complex{Float64}, Threads.nthreads()), zeros(Complex{Float64}, Threads.nthreads()))
end

struct fTimeRHS
    Kc::Vector{Float64}
    Ks::Vector{Complex{Float64}}
    ?????::Float64
    u????????::Float64
    ????????s::Vector{Float64}
    w???s::Vector{Float64}
    bs_cache::Matrix{Complex{Float64}}
    function fTimeRHS(Kc, Ks, ?????, u????????, ????????s, w???s)
        bs_cache = Matrix{Complex{Float64}}(undef, length(Kc), Threads.nthreads())
        return new(Kc, Ks, ?????, u????????, ????????s, w???s, bs_cache)
    end
end

struct fTimeRHS2{T1, T2}
    K::RealFFTOperator{T1,T2}
    ?????::Float64
    u????????::Float64
    ????????s::Vector{Float64}
    w???s::Vector{Float64}
    bs_cache::Matrix{Complex{Float64}}
    function fTimeRHS2(K::RealFFTOperator{T1,T2}, ?????, u????????, ????????s, w???s) where {T1,T2}
        bs_cache = Matrix{Complex{Float64}}(undef, K.N, Threads.nthreads())
        return new{T1,T2}(K, ?????, u????????, ????????s, w???s, bs_cache)
    end
end

mutable struct sTimeRHS_simple
    u????????::Float64
    ????????s::Vector{Float64}
    w???s::Vector{Float64}
    b_cache::Vector{Complex{Float64}}
    sTimeRHS_simple(u????????, ????????s, w???s) = new(u????????, ????????s, w???s, Vector{Complex{Float64}}(undef, Threads.nthreads()))
end

struct sTimeRHS_simple???
    u????????::Float64
    ????????s::Vector{Float64}
    w???s::Vector{Float64}
    b_cache::Vector{Complex{Float64}}
    sTimeRHS_simple???(u????????, ????????s, w???s) = new(u????????, ????????s, w???s, Vector{Complex{Float64}}(undef, Threads.nthreads()))
end

function (rhs::sTimeRHS)(du::T, u::T, p, t) where {T <: ArrayPartition} 
    b = u.x[1]
    Sp_x = u.x[2]
    Sp_y = u.x[3]
    Sp_z = u.x[4]
    db = du.x[1]
    dSp_x = du.x[2]
    dSp_y = du.x[3]
    dSp_z = du.x[4]
    b_cache1 = rhs.b_cache1
    b_cache2 = rhs.b_cache2
    w???s = rhs.w???s
    ????????s = rhs.????????s
    ??????? = rhs.???????
    u???????? = rhs.u????????
    u?????????? = rhs.u??????????
    ##################
    fill!(b_cache1, 0.0)
    fill!(b_cache2, 0.0)
    @tturbo for i = eachindex(Sp_x)
        dSp_x[i] = -2(b[1]*Sp_z[i] + ????????s[i]*Sp_y[i])
        dSp_y[i] = 2????????s[i]*Sp_x[i]
        dSp_z[i] = 2b[1]*Sp_x[i]
        b_cache1[Threads.threadid()] -= 4w???s[i]*????????s[i]*(b[1]*Sp_z[i] + ????????s[i]*Sp_y[i])
        b_cache2[Threads.threadid()] += w???s[i]*Sp_y[i]
    end
    db[1] = b[2]
    db[2] = u????????*sum(b_cache1) + u??????????*sum(b_cache2) - ???????*b[1]
end

function (rhs::fTimeRHS)(du::T, u::T, p, t) where {T<:ArrayPartition}
    bs = u.x[1]
    dbs = du.x[1]
    Sps = u.x[2]
    dSps = du.x[2]
    bs_cache = rhs.bs_cache
    fill!(bs_cache, 0.0)
    Threads.@threads for j = 1:size(Sps, 2)
        for i = 1:size(Sps, 1)
            B??? = SVector{3,Complex{Float64}}(0.0, -bs[i], rhs.????????s[j])
            dSps[i, j] = 2*cross_product(B???, Sps[i, j])
            bs_cache[i, Threads.threadid()] += rhs.w???s[j]*dSps[i, j][2]
        end
    end
    @views dbs .= bs_cache[:, 1]
    for i = 2:size(bs_cache, 2)
        @views dbs .+= bs_cache[:, i]
    end
    dbs .*= rhs.u????????
    @. dbs += rhs.Kc*sin(rhs.?????*t) - rhs.Ks*cos(rhs.?????*t)
end

function (rhs::fTimeRHS2)(du::T, u::T, p, t) where {T<:ArrayPartition}
    bs = u.x[1]
    dbs = du.x[1]
    Sps = u.x[2]
    dSps = du.x[2]
    bs_cache = rhs.bs_cache
    fill!(bs_cache, 0.0)
    Threads.@threads for j = 1:size(Sps,2)
        for i = 1:size(Sps, 1)
            B??? = SVector{3,Complex{Float64}}(0.0, -bs[i], rhs.????????s[j])
            dSps[i, j] = 2*cross_product(B???, Sps[i, j])
            bs_cache[i, Threads.threadid()] += rhs.w???s[j]*dSps[i, j][2]
        end
    end
    for j = 2:size(bs_cache, 2)
        @views bs_cache[:,1] .+= bs_cache[:, j]
    end
    mul!(dbs, rhs.K, view(bs_cache, :, 1))
    @views dbs  .+= rhs.u????????*bs_cache[:,1]
end

function (rhs::sTimeRHS_simple)(du::T, u::T, p, t) where {T<:ArrayPartition{Complex{Float64},Tuple{Array{Complex{Float64},1},Array{SArray{Tuple{3},Complex{Float64},1,3},1}}}}
    b = u[1]
    Sp = u.x[2]
    dSp = du.x[2]
    b_cache = rhs.b_cache
    fill!(b_cache, 0.0)
    Threads.@threads for i = 1:length(Sp)
    #for i = 1:length(Sp)
    B??? = SVector{3,Complex{Float64}}(0.0, -b, rhs.????????s[i])
        dSp[i] = 2*cross_product(B???, Sp[i])
        b_cache[Threads.threadid()] += rhs.w???s[i]*dSp[i][2]
    end
    du[1] = rhs.u????????*sum(b_cache)
end

function (rhs::sTimeRHS_simple???)(du::T, u::T, p, t) where {T<:ArrayPartition}
    b = u[1]
    Sp_x = u.x[2]
    Sp_y = u.x[3]
    Sp_z = u.x[4]
    dSp_x = du.x[2]
    dSp_y = du.x[3]
    dSp_z = du.x[4]
    b_cache = rhs.b_cache
    fill!(b_cache, 0.0)
    Threads.@threads for i = eachindex(Sp_x)
    #for i = 1:length(Sp)
        dSp_x[i] = -2(b[1]*Sp_z[i] + rhs.????????s[i]*Sp_y[i])
        dSp_y[i] = 2rhs.????????s[i]*Sp_x[i]
        dSp_z[i] = 2b[1]*Sp_x[i]
        b_cache[Threads.threadid()] += rhs.w???s[i]*dSp_y[i]
    end
    du[1] = rhs.u????????*sum(b_cache)
end

function cross_product(v1::T, v2::T) where {T<:SVector{3, Complex{Float64}}}
    return SVector{3,Complex{Float64}}(v1[2]*v2[3] - v1[3]*v2[2], v1[3]*v2[1] - v1[1]*v2[3], v1[1]*v2[2] - v1[2]*v2[1])
end

export process_configuration, initial_state

import Base.isnan

isnan(v::StaticArray) = any(isnan, v)


function LinearAlgebra.mul!(y::AbstractVector{Complex{Float64}}, O::RealFFTOperator, x::AbstractVector{Complex{Float64}})
    @assert length(y) == length(x) == O.N
    mul!(O.ys, O.rp, x)
    O.ys .*= O.ker_dft
    mul!(y, O.irp, O.ys)
end

function construct_FFT_operator(N, ????, W, ?????, u????????, a???)
    ker_dft = zeros(Complex{Float64}, N)
    ker_dft .= full_kernel_dft.(collect(0:N-1), ?????*W, N)*0.5?????*????*u????????*a???/(1+a???)
    return RealFFTOperator(ker_dft)
end


function retarded_integral!(sin_integral, cos_integral, b???s, ts, ??t, ???????)
    sin_integral[1] = 0
    cos_integral[1] = 0
    st = sin(???????*??t)
    ct = cos(???????*??t)
    ??t??? = ??t/2
    for i = 2:length(sin_integral)
        sin_integral[i] = sin_integral[i-1]*ct + cos_integral[i-1]*st + st*b???s[i-1]*??t???
        cos_integral[i] = cos_integral[i-1]*ct - sin_integral[i-1]*st + (b???s[i] + ct*b???s[i-1])*??t???
    end
    return sin_integral
end
function retarded_integral(b???s, ts, ??t, ???????)
    sin_integral = similar(b???s)
    cos_integral = similar(b???s)
    return retarded_integral!(sin_integral, cos_integral, b???s, ts, ??t, ???????)
end

function ie_rhs!(b??????s, Sp, sin_integral, cos_integral, b_cache, b???s, ts, ????????s, w???s, K, K???, ??t, ???????, u????????, a???)
    Sp = deepcopy(Sp)
    retarded_integral!(sin_integral, cos_integral, b???s, ts, ??t, ???????)
    a??? = a???*???????
    a?????? = a???*???????^2
    b??????s[1] = sum(w???s.*getindex.(Sp, 2))*u????????
    for i = 2:length(b??????s)
        fill!(b_cache, 0)
        Threads.@threads for j = eachindex(Sp)
            B??? = SVector{3,Complex{Float64}}(0.0, -b???s[i-1], ????????s[j])
            Sp[j] = Sp[j] + 2??t*cross_product(B???, Sp[j])
            b_cache[Threads.threadid()] += w???s[j]*Sp[j][2]
        end
        b??????s[i] = sum(b_cache)*u????????
    end
    b??????s .-= (sin_integral .+ (K???*im).*sin.(???????.*ts)).*a??? .+ cos.(???????.*ts).*(a??????*K)
    return b??????s, b???s
end

function setup_ie_params(sol, i::Int, bs, Sps, psamples_raw, ??s, ????, W, ?????, u????????, a???)
    ts = sol.t
    b???s = map(first, sol.u)
    ??t = ts[end]/(length(ts)-1)
    ????????s = [psample[2] for psample in psamples_raw]
    w???s = [psample[3]*0.5 for psample in psamples_raw]
    Sp = Sps[i, :]
    ??????? = ?????/sqrt(1+a???)
    K = dot(K???.(??s[i].-??s, ???????, W), bs)*????
    K??? = dot(K??????.(??s[i].-??s, ???????, W), bs)*????
    sin_integral = similar(b???s)
    cos_integral = similar(b???s)
    b??????s = similar(b???s)
    b_cache = similar(b???s, Threads.nthreads())
    return b??????s, Sp, sin_integral, cos_integral, b_cache, b???s, ts, ????????s, w???s, K, K???, ??t, ???????, u????????, a???
end

export retarded_integral!, retarded_integral, ie_rhs!, setup_ie_params

