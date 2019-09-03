using Statistics,QuadGK,FFTW

using Optim,Roots


mutable struct  EuropeanOption_crr
    St::Float64 #current assset price 当期价格
    K::Float64 #strike price 行权价
    T::Float64 #maturity time 到期时间
    t::Float64 #valuation date 有效日期
    r::Float64 #risk_less rate 无风险收益率
    σ::Float64 #volatility 波动率
    type::String #option type(call,put) 看涨、看跌
    # EuropeanOtion(St,K,T,t,r,σ,type)=new(St,K,T,t,r,σ,type)
end



function get_basic_parameters(o::EuropeanOption_crr,N::Int64)
    dt=o.T/N
    df=exp(-o.r*dt)
    u=exp(o.σ*√dt)
    d=1/u

    q=(exp(o.r*dt)-d)/(u-d)

    #初始化指数生成矩阵，即以u,d的概率生成二叉树网络，
    mu0=repeat(Vector(0:N),N+1)
    mu1=reshape(mu0,(N+1,N+1))
    md0=copy(mu1')
    mu2=mu1-md0
    mu=map(x->u^x,mu2)
    md=map(x->d^x,md0)
    S=o.St*(mu.*md)

    return dt,df,u,d,q,S

end

#CRR定价方法
function CRR_value(o::EuropeanOption_crr,N::Int64)

    dt,df,u,d,q,S=get_basic_parameters(o,N)

    if o.type=="call"
        V=map(x->max(x-o.K,0),S)
    else
        V=map(x->max(o.K-x,0),S)
    end

    VV=copy(V')
    z=0
    @simd for t= N:-1:1
        VV[1:N-z,t]=(q*VV[1:N-z,t+1]+(1-q)*VV[2:N-z+1,t+1])*df
        z+=1
    end
    return VV[1,1]
end


function CRR_call_value_FFT(o::EuropeanOption_crr,N::Int64)

    dt,df,u,d,q,S=get_basic_parameters(o,N)
    S0=copy(S')

    CT=map(x->max(x-o.K,0),S0[:,end])
    qv=zeros(N+1)
    qv[1]=q
    qv[2]=1-q

    C0_b=fft(exp(-o.r*o.T)*(ifft(CT).*fft(qv).^N))

    return real(C0_b[1])
end

#快速算法，如果不用struct,直接参数化输入，可大幅提升计算速度，6-7倍
function crr_value_fast(o::EuropeanOption_crr,N::Int64)
    Δt = o.T / N
    U = exp(o.σ * √Δt)
    D = 1 / U
    R = exp(o.r * Δt)
    p = (R - D) / (U - D)
    q = (U - R) / (U - D)

    if o.type == "put"
        Z = [max(0, o.K - o.St * exp((2 * i - N) *o.σ * √Δt)) for i = 0:N]
    else
        Z = [max(0, o.St * exp((2 * i - N) * σ * √Δt) - o.K) for i = 0:N]
    end

    @simd for n = N-1:-1:0
        for i = 0:n
            if o.type == "put"
                x = o.K - o.St * exp((2 * i - n) * o.σ * √Δt)
            else
                x = o.St * exp((2 * i - n) * o.σ * √Δt) - o.K
            end
            y = (q * Z[i+1] + p * Z[i+2]) / R
            Z[i+1] = max(x, y)
        end
    end

    return Z[1]
end




# @time t=crr(100,90,0.05,0.3,180/365,1000,"put")
o=EuropeanOption_crr(36.0,40.0,1.0,0,0.06,0.2,"put")
N=1000
#
# @time dt,df,u,d,q,S=get_basic_parameters(o,N)
# @time value=CRR_value(o,N)

# @time fft_value=CRR_value_FFT(o,N)

@time tg=crr_value_fast(o,N)
