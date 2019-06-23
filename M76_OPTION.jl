using Statistics,QuadGK,FFTW,Distributions
using BenchmarkTools,Random



#FFT算放中提及到的特征函数变化
function M76_CharacFun(u,x0,T,r,σ,λ,μ,δ)
    ω=x0/T+r-0.5*(σ^2)-λ*(exp(μ+0.5*(δ^2))-1)
    tmp1=λ*(map(x->exp(x)-1,1im*u*μ .-0.5*(u.^2)*δ^2))
    value=map(x->exp(x*T),(1im*u*ω .-0.5*(u .^2)*σ^2 .+tmp1))

    return value
end

#M76模型下的FFT方法计算欧式看涨期权
function M76_value_call_FFT(S0,K,T,r,σ,λ,μ,δ)

    """

    # Arguments
    - `S0`: 初始标的值.
    - `K`: 行权值
    - `T`: 到期时间，年为单位
    - `r`: 无风险利率
    - `σ`: 波动率
    - `λ`: 泊松过程强度
    - `μ`: 跳跃过程中均值
    - `δ`: 跳跃过程中方差

    """

    k=log(K/S0)
    x0=log(1)
    g=2
    N=g*4096
    eps0=(g*150.0)^-1
    eta=2*pi/(N*eps0)
    b=0.5*N*eps0-k
    u=Vector(1:N)
    vo=eta*(u .-1)

    if S0>0.95*K
        α=1.5
        v=vo .-(α+1)*1im
        mod_char_fun=exp(-r*T)*M76_CharacFun(v,x0,T,r,σ,λ,μ,δ) ./(α^2+α .-(vo .^2)+1im*(2*α+1)*vo)
    else
        α=1.1
        v=(vo .-1im*α) .-1im
        mod_char_fun_1=exp(-r*T)*(1 ./(1 .+1im*(vo .-α*1im))-exp(r*T) ./(1im*(vo .-α*1im))-M76_CharacFun(v,x0,T,r,σ,λ,μ,δ) ./((vo .-α*1im) .^2-1im*(vo .-α*1im)))
        v=(vo .+α*1im) .-1im
        mod_char_fun_2=exp(-r*T)*(1 ./(1 .+1im*(vo .+α*1im))-exp(r*T) ./(1im*(vo .+α*1im))-M76_CharacFun(v,x0,T,r,σ,λ,μ,δ) ./((vo .+α*1im) .^2-1im*(vo .+α*1im)))
    end
    δ=zeros(N)
    δ[1]=1
    J=Vector(1:N)
    SimpsonW=((3 .+((-1) .^J))-δ)/3

    if S0>=0.95*K
        fft_func=eta*map(x->exp(x),1im*b*vo) .*mod_char_fun .*SimpsonW
        payoff=real(fft(fft_func))
        call_value_m=exp(-α*k)/pi*payoff
    else
        fft_func=0.5*eta*map(x->exp(x),1im*b*vo) .*(mod_char_fun_1-mod_char_fun_2) .*SimpsonW
        payoff=real(fft(fft_func))
        call_value_m=payoff/(sinh(α*k)*pi)
    end

    pos=round(Int64,(k+b)/eps0)
    call_value=call_value_m[pos]

    return call_value*S0
end


#M76模型下的蒙特卡洛方法计算欧式看涨期权
function M76_value_call_MCS(S0,K,T,r,σ,λ,μ,δ,M,I,dist)

    """

    # Arguments
    - `S0`: 初始标的值.
    - `K`: 行权值
    - `T`: 到期时间，年为单位
    - `r`: 无风险利率
    - `σ`: 波动率
    - `λ`: 泊松过程强度
    - `μ`: 跳跃过程中均值
    - `δ`: 跳跃过程中方差
    - `I`: 模拟路径数
    - `M`: 时间步数
    """

    dt=T/M
    rj=λ*(exp(μ+0.5*δ^2)-1)

    S=zeros(Float64,M+1,I)
    S[1,:]=map(x->x=S0,S[1,:])

    Random.seed!(10000)

    rand1=rand(Normal(0,1),M+1,I)
    rand2=rand(Normal(0,1),M+1,I)
    rand3=rand(Poisson(λ*dt),M+1,I)

    for t=2:M+1
        if dist==1
            p0=(1+(r-rj)*dt) .+σ*√(dt)*rand1[t,:]
            p1=map(x->exp(μ+δ*x)-1,rand2[t,:]) .*rand3[t,:]
            S[t,:]=S[t-1,:] .*(p0+p1)
        else
            p_0=map(x->exp((r-rj-0.5*σ^2)*dt+σ*√(dt)*x),rand1[t,:])
            p_1=map(x->exp(μ+δ*x)-1,rand2[t,:]) .*rand3[t,:]
            S[t,:]=S[t-1,:] .*(p_0+p_1)
        end
    end

    h=map(x->max(x-K,0),S[end,:])
    res=exp(-r*T)*sum(h)/I

    return res

end



S0=3225.93
K=3225.93
T=0.22
r=0.005
σ=0.113
λ=3.559
μ=-0.075
δ=0.041

M=50
I=200000

dist=2


# test=M76_value_call_FFT(S0,K,T,r,σ,λ,μ,δ)

test=M76_value_call_MCS(S0,K,T,r,σ,λ,μ,δ,M,I,dist)
