using Statistics,QuadGK,FFTW
using BenchmarkTools




function M76_CharacFun(u,x0,T,r,σ,λ,μ,δ)
    ω=x0/T+r-0.5*(σ^2)-λ*(exp(μ+0.5*(δ^2))-1)
    tmp1=λ*(map(x->exp(x)-1,1im*u*μ .-0.5*(u.^2)*δ^2))
    value=map(x->exp(x*T),(1im*u*ω .-0.5*(u .^2)*σ^2 .+tmp1))

    return value
end


function M76_value_call_FFT(S0,K,T,r,σ,λ,μ,δ)
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
        print("ok")
        α=1.1
        v=(vo .-1im*α) .-1im
        mod_char_fun_1=exp(-r*T)*(1 ./(1 .+1im*(vo .-α*1im))-exp(r*T) ./(1im*(vo .-α*1im))-M76_CharacFun(v,x0,T,r,σ,λ,μ,δ) ./((vo .-α*1im) .^2-1im*(vo .-α*1im)))
        v=(vo .+α*1im) .-1im
        mod_char_fun_2=exp(-r*T)*(1 ./(1 .+1im*(vo .+α*1im))-exp(r*T) ./(1im*(vo .+α*1im))-M76_CharacFun(v,x0,T,r,σ,λ,μ,δ) ./((vo .+α*1im) .^2-1im*(vo .+α*1im)))
        print("ok")
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

S0=100.0
K=120.0
T=1.0
r=0.05
σ=0.4
λ=1.0
μ=-0.2
δ=0.1


test=M76_value_call_FFT(S0,K,T,r,σ,λ,μ,δ)
