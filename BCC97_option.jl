using Random,Distributions,Polynomials
using Statistics,QuadGK,FFTW
using Optim,Roots


Random.seed!(150000)


#ZSB 零息债券定价参数 gama,b1,b2
function gamma(κ_r,σ_r)
    return √(κ_r^2+2*σ_r^2)
end

function b1(κ_r,θ_r,σ_r,r0,T)
    g=gamma(κ_r,σ_r)
    result=((2*g*exp((κ_r+g)*T/2))/(2*g+(κ_r+g)*(exp(g*T)-1)))^(2*κ_r*θ_r/σ_r^2)

    return result
end

function b2(κ_r,θ_r,σ_r,r0,T)
    g=gamma(κ_r,σ_r)
    result=2*(exp(g*T)-1)/(2*g+(κ_r+g)*(exp(g*T)-1))
    return result
end

function B(κ_r,θ_r,σ_r,r0,T)

    b_1=b1(κ_r,θ_r,σ_r,r0,T)
    b_2=b2(κ_r,θ_r,σ_r,r0,T)
    result=b_1*exp(-b_2*r0)
    return result
end
###################end###############


#H93模型特征函数值
function H93_char_func(u,T,r,κ_v,θ_v,σ_v,ρ,v0)
    c1=κ_v*θ_v
    c2=-√((ρ*σ_v*u*1im-κ_v)^2-σ_v^2*(-u*1im-u^2))
    c3=(κ_v-ρ*σ_v*u*1im+c2)/(κ_v-ρ*σ_v*u*1im-c2)

    H1=(r*u*1im*T+(c1/σ_v^2)*((κ_v-ρ*σ_v*u*1im+c2)*T-2*log((1-c3*exp(c2*T))/(1-c3))))
    H2=((κ_v-ρ*σ_v*u*1im+c2)/σ_v^2*((1-exp(c2*T))/(1-c3*exp(c2*T))))

    char_fun_value=exp(H1+H2*v0)

    return char_fun_value
end



#M76模型特征函数值
function M76_char_func(u,T,λ,μ,δ)
        ω=-λ*(exp(μ+0.5*(δ^2))-1)
        tmp1=λ*(map(x->exp(x)-1,1im*u*μ .-0.5*(u.^2)*δ^2))
        value=map(x->exp(x*T),(1im*u*ω .+tmp1))
        return value
end



#BCC模型特征函数值
function BCC_char_func(u,T,r,κ_v,θ_v,σ_v,ρ,v0,λ,μ,δ)
    BCC1=H93_char_func(u,T,r,κ_v,θ_v,σ_v,ρ,v0)
    BCC2=M76_char_func(u,T,λ,μ,δ)

    return BCC1*BCC2
end


#H93模型内在价值
function H93_int_func(u,S0,K,T,r,κ_v,θ_v,σ_v,ρ,v0)
    char_fun_value=H93_char_func(u-1im*0.5,T,r,κ_v,θ_v,σ_v,ρ,v0)
    int_func_value=1/(u^2+0.25)*real(exp(1im*u*log(S0/K))*char_fun_value)
    return int_func_value
end


#BCC模型内在价值函数值
function BCC_int_func(u,S0,K,T,r,κ_v,θ_v,σ_v,ρ,v0,λ,μ,δ)
    char_fun_value=BCC_char_func(u-1im*0.5,T,r,κ_v,θ_v,σ_v,ρ,v0,λ,μ,δ)
    int_func_value=1/(u^2+0.25)*real(exp(1im*u*log(S0/K))*char_fun_value)

    return int_func_value
end


#H93模型欧式看涨期权值
function H93_call_func(S0,K,T,r,κ_v,θ_v,σ_v,ρ,v0)
    int_func_value=quadgk(x->H93_int_func(x,S0,K,T,r,κ_v,θ_v,σ_v,ρ,v0),0,Inf)[1]
    print("h93_int:",int_func_value)
    print("temp1:",exp(-r*T))
    print("temp2:",√(S0*K)/π*int_func_value)

    call_value=max(0,S0-exp(-r*T)*√(S0*K)/π*int_func_value)
    return call_value
end


#BCC模型欧式看涨期权值
function BCC_call_value(S0,K,T,r,κ_v,θ_v,σ_v,ρ,v0,λ,μ,δ)
    int_value=quadgk(x->BCC_int_func(x,S0,K,T,r,κ_v,θ_v,σ_v,ρ,v0,λ,μ,δ),0,Inf)[1]
    call_value=max(0,S0-exp(-r*T)*√(S0*K)/π*int_value)

    return call_value
end

κ_v,θ_v,σ_v,v0,T=1.5,0.02,0.15,0.01,1.0
ρ=0.1
λ = 0.25
μ = -0.2
δ = 0.1
σ = √(v0)
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05


# B_res=BCC_call_value(S0,K,T,r,κ_v,θ_v,σ_v,ρ,v0,λ,μ,δ)
h93=H93_call_func(S0,K,T,r,κ_v,θ_v,σ_v,ρ,v0)
# h93=π
print(h93)
# u=2
# h93=H93_char_func(u,T,r,κ_v,θ_v,σ_v,ρ,v0)
# h93_int=H93_int_func(u,S0,K,T,r,κ_v,θ_v,σ_v,ρ,v0)
# print("char",h93)
# print("int",h93_int)
