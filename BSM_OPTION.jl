using Statistics,QuadGK

using Optim,Roots


mutable struct  EuropeanOption
    St::Float64 #current assset price 当期价格
    K::Float64 #strike price 行权价
    T::Float64 #maturity time 到期时间
    t::Float64 #valuation date 有效日期
    r::Float64 #risk_less rate 无风险收益率
    σ::Float64 #volatility 波动率
    type::String #option type(call,put) 看涨、看跌
    # EuropeanOtion(St,K,T,t,r,σ,type)=new(St,K,T,t,r,σ,type)
end


#正态分布，数值累计积分算法
function CDF_dn1(x)
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    l = abs(x)
    k = 1 / (1 + 0.2316419 * l)

    CND = 1 - 1 / sqrt(2 * π) * exp(-l^2 / 2) * (a1 * k + a2 * k^2 + a3 * k^3 + a4 * k^4 + a5 * k^5)

    if x < 0
        return 1 - CND
    end

    return CND
end


#BSM公式定价
#dN(x)标准正太函数显示表达（即高斯函数）
#CDF_dn(x)高斯函数积分
function BSM_value(o::EuropeanOption,greeks::Bool)
    dN(x)=exp(-0.5*x^2)/sqrt(2*pi)
    CDF_dn(x)=quadgk((x)->dN(x),-20,x)[1]

    d1=(log(o.St/o.K)+(o.r+o.σ^2/2)*o.T)/(o.σ*√o.T)
    d2=d1-o.σ*√o.T

    option_value=0.0

    if o.type=="put"
        option_value=CDF_dn(-d2)*o.K*exp(-o.r*o.T)-CDF_dn(-d1)*o.St
    else
        option_value=o.St*CDF_dn(d1)-exp(-o.r*o.T)*o.K*CDF_dn(d2)
    end

    if greeks
        vega=o.St*dN(d1)*√o.T
        δ=CDF_dn(d1)
        γ=dN(d1)/(o.St*o.σ*√o.T)
        ρ=o.K*o.T*exp(-o.r*o.T)*CDF_dn(d2)
        θ=-(o.St*dN(d1)*o.σ/(2*√o.T)+o.r*o.K*exp(-o.r*o.T)*CDF_dn(d2))
        return [option_value,vega,δ,γ,ρ,θ]
    else
        return [option_value]
    end

end
#
#


#格式化函数为，find_zero提供标准输入
function formatfun(o::EuropeanOption,x)
    o.σ=x
    z=BSM_value(option)[1]
end


#隐含波动率计算
function imp_vol(o::EuropeanOption,x,c0)
    fi(x)=formatfun(o,x)-c0
    z=find_zero(fi,(0,1),Bisection())
    return z
end



# option=EuropeanOption(100.0,100.0,1,0,0.05,0.2,"call")
# option_values=BSM_value(option,false)
#
# vega=option_values[2]
