using Random,StatsBase,Distributions,Polynomials
Random.seed!(150000)

#LSM原始算法，针对美式PUT
function LSM_PRIMAL(S0::Float64,K::Float64,T::Float64,r::Float64,σ::Float64,I::Int,M::Int)
    """

    # Arguments
    - `S0`: 初始标的值.
    - `K`: 行权值
    - `T`: 到期时间，年为单位
    - `r`: 无风险利率
    - `σ`: 波动率
    - `I`: 原始算法中路径数
    - `M`: 时间步数

    """

    dt=T/M
    df=exp(-r*dt)


    p1=(r-0.5*(σ^2))*dt
    p2=rand(Normal(0,1),M+1,I)
    p3=p1 .+σ*√(dt)*p2
    p4=cumsum(p3,dims=1)

    S=S0*(p4.|>exp)
    S[1,:]=repeat([S0],I)
    h=map(x->max(K-x,0),S)

    V=copy(h[end,:])

    @simd for t=M:-1:1
        rg=polyfit(S[t,:],V*df,5)
        C=map(x->polyval(rg,x),S[t,:])
        res=h[t,:].>C
        for i=1:I
            if res[i]==true
                V[i]=h[t,i]
            else
                V[i]=V[i]*df
            end
        end
    end

    v0=df*sum(V)/I

end


#LSM对偶算法
function LSM_DUAL(S0::Float64,K::Float64,T::Float64,r::Float64,σ::Float64,J::Int,I1::Int,I2::Int,M::Int,reg::Int)
    """

    # Arguments
    - `S0`: 初始::Int标的值::Int.
    ::Int- ::Int`K`: 行权值
    - `T`: 到期时间，年为单位
    - `r`: 无风险利率
    - `σ`: 波动率
    - `J`: 对偶算法中子循环模拟路径
    - `I1`: 原始算法中路径数
    - `I2`: 对偶算法中路径数
    - `M`: 时间步数
    - `reg`: 多项是回归阶数

    """
    dt=T/M
    df=exp(-r*dt)

    #路径I1,产生回归系数
    S1=genrate_path(S0,r,σ,dt,M,I1)
    h1=map(x->max(K-x,0),S1)
    V1=map(x->max(K-x,0),S1)

    #路径I1,产生回归系数
    rg0=[]
    a=Poly(0)
    push!(rg0,a)
    @simd for t=M:-1:2
        rgr=polyfit(S1[t,:],V1[t+1,:]*df,5)
        push!(rg0,rgr)
    end
    push!(rg0,a)
    rg=reverse(rg0)

    #路径I2，对偶算法
    Q=zeros(M+1,I2)
    U=zeros(M+1,I2)
    S=genrate_path(S0,r,σ,dt,M,I2)
    h=map(x->max(K-x,0),S)
    V=map(x->max(K-x,0),S)


    @simd for t=2:M+1
        for i=1:I2
            tmp=polyval(rg[t],S[t,i])
            Vt=max(h[t,i],tmp)
            St=generate_nest_mc(r,σ,dt,S[t-1,i],J)
            Ct=map(x->polyval(rg[t],x),St)
            ht=map(x->max(K-x),St)
            res=ht.>Ct
            Vtj=zeros(J)
            for k=1:length(res)
                if res[k]==true
                    Vtj[k]=ht[k]
                else
                    Vtj[k]=Ct[k]
                end
            end
            sum_VT=sum(Vtj)/length(St)

            Q[t,i]=Q[t-1,i]/df+(Vt-sum_VT)
            U[t,i]=max(U[t-1,i]/df,(h[t,i]-Q[t,i]))

            if t==M+1
                U[t,i]=max(U[t-1,i]/df,mean(ht)-Q[t,i])
            end
        end
    end
    k=df^M
    U0=sum(U[M,:])/I2*(df^M)

end

#产生标的仿真路径-单路径
function generate_nest_mc(r::Float64,σ::Float64,dt::Float64,st::Float64,J::Int)

    p1=(r-0.5*(σ^2))*dt
    p2=rand(Normal(0,1),J)
    p3=p1 .+σ*√(dt)*p2
    S=st*(p3.|>exp)

    return S

end

#产生标的仿真路径-多路径
function genrate_path(S0::Float64,r::Float64,σ::Float64,dt::Float64,M::Int,I::Int)
    p1=(r-0.5*(σ^2))*dt
    S=zeros(M+1,I)
    S[1,:]=map(x->x=S0,S[1,:])
    @simd for t=2:M+1
        p2=rand(Normal(0,1),I)
        p3=p1 .+σ*√(dt)*p2
        S[t,:]=S[t-1,:].*(p3.|>exp)
    end

    return S
end

S0=36.0
K=40.0
T=1.0
r=0.06
σ=0.2
I1=16384
I2=1024

M=10
J=50
reg=5


res=LSM_DUAL(S0,K,T,r,σ,J,I1,I2,M,reg)
println(res)
