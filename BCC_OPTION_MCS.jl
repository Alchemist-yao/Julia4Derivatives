using Random,Distributions,Polynomials
using Statistics,QuadGK,FFTW,LinearAlgebra
using Optim,Roots
include("./BCC97_option.jl")


Random.seed!(50000)


function random_number_generator(symbol,normlization,M,I,mydim)
    if symbol==1
        rand0=rand(Normal(0,1),mydim,M+1,I)
        myrand=cat(myrand,-myrand,3)
    else
        myrand=rand(Normal(0,1),mydim,M+1,I)
    end # if

    if normlization==true
        myrand=(myrand .-mean(myrand)) ./std(myrand)
    end

    return myrand
end


function CIR_genetate_path(x0,κ,θ,σ,T,M,I,x_disc)
    dt=T/M
    x=zeros(M+1,I)
    x[1,:]=map(x->x=x0,x[1,:])
    xh=zeros(M+1,I)

    ran=rand(Normal(0,1),M+1,I)

    if isequal(x_disc,"exact")
        d=4*κ*θ/σ^2
        c=(σ^2*(1-exp(-κ*dt)))/(4*κ)
        if d>1
            for t in 2:M+1
                l=x[t-1,:]*(exp(-κ*dt))/c
                χ=rand(Chisq(d-1),I)
                x[t,:]=c*((ran[t,:] .+sqrt(l))^2 .+χ)
            end
        else
            for t in 2:M+1
                l=x[t-1,:]*(exp(-κ*dt))/c
                N=map(x->rand(Poisson(x/2)),l)
                χ=map(x->rand(Chisq(d+2*x)),N)
                x[t,:]=c*χ
            end
        end
    else
        for t in 2:M+1
            tmp1=map(x->dt*κ*(θ-max(0,x)),xh[t-1,:])
            tmp2=map(x->√(dt)*σ*sqrt(max(0,x)),xh[t-1,:])
            tmp3=ran[t,:] .*tmp2
            xh[t,:]=(xh[t-1,:] .+tmp1 .+tmp3)
            x[t,:]=map(x->max(0,x),xh[t,:])
        end
    end

    return x

end


function SRD_generate_path(x_disc,x0,κ,θ,σ,T,M,I,myrand,row,cho_matrix)
    dt=T/M
    x=zeros(M+1,I)
    x[1,:]=map(x->x=x0,x[1,:])
    xh=zeros(M+1,I)
    xh[1,:]=map(x->x=x0,xh[1,:])

    sdt=√(dt)

    @simd for t in 2:M+1
        ran=cho_matrix*myrand[:,t,:]
        if isequal(x_disc,"FULL Truncation")
            tmp1=map(x->dt*κ*(θ-max(0,x)),xh[t-1,:])
            tmp2=map(x->sdt*σ*sqrt(max(0,x)),xh[t-1,:])
            tmp3=ran[row,:] .*tmp2
            xh[t,:]=(xh[t-1,:] .+tmp1 .+tmp3)
            x[t,:]=map(x->max(0,x),xh[t,:])
        elseif isequal(x_disc,"Partial Truncation")
            tmp1=map(x->dt*κ*(θ-x),xh[t-1,:])
            tmp2=map(x->sdt*σ*sqrt(max(0,x)),xh[t-1,:])
            tmp3=ran[row,:] .*tmp2
            xh[t,:]=(xh[t-1,:] .+tmp1 .+tmp3)
            x[t,:]=map(x->max(0,x),xh[t,:])
        elseif isequal(x_disc,"Truncation")
            tmp1=map(x->dt*κ*(θ-x),x[t-1,:])
            tmp2=map(x->sdt*σ*sqrt(x),x[t-1,:])
            tmp3=ran[row,:] .*tmp2
            x[t,:]=map(x->max(0,x),(x[t-1,:] .+tmp1 .+tmp3))
        elseif isequal(x_disc,"Reflection")
            tmp1=map(x->dt*κ*(θ-abs(x)),xh[t-1,:])
            tmp2=map(x->sdt*σ*sqrt(abs(x)),xh[t-1,:])
            tmp3=ran[row,:] .*tmp2
            xh[t,:]=(xh[t-1,:] .+tmp1 .+tmp3)
            x[t,:]=map(x->abs(x),xh[t,:])
        elseif isequal(x_disc,"Higham-Mao")
            tmp1=map(x->dt*κ*(θ-x),xh[t-1,:])
            tmp2=map(x->sdt*σ*sqrt(abs(x)),xh[t-1,:])
            tmp3=ran[row,:] .*tmp2
            xh[t,:]=(xh[t-1,:] .+tmp1 .+tmp3)
            x[t,:]=map(x->abs(x),xh[t,:])
        elseif isequal(x_disc,"Simple Reflection")
            tmp1=map(x->dt*κ*(θ-x),x[t-1,:])
            tmp2=map(x->sdt*σ*sqrt(x),x[t-1,:])
            tmp3=ran[row,:] .*tmp2
            x[t,:]=map(x->abs(x),(x[t-1,:] .+tmp1 .+tmp3))
        elseif isequal(x_disc,"Absorption")
            tmp1=map(x->dt*κ*(θ-max(0,x)),xh[t-1,:])
            tmp2=map(x->sdt*σ*sqrt(max(0,x)),xh[t-1,:])
            tmp3=ran[row,:] .*tmp2
            xh[t,:]=map(x->max(0,x),xh[t-1,:] .+tmp1 .+tmp3)
            x[t,:]=map(x->max(0,x),xh[t,:])
        else
            print("wrong method!")
        end # if

    end

    return x


end


function H93_generate_path(S0,r,v,row,cho_matrix,M,I,T,myrand,s_disc)
    S=zeros(M+1,I)
    dt=T/M
    S[1,:]=map(x->x=S0,S[1,:])
    bias=0.0
    sdt=√(dt)
    @simd for t in 2:M+1
        ran=cho_matrix*myrand[:,t,:]
        # if true
        #     bias=mean(v[t,:] .*ran[row,:]*sdt)
        # end
        if isequal(s_disc,"Log")
            tmp0=(r .-0.5*v[t,:])*dt+map(x->sqrt(x),v[t,:]) .*ran[row,:]*sdt .-bias
            tmp1=map(x->exp(x),tmp0)
            S[t,:]=S[t-1,:] .*tmp1
        elseif isequal(s_disc,"Native")
            tmp2=exp(r*dt) .+map(x->sqrt(x),v[t,:]) .*ran[row,:]*sdt-bias
            S[t,:]=S[t-1,:] .*tmp2
        else
            print("wrong method1")
        end # if

    end # for
    return S
end

function ZCB_mcs(r0,κ_r,θ_r,σ_r,T,M,I,x_disc)
    dt=T/M
    r=CIR_genetate_path(r0,κ_r,θ_r,σ_r,T,M,I,x_disc)
    zcb=zeros(M+1,I)
    zcb[end,:]=map(x->x=1.0,zcb[1,:])
    for t=M+1:-1:2
        tmp1=map(x->exp(x/2*dt),-(r[t,:] .+r[t-1,:]))
        zcb[t-1,:]=zcb[t,:] .*tmp1
    end

    return sum(zcb,dims=2)/I
end



function H93_index_paths(S0,r,v,row,cho_matrix,M,I,T,myrand)
    S=zeros(M+1,I)
    dt=T/M
    S[1,:]=map(x->x=log(S0),S[1,:])
    sdt=√(dt)
    for t in 2:M+1
        ran=cho_matrix*myrand[:,t,:]
        S[t,:]+=S[t-1,:]
        S[t,:]+=((r[t,:]+r[t-1,:])/2 .-v[t,:]/2)*dt
        S[t,:]+=map(x->sqrt(x),v[t,:]) .*ran[row,:]*sdt
    end # for
    # if momath==1
    #     S[t,:]-=mean(map(x->sqrt(x),v[t,:]) .*ran[row,:]*sdt)
    # end
    S_index_path=map(x->exp(x),S)
    return S_index_path
end
#

function H93_euroption_mcs(args)
    cor_martix=[1.0 0.1;0.1 1.0]
    cho_matrix=LinearAlgebra.cholesky(cor_martix).L
    r=0.05
    x0=0.01
    κ=1.5
    θ=0.02
    σ=0.15
    T=1/12
    M=2
    dt=T/M
    S0=100.0
    I=25000
    x_disc="FULL Truncation"
    symbol=2
    normlization=false
    s_disc="Log"
    K=90
    mydim=2
    myrand=random_number_generator(symbol,normlization,M,I,mydim)
    v=SRD_generate_path(x_disc,x0,κ,θ,σ,T,M,I,myrand,2,cho_matrix)
    S=H93_generate_path(S0,r,v,1,cho_matrix,M,I,T,myrand,s_disc)
    h=map(x->max(x-K,0),S)
    H93_euroption_valution=sum(h[end,:]*exp(-r*T))/I

    return H93_euroption_valution
end


function get_condtion(A,B)
    len=length(A)
    C=zeros(sum(A))
    i=1
    for t in 1:len
        if A[t]==1
            C[i]=B[t]
            i+=1
        end
    end

    return C
end


function BCC97_SVSI_MCS(args)

    ρ=0.1
    cor_martix=[1.0 ρ 0.0;ρ 1.0 0.0;0.0 0.0 1.0]
    cho_matrix=LinearAlgebra.cholesky(cor_martix).L
    r=0.05
    x0=0.01
    κ=1.5
    θ=0.02
    σ=0.15
    T=1/12
    M=20
    dt=T/M
    S0=100.0
    I=25000
    x_disc="FULL Truncation"
    symbol=2
    normlization=false
    s_disc="Log"
    K=90
    row=1
    mydim=3
    D=10
    myrand=random_number_generator(symbol,normlization,M,I,mydim)
    r1=SRD_generate_path(x_disc,r,κ,θ,σ,T,M,I,myrand,1,cho_matrix)
    v=SRD_generate_path(x_disc,x0,κ,θ,σ,T,M,I,myrand,3,cho_matrix)
    S_path=H93_index_paths(S0,r1,v,row,cho_matrix,M,I,T,myrand)

    h=map(x->max(K-x,0),S_path)
    Va=map(x->max(K-x,0),S_path)

    for tt=M-1:-1:1
        df=map(x->exp(-x),(r1[tt,:].+r1[tt+1,:])/2*dt)
        itm=map(x->x>0 ? 1 : 0,h[tt,:])
        no_itm=sum(filter(!iszero,itm))

        println(tt)
        println(no_itm)
         rel_S=get_condtion(itm,S_path[tt,:])
        # rel_S=filter(!iszero,itm .*S_path[tt,:])
        if no_itm==0
            cv=zeros(I)
        else

            rel_v=get_condtion(itm,v[tt,:])
            rel_r=get_condtion(itm,r1[tt,:])
            rel_V0=get_condtion(itm,Va[tt,:])
            tmp000=get_condtion(itm,df)

            rel_V=rel_V0 .*tmp000
            mymatrix=zeros(D+1,no_itm)
            mymatrix[11,:]=rel_S .*rel_v .*rel_r
            mymatrix[10,:]=rel_S .*rel_v
            mymatrix[9,:]=rel_S .*rel_r
            mymatrix[8,:]=rel_v .*rel_r
            mymatrix[7,:]=rel_S .^2
            mymatrix[6,:]=rel_v .^2
            mymatrix[5,:]=rel_r .^2
            mymatrix[4,:]=rel_S
            mymatrix[3,:]=rel_v
            mymatrix[2,:]=rel_r
            mymatrix[1,:]=map(x->x=1,mymatrix[1,:])

            temp0=inv(mymatrix*mymatrix')*mymatrix
            reg=temp0*rel_V

            cv=reg'*mymatrix
        end
        erg=zeros(I)
        erg0=erg .*itm .*cv
        res=h[tt,:].>erg0
        @simd for i=1:I
            if res[i]==true
                global Va[tt,i]=h[tt,i]
            else
                global Va[tt,i]=Va[tt+1,i]*df[i]
            end
        end
    end

    df1=map(x->exp(-x),(r1[1,:]+r1[2,:])/2*dt)
    C0=H93_call_func(S0,K,T,r,κ,θ,σ,ρ,x0)
    κ_r,θ_r,σ_r,r0=0.3,0.04,0.1,0.04
    B0t=B(κ_r,θ_r,σ_r,r0,T)
    P0=C0+K*B0t-S0
    P0_MCS=sum(h[end,:]*B0t)/I


    x=B0t*h[end,:]
    y=Va[2,:] .*df1
    convar=true

    if convar
        x_mean=x .-mean(x)
        y_mean=y .-mean(y)
        b=sum(x .*y)/sum(x_mean .^2)
        y_cv=y .-1.0*(B0t*h[end,:] .-P0)
    else
        y_cv=y
    end

    SE=std(y)/√(I)

    V0_CV=max(sum(y_cv)/I,h[1,1])

    V0_LSM=max(sum(y)/I,h[1,1])

    return   V0_LSM
end

#
# res2=ZCB_mcs(r0,κ_r,θ_r,σ_r,T,M,I,x_disc)
