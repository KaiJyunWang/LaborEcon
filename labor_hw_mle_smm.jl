using Optim, LinearAlgebra, Statistics, Plots, DataFrames
using Distributions, Random, Parameters, LaTeXStrings
using Tullio, Interpolations, NLsolve, Roots, Expectations

#infinite horizon model
#to show the power of VFI
#similar structure to McCall job search model
#setting parameters
N = 5000
periods = 2
μ_w = 0.3
μ_ss = 1.0
σ = 0.3
β = 0.9
γ = 0.5
ν = 0.7
l = 8760
θ_p = 2080

Random.seed!(23)
ε_source = randn(N+150, periods)
ε = ε_source[1:N,:]
ε_mc = vec(ε_source[N+1:end,:])


rp = @with_kw (μ_w = μ_w, μ_ss = μ_ss, σ = σ, β = β, γ = γ, ν = ν, l = l, 
                θ_p = θ_p, ε_mc = ε_mc, wage = range(0.0, maximum(exp.(μ_w .+ σ*ε_source)), length = 1000),
                work = [0,1])
rp = rp()

#bellman operator
function T(v;rp)
    (;μ_w, μ_ss, σ, β, γ, ν, l, θ_p, ε_mc, wage, work) = rp
    u(c,p) = (ν != 1 ? 1/(1-ν)*((c+1e-9)^γ*(l-θ_p*p)^(1-γ))^(1-ν) : γ*log(c+1e-9)+(1-γ)*log(l-θ_p*p))
    v_func = LinearInterpolation((wage,work), v)
    v_candidate = zeros(length(wage), length(work), length(work))

    @tullio v_candidate[i,j,k] = u(wage[i]*work[j]*work[k]+μ_ss*(1-work[j]*work[k]), work[j]*work[k]) + β*mean(v_func(exp.(μ_w.+σ*ε_mc),work[k]*work[j]))
    @tullio w[i,j] := findmax(v_candidate[i,j,:])
    @tullio v1[i,j] := w[i,j][1]
    @tullio ind[i,j] := w[i,j][2]
    @tullio policy[i,j] := work[ind[i,j]]*work[j]
    return (v = v1, policy = policy)
end

#solve for the value function numerically
function solve(v_init, tol = 1e-6; rp)
    res = fixedpoint(v -> T(v;rp = rp).v, v_init, iterations = 1000, xtol = tol, m = 3)
    policy = T(res.zero;rp = rp).policy
    return (v = res.zero, policy = policy)
end

v_init = zeros(length(rp.wage), length(rp.work))
sol = solve(v_init;rp)
rw = rp.wage[findfirst(x -> x == 1, sol.policy[:,2])]
plt = plot(rp.wage, sol.v[:,2], label = L"V_{numerical}", lw = 2)
savefig(plt, "numerical_v.png")

#analytical solution (γ = 0.5, ν = 1)
u(c,p) = (ν != 1 ? 1/(1-ν)*(c^γ*(l-θ_p*p)^(1-γ))^(1-ν) : γ*log(c)+(1-γ)*log(l-θ_p*p))
vr = 1/(1-β)*u(μ_ss,0)
F(x) = cdf(LogNormal(μ_w,σ),x)
Ev(wr) = (wr>0 ? (vr*F(wr) + (l-θ_p)^((1-γ)*(1-ν))/(1-ν)*exp(γ*(1-ν)*μ_w-0.5*(σ*γ*(1-ν))^2)*(1-F(wr/exp(γ*(1-ν)*σ^2))))/((1-β*(1-F(wr)))) : -Inf)
vp(w) = u(w,1)+β*Ev(w)
rw_star = find_zero(x -> vp(x)-vr, 0.5)
U(w) = max(u(w,1)+β*Ev(rw_star),vr)

#comparison
plt = plot(rp.wage, U.(rp.wage), label = "analytical", lw = 2)
plot!(rp.wage, sol.v[:,2], label = "numerical", lw = 2)
savefig(plt, "comparison.png")

#however the model cannot be estimated because γ and ν are not identified
#we consider the two period model before to perform the estimations
#two period model
#simulate the model
wage = exp.(μ_w.+σ*ε)
s_1 = (1+β)/(1-ν)*(μ_ss^γ*l^(1-γ))^(1-ν)
s_2 = (wage[:,1].^γ.*(l-θ_p)^(1-γ)).^(1-ν)./(1-ν) .+ β*(μ_ss^γ*l^(1-γ))^(1-ν)/(1-ν)
s_3 = (wage[:,1].^γ.*(l-θ_p)^(1-γ)).^(1-ν)./(1-ν) .+ β/(1-ν)*((l-θ_p)^(1-γ)*exp(μ_w*γ+0.5*(1-ν)*(σ*γ)^2))^(1-ν)
work = zeros(Bool,N,2)
work[:,1] = max.(s_2, s_3) .≥ s_1
work[:,2] = work[:,1].*((wage[:,2].^γ)*((l-θ_p)^(1-γ)) .≥ μ_ss^γ*(l^(1-γ))) .== 1
con = wage.*work .+ μ_ss*(1 .- work)

df = DataFrame(id = 1:N, work_1 = work[:,1], work_2 = work[:,2], 
                con_1 = con[:,1], con_2 = con[:,2])
df1 = filter(row -> row.work_1 == 1, df)
rlc1 = sum(log.(df1.con_1))
rlcsq1 = sum(log.(df1.con_1).^2)
df2 = filter(row -> row.work_2 == 1, df)
rlc2 = sum(log.(df2.con_2))
rlcsq2 = sum(log.(df2.con_2).^2)
rlc = (rlc1 + rlc2)/(nrow(df1) + nrow(df2))
rlcsq = (rlcsq1 + rlcsq2)/(nrow(df1) + nrow(df2))
rlfpr1 = mean(df.work_1)
rlfpr2 = mean(df.work_2)/rlfpr1
m = [rlc, rlcsq, rlfpr1, rlfpr2]

df3 = filter(row -> row.work_1 == 0, df)
df4 = filter(row -> row.work_1 == 1 && row.work_2 == 0, df)

rlc1/nrow(df1)
minimum(df1.con_1)
#estimation
#MLE
#log likelihood function
function L(x)
    E = expectation(w -> 1/(1-x[4])*(max(w^x[3]*(l-θ_p)^(1-x[3]), μ_ss^x[3]*l^(1-x[3])))^(1-x[4]), LogNormal(x[1],x[2]))
    rw1 = ((1+β)/(1-x[4])*(μ_ss^x[3]*l^(1-x[3]))^(1-x[4])-β*E)
    retire_wage = ((1-x[4])*rw1)^(1/(1-x[4])/x[3])*(l-θ_p)^(1-1/x[3])
    L1 = sum(log.(pdf.(LogNormal(x[1],x[2]), df1.con_1))) + log(cdf(LogNormal(x[1],x[2]), retire_wage))*nrow(df3)
    L2 = sum(log.(pdf.(LogNormal(x[1],x[2]), df2.con_2))) + log(cdf(LogNormal(x[1],x[2]), μ_ss*(l/(l-θ_p))^(1/x[3]-1)))*nrow(df4)
    return -(L1+L2)
end

#optimization
res_mle = optimize(x -> L(x), [0.4,0.4,0.4,0.4])
res_mle.minimizer

rp_mle = @with_kw (μ_w = res_mle.minimizer[1], μ_ss = μ_ss, σ = res_mle.minimizer[2], β = β, 
                γ = res_mle.minimizer[3], ν = res_mle.minimizer[4], l = l, 
                θ_p = θ_p, ε_mc = ε_mc, 
                wage = range(0.0, maximum(exp.(μ_w .+ res_mle.minimizer[2]*ε_source)), length = 1000),
                work = [0,1])
rp_mle = rp_mle()
sol_mle = solve(v_init;rp = rp_mle)
plot(sol_mle.policy[:,2], label = "policy", lw = 2)

#sequential estimation
L_fs(x;df = df) = (x[2]>0 ? sum(log.(pdf.(LogNormal(x[1],x[2]), df1.con_1))) + log(cdf(LogNormal(x[1],x[2]), minimum(df1.con_1)))*nrow(df3) 
                        + sum(log.(pdf.(LogNormal(x[1],x[2]), df2.con_2))) + log(cdf(LogNormal(x[1],x[2]), minimum(df2.con_2)))*nrow(df4) : -1e10)
res_mle_fs = optimize(x -> -L_fs(x), [0.4,0.4])
z = res_mle_fs.minimizer
function L_ss(x;df=df,z=z)
    E = expectation(w -> 1/(1-x[2])*(max(w^x[1]*(l-θ_p)^(1-x[1]), μ_ss^x[1]*l^(1-x[1])))^(1-x[2]), LogNormal(z[1],z[2]))
    rw1 = ((1+β)/(1-x[2])*(μ_ss^x[1]*l^(1-x[1]))^(1-x[2])-β*E)
    retire_wage = ((1-x[2])*rw1)^(1/(1-x[2])/x[1])*(l-θ_p)^(1-1/x[1])
    L1 = sum(log.(pdf.(LogNormal(z[1],z[2]), df1.con_1))) + log(cdf(LogNormal(z[1],z[2]), retire_wage))*nrow(df3)
    L2 = sum(log.(pdf.(LogNormal(z[1],z[2]), df2.con_2))) + log(cdf(LogNormal(z[1],z[2]), μ_ss*(l/(l-θ_p))^(1/x[1]-1)))*nrow(df4)
    return -(L1+L2)
end
res_mle_ss = optimize(x -> L_ss(x), [0.4,0.4],LBFGS())
res_mle_ss.minimizer

zs = vcat(res_mle_fs.minimizer,res_mle_ss.minimizer)
simulator(zs)
plot(simulator(zs).works[:,1])

#SMM
#moments: E(log(w)|work = 1), E(log^2(w)|work = 1), lfpr_t|lfpr_t-1
function simulator(x)
    wages = exp.(x[1].+x[2]*ε)
    s_1 = (1+β)/(1-x[4])*(μ_ss^x[3]*l^(1-x[3]))^(1-x[4])
    s_2 = (wages[:,1].^x[3].*(l-θ_p)^(1-x[3])).^(1-x[4])./(1-x[4]) .+ β*(μ_ss^x[3]*l^(1-x[3]))^(1-x[4])/(1-x[4])
    s_3 = (wages[:,1].^x[3].*(l-θ_p)^(1-x[3])).^(1-x[4])./(1-x[4]) .+ β/(1-x[4])*((l-θ_p)^(1-x[3])*exp(x[1]*x[3]+0.5*(1-x[4])*x[2]^2*x[3]^2))^(1-x[4])
    works = zeros(Bool,N,2)
    works[:,1] = max.(s_2, s_3) .≥ s_1
    works[:,2] = works[:,1].*((wages[:,2].^x[3])*((l-θ_p)^(1-x[3])) .≥ μ_ss^x[3]*(l^(1-x[3]))) .== 1
    cons = wages.*works .+ μ_ss*(1 .- works)
    return (works = works, cons = cons)
end

#you may change the weighting matrix on your own
function moments(x;m = m)
    sim = simulator(x)
    #simulated moments
    dt = DataFrame(id = 1:N, work_1 = sim.works[:,1], work_2 = sim.works[:,2], 
                con_1 = sim.cons[:,1], con_2 = sim.cons[:,2])
    dt1 = filter(row -> row.work_1 == 1, dt)
    lc1 = sum(log.(dt1.con_1))
    lcsq1 = sum(log.(dt1.con_1).^2)
    dt2 = filter(row -> row.work_2 == 1, dt)
    lc2 = sum(log.(dt2.con_2))
    lcsq2 = sum(log.(dt2.con_2).^2)
    lc = (lc1 + lc2)/(nrow(dt1) + nrow(dt2))
    lcsq = (lcsq1 + lcsq2)/(nrow(dt1) + nrow(dt2))
    lfpr1 = mean(dt.work_1)
    lfpr2 = mean(dt.work_2)/lfpr1
    v = [lc, lcsq, lfpr1, lfpr2] - m
    return transpose(v)*opt_w_diag*v
end

#optimization
res_smm = optimize(x -> moments(x), [0.4,0.4,0.4,0.4])
res_smm.minimizer

#optimal weighting matrix
#bootstrap
ms = zeros(1000,4)
for i in 1:1000
    ds = df[shuffle(1:nrow(df)),:]
    ds1 = filter(row -> row.work_1 == 1, ds)
    rlc1 = sum(log.(ds1.con_1))
    rlcsq1 = sum(log.(ds1.con_1).^2)
    ds2 = filter(row -> row.work_2 == 1, ds)
    rlc2 = sum(log.(ds2.con_2))
    rlcsq2 = sum(log.(ds2.con_2).^2)
    rlc = (rlc1 + rlc2)/(nrow(ds1) + nrow(ds2))
    rlcsq = (rlcsq1 + rlcsq2)/(nrow(ds1) + nrow(ds2))
    rlfpr1 = mean(ds.work_1)
    rlfpr2 = mean(ds.work_2)/rlfpr1
    ms[i,:] = [rlc, rlcsq, rlfpr1, rlfpr2]
end
opt_w = inv(cov(ms))

#optimization
res_smm_opt = optimize(x -> moments(x), [0.4,0.4,0.4,0.4])
res_smm_opt.minimizer

opt_w_diag = Diagonal(opt_w)/tr(Diagonal(opt_w))