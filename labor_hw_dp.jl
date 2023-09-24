using Random, DataFrames, Plots, Statistics, Distributions
using Optim, Parameters, LaTeXStrings

#setting parameters
N = 5000 
μ_ss = 0.5
μ_w = 0.5
γ = 0.5
ν = 0.5
σ = 1
β = 0.9
Random.seed!(123)
ε = σ*randn(N)


work = (exp.(μ_w.+ε).^γ)*((8760-2080)^(1-γ)) .> μ_ss^γ*(8760^(1-γ))
wage = exp.(μ_w.+ε)
log_wage = μ_w.+ε

df = DataFrame(id = 1:N, work = work, wage = wage, 
                work_util = exp.(μ_w.+ε).^γ.*((8760-2080)^(1-γ)), 
                not_work_util = fill(μ_ss^γ*8760^(1-γ), N))
plt = scatter(wage, work, xlabel = "wage", label = "1:to work\n0:not to work")
savefig(plt, "wage_worked.png")

dt = filter(row -> row.work==1, df)
mean(log.(dt.wage))
a = filter(row -> row.work==1, df)
mean(a.wage)

#MLE
f(x, w) = pdf(LogNormal(x[1], σ), w)
F(x) = cdf(LogNormal(x[1], σ), μ_ss*(8760/(8760-2080))^(1-γ))
L(x) = sum(log.(f.(x, wage)).*work .+ log(F(x)).*(1 .- work))
L(0.5)
ŵ = Optim.maximizer(maximize(x -> L(x), [10.0], LBFGS()))

#new model, two period case
#setting parameters

Random.seed!(123)
new_ε = rand(N, 2)

ep = quantile.(Normal(0, σ), new_ε)
wage_1 = exp.(μ_w.+quantile.(Normal(0, σ), new_ε)[:,1])
wage_2 = exp.(μ_w.+quantile.(Normal(0, σ), new_ε)[:,2])

function work_solver(params)
    (;μ_w, γ, ν, σ, β, ε, μ_ss, wage_1, wage_2) = params
    c_1 = (1+β)/(1-ν)*(μ_ss^γ*8760^(1-γ))^(1-ν)
    c_2 = (wage_1.^γ.*(8760-2080)^(1-γ)).^(1-ν)./(1-ν) .+ β*(μ_ss^γ*8760^(1-γ))^(1-ν)/(1-ν)
    c_3 = (wage_1.^γ.*(8760-2080)^(1-γ)).^(1-ν)./(1-ν) .+ β/(1-ν)*((8760-2080)^(1-γ)*exp(μ_w*γ+0.5*(1-ν)*(σ*γ)^2))^(1-ν)
    work_1 = max.(c_2, c_3) .≥ c_1
    work_2 = work_1.*((wage_2.^γ)*((8760-2080)^(1-γ)) .≥ μ_ss^γ*(8760^(1-γ)))
    con_1 = wage_1.*work_1 .+ μ_ss*(1 .- work_1)
    con_2 = wage_2.*work_2 .+ μ_ss*(1 .- work_2)
    return (work_1, work_2, con_1, con_2)
end

wp = @with_kw (μ_w = 0.5, γ = 0.5, ν = 0.5, σ = 1, β = 0.9, ε = new_ε, μ_ss = 0.5,
                wage_1 = exp.(μ_w.+quantile.(Normal(0, σ), new_ε)[:,1]), 
                wage_2 = exp.(μ_w.+quantile.(Normal(0, σ), new_ε)[:,2]))

work_1, work_2, con_1, con_2 = work_solver(wp())


df = DataFrame(id = 1:N, work_1 = work_1, work_2 = work_2, 
                wage_1 = wage_1, wage_2 = wage_2, 
                con_1 = con_1, con_2 = con_2, 
                work_util_1 = wage_1.^γ.*(8760-2080)^(1-γ), 
                not_work_util_1 = fill(μ_ss^γ*8760^(1-γ), N), 
                work_util_2 = wage_2.^γ.*(8760-2080)^(1-γ), 
                not_work_util_2 = fill(μ_ss^γ*8760^(1-γ), N))
describe(df.con_1)
std(df.con_1)
describe(df.con_2)
std(df.con_2)
cor(df.con_1, df.work_1)
cor(df.con_2, df.work_2)

#plot function but still need to modify the parameters on your own
function plotter(vals, xlabel)
    plt = plot()
    employment_rate = zeros(length(vals),2)
    consumption = zeros(length(vals),2)
    cors = zeros(length(vals),2)
    for (i, v) in enumerate(vals)
        wp = @with_kw (μ_w = 0.5, γ = 0.5, ν = 0.5, σ = 0.9, β = 0.9, ε = new_ε, 
                μ_ss = v, wage_1 = exp.(μ_w.+quantile.(Normal(0, σ), new_ε)[:,1]), 
                wage_2 = exp.(μ_w.+quantile.(Normal(0, σ), new_ε)[:,2]))
        work_1, work_2, con_1, con_2 = work_solver(wp())
        employment_rate[i,1] = mean(work_1)
        employment_rate[i,2] = mean(work_2)
        consumption[i,1] = mean(con_1)
        consumption[i,2] = mean(con_2)
        cors[i,1] = cor(work_1, con_1)
        cors[i,2] = cor(work_2, con_2)
    end
    plot!(vals, employment_rate[:,1], xlabel = xlabel, ylabel = "employment rate", label = L"emp_1", legend = :topleft)
    plot!(vals, employment_rate[:,2], ylabel = "employment rate", label = L"emp_2", legend = :topleft)
    p = twinx()
    plot!(p, vals, consumption[:,1], ylabel = "consumption", label = L"con_1", color = :purple, legend = :bottomright)
    plot!(p, vals, consumption[:,2], ylabel = "consumption", label = L"con_2", color = :green, legend = :bottomright)
    plot!(vals, cors[:,1], label = L"cor_1", color = :brown, lw = 2)
    plot!(vals, cors[:,2], label = L"cor_2", color = :black, lw = 2)
    return plt
end

vals1 = range(0.1, 1.0, length = 101)
vals2 = range(0.0, 1.0, length = 101)
vals3 = range(0.1, 2.0, length = 101)
vals4 = range(0.1, 2.0, length = 101)
vals5 = range(0.1, 1.0, length = 101)
vals6 = range(0.1, 1.0, length = 101)
plt1 = plotter(vals1, L"μ_w")
plt2 = plotter(vals2, "γ")
plt3 = plotter(vals3, "ν")
plt4 = plotter(vals4, "σ")
plt5 = plotter(vals5, "β")
plt6 = plotter(vals6, L"μ_{ss}")
plt = plot(plt1, plt2, plt3, plt4, plt5, plt6, layout = (3,2), size = (800, 600))

savefig(plt, "cs.png")


