#programming and roy model

using Distributions, Plots, Random, DataFrames, CSV, GLM, Statistics, StatsPlots, Tables, StatFiles

#loading file
dt = load(raw"C:\Users\kevin\Downloads\rr2007_e_v201101(stata).dta") |> DataFrame
dt2 = DataFrame(work = (dt.a05 .== 1), age = dt.a02a) |> dropmissing!
#labor force participation conditional on age
labor_force_participation = combine(groupby(dt2, :age), :work => mean)
fig = plot(labor_force_participation.work_mean, labor_force_participation.age, label = "", ylabel = "age", xlabel = "labor force participation", title = "labor force participation conditional on age")
savefig(fig, "labor_force_participation.png")
#parameters

μ_0 = 0.0
μ_1 = 1.0
σ_0 = 1.0
σ_1 = 2.0
σ_01 = 1.5
c = 0.3
ρ = σ_01 / (σ_0 * σ_1)
σ_v = sqrt(σ_0^2 + σ_1^2 - 2*σ_01)

n = 10_000_000

#simulate data
ϵ_0 = σ_0 * randn(n)
ϵ_1 = σ_1 * randn(n)
w_0 = μ_0 .+ ϵ_0
w_1 = μ_1 .+ ϵ_1 
I = w_1 .- w_0 .> c

df = DataFrame(w_0 = w_0, w_1 = w_1, ϵ_0 = ϵ_0, ϵ_1 = ϵ_1, I = I)

#E[w_0|I]
avg_w_0_I = df[df.I .== 1, :].w_0 |> mean
#E[w_1|I]
avg_w_1_I = df[df.I .== 1, :].w_1 |> mean
#Q_0
Q_0 = df[df.I .== 1, :].ϵ_0 |> mean
#Q_1
Q_1 = df[df.I .== 1, :].ϵ_1 |> mean
#theoretical values
avg_w_0_I_theoretical = μ_0 + σ_0*σ_1/σ_v * (ρ - σ_0/σ_1) * pdf(Normal(0, 1), (μ_0-μ_1+c)/σ_v)/(1-cdf(Normal(0, 1), (μ_0-μ_1+c)/σ_v))
avg_w_1_I_theoretical = μ_1 + σ_0*σ_1/σ_v * (σ_1/σ_0 - ρ) * pdf(Normal(0, 1), (μ_0-μ_1+c)/σ_v)/(1-cdf(Normal(0, 1), (μ_0-μ_1+c)/σ_v))



