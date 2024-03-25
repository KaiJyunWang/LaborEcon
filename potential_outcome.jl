using Distributions, Plots, LinearAlgebra, Statistics, Optim, DataFrames
using GLM, Tables, RegressionTables

#1.4
σ_1 = 1
σ_2 = 2
function y_generator(n; σ_1 = σ_1, σ_2 = σ_2)
    x_1 = rand(Normal(0, σ_1), n)
    x_2 = rand(Normal(0, σ_2), n)
    return x_1 + x_2
end

y = y_generator(2)

#likelihood function
L(s) = sum(log.(pdf.(Normal(0, sqrt(s[1]^2 + s[2]^2)), y)))
L([1.0, 1.0])
res = optimize(s -> -L(s), [2.0, 1.0], LBFGS())
res.minimizer

#3.2
μ_0 = 0.0
μ_1 = 1.0
β_1 = 1.0
β_2 = 2.0
n = 5000
σ_0 = 0.5
σ_1 = 1.0
ρ = 0.3
ϵ = rand(MvNormal([σ_0^2 ρ*σ_0*σ_1; ρ*σ_0*σ_1 σ_1^2]), n)
x_1 = rand(Normal(0, 1), n)
x_2 = rand(Normal(0, 1), n)
w_0 = μ_0 .+ β_1*x_1 + ϵ[1, :]
w_1 = μ_1 .+ β_1*x_1 + β_2*x_2 + ϵ[2, :]
i = w_1 .> w_0
df = DataFrame(x_1 = x_1, x_2 = x_2, w_0 = w_0, w_1 = w_1, i = i, w = w_1 .* i .+ w_0 .*(1 .- i), p = cdf.(Normal(),(μ_1 - μ_0 .+ β_2*x_2)./sqrt(σ_0^2 + σ_1^2 - 2*ρ*σ_0*σ_1)))

#logit estimated prospencity score
res = glm(@formula(i ~ x_1 + x_2), df, Binomial(), LogitLink())
p_hat = predict(res)
df[!,:p_hat] = p_hat
cor(p_hat, df.p)
ipw = mean(df.i .* df.w ./ df.p .+ (1 .- df.i) .* df.w ./ (1 .- df.p))
ipw_hat = mean(df.i .* df.w ./ p_hat .+ (1 .- df.i) .* df.w ./ (1 .- p_hat))
lm1 = lm(@formula(w ~ x_1 + x_2), df)
lm2 = lm(@formula(w ~ x_1 + x_2 + p_hat), df)
