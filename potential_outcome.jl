using Distributions, Plots, LinearAlgebra, Statistics, Optim

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
