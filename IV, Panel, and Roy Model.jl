using DataFrames, GLM, FixedEffectModels, Tables, DiffinDiffs, FixedEffects, Vcov, RCall, Statistics

#simulate the data
n = 10000
T = 10
β = -2.0
σ_ϵ = 1.5
σ_α = 2.5
σ_δ = 0.5
ρ = 0.2
ϵ = σ_ϵ*randn(n,T)
for t in 2:T
    ϵ[:,t] = ρ*ϵ[:,t-1] + randn(n)
end
d = floor.(Int,T*rand(n)) .+2
α = σ_α*randn(n)
δ = σ_δ*randn(T)
y = zeros(n,T)
for t in 1:T
    y[:,t] = 1.0 .+ α .+ δ[t] .+ β*(t .≥ d).*(t .- d .+ 1) .+ ϵ[:,t]
end

#construct panel 
df = DataFrame(id = vec(transpose(repeat(1:n, 1, T))), t = repeat(1:T,n), d = vec(transpose(repeat(d,1,T))), y = vec(transpose(y)))
treated = df.d .≤ df.t
df.treated = treated
vscodedisplay(df)

#run two way fixed effects
reg(df, @formula(y ~ 1 + fe(t) + fe(id) + treated))
#ur... Not bad?

#run two way fixed effects dropping the post-treated group
post_treated = df.t .> df.d
df.post_treated = post_treated
dt = filter(row -> row.t ≤ row.d, df)
reg(dt, @formula(y ~ 1 + fe(t) + fe(id) + treated), Vcov.cluster(:id))

#somehow there are some problems with dof_stat in Vcov.jl
#somehow fixed! such a weird thing
res = @did(Reg, data = df, dynamic(:t, -1), notyettreated(11), 
    vce = Vcov.cluster(:id), yterm = term(:y), treatname = :d, 
    treatintterms=(), xterms=(fe(:t)+fe(:id)))
agg(res, :rel)

#example codes
hrs = DiffinDiffsBase.exampledata("hrs")
r = @did(Reg, data=hrs, dynamic(:wave, -1), notyettreated(11),
    vce=Vcov.cluster(:hhidpn), yterm=term(:oop_spend), treatname=:wave_hosp,
    treatintterms=(), xterms=(fe(:wave)+fe(:hhidpn)))
agg(r)