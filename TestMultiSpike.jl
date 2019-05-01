push!(LOAD_PATH, Base.Filesystem.dirname(
                 Base.Filesystem.dirname(
                 Base.source_path())))

using Tempotrons
using Tempotrons.Utils
using Tempotrons.Plots
using Plots
pyplot(size = (500, 1000))

N = 10
T = 500
dt = 1
t = collect(0:dt:T)
ν = 3
n_samples = 10
n_classes = 3
n_steps = 500
tmp = Tempotron(N = N)

C(g::ColorGradient) = RGB[g[z]
    for z = range(0, stop = 1, length = n_classes)]
clibrary(:misc)
g = :rainbow
cols = cgrad(g) |> C

samples = [([PoissonSpikeTrain(ν = ν, T = T)
             for i = 1:N], rand(1:n_classes) - 1)
           for j = 1:n_samples]
inp_plots = [PlotInputs(s[1], T_max = T, color = cols[s[2] + 1])
             for s ∈ samples]

out_b = [tmp(s[1], t = t) for s ∈ samples]

@time for i = 1:n_steps
    println("Sample: ", i)
    s = rand(samples)
    Train!(tmp, s[1], s[2], T_max = T)
end

out_a = [tmp(s[1], t = t) for s ∈ samples]

train_plots = [PlotPotential(tmp, out_b = out_b[i], out_a = out_a[i],
                             t = t, color = cols[samples[i][2] + 1])
                for i = 1:length(samples)]
ps = vcat(reshape(inp_plots, 1, :), reshape(train_plots, 1, :))
plot(ps[:]..., layout = (length(inp_plots), 2))
