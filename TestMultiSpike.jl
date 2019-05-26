push!(LOAD_PATH, Base.Filesystem.dirname(
                 Base.Filesystem.dirname(
                 Base.source_path())))

using Tempotrons
using Tempotrons.Utils
using Tempotrons.Plots
using Tempotrons.Optimizers
using Plots
pyplot(size = (500, 1000))

N = 10
T = 500
dt = 1
t = collect(0:dt:T)
ν = 3
λ = 0.001
opt = Adam(λ)
n_samples = 10
n_classes = 5
n_steps = 2500
tmp = Tempotron(N = N)

C(g::ColorGradient) = RGB[g[z]
    for z = range(0, stop = 1, length = n_classes)]
clibrary(:misc)
g = :rainbow
cols = cgrad(g) |> C

base_samples = [[PoissonSpikeTrain(ν = ν, T = T)
                 for i = 1:N]
                for j = 1:n_classes]
samples = [([SpikeJitter(s, T = T, σ = 5)
             for s ∈ base_samples[n_classes*(j - 1)÷n_samples + 1]],
            n_classes*(j - 1)÷n_samples)
           for j = 1:n_samples]
inp_plots = [PlotInputs(s[1], T_max = T, color = cols[s[2] + 1])
             for s ∈ samples]

out_b = [tmp(s[1], t = t) for s ∈ samples]

@time for i = 1:n_steps
    println("Sample: ", i)
    s = rand(samples)
    Train!(tmp, s[1], s[2], optimizer = opt, T_max = T)
end

out_a = [tmp(s[1], t = t) for s ∈ samples]

train_plots = [PlotPotential(tmp, out_b = out_b[i], out_a = out_a[i],
                             t = t, color = cols[samples[i][2] + 1])
                for i = 1:length(samples)]
ps = vcat(reshape(inp_plots, 1, :), reshape(train_plots, 1, :))
p = plot(ps[:]..., layout = (length(inp_plots), 2))
display(p)
filename(i) = "Results\\results" * string(i) * ".png"
let i = 0
    while isfile(filename(i))
        i += 1
    end
    savefig(filename(i))
end
