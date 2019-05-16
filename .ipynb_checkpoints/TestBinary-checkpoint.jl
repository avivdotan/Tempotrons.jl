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
dt = 0.1
t = collect(0:dt:T)
ν = 3
n_samples = 10
n_steps = 5000
tmp = Tempotron(N = N)

# inp1 = [[70, 200, 400], [], [400, 420], [], [110], [230], [240, 260, 340], [380], [300], [105]]
# inp2 = [[], [395], [50, 170], [], [70, 280], [], [290], [115], [250, 320], [225, 330]]
# samples = [(inp1, true), (inp2, false)]
samples = [([PoissonSpikeTrain(ν = ν, T = T)
             for i = 1:N],
            rand(Bool))
           for j = 1:n_samples]
inp_plots = [PlotInputs(s[1], T_max = T, color = (s[2] ? :red : :blue))
             for s ∈ samples]

out_b = [tmp(s[1], t = t) for s ∈ samples]

@time for i = 1:n_steps
    s = rand(samples)
    Train!(tmp, s[1], s[2], T_max = T)
end

out_a = [tmp(s[1], t = t) for s ∈ samples]

train_plots = [PlotPotential(tmp, out_b = out_b[i], out_a = out_a[i],
                             t = t, color = (samples[i][2] ? :red : :blue))
                for i = 1:length(samples)]
ps = vcat(reshape(inp_plots, 1, :), reshape(train_plots, 1, :))
plot(ps[:]..., layout = (length(inp_plots), 2))
