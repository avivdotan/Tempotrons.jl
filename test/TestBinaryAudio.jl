# Imports
using Random, MAT
using Tempotrons
using Tempotrons.InputGen
using Tempotrons.Plots
using Tempotrons.Optimizers
using Plots

function LoadDataFile(filename, batch_num)

data_dict = matread(filename)
spikeTrains = data_dict["batch_items"][batch_num, :, :]
t_vec = collect(0:data_dict["dt_sim"]:data_dict["t_final"])
@assert size(spikeTrains, 2) == length(t_vec)

return (t = t_vec,
        x = [t_vec[spikeTrains[n, :]] for n = 1:size(spikeTrains, 1)])

end

# Set parameters
data_path = joinpath("test", "Data")
files = [(fname = "44_-4SYC2YgzL8.mat", y0 = true),
         (fname = "78_-9phJ0sJrXg.mat", y0 = false)]
n_base_samples = length(files)
n_samples_per_base = 5
n_samples = n_base_samples * n_samples_per_base

λ = 0.05
opt = SGD(λ)
n_epochs = 50

# Get input samples
base_samples = [(data = LoadDataFile(joinpath(data_path, f.fname), 1),
                 y0 = f.y0)
                for f ∈ files]
base_samples = [(t = s.data.t,
                 x = s.data.x,
                 y = s.y0)
                for s ∈ base_samples]
samples = [(t = bs.t,
            x = [SpikeJitter(st, T = maximum(bs.t), σ = 5)
                  for st ∈ bs.x],
            y = bs.y)
           for bs ∈ base_samples
           for j = 1:n_samples_per_base]

# Create a tempotron
N = length(samples[1].x)
for s ∈ samples
    @assert length(s.x) == N
end
tmp = Tempotron(N = N)

# Get the tempotron's output before training
out_b = [tmp(s.x, t = s.t).V for s ∈ samples]

# Train the tempotron
@time for i = 1:n_epochs
    p = randperm(n_samples)
    for s ∈ samples[p]
        Train!(tmp, s.x, s.y, optimizer = opt)
    end
end

# Get the tempotron's output after training
out_a = [tmp(s.x, t = s.t).V for s ∈ samples]

# Plots
pyplot(size = (700, 1000))
inp_plots = [PlotInputs(s.x, color = (s.y ? :red : :blue))
             for s ∈ samples]
train_plots = [PlotPotential(tmp, out_b = out_b[i], out = out_a[i],
                             t = samples[i].t,
                             color = (samples[i].y ? :red : :blue))
               for i = 1:length(samples)]
ps = [reshape(inp_plots, 1, :);
      reshape(train_plots, 1, :)]
p = plot(ps[:]..., layout = (length(inp_plots), 2), link = :x)
display(p)
