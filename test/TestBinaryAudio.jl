# Imports
using Tempotrons
using Tempotrons.InputGen
using Tempotrons.Plots
using Tempotrons.Optimizers
using MAT
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
n_samples = length(files)

λ = 0.05
opt = SGD(λ)
n_steps = 50

# Generate input samples
samples = [(data = LoadDataFile(joinpath(data_path, f.fname), 1),
            y0 = f.y0)
           for f ∈ files]
samples = [(t = s.data.t,
            x = s.data.x,
            y = s.y0)
           for s ∈ samples]

# Create a tempotron
N = length(samples[1].x)
for s ∈ samples
    @assert length(s.x) == N
end
tmp = Tempotron(N = N)

# Get the tempotron's output before training
out_b = [tmp(s.x, t = s.t).V for s ∈ samples]

# Train the tempotron
@time for i = 1:n_steps
    s = rand(samples)
    Train!(tmp, s.x, s.y, optimizer = opt)
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
