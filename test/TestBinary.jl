# Imports
using Tempotrons
using Tempotrons.InputGen
using Tempotrons.Plots
using Tempotrons.Optimizers
using Plots

# Set parameters
N = 10
T = 500
dt = 1
t = collect(0:dt:T)
method = :∇
# method = :corr
λ = 0.01
opt = SGD(λ)
ν = 3
n_samples = 10
n_steps = 5000
tmp = Tempotron(N = N)

# Generate input samples
base_samples = [[PoissonProcess(ν = ν, T = T)
                 for i = 1:N]
                for j = 1:2]
samples = [(x = [SpikeJitter(s, T = T, σ = 5)
                 for s ∈ base_samples[2(j-1)÷n_samples + 1]],
            y = Bool(2(j-1)÷n_samples))
           for j = 1:n_samples]

# Get the tempotron's output before training
out_b = [tmp(s.x, t = t) for s ∈ samples]

# Train the tempotron
@time for i = 1:n_steps
    s = rand(samples)
    Train!(tmp, s.x, s.y, optimizer = opt, method = method)
end

# Get the tempotron's output after training
out_a = [tmp(s.x, t = t) for s ∈ samples]

# Plots
gr(size = (800, 1200))
cols = collect(1:2)#palette(:rainbow, 2)

inp_plots = [PlotInputs(s.x, color = cols[1 + s.y])
             for s ∈ samples]
train_plots = [PlotPotential(tmp, out_b = ob.V, out = oa.V,
                             N_b = length(ob.spikes), N = length(oa.spikes),
                             t = t, color = cols[1 + s.y])
               for (s, ob, oa) ∈ zip(samples, out_b, out_a)]
ip = plot(inp_plots..., layout = (length(inp_plots), 1), link = :all)
tp = plot(train_plots..., layout = (length(train_plots), 1), link = :all)
p = plot(ip, tp, layout = (1, 2))
display(p)
