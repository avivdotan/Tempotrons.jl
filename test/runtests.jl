# Imports
using Tempotrons
using Tempotrons.InputGen
using Tempotrons.Plots
using Tempotrons.Optimizers
using Plots

# Set parameters
N = 10
T = 500
dt = 0.1
t = collect(0:dt:T)
λ = 0.01
opt = RMSprop(λ)
ν = 3
n_samples = 10
n_classes = 5
n_steps = 1000
tmp = Tempotron(N = N)

# Generate binary input samples
base_samples = [[PoissonSpikeTrain(ν = ν, T = T)
                 for i = 1:N]
                for j = 1:2]
samples = [(x = [SpikeJitter(s, T = T, σ = 5)
                 for s ∈ base_samples[2(j-1)÷n_samples + 1]],
            y = Bool(2(j-1)÷n_samples))
           for j = 1:n_samples]

# Train the tempotron
print("Training binary tempotron... ")
@time for i = 1:n_steps
    s = rand(samples)
    Train!(tmp, s.x, s.y, optimizer = opt)
end

# Generate multi-spike input samples
base_samples = [[PoissonSpikeTrain(ν = ν, T = T)
                 for i = 1:N]
                for j = 1:n_classes]
samples = [(x = [SpikeJitter(s, T = T, σ = 5)
                 for s ∈ base_samples[n_classes*(j - 1)÷n_samples + 1]],
            y = n_classes*(j - 1)÷n_samples)
           for j = 1:n_samples]

# Train the tempotron
print("Training multi-spike tempotron... ")
@time for i = 1:n_steps
    s = rand(samples)
    Train!(tmp, s.x, s.y, optimizer = opt)
end
