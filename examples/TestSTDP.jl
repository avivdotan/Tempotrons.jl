# Imports
using Tempotrons
using Tempotrons.InputGen
using Tempotrons.Optimizers
using Plots

# Set parameters
N = 100
T = 500
μ = 0.02
method = :corr
λ = 1e-3
opt = SGD(λ)
ν = 10
n_samples = 1000
tmp = Tempotron(N)
tmp.w .= 0.5ones(N)

# Generate input sample
samples = [poisson_spikes_input(N, ν = ν, T = T) for i = 1:n_samples]

# Train the neuron
@time train!(tmp, samples, optimizer = opt, method = method, μ = μ)

# Plot
gr(size = (800, 400))
hist_plot =
    histogram(tmp.w, bins = -0.1:0.1:1.1, xlabel = "weight", ylabel = "count", label = "")
display(hist_plot)
