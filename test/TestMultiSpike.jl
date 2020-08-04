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
ν = 3
λ = 0.01
opt = SGD(λ)
n_samples = 10
n_classes = 5
n_steps = 5000
tmp = Tempotron(N = N)

# Generate input samples
base_samples = [[PoissonProcess(ν = ν, T = T)
                 for i = 1:N]
                for j = 1:n_classes]
samples = [(x = [SpikeJitter(s, T = T, σ = 5)
                 for s ∈ base_samples[n_classes*(j - 1)÷n_samples + 1]],
            y = n_classes*(j - 1)÷n_samples)
           for j = 1:n_samples]

# Get the tempotron's output before training
out_b = [tmp(s.x, t = t) for s ∈ samples]

# Get STS before training
θ⃰_b = [GetSTS(tmp, s.x) for s ∈ samples]

# Train the tempotron
@time for i = 1:n_steps
    s = rand(samples)
    Train!(tmp, s.x, s.y, optimizer = opt)
end

# Get the tempotron's output after training
out_a = [tmp(s.x, t = t) for s ∈ samples]

# Get STS after training
θ⃰_a = [GetSTS(tmp, s.x) for s ∈ samples]

# Prepare to plot
pyplot(size = (1000, 1200))
cols = collect(1:n_classes)#palette(:rainbow, n_classes)

# Plots
inp_plots = [PlotInputs(s.x, color = cols[s.y + 1])
             for s ∈ samples]
train_plots = [PlotPotential(tmp, out_b = ob.V, out = oa.V,
                             N_b = length(ob.spikes), N = length(oa.spikes),
                             N_t = s.y, t = t, color = cols[s.y + 1])
               for (s, ob, oa) ∈ zip(samples, out_b, out_a)]
STS_plots = [PlotSTS(tmp, θ⃰_b = θ⃰_o, θ⃰ = θ⃰_n,
                     color = cols[s.y + 1])
             for (s, θ⃰_o, θ⃰_n) ∈ zip(samples, θ⃰_b, θ⃰_a)]
ps = [reshape(inp_plots, 1, :);
      reshape(train_plots, 1, :);
      reshape(STS_plots, 1, :)]
p = plot(ps[:]..., layout = (length(inp_plots), 3), link = :x)
display(p)

# # Save plots
# filename(i) = "Results\\results" * string(i) * ".png"
# let i = 0
#     while isfile(filename(i))
#         i += 1
#     end
#     savefig(p, filename(i))
# end
