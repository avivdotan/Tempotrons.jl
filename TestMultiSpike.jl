# Imports
push!(LOAD_PATH, abspath("../"))
using Tempotrons
using Tempotrons.InputGen
using Tempotrons.Plots
using Tempotrons.Optimizers
using Plots
pyplot()

# Set parameters
N = 10
T = 500
dt = 0.1
t = collect(0:dt:T)
ν = 3
λ = 0.1
opt = RMSprop(λ)
n_samples = 10
n_classes = 5
n_steps = 1000
tmp = Tempotron(N = N)

# Generate input samples
base_samples = [[PoissonSpikeTrain(ν = ν, T = T)
                 for i = 1:N]
                for j = 1:n_classes]
samples = [(x = [SpikeJitter(s, T = T, σ = 5)
                 for s ∈ base_samples[n_classes*(j - 1)÷n_samples + 1]],
            y = n_classes*(j - 1)÷n_samples)
           for j = 1:n_samples]

# Get the tempotron's output before training
out_b = [tmp(s.x, t = t)[2] for s ∈ samples]

# Train the tempotron
@time for i = 1:n_steps
    @debug "Sample: $i"
    s = rand(samples)
    Train!(tmp, s.x, s.y, optimizer = opt, T_max = T)
end

# Get the tempotron's output after training
out_a = [tmp(s.x, t = t)[2] for s ∈ samples]

# # Prepare to plot
# pyplot(size = (700, 1000))
# C(g::ColorGradient) = RGB[g[z]
#     for z = range(0, stop = 1, length = n_classes)]
# clibrary(:misc)
# g = :rainbow
# cols = cgrad(g) |> C
#
# # Plots
# inp_plots = [PlotInputs(s.x, T_max = T, color = cols[s.y + 1])
#   for s ∈ samples]
# train_plots = [PlotPotential(tmp, out_b = out_b[i], out_a = out_a[i],
#                              t = t, color = cols[samples[i].y + 1])
#                 for i = 1:length(samples)]
# ps = vcat(reshape(inp_plots, 1, :), reshape(train_plots, 1, :))
# p = plot(ps[:]..., layout = (length(inp_plots), 2))
# display(p)
#
# # Save plots
# filename(i) = "Results\\results" * string(i) * ".png"
# let i = 0
#     while isfile(filename(i))
#         i += 1
#     end
#     savefig(p, filename(i))
# end
