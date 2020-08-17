# Imports
using Tempotrons
using Tempotrons.InputGen
using Tempotrons.Optimizers
using Plots

# Set parameters
N = 10
T = 500
dt = 1
t = collect(0:dt:T)
ν = 3
method = :∇
# method = :corr
λ = 1e-4
opt = SGD(λ, momentum = 0.99)
n_samples = 10
n_classes = 5
n_epochs = 2000
tmp = Tempotron(N)

# Generate input samples
base_samples = [poisson_spikes_input(N, ν = ν, T = T) for j = 1:n_classes]
samples = [(x = spikes_jitter(base_samples[n_classes * (j - 1) ÷ n_samples + 1],
                              σ = 5), y = n_classes * (j - 1) ÷ n_samples)
           for j = 1:n_samples]

# Get the tempotron's output before training
out_b = tmp([s.x for s ∈ samples], t = t)

# Get STS before training
θ⃰_b = [get_sts(tmp, s.x) for s ∈ samples]

# Train the tempotron
@time train!(tmp, samples, epochs = n_epochs, optimizer = opt, method = method)

# Get the tempotron's output after training
out_a = tmp([s.x for s ∈ samples], t = t)

# Get STS after training
θ⃰_a = [get_sts(tmp, s.x) for s ∈ samples]

# Prepare to plot
gr(size = (1000, 1200))
cols = collect(1:n_classes)#palette(:rainbow, n_classes)

# Plots
inp_plots = map(samples) do s
    return plot(s.x, color = cols[1 + s.y], markersize = sqrt(5))
end
train_plots = map(zip(samples, out_b, out_a)) do (s, ob, oa)
    p = plot(tmp, t, oa.V, color = cols[1 + s.y])
    plot!(tmp, t, ob.V, color = cols[1 + s.y], linestyle = :dash)
    txt, clr = Tempotrons.get_progress_annotations(length(oa.spikes),
                                                   N_b = length(ob.spikes),
                                                   N_t = s.y)
    annotate!(xlims(p)[1], ylims(p)[2], text(txt, 10, :left, :bottom, clr))
    return p
end
STS_plots = map(zip(samples, θ⃰_b, θ⃰_a)) do (s, θb, θa)
    p = plotsts(tmp, θa, color = cols[1 + s.y])
    plotsts!(tmp, θb, color = cols[1 + s.y], linestyle = :dash)
    return p
end
ip = plot(inp_plots..., layout = (length(inp_plots), 1), link = :all)
tp = plot(train_plots..., layout = (length(train_plots), 1), link = :all)
sp = plot(STS_plots..., layout = (length(STS_plots), 1), link = :all)
p = plot(ip, tp, sp, layout = (1, 3))
display(p)

# # Save plots
# filename(i) = "Results\\results" * string(i) * ".png"
# let i = 0
#     while isfile(filename(i))
#         i += 1
#     end
#     savefig(p, filename(i))
# end
