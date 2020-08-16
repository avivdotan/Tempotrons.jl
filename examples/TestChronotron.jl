# Imports
using Tempotrons
using Tempotrons.InputGen
using Tempotrons.Optimizers
using Plots

# Set parameters
N = 100
T = 500
dt = 1
t = collect(0:dt:T)
method = :∇
λ = 0.01
opt = SGD(λ)
ν = 5
n_samples = 10
n_epochs = 500
tmp = Tempotron(N)

# Generate input samples
samples = [(x = poisson_spikes_input(N, ν = ν, T = T),
            y = poisson_spikes_input(1, ν = 3, T = TimeInterval(0.1T, T)))
           for j = 1:n_samples]

# Get the tempotron's output before training
out_b = tmp([s.x for s ∈ samples], t = t)

# Train the tempotron
@time train!(tmp, samples, epochs = n_epochs, optimizer = opt, method = method)

# Get the tempotron's output after training
out_a = tmp([s.x for s ∈ samples], t = t)

# Plots
gr(size = (800, 1200))

train_plots = map(zip(samples, out_b, out_a)) do (s, ob, oa)
    p = plot(tmp, t, oa.V)
    plot!(tmp, t, ob.V, linestyle = :dash)
    yl = ylims()
    for t_spk ∈ s.y[1]
        plot!(t_spk .* [1, 1], [yl[1], yl[2]], color = :salmon,
              linestyle = :dash, linewidth = 2)
    end
    txt, clr = Tempotrons.get_progress_annotations(vp_distance(oa.spikes,
                                                               s.y[1],
                                                               τ_q = tmp.τₘ),
                                                   N_b = vp_distance(ob.spikes,
                                                                     s.y[1],
                                                                     τ_q = tmp.τₘ),
                                                   desc = "VP distance")
    annotate!(xlims(p)[1], ylims(p)[2], text(txt, 10, :left, :bottom, clr))
    return p
end
tp = plot(train_plots..., layout = (length(train_plots), 1), link = :all)
display(tp)
