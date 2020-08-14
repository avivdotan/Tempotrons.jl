# Imports
using Tempotrons
using Tempotrons.InputGen
using Tempotrons.Optimizers
using Random
using ProgressMeter
using Plots
using Plots.PlotMeasures

# Set parameters
N = 500
Tᶲ = 500
dt = 1
ν = 5
λ = 1e-5
opt = SGD(λ, momentum = 0.99)
Nᶠ = 10
Tᶠ = 50
Cᶠ_mean = 0.5
n_steps = 20000
n_train_samples = 10000
n_test_samples = 10
tmp = Tempotron(N, τₘ = 20)
pretrain!(tmp)

# Set teacher's rule
teacher_type = 1
teachers = []
push!(teachers, tps -> isempty(tps) ? 0 : length(filter(x -> x == 2, tps)))
push!(teachers, tps -> isempty(tps) ? 0 : 5 * length(filter(x -> x == 2, tps)))
push!(teachers, tps -> isempty(tps) ? 0 : length(filter(x -> x % 2 == 0, tps)))
push!(teachers, tps -> isempty(tps) ? 0 : sum(filter(x -> x % 2 == 0, tps) / 2))
y₀ = teachers[teacher_type]

# Input features
features = get_features(Nᶠ = Nᶠ, Tᶠ = Tᶠ, N = N, ν = ν)

# Test samples
test_samples = [get_embedded_events_sample(features, Tᶠ = Tᶠ, Cᶠ_mean = Cᶠ_mean,
                                           ν = ν, Tᶲ = Tᶲ)
                for j = 1:(n_test_samples - 1)]
push!(test_samples,
      get_embedded_events_sample(features, Tᶠ = Tᶠ, Cᶠ_mean = 0.1, ν = ν,
                                 Tᶲ = 5Tᶠ, test = true))
test_samples = [(ts..., y = y₀([f.type for f ∈ ts.features]),
                 t = collect((ts.x.duration.from):dt:(ts.x.duration.to)))
                for ts ∈ test_samples]

# Plot inputs
gr(size = (800, 1500))
cols = map(c -> mapc(x -> min(x + 0.15, 1), c), palette(:rainbow, Nᶠ))
neur_disp = randsubseq(1:N, 0.1)

inp_plots = map(test_samples) do s
    p = plot(s.x, reduce_afferents = neur_disp, markersize = sqrt(5))
    for f ∈ s.features
        plot!(f.duration, color = cols[f.type])
    end
    return p
end
p = plot(inp_plots[1:(end - 1)]..., layout = (length(inp_plots) - 1, 1),
         link = :all)
p = plot(p, inp_plots[end],
         layout = grid(2, 1,
                       heights = [1 - 1 / length(inp_plots),
                                  1 / length(inp_plots)]), left_margin = 8mm)
display(p)

# Training inputs
train_samples = @showprogress(1, "Generating samples...",
                              [get_embedded_events_sample(features, Tᶠ = Tᶠ,
                                                          Cᶠ_mean = Cᶠ_mean,
                                                          ν = ν, Tᶲ = Tᶲ)
                               for j = 1:n_train_samples])
train_samples = [(ts..., y = y₀([f.type for f ∈ ts.features]))
                 for ts ∈ train_samples]

# Train the tempotron
@showprogress 1 "Training..." for i = 1:n_steps
    s = rand(train_samples)
    train!(tmp, s.x, s.y, optimizer = opt)
end

# Voltage traces
out_a = @showprogress 1 "Evaluating test samples..." [tmp(s.x, t = s.t)
                                                      for s ∈ test_samples]

# Plot
inp_plots = map(test_samples) do s
    p = plot(s.x, reduce_afferents = neur_disp, markersize = sqrt(5))
    for f ∈ s.features
        plot!(f.duration, color = cols[f.type])
    end
    return p
end
train_plots = map(zip(test_samples, out_a)) do (s, oa)
    p = plot(tmp, s.t, oa.V)
    for f ∈ s.features
        plot!(f.duration, color = cols[f.type])
    end
    txt, clr = Tempotrons.get_progress_annotations(length(oa.spikes), N_t = s.y)
    annotate!(xlims(p)[1], ylims(p)[2], text(txt, 10, :left, :bottom, clr))
    return p
end
ip = plot(inp_plots..., layout = (length(inp_plots), 1), link = :all)
tp = plot(train_plots..., layout = (length(train_plots) - 1, 1), link = :all)
p = plot(ip, tp, layout = (1, 2), link = :all)
display(p)

# Save plots
filename(i) = "examples\\Results\\AggLabels_T" *
              string(teacher_type) *
              "_" *
              string(i)
let i = 0
    while isfile(filename(i) * ".png") || isfile(filename(i) * ".svg")
        i += 1
    end
    savefig(p, filename(i) * ".png")
    savefig(p, filename(i) * ".svg")
end
