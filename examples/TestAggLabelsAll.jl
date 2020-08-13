# Imports
using Tempotrons
using Tempotrons.InputGen
using Tempotrons.Optimizers
using Random
using Statistics
using ProgressMeter
using Plots
using Plots.PlotMeasures

# Set parameters
N       = 500
τₘ      = 20
Tᶲ      = 500
dt      = 1
ν       = 5
λ       = 1e-5
opt     = SGD(λ, momentum = 0.99)
Nᶠ      = 10
Tᶠ      = 50
T_probe = 2000
N_probe = 50
ν_probe = 1000
Cᶠ_mean = 0.5
n_steps = 20000
tmp     = Tempotron(N, τₘ = τₘ)

# Set teacher's rule
teacher_type = 1
teachers = []
push!(teachers, tps -> isempty(tps) ? 0 : length(filter(x -> x == 2, tps)))
push!(teachers, tps -> isempty(tps) ? 0 : 5*length(filter(x -> x == 2, tps)))
push!(teachers, tps -> isempty(tps) ? 0 : length(filter(x -> x%2 == 0, tps)))
push!(teachers, tps -> isempty(tps) ? 0 : sum(filter(x -> x%2 == 0, tps))÷2)

# Input features
features = get_features(Nᶠ = Nᶠ, Tᶠ = Tᶠ, N = N, ν = ν)

# validation samples
fᶲ = poisson_spikes_input(N, ν = ν, T = Tᶠ)
# valid_samples = [poisson_spikes_input(N, ν = ν, T = T_probe) for i = 1:N_probe]
# valid_samples = [[Tempotrons.Inputs.insert_spikes_input!(vs, f, T_probe/2)
#                   for f ∈ [features..., fᶲ]]
#                  for vs ∈ valid_samples]
valid_samples = map(1:N_probe) do i
    vs = poisson_spikes_input(N, ν = ν, T = T_probe)
    return map([features..., fᶲ]) do f
        vsf = deepcopy(vs)
        Tempotrons.Inputs.insert_spikes_input!(vsf, f, T_probe/2)
        return vsf
    end
end

# Test sample
test_sample = get_embedded_events_sample(features, Tᶠ = Tᶠ, Cᶠ_mean = Cᶠ_mean,
                                             ν = ν, Tᶲ = 5Tᶠ, test = true)
test_sample = (test_sample...,
               t = collect(test_sample.x.duration.from:dt:test_sample.x.duration.to))

test_out_b = tmp(test_sample.x, t = test_sample.t)

# Train the tempotrons
valid_R = []
res = map(1:length(teachers)) do k

    # Set the teacher
    y₀ = teachers[k]

    # Pretrain the tempotron
    # Pretrain!(tmp)

    # Train the tempotron
    @showprogress 1 "Training tempotron #$k..." for i = 1:n_steps
        s = get_embedded_events_sample(features, Tᶠ = Tᶠ, Cᶠ_mean = Cᶠ_mean,
                                       ν = ν, Tᶲ = Tᶲ)
        Train!(tmp, s.x, y₀([f.type for f ∈ s.features]), optimizer = opt)

        if i % ν_probe == 0
            valid_spk = [[length(tmp(f).spikes) for f ∈ vs]
                         for vs ∈ valid_samples]
            valid_spk = hcat(valid_spk...)
            foreach(f -> valid_spk[f, :] .- valid_spk[end, :], 1:Nᶠ)
            R_mean = dropdims(mean(valid_spk[1:Nᶠ, :], dims = 2), dims = 2)
            push!(R_mean, mean(valid_spk[end, :])./T_probe)
            R_std = dropdims(std(valid_spk[1:Nᶠ, :], mean = R_mean[1:Nᶠ], dims = 2), dims = 2)
            push!(R_std, std(valid_spk[end, :])./T_probe)
            push!(valid_R, (mean = R_mean, std = R_std))
        end
    end

    out_a = tmp(test_sample.x, t = test_sample.t)

    return (test_out_a  = out_a,
            test_teach  = y₀([f.type for f ∈ test_sample.features]),
            teacher     = y₀)

end

# Rearrange validation results
R_means = hcat([r.mean for r ∈ valid_R]...)'
R_stds  = hcat([r.std  for r ∈ valid_R]...)'

# Plot inputs
gr(size = (700, 1000))
cols = [mapc(x -> min(x + 0.15, 1), c)
        for c ∈ [palette(:rainbow, Nᶠ)...,
                 Colors.parse(Colors.Colorant, Tempotrons.fg_color())]]
neur_disp = randsubseq(1:N, 0.1)

# Plot
inp_plot = plot(test_sample.x,
                reduce_afferents = neur_disp,
                markersize = sqrt(5))
for f ∈ test_sample.features
    plot!(f.duration, color = cols[f.type])
end
pre_plot = plot(tmp, test_sample.t, test_out_b.V)
train_plots = map(res) do r
    p = plot(tmp, test_sample.t, r.test_out_a.V)
    for f ∈ test_sample.features
        if r.teacher([f.type]) > 0
            plot!(f.duration, color = cols[f.type])
        end
    end
    txt, clr = Tempotrons.get_progress_annotations(length(r.test_out_a.spikes),
                                                   N_t = r.test_teach)
    annotate!(xlims(p)[1], ylims(p)[2],
              text(txt, 10, :left, :bottom, clr))
    return p
end
hist_t = vcat([collect((1:(n_steps÷ν_probe))*ν_probe .+ (i - 1)*n_steps)
          for i = 1:length(teachers)]...)
hist_t = [hist_t for i = 1:(Nᶠ + 1)]
hist_plot = plot(hist_t, R_means, ribbon = R_stds,
                 labels = hcat([["feature $i" for i = (1:Nᶠ)]...,
                                "background"]...),
                 color = cols', legend = :outerright)
yl = ylims()
for k ∈ 1:length(teachers) - 1
    plot!(k*n_steps*[1, 1], [yl[1], yl[2]], color = Tempotrons.fg_color(),
          label = "")
end
xlabel!("# of training steps")
ylabel!("# of spikes")
ps = [inp_plot, pre_plot, train_plots...]
ps = plot(ps..., layout = (length(ps), 1), link = :all, left_margin = 8mm)
p = plot(ps, hist_plot, layout = grid(2, 1,
                                      heights = [1 - 1/length(ps),
                                                 1/length(ps)]))
display(p)

# Save plots
filename(i) = "examples\\Results\\AggLabelsAll_" * string(i)
let i = 0
    while isfile(filename(i) * ".png") || isfile(filename(i) * ".svg")
        i += 1
    end
    savefig(p, filename(i) * ".png")
    savefig(p, filename(i) * ".svg")
end
