"""
Test the values of ∇θ⃰ for different random samples
"""

using Tempotrons
using Tempotrons.InputGen
using Plots

function get_∇θ⃰(m::Tempotron{N}, inp::SpikesInput{T,N}, k::Integer) where {T<:Real,N}

    # Get the PSPs
    PSPs = sort(Tempotrons.get_psps(m, inp), by = x -> x.time)

    # Get the spike times elicited by the critical threshold up to the new
    # elicited spike (inclusive)
    spk, t⃰ = [], 0
    try
        t⃰, ~, spk = Tempotrons.get_critical_threshold(m, PSPs, k)
    catch ex
        # Ignore samples with non-positive voltage trace
        if isa(ex, NonPositiveVoltageError)
            return
        else
            rethrow(ex)
        end
    end
    push!(spk, t⃰)

    # Get the gradient of θ⃰ w.r.t. the tempotron's weights
    # TODO: Performance?
    PSP(t::Real)::Real = sum(x -> x.ΔV(t), PSPs)
    dPSP(t::Real)::Real = sum(x -> m.w[x.neuron] * m.K̇(t - x.time), PSPs)
    ∇θ⃰ = Tempotrons.get_θ_gradient(m, inp, spk, PSP, dPSP)

    return ∇θ⃰

end

# Set parameters
N = 10
T = 500
dt = 1
t = collect(0:dt:T)
ν = 10
λ = 1e-4
n_samples = 100
nk = 20
tmp = Tempotron(N)

# Generate input samples
samples = [poisson_spikes_input(N, ν = ν, T = T) for j = 1:n_samples]


# Get the gradients
res = []
for s ∈ samples
    for k = 1:nk
        ∇θ⃰ = get_∇θ⃰(tmp, s, k)
        push!(res, ∇θ⃰)
    end
end
res = hcat(res...)

ind = findall(x -> x < 0, res)
println(length(ind), '/', length(res))

scatter(res, legend = false)