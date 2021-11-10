#-------------------------------------------------------------------------------
# Correlation-based training methods
#-------------------------------------------------------------------------------
"""
    get_eligibilities(m::Tempotron, inp, PSPs, spikes)

Get the eligibility of each input neuron:

```math
\\mathcal{E}_i = \\sum_{t_i^j} \\mathcal{V}_i^j,
```

where:

```math
\\mathcal{V}_i^j = \\int_{t_i^j}^{\\infty} V\\left(t\\right)K\\left(t - t_i^j\\right) \\mathrm{d}t.
```

# Arguments

  - `m::Tempotron`: a tempotron.
  - `inp`: an input vector of spike trains.
  - `PSPs`: the PSPs elicited by the input.
  - `spikes`: the spikes elicited by the input.
"""
function get_eligibilities(
    m::Tempotron{N},
    inp::SpikesInput{T1,N},
    PSPs::Array{T2,1},
    spikes::Array{T3,1},
)::Array{Real,1} where {T1<:Real,T2<:NamedTuple{(:time, :ΔV, :neuron)},T3,N}

    # Set constants
    C₁::Real = (m.α - 1) / (2m.K_norm * (m.α + 1))
    C₂::Real = 1 / (m.α + 1)
    W::Array{Real,1} = m.w ./ m.K_norm

    # The correlation of the voltage trace with a single PSP
    # This calculation follows an explicit analytical expression derived from
    # the original time convolution.
    @inline function 𝒱(tᵢʲ::Real)::Real
        spikes_b = filter(x -> x.time < tᵢʲ, spikes)
        spikes_a = filter(x -> x.time ≥ tᵢʲ, spikes)
        Σ₁ =
            isempty(PSPs) ? 0.0 :
            sum(PSPs) do x
                absdiff = abs(x.time - tᵢʲ)
                return W[x.neuron] *
                       (m.τₘ * exp(-absdiff / m.τₘ) - m.τₛ * exp(-absdiff / m.τₛ))
            end
        Σ₂ = isempty(spikes_b) ? 0.0 : sum(x -> exp(-(tᵢʲ - x.time) / m.τₘ), spikes_b)
        Σ₃ =
            isempty(spikes_a) ? 0.0 :
            sum(spikes_a) do x
                absdiff = x.time - tᵢʲ
                return exp(-absdiff / m.τₘ) / 2 - C₂ * exp(-absdiff / m.τₛ)
            end
        return C₁ * Σ₁ - (m.θ - m.V₀) * m.τₘ * (C₁ * Σ₂ + Σ₃ / m.K_norm)
    end

    # The eligibility of each input neuron
    return [isempty(x) ? 0.0 : sum(𝒱, x) for x ∈ inp]

end

"""
    train_corr!(m::Tempotron, inp, y₀::Integer)

Trains a tempotron `m` to fire y₀ spikes in response to an input vector of spike
trains `inp`.

# Optional arguments

  - `optimizer::Optimizers.Optimizer = Optimizers.SGD(0.01)`: a gradient-based optimization method (see [`Optimizers`](@ref)).
  - `top_elig_update::Real = 0.1`: the proportion of top eligibilities to be updated.

# Learning rule

Get the eligibility of each input neuron:

```math
\\mathcal{E}_i = \\sum_{t_i^j} \\mathcal{V}_i^j,
```

where (eq. 3 in [1]):

```math
\\mathcal{V}_i^j = \\int_{t_i^j}^{\\infty} V\\left(t\\right)K\\left(t - t_i^j\\right) \\mathrm{d}t,
```

then update the `top_elig_update` of the neurons with the highest eligibilities
(default is `0.1`, i.e. 10%).

Assuming SGD, the update rule in case of an error is (eq. 8 in [2]):

```math
\\Delta w_i=
\\begin{cases}
    \\pm\\lambda_{\\mathcal{V}} & ,\\quad \\mathscr{E}_i \\gt Q \\\\
    0                           & ,\\quad \\mathscr{E}_i \\le Q
\\end{cases},
```

where the sign (±) of the update is determined by the teacher and ``Q`` is
determined by `top_elig_update`.

For further details:

  - For the binary tempotron, check the "Implementation by voltage convolution" subsection under the "Results" section in [1].
  - For the Multi-spike tempotron, check the "Correlation-based learning" subsection under the "Materials and methods" section in [2].

# References

## Binary tempotron:

[1] [Gütig, R., & Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing–based decisions. Nature neuroscience, 9(3), 420.](https://www.nature.com/articles/nn1643)    # Get the current number of spikes and voltage trace

## Multi-spike tempotron:

[2] [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113)    # If the tempotron's number of spikes matches the teacher, do not learn.
"""
function train_corr!(
    m::Tempotron{N},
    inp::SpikesInput{T,N},
    y₀::Union{Bool,Integer};
    optimizer::Optimizer = SGD(0.001),
    top_elig_update::Real = 0.1,
) where {T<:Real,N}

    # Get the PSPs
    PSPs = sort(get_psps(m, inp), by = x -> x.time)

    # Get the current number of spikes and voltage trace
    spikes = get_spikes(m, PSPs).spikes
    k = min(length(spikes), typemax(typeof(y₀)))

    # If the tempotron's number of spikes matches the teacher, do not learn.
    if k == y₀
        return
    end

    ℰ = get_eligibilities(m, inp, PSPs, spikes)
    max_k = Int(round(top_elig_update * length(ℰ)))
    idx = partialsortperm(ℰ, 1:max_k, rev = true)

    # Get the weight changes
    ∇ = zeros(N)
    ∇[idx] .= 1

    # Change tempotron's weights
    m.w .+= optimizer((y₀ > k ? -1 : 1) .* ∇)

    return

end
