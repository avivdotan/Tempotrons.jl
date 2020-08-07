using ..Optimizers

function GetEligibilities(m::Tempotron,
                          inp::Array{Array{T1, 1}, 1},
                          PSPs::Array{T2, 1},
                          spikes::Array{T3, 1},
                          V) where {T1 <: Real,
                                    T2 <: NamedTuple,
                                    T3 <: Any}

    C₁ = (m.α - 1)/(2m.K_norm*(m.α + 1))
    C₂ = 1/(m.α + 1)
    W = m.w./m.K_norm
    function 𝒱(tᵢʲ::Real)::Real
        spikes_b = filter(x -> x.time < tᵢʲ, spikes)
        spikes_a = filter(x -> x.time ≥ tᵢʲ, spikes)
        Σ₁ = isempty(PSPs) ? 0.0 : sum(PSPs) do x
            absdiff = abs(x.time - tᵢʲ)
            return W[x.neuron]*(m.τₘ*exp(-absdiff/m.τₘ) -
                                m.τₛ*exp(-absdiff/m.τₛ))
        end
        Σ₂ = isempty(spikes_b) ? 0.0 : sum(x -> exp(-(tᵢʲ - x.time)/m.τₘ),
                                           spikes_b)
        Σ₃ = isempty(spikes_a) ? 0.0 : sum(spikes_a) do x
            absdiff = x.time - tᵢʲ
            return exp(-absdiff/m.τₘ)/2 - C₂*exp(-absdiff/m.τₛ)
        end
        return C₁*Σ₁ - (m.θ - m.V₀)*m.τₘ*(C₁*Σ₂ + Σ₃/m.K_norm)
    end
    return [isempty(x) ? 0.0 : sum(𝒱, x) for x ∈ inp]

end

"""
    Train_corr!(m::Tempotron, inp, y₀::Integer)
Train a tempotron `m` to fire y₀ spikes in response to an input vector of spike
trains `inp`.
For further details see [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113).
"""
function Train_corr!(m::Tempotron,
                    inp::Array{Array{Tp, 1}, 1},
                    y₀::TrgtT;
                    optimizer::Optimizer = SGD(0.001),
                    top_elig_update::Real = 0.1) where {Tp <: Real,
                                                        TrgtT <: Integer}

    ∇ = zeros(size(m.w))

    # Get the PSPs
    PSPs = sort(GetPSPs(m, inp), by = x -> x.time)

    # Get the current number of spikes and voltage trace
    spikes, V = GetSpikes(m, PSPs, (m.θ - m.V₀), return_V = true)
    k = min(length(spikes), typemax(TrgtT))

    # If the tempotron's number of spikes matches the teacher, do not learn.
    if k == y₀
        optimizer(∇)
        return
    end

    ℰ = GetEligibilities(m, inp, PSPs, spikes, V)
    max_k = Int(round(top_elig_update*length(ℰ)))
    idx = partialsortperm(ℰ, 1:max_k, rev = true)

    # Get the weight changes
    ∇[idx] .= 1;

    # Change tempotron's weights
    m.w .+= (y₀ > k ? -1 : 1).*optimizer(∇)

end
