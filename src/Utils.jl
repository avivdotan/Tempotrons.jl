module Utils

using Distributions

export PoissonSpikeTrain, SpikeJitter

function PoissonSpikeTrain(; ν::Real, T::Real)
    ξ = 1000cumsum(rand(Exponential(1/ν), 10))
    return ξ[ξ.<T]
end

function SpikeJitter(SpikeTrain::Array{T1};
                        T::Real,
                        σ::Real = 1) where T1 <: Real
    n = rand(Normal(0, σ), size(SpikeTrain))
    ξ = SpikeTrain + n
    ξ = ξ[ξ.<T]
    ξ = ξ[ξ.>0]
    return ξ
end

end
