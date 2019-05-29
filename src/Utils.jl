"""
Input generating helper gunctions for the Tempotrons.jl package.
"""
module InputGen

using Distributions

export PoissonSpikeTrain, SpikeJitter

"""
    PoissonSpikeTrain([ν][, T])
Generate a Poisson spike train's times with frequency ν in (0, T).
"""
function PoissonSpikeTrain(; ν::Real, T::Real)
    ξ = 1000cumsum(rand(Exponential(1/ν), 10))
    return ξ[ξ.<T]
end
"""
    SpikeJitter(SpikeTrain, T, [σ])
Add a Gaussian jitter with s.t.d. σ in time to an existing spike train's times
in (0, T).
"""
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
