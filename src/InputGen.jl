"""
Input generating helper gunctions for the Tempotrons.jl package.
"""
module InputGen

using Distributions

export PoissonSpikeTrain, SpikeJitter

"""
    PoissonSpikeTrain([ν][, T])
Generate a Poisson spike train's times with frequency `ν` in (`0`, `T`).
"""
function PoissonSpikeTrain(; ν::Real, T::Real)::Array{Real, 1}
    return rand(Uniform(0, T), rand(Poisson(0.001ν*T)))
end

"""
    SpikeJitter(SpikeTrain, T, [σ])
Add a Gaussian jitter with s.t.d. `σ` in time to an existing spike train's times
in (`0`, `T`).
"""
function SpikeJitter(SpikeTrain::Array{T1, N};
                        T::Real = typemax(T1),
                        σ::Real = 1)::Array{T1, N} where {T1 <: Real, N}
    n = rand(Normal(0, σ), size(SpikeTrain))
    ξ = SpikeTrain + n
    ξ = ξ[ξ.<T]
    ξ = ξ[ξ.>0]
    return ξ
end

end
