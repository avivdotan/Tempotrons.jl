module Utils

    using Distributions

    export PoissonSpikeTrain

    function PoissonSpikeTrain(; ν::Real, T::Real)
        ξ = 1000cumsum(rand(Exponential(1/ν), 10))
        return ξ[ξ.<T]
    end

end
