"""
Input generating helper gunctions for the Tempotrons.jl package.
"""
module InputGen

using Distributions

export PoissonProcess,
       SpikeJitter,
       GetEvents,
       GenerateSampleWithEmbeddedEvents

"""
    PoissonSpikeTrain(ν, T)
Generate a Poisson spike train's times with frequency `ν` in (`0`, `T`).
"""
function PoissonProcess(; ν::Real, T::Real)::Array{Real, 1}
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

"""
    GetEvents(;Nᶠ, Tᶠ N, ν)
Get `Nᶠ` distinct events, each of them is composed of `N` Poisson
spike trains of frequency `ν` (in Hz) and length `Tᶠ` (in ms).
"""
function GetEvents(;Nᶠ::Integer,
                   Tᶠ::Real,
                   N::Integer,
                   ν::Real)::Array{Array{Array{Real, 1}, 1}, 1}
    # TODO: Variable event lengths
    events = [[Real[] for i = 1:N] for j = 1:Nᶠ]
    for k = 1:length(events)

        # Make sure to to return empty events
        while all([isempty(eki) for eki ∈ events[k]])
            events[k] = [PoissonProcess(ν = ν, T = Tᶠ) for i = 1:N]
        end

    end
    return events
end

"""
    GenerateSampleWithEmbeddedEvents(events, Tᶠ, Cᶠ_mean, ν, T)
Generate Poisson spike trains of frequency `ν` (in Hz) and base length `T` (in
ms) embedded with events of length `Tᶠ` (in ms) drawn from `events` using a
Poisson process with frequency `Cᶠ_mean` (in Hz) for each event type.
Returns the spike train `x`, a list of event types `event_types` ordered by
occurance and a list of event times `event_times` (also ordered by occurance).
The mean duration of the resulted spike train is `T + Nᶠ*Cᶠ_mean*Tᶠ`, where
Nᶠ is the number of distinct events (see the original paper for details).
"""
function GenerateSampleWithEmbeddedEvents(events::Array{Array{Array{Real, 1}, 1}, 1};
                                          Tᶠ::Real,
                                          Cᶠ_mean::Real,
                                          ν::Real,
                                          T::Real,
                                          test::Bool = false)::NamedTuple where Tp <: Real
    # TODO: Variable event lengths

    N   = length(events[1])
    Nᶠ  = length(events)

    # Helpers for woriking with spike times input:
    #   `Add.(x, y)` will add `y` to a jagged array `x`
    Add(x, y)       = x .+ y
    # `Insert.(x, y, z)` will add `z` to all `x[i][j] > y` in a jagged array `x`
    Insert(x, y, z) = ((a, b, c) -> a ≥ b ? a + c : a).(x, y, z)

    # Get event times
    event_times = sort(rand(Uniform(0, T),
                            rand(Poisson(Cᶠ_mean*length(events)))))
    event_types = rand(1:length(events), size(event_times))

    # Add test events
    if test
        test_time = rand(Uniform(0, T))
        test_events_times = test_time.*ones(length(events), 1)
        event_times = [event_times..., test_events_times...]
        event_types = [event_types..., collect(1:length(events))...]
        ind = sortperm(event_times)
        event_times = event_times[ind]
        event_types = event_types[ind]
    end

    # Generate Poisson noise
    inp = [PoissonProcess(ν = ν, T = T) for i = 1:N]

    # If there are any events
    if !isempty(event_times)
        for k = 1:length(event_times)

            # Insert an event
            event   = Add.(events[event_types[k]], event_times[k])
            inp    .= Insert.(inp, event_times[k], Tᶠ)
            append!.(inp, event)

            # Delay later events
            event_times[(k + 1):end] = Insert(event_times[(k + 1):end],
                                              event_times[k], Tᶠ)

        end
    end

    return (x = inp, event_types = event_types, event_times = event_times)
end

end
