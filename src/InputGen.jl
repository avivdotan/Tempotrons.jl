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
    GetEvents(n_event_types, event_length; N, ν)
Get a `n_event_types` distinct events, each of them is composed of `N` Poisson
spike trains of frequency `ν` (in Hz) and length `event_length` (in ms).
"""
function GetEvents(n_event_types::Integer,
                   event_length::Real;
                   N::Integer,
                   ν::Real)::Array{Array{Array{Real, 1}, 1}, 1}
    # TODO: Variable event lengths
    events = [[Real[] for i = 1:N] for j = 1:n_event_types]
    for k = 1:length(events)

        # Make sure to to return empty events
        while all([isempty(eki) for eki ∈ events[k]])
            events[k] = [PoissonProcess(ν = ν, T = event_length)
                         for i = 1:N]
        end

    end
    return events
end

"""
    GenerateSampleWithEmbeddedEvents(events, event_length, event_freq, ν, T)
Generate Poisson spike trains of frequency `ν` (in Hz) and length `T` (in ms)
embedded with events of length `event_length` (in ms) drawn from `events` with
frequency `event_freq` (in Hz).
"""
function GenerateSampleWithEmbeddedEvents(events::Array{Array{Array{Real, 1}, 1}, 1};
                                          event_length::Real,
                                          event_freq::Real,
                                          ν::Real,
                                          T::Real)::NamedTuple where Tp <: Real
    # TODO: Variable event lengths

    N = length(events[1])

    # Helpers for woriking with spike times input:
    #   `Add.(x, y)` will add `y` to a jagged array `x`
    Add(x, y)       = x .+ y
    # `Insert.(x, y, z)` will add `z` to all `x[i][j] > y` in a jagged array `x`
    Insert(x, y, z) = ((a, b, c) -> a > b ? a + c : a).(x, y, z)

    # Get event times
    event_times = sort(PoissonProcess(ν = event_freq, T = T))
    event_types = rand(1:length(events), size(event_times))

    # Generate Poisson noise
    inp = [PoissonProcess(ν = ν, T = T) for i = 1:N]

    # If there are any events
    if !isempty(event_times)
        for k = 1:length(event_times)

            # Insert an event
            event   = Add.(events[event_types[k]], event_times[k])
            inp    .= Insert.(inp, event_times[k], event_length)
            append!.(inp, event)

            # Delay later events
            event_times = Insert(event_times, event_times[k], event_length)

        end
    end

    return (x = inp, event_types = event_types, event_times = event_times)
end

end
