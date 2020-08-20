"""
Input generating helper functions for the Tempotrons.jl package.
"""
module InputGen

using ..Inputs
using Distributions

export poisson_process, poisson_spikes_input, spikes_jitter, spikes_jitter!,
       get_features, get_embedded_events_sample

"""
poisson_process(;ν, T)
Generate a Poisson spike train's times with frequency `ν` in `T`.
"""
function poisson_process(; ν::Real, T::Union{TimeInterval,Real})::Array{Real,1}
    τ = TimeInterval(T)
    return rand(Uniform(τ.from, τ.to), rand(Poisson(ν * 0.001abs(τ))))
end

"""
    poisson_spikes_input(N; ν, T)

Generate `N` Poisson spike trains with frequency `ν` in `T`.
"""
function poisson_spikes_input(N::Integer; ν::Real,
                              T::Union{TimeInterval,Real})::SpikesInput

    τ = TimeInterval(T)
    let valid = false
        global si
        while !valid
            si = [poisson_process(ν = ν, T = τ) for i = 1:N]
            valid = isvalidinput(si)
        end
    end
    return SpikesInput(si, duration = τ)
end

"""
    spikes_jitter(SpikeTrain, T, [σ])

Add a Gaussian jitter with s.t.d. `σ` in time to an existing spikes input.
Limit in the result spike times to `T` if supplied.
"""
function spikes_jitter(si::SpikesInput{T1,N};
                       T::Union{TimeInterval,Nothing} = si.duration,
                       σ::Real = 1)::SpikesInput where {T1<:Real,N}
    let valid = false
        global out
        while !valid
            out = map(si) do x
                ξ = x .+ rand(Normal(0, σ), length(x))
                if T ≢ nothing
                    filter!(ζ -> ζ ∈ T, ξ)
                end
                return ξ
            end
            out = Array{Array{Any,1},1}(out)
            valid = isvalidinput(out)
        end
    end
    return SpikesInput(out, duration = T)
end
function spikes_jitter(si::Union{Array{Array{T,1} where T,1},
                                 Array{Array{T,1},1} where T}; kwargs...)
    return spikes_jitter(convert(SpikesInput, si); kwargs...)
end

"""
    spikes_jitter!(SpikeTrain, T, [σ])

Add a Gaussian jitter with s.t.d. `σ` in time to an existing spikes input.
Limit in the result spike times to `T` if supplied.
"""
function spikes_jitter!(si::SpikesInput{T1,N};
                        T::Union{TimeInterval,Nothing} = si.duration,
                        σ::Real = 1) where {T1<:Real,N}
    valid = false
    while !valid
        for x ∈ si
            x .+= rand(Normal(0, σ), length(x))
            if T ≢ nothing
                filter!(z -> z ∈ T, x)
            end
        end
        valid = isvalid(si)
    end
    if T ≢ nothing
        return
    end
    dur = get_duration(si.input)
    si.from = min(si.from, dur.from)
    si.from = max(si.to, dur.to)
    return
end
function spikes_jitter!(si::Union{Array{Array{T,1} where T,1},Array{Array{T,1},1} where T}; kwargs...)
    spikes_jitter!(convert(SpikesInput, si); kwargs...)
    return
end

"""
    get_features(;Nᶠ, Tᶠ N, ν)

Get `Nᶠ` distinct events, each of them is composed of `N` Poisson
spike trains of frequency `ν` (in Hz) and length `Tᶠ` (in ms).
"""
function get_features(; Nᶠ::Integer, Tᶠ::Union{Real,Array{T1,1}}, N::Integer,
                      ν::Real)::Array{SpikesInput{Real,N},1} where {T1<:Real}
    τᶠ = isa(Tᶠ, Real) ? fill(Tᶠ, Nᶠ) : Tᶠ
    return [poisson_spikes_input(N, ν = ν, T = τᶠ[i]) for i = 1:Nᶠ]

end

"""
    get_embedded_events_sample(events, Tᶠ, Cᶠ_mean, ν, T)

Generate Poisson spike trains of frequency `ν` (in Hz) and base length `T` (in
ms) embedded with events of length `Tᶠ` (in ms) drawn from `events` using a
Poisson process with frequency `Cᶠ_mean` (in Hz) for each event type.
Returns the spike train `x`, a list of event types `event_types` ordered by
occurance and a list of event times `event_times` (also ordered by occurance).
The mean duration of the resulted spike train is `T + Nᶠ*Cᶠ_mean*Tᶠ`, where
Nᶠ is the number of distinct events (see the original paper for details).
"""
function get_embedded_events_sample(features::Array{SpikesInput{T1,N},1};
                                    Tᶠ::Real, Cᶠ_mean::Real, ν::Real,
                                    Tᶲ::Union{TimeInterval,Real},
                                    test::Bool = false)::NamedTuple{(:x,
                                                                     :features)} where {T1<:Real,
                                                                                        N}
    Nᶠ = length(features)
    T = TimeInterval(Tᶲ)

    feat_times = sort(rand(Uniform(T.from, T.to), rand(Poisson(Cᶠ_mean * Nᶠ))))
    feat_types = rand(1:Nᶠ, size(feat_times))
    feats = NamedTuple{(:time, :type)}.(zip(feat_times, feat_types))

    # Add test features
    if test
        test_feat_times = rand(Uniform(T.from, T.to)) .* ones(Nᶠ, 1)
        test_feats = NamedTuple{(:time, :type)}.(zip(test_feat_times,
                                                     collect(1:Nᶠ)))
        append!(feats, test_feats)
        feats = sort!(feats, by = f -> f.time)
    end

    # Generate Poisson noise
    si = poisson_spikes_input(N, ν = ν, T = T)

    # If there are any events
    if !isempty(feats)
        for k = 1:length(feats)

            # Get current feature
            feat = features[feats[k].type]

            # Insert an event
            insert_spikes_input!(si, feat, feats[k].time)

            # Delay later events
            feats[(k + 1):end] = map(f -> (time = f.time + abs(feat.duration),
                                           type = f.type), feats[(k + 1):end])

        end
    end

    return (x = si,
            features = map(feats) do f
                return (duration = delay(TimeInterval(abs(features[f.type].duration)),
                                         f.time), type = f.type)
            end)

end

end
