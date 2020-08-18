"""
Input structures for the Tempotrons.jl package:
`TimeInterval`, `SpikesInput`.
"""
module Inputs

# Exports
export TimeInterval, SpikesInput
export get_duration, set_duration!, delay, delay!, insert_spikes_input!,
       isvalid, isvalidinput

#-------------------------------------------------------------------------------
# TimeInterval definition
#-------------------------------------------------------------------------------

"""
    TimeInterval
A `TimeInterval` [`from`, `to`].
"""
mutable struct TimeInterval{T <: Real}

    """
    Start time
    """
    from::T

    """
    End time
    """
    to::T

    """
        TimeInterval(from, to)
    Create a new `TimeInterval` [`from`, `to`].
    """
    function TimeInterval{T}(from::T, to::T) where {T<:Real}
        @assert from ≤ to "ill-defined interval. "
        return new(from, to)
    end
end

"""
    TimeInterval(from, to)
Create a new `TimeInterval` [`from`, `to`].
"""
function TimeInterval(from::T, to::T) where {T<:Real}
    return TimeInterval{T}(from, to)
end

"""
    TimeInterval(from, to)
Create a new `TimeInterval` [`from`, `to`].
"""
function TimeInterval(from::T1, to::T2) where {T1<:Real,T2<:Real}
    return TimeInterval(promote(from, to)...)
end

"""
    TimeInterval((from, to))
Create a new `TimeInterval` [`from`, `to`].
"""
function TimeInterval(interval::Tuple{<:Real,<:Real})
    return TimeInterval(interval...)
end

"""
    TimeInterval(::TimeInterval)
Returns the input as is.
"""
function TimeInterval(ti::TimeInterval) where {T<:Real}
    return ti
end

"""
    TimeInterval(t)
Create a new `TimeInterval` [`0`, `t`].
"""
function TimeInterval(t::Real)
    return TimeInterval(0, t)
end

"""
    delay!(::TimeInterval, d)
Delays a tie interval by `d` (inplace).
"""
function delay!(ti::TimeInterval, d::Real)
    ti.from += d
    ti.to += d
    return
end

"""
    delay!(::TimeInterval, d)
Delays a time interval by `d`.
"""
function delay(ti::TimeInterval, d::Real)
    return TimeInterval(ti.from + d, ti.to + d)
end

"""
    Base.abs(::TimeInterval)
TimeInterval's length (`t - from`).
"""
Base.abs(ti::TimeInterval) = ti.to - ti.from

Base.in(t::Real, ti::TimeInterval)::Bool = ti.from ≤ t ≤ ti.to
function Base.in(ti1::TimeInterval, ti2::TimeInterval)::Bool
    return ti1.from ∈ ti2 && ti1.to ∈ ti2
end

#-------------------------------------------------------------------------------
# SpikesInput definition
#-------------------------------------------------------------------------------

"""
    SpikesInput
A structure containing a sries of spike trains.
"""
struct SpikesInput{T <: Real,N} <: AbstractArray{T,1}

    """
    An array of spike-trains' times.
    """
    input::Array{Array{T,1},1}

    """
    The total input duration.
    """
    duration::TimeInterval

    function SpikesInput{T,N}(input::Array{Array{T,1},1},
                              duration::Union{TimeInterval,Nothing} = nothing) where {T<:Real,
                                                                                      N}

        # Validate inputs
        @assert length(input) == N "incompatible input size. "
        dur = get_duration(input)
        if duration ≢ nothing
            @assert dur ∈ duration "input overflows specified duration. "
            dur = duration
        end
        si = new(input, dur)
        @assert isvalid(si)
        [!isempty(x) && sort!(x) for x ∈ si.input]
        return si
    end

end
Broadcast.broadcastable(si::SpikesInput) = Ref(si)

"""
    SpikesInput(input[, duration])
Create a new `SpikesInput` structure. `input` is a series of spike-trains (each
represented by spike times), `duration` is the total time interval of the input
(infernced from `input` if not supplied).

Usage:

```julia
inp = SpikesInput([[1,2,3,4], [2.3,5,7]])
inp = SpikesInput([[1,2,3,4], [2.3,5,7]], duration = (0, 10))
inp[1][3] = 3.5
inp[1][3] = inp[2][1]
```

"""
function SpikesInput(input::Array{Array{T,1},1};
                     duration::Union{TimeInterval,Tuple{T2,T2},Nothing} = nothing) where {T,
                                                                                          T2<:Real}
    if T <: Real
        iT, inp = T, input
    else
        iT, inp = Real, Array{Array{Real,1},1}(input)
    end
    dur = isa(duration, Tuple) ? TimeInterval(duration) : duration
    return SpikesInput{iT,length(input)}(inp, dur)
end

"""
    SpikesInput(::SpikesInput)
Typecasting `SpikesInput`.
"""
function SpikesInput{T}(si::SpikesInput{TT,N}) where {T<:Real,TT<:Real,N}
    return T ≡ TT ? si :
           SpikesInput{T,N}(Array{Array{T,1},1}(si.input), si.duration)
end

#-------------------------------------------------------------------------------
# AbstractArray interface for SpikesInput
#-------------------------------------------------------------------------------
Base.size(::SpikesInput{T,N}) where {T<:Real,N} = (N,)
Base.IndexStyle(::Type{<:SpikesInput}) = IndexLinear()
function Base.getindex(si::SpikesInput{T,N},
                       i::Int)::Array{T,1} where {T<:Real,N}
    return si.input[i]
end
function Base.setindex!(si::SpikesInput{T,N}, v::Array{T,1},
                        i::Int) where {T<:Real,N}
    si.input[i] = sort(v)
    si.duration = TimeInterval(min(minimum(v), si.duration[1]),
                               max(maximum(v), si.duration[2]))
    return
end

#-------------------------------------------------------------------------------
# Methods
#-------------------------------------------------------------------------------
function get_duration(input::Array{Array{T,1},1})::TimeInterval where {T<:Real}
    @assert !isempty(input) "input must have at least a single neuron. "
    @assert !all(isempty.(input)) "input must have at least a single spike. "
    return TimeInterval(minimum((x -> isempty(x) ? Inf : minimum(x)).(input)),
                        maximum((x -> isempty(x) ? -Inf : maximum(x)).(input)))
end

"""
    set_duration!(::SpikesInput, from, to)
Sets the duration of a given `SpikesInput`.
"""
function set_duration!(si::SpikesInput{T,N}, from::T2,
                       to::T2) where {T<:Real,N,T2<:Real}
    dur = get_duration(si.input)
    @assert from ≤ dur.from "input underflows specified duration. "
    @assert to ≥ dur.to "input overflows specified duration. "
    si.duration.from = from
    si.duration.to = to
    return
end

"""
    set_duration!(::SpikesInput, ::TimeInterval)
Sets the duration of a given `SpikesInput`.
"""
function set_duration!(si::SpikesInput{T,N},
                       duration::TimeInterval) where {T<:Real,N}
    set_duration!(si, duration.from, duration.to)
    return
end

"""
    set_duration!(::SpikesInput, (from, to))
Sets the duration of a given `SpikesInput`.
"""
function set_duration!(si::SpikesInput{T,N},
                       duration::Tuple{T2,T2}) where {T<:Real,N,T2<:Real}
    set_duration!(si, duration...)
    return
end

"""
    delay!(::SpikesInput, d)
Delays a `SpikesInput` by `d` (inplace).
"""
function delay!(si::SpikesInput{T,N}, d::T) where {T<:Real,N}
    [x .+= d for x ∈ si.input]
    delay!(si.duration, d)
    return
end

"""
    delay!(::SpikesInput, d)
Delays a `SpikesInput` by `d`.
"""
function delay(si::SpikesInput{T,N}, d::T)::SpikesInput{T,N} where {T<:Real,N}
    return SpikesInput(Array{Array{T,1},1}([x .+ d for x ∈ si.input]),
                       duration = delay(si.duration, d))
end

"""
    insert_spikes_input!(si1::SpikesInput, si2::SpikesInput, t)
Inserts a `si2` in the middle of `si1` at time `t` (inplace).
"""
function insert_spikes_input!(si1::SpikesInput{T,N}, si2::SpikesInput{T,N},
                              t::T) where {T<:Real,N}
    add_break!(x::Array{T,1}, d::T, from::T) = x[x .> from] .+= d
    add_break!.(si1.input, abs(si2.duration), t)
    append!.(si1.input, delay(si2, t).input)
    sort!.(si1.input)
    si1.duration.to += abs(si2.duration)
    return
end

"""
    isvalidinput(input::Array{Array{<:Real,1},1})
Checks that `input` is not empty.
"""
function isvalidinput(input::Array{Array{T,1},1}) where {T}
    return !all(isempty.(input))
end

"""
    isvalidinput(si::SpikesInput)
Checks that `si` is not empty.
"""
function isvalid(si::SpikesInput)
    return isvalidinput(si.input)
end

end
