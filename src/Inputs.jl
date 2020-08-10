module Inputs

export TimeInterval, SpikesInput
export get_duration,
       set_duration!,
       delay, delay!,
       insert!,
       isvalid, isvalidinput

#-------------------------------------------------------------------------------
# TimeInterval definition
#-------------------------------------------------------------------------------
mutable struct TimeInterval{T <: Real}
    from::T
    to::T
    function TimeInterval{T}(from::T, to::T) where {T <: Real}
        @assert from ≤ to "ill-defined interval. "
        return new(from, to)
    end
end


function TimeInterval(from::T, to::T) where {T <: Real}
    return TimeInterval{T}(promote(from, to)...)
end
function TimeInterval(from::T1, to::T2) where {T1 <: Real, T2 <: Real}
    return TimeInterval(promote(from, to)...)
end
function TimeInterval(interval::Tuple{<:Real, <:Real})
    return TimeInterval(interval...)
end

#-------------------------------------------------------------------------------
# SpikesInput definition
#-------------------------------------------------------------------------------
struct SpikesInput{T <: Real, N} <: AbstractArray{T, 1}

    input::Array{Array{T, 1}, 1}
    duration::TimeInterval

    function SpikesInput{T, N}(input::Array{Array{T, 1}, 1},
                               duration::Union{TimeInterval, Nothing} = nothing
                               ) where {T <: Real, N}

        # Validate inputs
        @assert length(input) == N "incompatible input size. "
        dur = get_duration(input)
        if duration ≢ nothing
            @assert duration.from ≤ dur.from "input underflows specified duration. "
            @assert duration.to ≥ dur.to "input overflows specified duration. "
            dur = duration
        end
        si = new(input, dur)
        @assert isvalid(si)
        [!isempty(x) && sort!(x) for x ∈ si.input]
        return si
    end

end
Broadcast.broadcastable(si::SpikesInput) = Ref(si)

function SpikesInput(input::Array{Array{T, 1}, 1};
                     duration::Union{TimeInterval, Tuple{T2, T2}, Nothing} = nothing
                     ) where {T, T2 <: Real}
    if T <: Real
        iT, inp = T, input
    else
        iT, inp = Real, Array{Array{Real, 1}, 1}(input)
    end
    dur = isa(duration, Tuple) ? TimeInterval(duration) : duration
    return SpikesInput{iT, length(input)}(inp, dur)
end
function SpikesInput{T}(si::SpikesInput{TT, N}) where {T <: Real, TT <: Real, N}
    return T ≡ TT ? si : SpikesInput{T, N}(Array{Array{T, 1}, 1}(si.input),
                                           si.duration)
end

#-------------------------------------------------------------------------------
# AbstractArray interface for SpikesInput
#-------------------------------------------------------------------------------
Base.size(::SpikesInput{T, N}) where {T <: Real, N} = (N, )
Base.IndexStyle(::Type{<:SpikesInput}) = IndexLinear()
function Base.getindex(si::SpikesInput{T, N},
                       i::Int)::Array{T, 1} where {T <: Real, N}
    return si.input[i]
end
function Base.setindex!(si::SpikesInput{T, N},
                        v::Array{T, 1},
                        i::Int) where {T <: Real, N}
    si.input[i] = sort(v)
    si.duration = TimeInterval(min(minimum(v), si.duration[1]),
                               max(maximum(v), si.duration[2]))
    return
end

#-------------------------------------------------------------------------------
# Methods
#-------------------------------------------------------------------------------
function get_duration(input::Array{Array{T, 1}, 1})::TimeInterval where {T <: Real}
    @assert !isempty(input) "input must have at least a single neuron. "
    @assert !all(isempty.(input)) "input must have at least a single spike. "
    return TimeInterval(minimum((x -> isempty(x) ?  Inf : minimum(x)).(input)),
                        maximum((x -> isempty(x) ? -Inf : maximum(x)).(input)))
end
function set_duration!(si::SpikesInput{T, N},
                       from::T2,
                       to::T2) where {T <: Real, N, T2 <: Real}
    dur = get_duration(si.input)
    @assert from ≤ dur.from "input underflows specified duration. "
    @assert to ≥ dur.to "input overflows specified duration. "
    si.duration.from = from
    si.duration.to = to
    return
end
function set_duration!(si::SpikesInput{T, N},
                      duration::TimeInterval) where {T <: Real, N}
    set_duration!(si, duration.from, duration.to)
    return
end
function set_duration!(si::SpikesInput{T, N},
                      duration::Tuple{T2, T2}) where {T <: Real, N, T2 <: Real}
    set_duration!(si, duration...)
    return
end
function delay!(si::SpikesInput{T, N}, d::T) where {T <: Real, N}
    [x .+= d for x ∈ si.input]
    si.duration.from += d
    si.duration.to += d
    return
end
function delay(si::SpikesInput{T, N},
               d::T)::SpikesInput{T, N} where {T <: Real, N}
    return SpikesInput([x .+= d for x ∈ si.input],
                       duration = TimeInterval(si.duration.from + d,
                                               si.duration.to + d))
end
function insert!(si1::SpikesInput{T, N},
                 si2::SpikesInput{T, N},
                 t::T) where {T <: Real, N}
    add_break!(x::Array{T, 1}, d::T, from::T) = x[x .> from] .+= d
    add_break!.(si1.input, si2.duration.to - si2.duration.from, t)
    append!.(si1.input, delay(si2, t).input)
    sort!.(si1.input)
    si1.duration.from += si2.duration.from
    si1.duration.to += si2.duration.to
    return
end
function isvalidinput(si::Array{Array{T, 1}, 1}) where T
    return !all(isempty.(si))
end
function isvalid(si::SpikesInput)
    return isvalidinput(si.input)
end

end
