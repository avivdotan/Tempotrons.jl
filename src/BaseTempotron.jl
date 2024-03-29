#-------------------------------------------------------------------------------
# Tempotron definition
#-------------------------------------------------------------------------------
struct Tempotron{N}
    """
    Membrane time constant [ms]
    """
    τₘ::Real

    """
    Sytaptic time constant [ms]
    """
    τₛ::Real

    """
    Voltage threshold [mV]
    """
    θ::Real

    """
    Reset voltage [mV]
    """
    V₀::Real

    """
    Synaptic weights
    """
    w::Array{Real,1}

    # Precalculated constants
    α::Real
    K_norm::Real
    A::Real
    log_α::Real

    """
    Input kernel `K(t)`
    """
    K::Function

    """
    Input kernel's time derivative `K̇(t)`
    """
    K̇::Function

    """
    Spike kernel `η(t)`
    """
    η::Function

    """
    Numerical hack [ms]
    """
    Tϵ::Real

    """
        Tempotron{N}(τ_m = 15, τₛ = τₘ/4, θ = 1, V₀ = 0, Tϵ = 700τₛ)
    Create a new tempotron.

    # Arguments
    - `τₘ::Real = 15`: the membrane's time constant (in ms).
    - `τₛ::Real = τₘ/4`: the synaptic time constant (in ms).
    - `θ::Real = 1`: the threshold (in mV).
    - `V₀::Real = 0`: the rest potential (in mV).
    - `Tϵ::Real = 700τₛ`: a time constant (in ms) used to prevent numerical overflows. Should not exceed `700τₛ`.
    """
    function Tempotron{N}(τₘ::Real = 15.0,
                          τₛ::Real = τₘ / 4,
                          θ::Real = 1.0,
                          V₀::Real = 0.0,
                          Tϵ::Real = 700τₛ) where {N}

        # Validate inputs
        @assert N > 0 "There must be at least one input neuron. "
        @assert τₘ > 0 "Membrane's time constant must be positive. "
        @assert τₛ > 0 "Synaptic time constant must be positive. "
        @assert τₛ < τₘ "Synaptic time constant must be lower than " *
                        "membrane's time constant. "
        @assert V₀ < θ "Firing threshold must be above rest potential. "

        # Pre-calculate constants
        α = τₘ / τₛ
        K_norm = α^(-α / (α - 1)) * (α - 1)
        A = τₘ * τₛ / (τₘ - τₛ)
        log_α = log(α)

        # The input kernel `K(t)` and its derivative `k̇(t)`.
        K(t::Real) = t < 0 ? 0.0 :
                     ((exp(-t / τₘ) - exp(-t / τₛ)) / K_norm)
        K̇(t::Real) = t < 0 ? 0.0 :
                      ((-exp(-t / τₘ) / τₘ + exp(-t / τₛ) / τₛ) /
                       K_norm)

        #The spike kernel `η(t)`.
        η(t::Real) = t < 0 ? 0.0 : exp(-t / τₘ)

        # Initialize weights
        w = Array{Real,1}(undef, N)

        return new(τₘ, τₛ, θ, V₀, w, α, K_norm, A, log_α, K, K̇, η, Tϵ)
    end
end
Broadcast.broadcastable(m::Tempotron) = Ref(m)

"""
    Tempotron(N; τ_m = 15, τₛ = τₘ/4, θ = 1, V₀ = 0, Tϵ = 700τₛ)

Create a new tempotron.
The neuron model used is [1]:

```math
V\\left(t\\right) = V_0 + \\sum_{i=1}^N w_i \\sum_{t_i^j<t} K\\left(t-t_i^j\\right)
-θ \\sum_{t_{spk}^j<t} \\exp{\\left(-\\frac{t-t_{spk}^j}{\\tau_m}\\right)}
```

where ``V_0`` is the rest potential, ``\\theta`` is the firing threshold, ``w_i``
is the weight of the ``i``<sup>th</sup> input neuron, ``t_i^j`` is the time of the
``j``<sup>th</sup> spike in the ``i``<sup>th</sup> input neuron and ``t_{spk}^j``
is the time of the ``j``<sup>th</sup> spike in the output neuron.

The voltage kernel function used is:

```math
K\\left(t\\right) = V_{norm} \\left(\\exp{\\left(-\\frac{t}{\\tau_m}\\right)}
- \\exp{\\left(-\\frac{t}{\\tau_s}\\right)}\\right)
```

# Arguments

  - `N::Integer`: the number of input neurons.
  - `τₘ::Real = 15`: the membrane's time constant (in ms).
  - `τₛ::Real = τₘ/4`: the synaptic time constant (in ms).
  - `θ::Real = 1`: the threshold (in mV).
  - `V₀::Real = 0`: the rest potential (in mV).
  - `weights::Array{<: Real, 1}`: initial weights vector.
  - `Tϵ::Real = 700τₛ`: a time constant (in ms) used to prevent numerical overflows. Should not exceed `700τₛ`.

# Examples

```julia
tmp = Tempotron(10)
tmp = Tempotron(10; τₘ = 20)
tmp = Tempotron(10; θ = -55, V₀ = -70)
```

# References

[1] [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113)
"""
function Tempotron(N::Integer;
                   τₘ::Real = 15.0,
                   τₛ::Real = τₘ / 4,
                   θ::Real = 1.0,
                   V₀::Real = 0.0,
                   weights::Array{T,1} = (12rand(Float64, N) .- 3) .*
                                         (θ - V₀) ./ N,
                   Tϵ::Real = 700τₛ)::Tempotron where {T<:Real}

    # Validate inputs
    @assert N > 0 "There must be at least one input neuron. "
    @assert length(weights) == N "weights vector must have exaclty N elements. "

    tmp = Tempotron{N}(τₘ, τₛ, θ, V₀, Tϵ)

    # Initialize weights
    tmp.w .= weights

    return tmp
end

#-------------------------------------------------------------------------------
# Tempotron core methods
#-------------------------------------------------------------------------------

"""
    (m::Tempotron)(inp[; t])

Get the tempotron `m`'s output spike times and voltage for an input vector of
spike trains `inp`. An optional parameter is a time grid `t`, at which to sample
the voltage function to be returned as a second output argument.

# Examples

```julia
input = InputGen.poisson_spikes_input(10; ν = 5, T = 500)
tmp = Tempotron(10)
output = tmp(inp).spikes
voltage = tmp(inp; t = 0:500).V
output, voltage = tmp(inp; t = 0:500)
```
"""
function (m::Tempotron{N})(inp::SpikesInput{T1,N};
                           t::Union{Array{T2,1},Nothing} = nothing)::NamedTuple where {T1<:Real,
                                                                                       T2<:Real,
                                                                                       N}

    # Get the PSPs
    PSPs = sort(get_psps(m, inp); by = x -> x.time)

    # Get the spike times and voltage function
    if t ≡ nothing
        spikes = get_spikes(m, PSPs).spikes
    else
        spikes, V = get_spikes(m, PSPs; return_V = true)
    end
    spikes = [s.time for s ∈ spikes]

    # If no time grid was given, return spike times
    ret = (spikes = spikes,)
    if t ≡ nothing
        return ret
    end

    # calculate the voltage over the time grid
    Vt = V.(t)

    # Improve spikes visibility
    for j ∈ spikes
        if t[begin] < j < t[end]
            k = findfirst(t .> j)
            Vt[k] = m.θ + 0.3(m.θ - m.V₀)
        end
    end

    ret = merge(ret, (V = Vt,))
    return ret
end
function (m::Tempotron)(inp::Array{Array{T,1},1} where {T};
                        kwargs...)::NamedTuple
    return m(convert(SpikesInput, inp); kwargs...)
end
function (m::Tempotron)(inp::Array{Array{T,1} where T,1}; kwargs...)::NamedTuple
    return m(convert(SpikesInput, inp); kwargs...)
end

"""
    (m::Tempotron)(inp::Array{SpikesInput, 1}[; ...])

Get the tempotron `m`'s output for each input is the list `inp`.
Any additional keyworkd arguments are passed to the single-input method.

# Examples

```julia
inp1 = InputGen.poisson_spikes_input(10; ν = 5, T = 500)
inp2 = InputGen.poisson_spikes_input(10; ν = 5, T = 500)
tmp = Tempotron(10)
outputs = tmp([inp1, inp2])
```
"""
function (m::Tempotron{N})(inp::Array{S,1};
                           kwargs...) where {N,T<:Real,T2,
                                             S<:Union{SpikesInput{T,N},
                                                      Array{Array{T2,1},1}}}
    return [m(i; kwargs...) for i ∈ inp]
end
function (m::Tempotron{N})(inp::Array{Array{T,1} where T,2};
                           kwargs...) where {N}
    @assert size(inp, 2) == N
    return [m(inp[i, :]; kwargs...) for i = 1:size(inp, 1)]
end

"""
    get_psps(m::Tempotron, inp)

Generate a list of PSPs for a given input vector of spike trains `inp` and
tempotron `m`. Each PSP in the list is a named tuple `(time = j, ΔV(t) = wᵢ⋅K(t - j), neuron = i)`, where `j` is the input spike time, `ΔV(t)`
is the properly weighted and shifted voltage kernel ``K(t)``  and `i` is the
index of the generating input neuron.
"""
function get_psps(m::Tempotron{N},
                  inp::SpikesInput{T,N})::Array{NamedTuple{(:time, :ΔV,
                                                            :neuron)},
                                                1} where {T<:Real,N}
    PSPs = [(time = j::Real, ΔV = t::Real -> m.w[i] .* m.K.(t - j),
             neuron = i::Integer) for
            i = 1:N for j ∈ inp[i]]
    return PSPs
end

"""
    get_unresetted_v_max(m::Tempotron, inp)

Get the tempotron `m`'s maximal unresetted voltage for agiven sample.

# Arguments

  - `m::Tempotron`: a tempotron
  - `PSPs`: a list of PSPs, formatted same as the output of ['get_psps'](@ref). Assumed to be sorted.

Returns a tuple containing:

  - `t_max`: the time of the maximal PSP.
  - `v_max`: the maximal PSP.
"""
function get_unresetted_v_max(m::Tempotron,
                              PSPs::Array{Tp,1})::Tuple{Real,
                                                        Real} where {Tp<:NamedTuple{(:time,
                                                                                     :ΔV,
                                                                                     :neuron)}}

    # The normalized weights
    W = m.w / m.K_norm

    # A temporary voltage function
    function Vt(t::Real)::Real
        tt = t - ΔTϵ
        emt, est = exp(-tt / m.τₘ), exp(-tt / m.τₛ)
        return (emt * sum_m - est * sum_s)
    end

    # Numerical hack
    Nϵ, ΔTϵ = 0, 0

    sum_m, sum_s = 0, 0
    t_max, V_max = 0, 0.0
    for P = 1:length(PSPs)
        (j, ΔV, i) = PSPs[P]

        # Get the next PSP's time
        next = (P < length(PSPs) ? PSPs[P + 1].time : j + 3m.τₘ)

        # Numerical hack
        N_ϵ = floor(j / m.Tϵ)
        if N_ϵ > Nϵ
            ΔNϵ = N_ϵ - Nϵ
            ΔT_ϵ = ΔNϵ * m.Tϵ
            sum_m *= exp(-ΔT_ϵ / m.τₘ)
            sum_s *= exp(-ΔT_ϵ / m.τₛ)
            Nϵ = N_ϵ
            ΔTϵ = Nϵ * m.Tϵ
        end
        jt = j - ΔTϵ

        # Analitically find the next local extermum
        sum_m += W[i] * exp(jt / m.τₘ)
        sum_s += W[i] * exp(jt / m.τₛ)
        t_max_c = get_next_t_max(m, j, next, ΔTϵ, sum_m, sum_s)[1]
        V_max_c = Vt(t_max_c)

        # Save the maximal local extermum
        if V_max_c > V_max
            t_max, V_max = t_max_c, V_max_c
        end
    end

    return t_max, V_max
end

"""
    get_spikes(m::Tempotron, PSPs, θ, max_spikes = typemax(Int); return_V = false, return_v_max = false)

Get the spike times for a given tempotron `m` and `PSPs` list.

# Arguments

  - `m::Tempotron`: a tempotron
  - `PSPs`: a list of PSPs, formatted same as the output of ['get_psps'](@ref). Assumed to be sorted.
  - `θ::Real = (m.θ - m.V₀)`: a voltage threshold (different than the tempotron's one).
  - `max_spikes::Integer = typemax(Int)`: a number of spikes to stop the search at.
  - `return_V::Bool = false`: add a `V` field to the output, containing the voltage trace function ``V(t)``.
  - `return_v_max::Bool = false`: add a `v_max` field to the output, containing information about the maxiumum subthreshold voltage.

Returns a named tuple:

  - `spikes` is a list of named tuples, where each one is of the form `(time, ΔV, psp)`. `time` is the spike's time, `ΔV` is the change in voltage incurred by the spike (as a function of time)m and `psp` is a tuple containing the time of the last PSP before the spike and the input neuron which elicited that PSP.

  - `V`: If `return_V` is set, also return the voltage function.
  - `v_max`: If `return_v_max` is set, also return the information about the maximal subthreshold local voltage maximum:
    
      + `psp`: the last PSP before the local maximum (time and neuron, same as spikes).
      + `t_max`: the time of the local maximum ``t_{max}``.
      + `next_psp`: the first PSP after the local maximum.
      + `v_max`: the voltage at the local maximum.
      + `sum_m`: ``V_{norm}\\sum_i w_i \\sum_{t_i^j \\le t_{max}} \\exp{\\left(\\frac{t_i^j - \\Delta T_{\\varepsilon}}{\\tau_m}\\right)}``
      + `sum_s`: ``V_{norm}\\sum_i w_i \\sum_{t_i^j \\le t_{max}} \\exp{\\left(\\frac{t_i^j - \\Delta T_{\\varepsilon}}{\\tau_s}\\right)}``
      + `ΔTϵ`: a time bias in `sum_m` and `sum_s`, introduced for numerical stability.
"""
function get_spikes(m::Tempotron,
                    PSPs::Array{Tp,1},
                    θ::Real = (m.θ - m.V₀),
                    max_spikes::Integer = typemax(Int);
                    return_V::Bool = false,
                    return_v_max::Bool = false)::NamedTuple where {Tp<:NamedTuple{(:time,
                                                                                   :ΔV,
                                                                                   :neuron)}}

    # Numerical constants
    ϵ = eps(Float64)

    # The normalized weights
    W = m.w ./ m.K_norm

    # Sums used to get local voltage maxima
    sum_m, sum_s, sum_e = 0.0, 0.0, 0.0

    # Numerical hack
    Nϵ, ΔTϵ = 0, 0.0

    # A list of spikes
    spikes = []

    # A temporary voltage function
    function Vt(t::Real)::Real
        tt = t - ΔTϵ
        emt, est = exp(-tt / m.τₘ), exp(-tt / m.τₛ)
        return (emt * sum_m - est * sum_s - θ * emt * sum_e)
    end

    # Save monotonous intervals
    if return_v_max
        mon_int_from = 0.0
        mon_int_prev = (spk = false,
                        asc = false,
                        v_e = -Inf)
        mon_int_max = (v_e = -Inf,)
        function push_mon_int(e::Real,
                              asc::Bool,
                              next::Real,
                              spk::Bool,
                              v_e::Real,
                              Σₘ::Real,
                              Σₛ::Real,
                              ΔTₑ::Real,
                              gen::Integer,
                              s::Real = mon_int_from)
            if !mon_int_prev.spk &&
               mon_int_prev.asc != asc &&
               mon_int_prev.v_e > mon_int_max.v_e
                mon_int_max = mon_int_prev
            end
            mon_int_prev = (s = s,
                            e = e,
                            asc = asc,
                            next = next,
                            spk = spk,
                            v_e = v_e,
                            gen = gen,
                            sum_m = Σₘ,
                            sum_s = Σₛ,
                            ΔTϵ = ΔTₑ)
            mon_int_from = e
        end
    end

    # Loop over PSPs
    for P = 1:length(PSPs)
        (j, ~, i) = PSPs[P]

        # If `max_spikes` is met, stop looking for spikes
        if length(spikes) ≥ max_spikes
            break
        end

        # Get the next PSP's time
        next = (P < length(PSPs) ? PSPs[P + 1].time : j + 3m.τₘ)

        # Numerical hack
        N_ϵ = Int(floor(j / m.Tϵ))
        if N_ϵ > Nϵ
            ΔNϵ = N_ϵ - Nϵ
            ΔT_ϵ = ΔNϵ * m.Tϵ
            em, es = exp(-ΔT_ϵ / m.τₘ), exp(-ΔT_ϵ / m.τₛ)
            sum_m *= em
            sum_s *= es
            sum_e *= em
            Nϵ = N_ϵ
            ΔTϵ = Nϵ * m.Tϵ
        end
        jt = j - ΔTϵ

        # Get the next local maximum
        sum_m += W[i] * exp(jt / m.τₘ)
        sum_s += W[i] * exp(jt / m.τₛ)
        t_max_j, l_max = get_next_t_max(m, j, next, ΔTϵ, sum_m, sum_s, sum_e, θ)
        v_max_j = Vt(t_max_j)

        # Start point from which to search for a spike
        s = j

        # Find the spike(s) time
        while v_max_j ≥ θ

            # Numerically find the spike time
            if v_max_j == θ # Extreme case.
                t_spk = t_max_j
            else
                st = s - ΔTϵ
                em, es = exp(-st / m.τₘ), exp(-st / m.τₛ)
                sum_s_m, sum_s_s, sum_s_e = em * sum_m, es * sum_s, em * sum_e
                function ΔVθ(t::Real)
                    emt, est = exp(-t / m.τₘ), exp(-t / m.τₛ)
                    V = emt * sum_s_m - est * sum_s_s - θ * emt * sum_s_e
                    return (V - θ)
                end
                t_spk = s + find_zero(ΔVθ, (0.0, t_max_j - s),
                                      Roots.AlefeldPotraShi())
            end

            # Add a new monotonous for the new spike.
            if return_v_max
                push_mon_int(t_spk, true, next, true, -Inf, sum_m, sum_s, ΔTϵ,
                             i, j)
            end

            # Update the voltage function
            ΔV(t) = -θ * m.η.(t .- t_spk)

            # Update spikes' sum
            sum_e += exp((t_spk - ΔTϵ) / m.τₘ)

            # Save spike
            push!(spikes, (time = t_spk, ΔV = ΔV, psp = (time = j, neuron = i)))

            # If `max_spikes` is met, stop looking for spikes
            if length(spikes) ≥ max_spikes
                break
            end

            # Set next starting point
            s = t_spk + ϵ

            # Check for immediate next spike
            t_max_j, l_max = get_next_t_max(m, t_spk, next, ΔTϵ, sum_m, sum_s,
                                            sum_e, θ)
            v_max_j = Vt(t_max_j)
        end

        # Set the next monotonous interval(s)
        if return_v_max
            asc = Vt(j) < v_max_j
            push_mon_int(t_max_j, asc, next, false, v_max_j, sum_m, sum_s, ΔTϵ,
                         i, j)
            if l_max
                push_mon_int(next, !asc, next, false, -Inf, sum_m, sum_s, ΔTϵ,
                             i)
            end
        end
    end

    # Return results
    ret = (spikes = spikes,)

    # Add the voltage function
    if return_V
        Vpsp(t::Real)::Real = sum(x -> x.ΔV(t), PSPs)
        Vspk(t::Real)::Real = (isempty(spikes) ? 0.0 :
                               sum(x -> x.ΔV(t), spikes))
        V(t::Real)::Real = m.V₀ + Vpsp(t) + Vspk(t)
        ret = merge(ret, (V = V,))
    end

    # Add the maximal subthreshold local voltage maximum
    if return_v_max
        v_max = (psp = (time = mon_int_max.s, neuron = mon_int_max.gen),
                 t_max = mon_int_max.e,
                 next_psp = mon_int_max.next,
                 v_max = mon_int_max.v_e,
                 sum_m = mon_int_max.sum_m,
                 sum_s = mon_int_max.sum_s,
                 ΔTϵ = mon_int_max.ΔTϵ)
        ret = merge(ret, (v_max = v_max,))
    end

    return ret
end

"""
    get_next_t_max(m::Tempotron, from, to, ΔTϵ, sum_m, sum_s, sum_e = 0, θ = (m.θ - m.V₀))

Get the next time suspected as a local extermum.

# Arguments

  - `m::Tempotron`: a tempotron.
  - [`from::Real`, `to::Real`]: a time interval ``[t_f, t_t]``.
  - `ΔTϵ`: a time interval introduced to solve some numerical instabilities.
  - `sum_m::Real`: ``V_{norm}\\sum_i w_i \\sum_{t_i^j \\le t_f} \\exp{\\left(\\frac{t_i^j - \\Delta T_{\\varepsilon}}{\\tau_m}\\right)}``
  - `sum_s::Real`: ``V_{norm}\\sum_i w_i \\sum_{t_i^j \\le t_f} \\exp{\\left(\\frac{t_i^j - \\Delta T_{\\varepsilon}}{\\tau_s}\\right)}``
  - `sum_e::Real = 0`: ``\\sum_{t_{spk}^j \\le t_f} \\exp{\\left(\\frac{t_{spk}^j - \\Delta T_{\\varepsilon}}{\\tau_m}\\right)}``
  - `θ::Real = (m.θ - m.V₀)`: the voltage threshold.

Returns the time of next suspected local maximum
``t_{max} \\in \\left[t_f, t_t\\right]``and an indicator whether the time
returned has a zero voltage derivative (i.e. ``t_{max} \\in \\left(t_f, t_t\\right)``).
"""
function get_next_t_max(m::Tempotron,
                        from::Real,    # TODO: use TimeInterval
                        to::Real,
                        ΔTϵ::Real,
                        sum_m::Real,
                        sum_s::Real,
                        sum_e::Real = 0,
                        θ::Real = (m.θ - m.V₀))::Tuple

    # Get next local extermum
    rem = (sum_m - θ * sum_e) / sum_s
    l_max = true
    if rem ≤ 0  # If there is no new local extermum
        l_max = false
    else
        t_max = ΔTϵ + m.A * (m.log_α - log(rem))
    end

    # Clamp the local maximum to the search interval
    l_max = l_max && (from < t_max < to)
    if !l_max
        t_max = to
    end

    return t_max, l_max
end

#-------------------------------------------------------------------------------
# Tempotron training methods API
#-------------------------------------------------------------------------------
"""
    TRAINING_METHODS

A list of acceptable training methods:

  - `:∇`: Gradient-based learning rules.
  - `:corr`: Correlation-based learning rules.
    For further details, see [`Tempotrons.train_∇!`](@ref) and
    [`Tempotrons.train_corr!`](@ref).
"""
const TRAINING_METHODS = [:∇, :corr]

"""
    train!(m::Tempotron, inp, y₀; method = :∇, kwargs...)

Trains a tempotron.

# Arguments

  - `m::Tempotron`: the tempotron to be trained.

  - `inp`: an input vector of spike trains.
  - `y₀`: a teacher's signal. The type of `y₀` determines the learning rules:
    
      + `y₀::Bool`: use the binary tempotron's learning rule.
      + `y₀::Int`: use the multi-spike tempotron's learning rule.
      + `y₀::SpikesInput{<:Real, 1}`: use the chronotron's learning rule for `method = :∇` or ReSuMe's learning rule for `method = :corr`.
  - `method::Symbol = :∇`: the training method (see [`Tempotrons.training_methods`](@ref)).
  - `kwargs...`: additional parameters (depend on the chosen method), i.e `optimizer::Optimizers.Optimizer`. See [`Tempotrons.Train_∇!`](@ref) or [`Tempotrons.Train_corr!`](@ref).

# Examples

```julia
input = InputGen.poisson_spikes_input(10; ν = 5, T = 500)
tmp = Tempotron(; N = 10)                                         # Create a tempotron
train!(tmp, input, true)                                        # Binary tempotron
train!(tmp, input, true; method = :corr)                        # Binary correlation-based
train!(tmp, input, 3)                                           # Multi-spike tempotron
train!(tmp, input, 5; method = :corr)                           # Multi-spike correlation-based
train!(tmp, input, SpikesInput([[50, 100]]))                    # Chronotron
train!(tmp, input, SpikesInput([[50, 100]]); method = :corr)    # ReSuMe
train!(tmp, input, 7; optimizer = Optimizers.SGD(0.01; momentum = 0.99))
train!(tmp, input, true; optimizer = Optimizers.Adam(0.001))
```

# References

## Binary tempotron:

[1] [Gütig, R., & Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing–based decisions. Nature neuroscience, 9(3), 420.](https://www.nature.com/articles/nn1643)

## Multi-spike tempotron:

[2] [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113)

## Chronotron

[3] [Florian R.V. (2012) The Chronotron: A Neuron That Learns to Fire Temporally Precise Spike Patterns. PLOS ONE, 7(8), e40233.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0040233)

## ReSuMe

[4] [Ponulak F. and Kasiński A. (2010). Supervised Learning in Spiking Neural Networks with ReSuMe: Sequence Learning, Classification, and Spike Shifting. Neural Computation, 22(2), 467-510](https://www.mitpressjournals.org/doi/abs/10.1162/neco.2009.11-08-901)
"""
function train!(m::Tempotron{N},
                inp::SpikesInput{T,N},
                y₀::Union{Bool,Integer,SpikesInput{T2,1}};
                method::Symbol = :∇,
                kwargs...) where {T<:Real,T2<:Real,N}
    if !(method ∈ TRAINING_METHODS)
        throw(ArgumentError("invalid method: $method. " *
                            "method must be one of $TRAINING_METHODS"))
    end
    train_func = Symbol(:train_, method, :!)
    train_func = eval(:($train_func))
    train_func(m, inp, y₀; kwargs...)
    return
end
function train!(m::Tempotron{N},
                inp::SpikesInput{T,N};
                method::Symbol = :∇,
                kwargs...) where {T<:Real,N}
    if !(method ∈ TRAINING_METHODS)
        throw(ArgumentError("invalid method: $method. " *
                            "method must be one of $TRAINING_METHODS"))
    end
    train_func = Symbol(:train_, method, :!)
    train_func = eval(:($train_func))
    train_func(m, inp; kwargs...)
    return
end
function train!(m::Tempotron{N},
                inp::Union{Array{Array{T,1} where T,1},
                           Array{Array{T,1},1} where T},
                args...;
                kwargs...) where {N}
    train!(m, convert(SpikesInput, inp), args...; kwargs...)
    return
end

"""
    train!(m::Tempotron, inp; epochs = 1[, kwargs...])

Trains a tempotron for each input is the list `inp`.
Any input sample in the list is assumed to be a named tuple `(:x, :y)` where `x`
is a `SpikesInput` and `y` is one of the acceptable teachers type.

# Examples

```julia
inp1 = InputGen.poisson_spikes_input(10; ν = 5, T = 500)
inp2 = InputGen.poisson_spikes_input(10; ν = 5, T = 500)
tmp = Tempotron(10)
train!(tmp, [(x = inp1, y = true), (x = inp2, y = false)])
train!(tmp, [(x = inp1, y = 7), (x = inp2, y = 2)])
```
"""
function train!(m::Tempotron{N},
                inp::Array{S,1};
                epochs::Integer = 1,
                kwargs...) where {N,
                                  T<:Real,
                                  Tx<:Union{SpikesInput{T,N},
                                            Array{Array{T1,1},1} where T1,
                                            Array{Array{T1,1} where T1,1},
                                            Array{T1,1} where T1},
                                  Ty<:Union{Integer,Bool,
                                            SpikesInput{T1,1} where T1,
                                            Array{T1,1} where T1},
                                  S<:NamedTuple{(:x, :y),Tuple{Tx,Ty}}}
    @assert epochs > 0 "At least one training epoch is required."

    inputs = copy(inp)
    for e = 1:epochs
        shuffle!(inputs)
        for i ∈ inputs
            train!(m, i.x, i.y; kwargs...)
        end
    end

    return
end
function train!(m::Tempotron{N},
                inp::Array{S,1};
                epochs::Integer = 1,
                kwargs...) where {N,
                                  T<:Real,
                                  S<:Union{SpikesInput{T,N},
                                           Array{Array{T1,1},1} where T1,
                                           Array{Array{T1,1} where T1,1},
                                           Array{T1,1} where T1}}
    @assert epochs > 0 "At least one training epoch is required."

    inputs = copy(inp)
    for e = 1:epochs
        shuffle!(inputs)
        for i ∈ inputs
            train!(m, i; kwargs...)
        end
    end

    return
end
function train!(m::Tempotron{N}, inp::Array{S,1};
                kwargs...) where {N,Tx,Ty,S<:Tuple{Tx,Ty}}
    train!(m, [NamedTuple{(:x, :y),S}(i) for i ∈ inp]; kwargs...)
    return
end
