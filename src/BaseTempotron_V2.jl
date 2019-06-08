"""
A (binary/multi-spike) tempotron.
"""
struct Tempotron
    """
    Membrane time constant [ms]
    """
    τₘ :: Real

    """
    Sytaptic time constant [ms]
    """
    τₛ :: Real

    """
    Voltage threshold [mV]
    """
    θ :: Real

    """
    Reset voltage [mV]
    """
    V₀ :: Real

    """
    Synaptic weights
    """
    w :: Array{Real}

    # Precalculated constants
    α :: Real
    K_norm :: Real
    A :: Real
    log_α :: Real

    """
    Input kernel `K(t)`
    """
    K

    """
    Spike kernel `η(t)`
    """
    η
end
Broadcast.broadcastable(m::Tempotron) = Ref(m)

"""
    Tempotron(N[, τ_m = 15][, τₛ = τₘ/4][, θ = 1][, V₀ = 0])
Create a new tempotron win `N` input neurons, membrane time constant `τₘ`,
synaptic time constant `τₛ`, voltage threshold `θ` and rest potential `V₀`.
"""
function Tempotron(; N :: Integer,
                    τₘ :: Real = 15,
                    τₛ :: Real = τₘ/4,
                    θ :: Real = 1,
                    V₀ :: Real = 0)
    # Validate inputs
    if N < 1
        error("There must be at least one input neuron. ")
    end
    if τₘ ≤ 0
        error("Membrane's time constant must be positive. ")
    end
    if τₛ ≤ 0
        error("Synaptic time constant must be positive. ")
    end
    if V₀ ≥ θ
        error("Firing threshold must be above rest potential. ")
    end

    # Pre-calculate constants
    α = τₘ / τₛ
    K_norm = α^(-1/(α - 1)) - α^(-α/(α - 1))
    A = τₘ * τₛ / (τₘ - τₛ)
    log_α = log(α)

    # The input kernel `K(t)`.
    K(t::Real) = t < 0 ? 0 : ((exp(-t/τₘ) - exp(-t/τₛ)) / K_norm)

    #The spike kernel `η(t)`.
    η(t::Real) = t < 0 ? 0 : exp(-t/τₘ)

    # Initialize weights
    w = (1.2.*rand(Float64, N) .- 0.3).*(θ - V₀)

    return Tempotron(τₘ, τₛ, θ, V₀, w, α, K_norm, A, log_α, K, η)
end

"""
    ValidateInput(m::Tempotron, inp[, T_max])
Validates an input vector of spike trains `inp` for a given tempotron `m` and
sets default values for the number of ninput neurons `N` and the maximal time
`T`.
"""
function ValidateInput(m::Tempotron,
                        inp::Array{Array{Tp, 1}, 1},
                        T_max::Real = 0) where Tp <: Any
    # N
    N = length(m.w)
    if length(inp) != N
        error("The number of input neurons is incompatible with the input. ")
    end

    # T
    jmin(x) = minimum(Iterators.flatten(x))
    jmax(x) = maximum(Iterators.flatten(x))
    T = (T_max == 0) ? (maximum(jmax(inp)) + 3m.τₘ) : T_max
    if T < maximum(jmax(inp)) || minimum(jmin(inp)) < 0
        error("There are input spike times outside of the simulation's skope")
    end

    return N, T
end

"""
    GetPSPs(m::Tempotron, inp)
Get a list of PSPs for a given input vector of spike trains `inp` and tempotron
`m`. Each PSP in the list is a named tuple `(time = j, ΔV = K(t - j), neuron = i)`,
where `j` is the input spike time, `K(t)` is the properly weighted and shifted
kernel of the tempotron `m` and `i` is the index of the generating input neuron.
"""
function GetPSPs(m::Tempotron,
                 inp::Array{Array{Tp, 1}, 1}) where Tp <: Any

    PSPs = hcat([(time      = j,
                  ΔV        = t -> m.w[i].*m.K.(t - j),
                  neuron    = i)
                 for i = 1:length(m.w)
                 for j ∈ inp[i]])
    return PSPs[:]
end

"""
    (m::Tempotron)(inp[, t][, dt = 1][, T_max])
Get the tempotron `m`'s output voltage for an input vector of spike trains `inp`.
Optional parameters are the time grid `t` or its density `dt` and its maximum
`T_max`.
"""
function (m::Tempotron)(inp::Array{Array{Tp1, 1}, 1};
                        t::Array{Tp2, 1} = nothing) where {Tp1 <: Any,
                                                            Tp2 <: Real}
    T_max = (t ≡ nothing ? 0 : maximum(t))
    N, T = ValidateInput(m, inp, T_max)

    # Get the PSPs
    PSPs = sort(GetPSPs(m, inp), by = x -> x.time)

    # Get the spike times
    (spikes, V) = GetSpikes(m, PSPs, return_V = true)
    spikes = [s.time for s ∈ spikes]

    # If no time grid was given, return spike times
    if t ≡ nothing
        return spikes
    end

    # calculate the (unresetted) volage over the time grid
    Vt = V.(t)

    # Improve spikes visibility
    for j ∈ spikes
        if j < maximum(t)
            k = findfirst(t .> j)
            Vt[k] = m.θ + 0.3(m.θ - m.V₀)
        end
    end

    return (spikes, Vt)
end

"""
    GetSpikes(m::Tempotron, PSPs[, PSP][, θ::Real])
Get the spike times for a given tempotron `m` and PSPs list. `PSPs` is assumed
to be sorted by the input spike times. The unresetted total voltage function can
be also supplied to increase preformance. For the multi-spike tempotron there is
an optonal parameter `θ` for the voltage threshold (default is the tempotron's
threshold `m.θ`).
Returns a list of named tuples, where each one is of the form
`(time = spike_time, ΔV = -θ*η(t - spike_time), neuron = 0)`.
If `return_V == true`, also return the voltage function.
If `return_v_max == true`, also return the maximal local voltage maximum along
with some of its properties (see [`AddVmax!`](@ref) for details).
"""
function GetSpikes(m::Tempotron,
                    PSPs,
                    PSP = (t -> m.V₀ + sum(x -> x.ΔV(t), PSPs)),
                    θ::Real = m.θ,
                    max_spikes::Integer = typemax(Int);
                    return_V::Bool = false,
                    return_v_max = false) where {T1 <: Real,
                                                T2 <: Real}

    # A small perturbation
    ϵ = eps(Float64)

    # The normalized weights
    W = m.w / m.K_norm

    # Sums used to get local voltage maxima
    sum_m = 0
    sum_s = 0
    sum_e = 0

    # A list of spikes
    spikes = []

    # Voltage function
    Vspk(t) = (isempty(spikes) ? 0 : sum(x -> x.ΔV(t), spikes))
    V(t) = PSP(t) + (isempty(spikes) ? 0 : sum(x -> x.ΔV(t), spikes))

    # Voltage derivative
    der(f) = x -> ForwardDiff.derivative(f, float(x))
    dVpsp = der(PSP)
    dVspk = der(Vspk)
    V̇(t) = dVpsp(t) + dVspk(t)

    # Start point from which to search for a spike
    s = 0

    # Save monotonous intervals
    if return_v_max
        mon_int = []
        mon_int_last = 0
        function push_mon_int(e::Real, asc::Bool, l_max::Bool, spk::Bool,
                                v_e::Real, sum_m::Real, sum_s::Real,
                                s::Real = mon_int_last)
            push!(mon_int, (s = s, e = e, asc = asc, l_max = l_max,
                            spk = spk, v_e = v_e, sum_m = sum_m, sum_s = sum_s))
            mon_int_last = e
        end
        function pop_mon_int()
            tmp = pop!(mon_int)
            mon_int_last = tmp.s
        end
        push_mon_int(PSPs[1].time, true, false, false, -Inf, sum_m, sum_s)
    end

    # Loop over PSPs
    for P = 1:length(PSPs)
        (j, ~, i) = PSPs[P]

        if length(spikes) ≥ max_spikes
            break
        end

        # Get the next local maximum
        sum_m += W[i]*exp(j/m.τₘ)
        sum_s += W[i]*exp(j/m.τₛ)
        rem = (sum_m - θ*sum_e)/sum_s
        l_max = true
        if rem ≤ 0  #TODO: Remove?
            l_max = false
        else
            t_max_j = m.A*(m.log_α - log(rem))
        end
        l_max = l_max && !(t_max_j < j ||
                  (P < length(PSPs) && PSPs[P + 1].time < t_max_j))
        if !l_max
            t_max_j = (P < length(PSPs) ? PSPs[P + 1].time : j + 3m.τₘ)
        end
        v_max_j = V(t_max_j)
        if return_v_max
            v_j = V(j)
            asc = v_j < v_max_j
            push_mon_int(t_max_j, asc, l_max, false, v_max_j, sum_m, sum_s)
            if l_max
                tmp = (P < length(PSPs) ? PSPs[P + 1].time : j + 3m.τₘ)
                push_mon_int(tmp, !asc, false, false, -Inf, sum_m, sum_s)
            end
        end

        # Find the spike(s) time
        s = j
        while v_max_j > θ

            # Numerically find the spike time
            # TODO: Remove debug prints once stable
            t_spk = 0
            try
                t_spk = find_zero(t -> V(t) - θ, (s, t_max_j), Roots.A42())
            catch ex
                Vs, V_t_max = V(s), V(t_max_j)
                spsps = [sign(m.w[x.neuron])*x.time for x ∈ PSPs]
                spks = [x.time for x ∈ spikes]
                @debug ("θ = $θ: [$s, $t_max_j] -> [$Vs, $V_t_max]\n" *
                "PSPs: $spsps" *
                "spikes (partial): $spks")
                throw(ex)
            end

            if return_v_max
                pop_mon_int()
                if l_max
                    pop_mon_int()
                end
                push_mon_int(t_spk, true, false, true, -Inf, sum_m, sum_s)
            end

            # Update the voltage function and derivative
            ΔV(t) = -θ*m.η.(t .- t_spk)
            dVspk = der(Vspk)

            # Update spikes' sum
            sum_e += exp(t_spk/m.τₘ)

            # Save spike
            push!(spikes, (time = t_spk, ΔV = ΔV, neuron = 0, psp = j))

            if length(spikes) ≥ max_spikes
                break
            end

            # Set next starting point
            s = t_spk + ϵ

            # Check for immediate next spike
            rem = (sum_m - θ*sum_e)/sum_s
            l_max = true
            if rem ≤ 0  #TODO: Remove?
                l_max = false
            else
                t_max_j = m.A*(m.log_α - log(rem))
            end
            l_max = l_max && !(t_max_j < t_spk ||
                      (P < length(PSPs) && PSPs[P + 1].time < t_max_j))
            if !l_max
                t_max_j = (P < length(PSPs) ? PSPs[P + 1].time : j + 3m.τₘ)
            end
            v_max_j = V(t_max_j)
            if return_v_max
                v_j = V(j)
                asc = v_j < v_max_j
                push_mon_int(t_max_j, asc, l_max, false, v_max_j, sum_m, sum_s,
                                j)
                if l_max
                    (P < length(PSPs) ? PSPs[P + 1].time : j + 3m.τₘ)
                    push_mon_int(tmp, !asc, false, false, -Inf, sum_m, sum_s)
                end
            end
        end
    end

    # Return results
    if !return_V && !return_v_max
        return spikes
    else
        ret = (spikes, )
        if return_V
            ret = (ret..., V)
        end
        if return_v_max
            v_max = (v_e = -Inf, )
            for k = 1:length(mon_int) - 1
                if !mon_int[k].spk &&
                    mon_int[k].asc != mon_int[k + 1].asc &&
                    mon_int[k].v_e > v_max.v_e
                    v_max = mon_int[k]
                end
            end
            # TODO: Remove debug prints once stable
            # @debug "mon_int: $mon_int\n
            # v_max: $v_max"
            ret = (ret..., v_max)
        end
        return ret
    end
end
