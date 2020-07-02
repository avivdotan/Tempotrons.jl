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
    w :: Array{Real, 1}

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
    Input kernel's time derivative `K̇(t)`
    """
    K̇

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
# The input kernel `K(t)` and its derivative `k̇(t)`.
    K(t::Real) = t < 0 ? 0 : ((exp(-t/τₘ) - exp(-t/τₛ)) / K_norm)
    K̇(t::Real) = t < 0 ? 0 : ((-exp(-t/τₘ)/τₘ + exp(-t/τₛ)/τₛ) / K_norm)

    #The spike kernel `η(t)`.
    η(t::Real) = t < 0 ? 0 : exp(-t/τₘ)

    # Initialize weights
    w = (12rand(Float64, N) .- 3).*(θ - V₀)./N

    return Tempotron(τₘ, τₛ, θ, V₀, w, α, K_norm, A, log_α, K, K̇, η)
end

"""
    ValidateInput(m::Tempotron, inp[, T_max])
Validates an input vector of spike trains `inp` for a given tempotron `m` and
sets default values for the number of input neurons `N`.
"""
function ValidateInput(m::Tempotron,
                        inp::Array{Array{Tp, 1}, 1}) where Tp <: Real

    # N
    N = length(m.w)
    if length(inp) != N
        error("The number of input neurons is incompatible with the input. ")
    end

    # Valid
    valid = !all([isempty(inp[i]) for i = 1:N])
    if !valid
        return valid, N
    end

    return valid, N
end

"""
    GetPSPs(m::Tempotron, inp)
Get a list of PSPs for a given input vector of spike trains `inp` and tempotron
`m`. Each PSP in the list is a named tuple `(time = j, ΔV = K(t - j), neuron = i)`,
where `j` is the input spike time, `K(t)` is the properly weighted and shifted
kernel of the tempotron `m` and `i` is the index of the generating input neuron.
"""
function GetPSPs(m::Tempotron,
                 inp::Array{Array{Tp, 1}, 1}) where Tp <: Real

    PSPs = [(time      = j::Real,
             ΔV        = t::Real -> m.w[i].*m.K.(t - j),
             neuron    = i::Integer)
            for i = 1:length(m.w)
            for j ∈ inp[i]]
    return PSPs
end

"""
    (m::Tempotron)(inp[, t][, dt = 1][, T_max])
Get the tempotron `m`'s output voltage for an input vector of spike trains `inp`.
An optional parameter is a time grid `t`, at which to sample the voltage
function to be returned as a second output argument.
"""
function (m::Tempotron)(inp::Array{Array{Tp1, 1}, 1};
                        t::Union{Array{Tp2, 1}, Nothing} = nothing) where {Tp1 <: Real,
                                                                           Tp2 <: Real}
    ~, N = ValidateInput(m, inp)

    # Get the PSPs
    PSPs = sort(GetPSPs(m, inp), by = x -> x.time)

    # Get the spike times and voltage function
    (spikes, V) = GetSpikes(m, PSPs, return_V = true)
    spikes = [s.time for s ∈ spikes]

    # If no time grid was given, return spike times
    ret = (spikes = spikes, )
    if t ≡ nothing
        return ret
    end

    # calculate the voltage over the time grid
    Vt = V.(t)

    # Improve spikes visibility
    for j ∈ spikes
        if j < maximum(t)
            k = findfirst(t .> j)
            Vt[k] = m.θ + 0.3(m.θ - m.V₀)
        end
    end

    ret = merge(ret, (V = Vt, ))
    return ret
end

"""
    GetSpikes(m::Tempotron, PSPs[, θ::Real])
Get the spike times for a given tempotron `m` and PSPs list. `PSPs` is assumed
to be sorted by the input spike times. For the multi-spike tempotron there are
optional parameters `θ` for the voltage threshold (default is the tempotron's
threshold `m.θ`) and the maximum number of spikes to look for (`max_spikes`).
Returns a list of named tuples, where each one is of the form
`(time, ΔV, psp)`, where `time` is the spike's time, `ΔV` is the change
in voltage incurred by the spike (as a function of time)m and `psp` is a tuple
containing the time of the last PSP before the spike and the input neuron which
elicited that PSP.
If `return_V == true`, also return the voltage function.
If `return_v_max == true`, also return the maximal subthreshold local voltage
maximum in the format `(psp, t_max, next_psp, v_max, sum_m, sum_s)`, where `psp`
is the last PSP before the local maximum (time and neuron, same as spikes),
`t_max` is the time of the local maximum, `next_psp` is the first PSP after the
local maximum, `v_max` is the voltage at the local maximum and `sum_m` and
`sum_s` are sums of exponents at the local maximum used for recalculating it and
`Nϵ` and `Tϵ` stand for bias in `sum_m` and `sum_s`, meant soley for numerical
stability.
"""
function GetSpikes(m::Tempotron,
                    PSPs::Array{Tp, 1},
                    θ::Real = (m.θ - m.V₀),
                    max_spikes::Integer = typemax(Int);
                    return_V::Bool = false,
                    return_v_max::Bool = false) where Tp <: NamedTuple

    # Numerical constants
    ϵ    = eps(Float64)
    Tϵ   = 1000
    e_m  = exp(-Tϵ/m.τₘ)
    e_s  = exp(-Tϵ/m.τₛ)

    # The normalized weights
    W = m.w / m.K_norm

    # Sums used to get local voltage maxima
    sum_m, sum_s, sum_e = 0, 0, 0
    Nϵ = 0

    # A list of spikes
    spikes = []

    # A temporary voltage function
    function Vt(t::Real)::Real
        t_tmp = t - Nϵ*Tϵ
        emt, est = exp(-t_tmp/m.τₘ), exp(-t_tmp/m.τₛ)
        return (emt*sum_m - est*sum_s - θ*emt*sum_e)
    end

    # Save monotonous intervals
    if return_v_max
        mon_int = []
        mon_int_last = 0
        function push_mon_int(e::Real, asc::Bool, next::Real, spk::Bool,
                                v_e::Real, sum_m::Real, sum_s::Real, Nϵ::Real,
                                gen::Integer, s::Real = mon_int_last)
            push!(mon_int, (s = s, e = e, asc = asc, next = next, spk = spk,
                            v_e = v_e, gen = gen, sum_m = sum_m, sum_s = sum_s,
                            Nϵ = Nϵ))
            mon_int_last = e
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

        # Handle numerical stability issues (exponents returned `Inf`)
        j_tmp = j - Tϵ*Nϵ
        while j_tmp > Tϵ
            Nϵ     += 1
            j_tmp  -= Tϵ
            sum_m  *= e_m
            sum_s  *= e_s
            sum_e  *= e_m
        end

        # Get the next local maximum
        sum_m += W[i]*exp(j_tmp/m.τₘ)
        sum_s += W[i]*exp(j_tmp/m.τₛ)
        t_max_j, l_max = GetNextTmax(m, j, next, sum_m, sum_s, sum_e, θ)
        v_max_j = Vt(t_max_j)

        # Start point from which to search for a spike
        s = j

        # Find the spike(s) time
        while v_max_j ≥ θ

            # Numerically find the spike time
            if v_max_j == θ # Extreme case, when two spikes are
                            # generated together.
                t_spk = t_max_j
            else
                t_spk = find_zero(t -> Vt(t) - θ, (s, t_max_j), Roots.A42())
            end

            # Add a new monotonous for the new spike.
            if return_v_max
                push_mon_int(t_spk, true, next, true, -Inf,
                             sum_m, sum_s, Nϵ, i, j)
            end

            # Update the voltage function
            ΔV(t) = -θ*m.η.(t .- t_spk)

            # Update spikes' sum
            t_spk_tmp = t_spk - Nϵ*Tϵ
            sum_e += exp(t_spk_tmp/m.τₘ)

            # Save spike
            push!(spikes, (time = t_spk, ΔV = ΔV, psp = (time = j, neuron = i)))

            # If `max_spikes` is met, stop looking for spikes
            if length(spikes) ≥ max_spikes
                break
            end

            # Set next starting point
            s = t_spk + ϵ

            # Check for immediate next spike
            t_max_j, l_max = GetNextTmax(m, t_spk, next, sum_m, sum_s, sum_e, θ)
            v_max_j = Vt(t_max_j)

        end

        # Set the next monotonous interval(s)
        if return_v_max
            v_j = Vt(j)
            asc = v_j < v_max_j
            push_mon_int(t_max_j, asc, next, false, v_max_j,
                         sum_m, sum_s, Nϵ, i, j)
            if l_max
                push_mon_int(next, !asc, next, false, -Inf,
                             sum_m, sum_s, Nϵ, i)
            end
        end

    end

    # Return results
    ret = (spikes = spikes, )

    # Add the voltage function
    if return_V
        Vpsp(t::Real)::Real = sum(x -> x.ΔV(t), PSPs)
        Vspk(t::Real)::Real = (isempty(spikes) ? 0 : sum(x -> x.ΔV(t), spikes))
        V(t::Real)::Real = m.V₀ + Vpsp(t) + Vspk(t)
        ret = merge(ret, (V = V, ))
    end

    # Add the maximal subthreshold local voltage maximum
    if return_v_max
        mon_int_max = (v_e = -Inf, )
        for k = 1:length(mon_int) - 1
            if !mon_int[k].spk &&
                mon_int[k].asc != mon_int[k + 1].asc &&
                mon_int[k].v_e > mon_int_max.v_e
                mon_int_max = mon_int[k]
            end
        end
        v_max = (psp        = (time   = mon_int_max.s,
                               neuron = mon_int_max.gen),
                 t_max      = mon_int_max.e,
                 next_psp   = mon_int_max.next,
                 v_max      = mon_int_max.v_e,
                 sum_m      = mon_int_max.sum_m,
                 sum_s      = mon_int_max.sum_s,
                 Nϵ         = mon_int_max.Nϵ,
                 Tϵ         = Tϵ)
        ret = merge(ret, (v_max = v_max, ))
    end

    return ret

end

"""
    GetNextTmax(m::Tempotron, from, to, sum_m, sum_s, sum_e, θ)
Get the next time suspected as a local extermum. Receiving a Tempotron `m`, an
interval [`from`, `to`], relevant sums of exponents `sum_m`, `sum_s`, `sum_e`
and the voltage threshold `θ`. Returns the next suspected local maximum and an
indicator whether the time returned has a zero voltage derivative.
"""
function GetNextTmax(m::Tempotron,
                    from::Real,
                    to::Real,
                    sum_m::Real,
                    sum_s::Real,
                    sum_e::Real = 0,
                    θ::Real = m.θ)

    # Get next local extermum
    rem = (sum_m - θ*sum_e)/sum_s
    l_max = true
    if rem ≤ 0  # If there is no new local extermum
        l_max = false
    else
        t_max = m.A*(m.log_α - log(rem))
    end

    # Clamp the local maximum to the search interval
    l_max = l_max && (from < t_max < to)
    if !l_max
        t_max = to
    end

    return t_max, l_max
end
