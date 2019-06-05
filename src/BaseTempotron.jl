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
                    θ::Real = m.θ;
                    return_V::Bool = false,
                    return_v_max = false) where {T1 <: Real,
                                                T2 <: Real}

    # A small perturbation
    ϵ = eps(Float64)

    # The normalized weights
    W = m.w / m.K_norm

    # Generate an alternative list of PSPs along with the last exitatory PSP's
    # time
    # TODO: Prebuild (for speed)
    Ps = [(time = j, ΔV = ΔV, neuron = i, lex = j) for (j, ΔV, i) ∈ PSPs]
    for a = 1:length(Ps)
        # For inhibitory input, get the last exitatory input time
        if m.w[Ps[a].neuron] < 0
            Ps[a] = merge(Ps[a], (lex = (a == 1 ? 0 : Ps[a - 1].lex), ))
        end
    end

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

    # Local subthreshold voltage maxima (and, sometimes, minima)
    v_max_hist = []

    # Loop over PSPs
    for P = 1:length(Ps)
        (j, ~, i, k) = Ps[P]

        # Gt the next local maximum
        sum_m += W[i]*exp(j/m.τₘ)
        sum_s += W[i]*exp(j/m.τₛ)
        ex_max = !(W[i] < 0 && V̇(j + ϵ) < 0)
        t_max_j = NextTmax(m, j, ex_max, sum_m, sum_s, sum_e, θ)
        if t_max_j ≡ nothing
            continue
        end
        v_max_j = V(t_max_j)

        # Add possible local maximum
        if return_v_max && v_max_j < θ
            AddVmax!(v_max_hist, t_max_j, v_max_j, ex_max, V̇, k, sum_m, sum_s)
        end

        # Only update the start point if the PSP started with subthreshold
        # voltage
        if V(k) < θ
            s = k
        end

        # Find the spike(s) time
        while v_max_j > θ

            # Numerically find the spike time
            # TODO: Remove debug prints once stable
            t_spk = 0
            try
                t_spk = find_zero(t -> V(t) - θ, (s, t_max_j), Roots.A42())
            catch ex
                Vs, V_t_max = V(s), V(t_max_j)
                spsps = [sign(m.w[x.neuron])*x.time for x ∈ PSPs]
                spks = [x[1] for x ∈ spikes]
                @debug ("θ = $θ: [$s, $t_max_j] -> [$Vs, $V_t_max]\n" *
                "PSPs: $spsps" *
                "spikes (partial): $spks")
                throw(ex)
            end

            # Update the voltage function and derivative
            ΔV(t) = -θ*m.η.(t .- t_spk)
            dVspk = der(Vspk)

            # Update spikes' sum
            sum_e += exp(t_spk/m.τₘ)

            # Link to last exitatory PSP prior to the spike
            Q = P + 1
            while Q ≤ length(Ps) && Ps[Q].time < t_spk
                Q += 1
            end
            c = Ps[Q - 1].lex

            # Save spike
            push!(spikes, (time = t_spk, ΔV = ΔV, neuron = 0, lex = c))

            # Set next starting point
            s = t_spk + ϵ

            # Check for immediate next spike
            t_max_t = NextTmax(m, j, ex_max, sum_m, sum_s, sum_e, θ)
            if t_max_t ≡ nothing # TODO: remove?
                break
            end
            t_max_j = t_max_t
            v_max_j = V(t_max_j)

            # Add possible local maximum
            if return_v_max && v_max_j < θ
                AddVmax!(v_max_hist, t_max_j, v_max_j, ex_max, V̇, k, sum_m, sum_s)
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
            v_max = v_max_hist[argmax([vm.v_max_j for vm ∈v_max_hist])]
            ret = (ret..., v_max)
        end
        return ret
    end
end

"""
    NextTmax(m::Tempotron, w, V̇, sum_m, sum_s[, sum_e = 0][, θ = m.θ])
Search for the next possible point of local maximum. Recieving a Tempotron `m`,
last event time `j`, A Boolean `ex_max` indicating an exitatory PSP treatment,
relevant sums (`sum_m`, `sum_s`, `sum_e`) and optionally a threshold `θ`.
"""
function NextTmax(m::Tempotron,
                    j::Real,
                    ex_max::Bool,
                    sum_m::Real,
                    sum_s::Real,
                    sum_e::Real = 0,
                    θ::Real = m.θ)

    # Inhibitory PSP that decreases the voltage
    if !ex_max
        # The local maximum is the input spike time
        t_max_j = j
    else
        # Get the local maximum's time
        rem = (sum_m - θ*sum_e)/sum_s
        if rem ≤ 0  #TODO: Remove?
            return nothing
        end
        t_max_j = m.A*(m.log_α - log(rem))
    end
    return t_max_j
end

# TODO: Docs
function AddVmax!(v_max_hist::Array{Tp, 1},
                t_max_j::Real,
                v_max_j::Real,
                ex_max::Bool,
                V̇,
                k::Real,
                sum_m::Real,
                sum_s::Real) where Tp <: Any

    # A small preturbation
    ϵ = eps(Float64)

    # Check if local maximum
    if (ex_max && abs(V̇(t_max_j)) < 1e-5 ||
        !ex_max && V̇(t_max_j + ϵ) < 0)
        v_max = (v_max_j = v_max_j, t_max_j = t_max_j, lex = k, ex_max = ex_max,
                sum_m = sum_m, sum_s = sum_s)
        push!(v_max_hist, v_max)
    end
end
