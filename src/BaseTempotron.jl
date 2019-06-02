"""
A (binary/multi-spike) tempotron.
"""
struct Tempotron
    τₘ :: Real # ms
    τₛ :: Real # ms
    θ :: Real # mV
    V₀ :: Real # mV
    w :: Array{Real}
    α :: Real
    K_norm :: Real
    A :: Real
    log_α :: Real
    K
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
    GetPSPs(m::Tempotron, inp[, T_max])
Get a list of PSPs for a given input vector of spike trains `inp` and tempotron
`m`. Each PSP in the list is a tuple `(j, K(t), i)`, where `j` is the input spike
time, `K(t)` is the properly weighted and shifted kernel of the tempotron `m`
and `i` is the index of the generating input neuron.
"""
function GetPSPs(m::Tempotron,
                    inp::Array{Array{Tp, 1}, 1},
                    T_max::Real = 0) where Tp <: Any
    N, T = ValidateInput(m, inp, T_max)

    PSPs = hcat([(j, t -> m.w[i].*m.K.(t - j), i)
                 for i = 1:N
                 for j ∈ inp[i]])
    return PSPs
end

"""
    (m::Tempotron)(inp[, t][, dt = 1][, T_max])
Get the tempotron `m`'s output voltage for an input vector of spike trains `inp`.
Optional parameters are the time grid `t` or its density `dt` and its maximum
`T_max`.
"""
function (m::Tempotron)(inp::Array{Array{Tp1, 1}, 1};
                        t::Array{Tp2, 1} = nothing,
                        dt::Real = 1,
                        T_max::Real = 0) where {Tp1 <: Any,
                                                Tp2 <: Real}
    N, T = ValidateInput(m, inp, T_max)

    # Get the unresetted voltage
    PSPs = GetPSPs(m, inp, T)
    PSP(t) = m.V₀ + sum(x -> x[2](t), PSPs)

    # Generate a default time grid
    if t ≡ nothing
        t = 0:dt:T
    end

    # calculate the (unresetted) volage over the time grid
    V = PSP.(t)

    # Search for spikes
    for j = 1:length(t)
        if V[j] > m.θ && j < length(t)
            # Add a spike (for visualization)
            V[j] = m.θ + 0.3(m.θ - m.V₀)
            # Reset the voltage
            V -= (m.θ - m.V₀).*m.η.(t .- t[j + 1])
        end
    end

    return V
end
