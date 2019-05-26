mutable struct Tempotron
    τₘ :: Real # ms
    τₛ :: Real # ms
    θ :: Real # mV
    V₀ :: Real # mV
    w :: Array{Real}
end
Broadcast.broadcastable(m::Tempotron) = Ref(m)

function Tempotron(; N :: Integer,
                    τₘ :: Real = 15,
                    τₛ :: Real = τₘ/4,
                    θ :: Real = 1,
                    V₀ :: Real = 0)
    w = (1.2.*rand(Float64, N) .- 0.3).*(θ - V₀)
    # w = 0.8.*rand(Float64, N).*(θ - V₀)
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
    return Tempotron(τₘ, τₛ, θ, V₀, w)
end

function ValidateInput(m::Tempotron,
    inp::Array{Array{Tp, 1}, 1},
    T_max::Real = 0) where Tp <: Any
    N = length(m.w)
    if length(inp) != N
        error("The number of input neurons is incompatible with the input. ")
    end
    jmin(x) = minimum(Iterators.flatten(x))
    jmax(x) = maximum(Iterators.flatten(x))
    T = (T_max == 0) ? (maximum(jmax(inp)) + 3m.τₘ) : T_max
    if T < maximum(jmax(inp)) || minimum(jmin(inp)) < 0
        error("There are input spike times outside of the simulation's skope")
    end
    return N, T
end

function K(m::Tempotron, t::Real)
    α = m.τₘ / m.τₛ
    K_norm = α^(-1/(α - 1)) - α^(-α/(α - 1))
    return  ((t < 0) ? 0 : ((exp(-t/m.τₘ) - exp(-t/m.τₛ)) / K_norm))
end

function GetPSPs(m::Tempotron,
                    inp::Array{Array{Tp, 1}, 1},
                    T_max::Real = 0) where Tp <: Any
    N, T = ValidateInput(m, inp, T_max)

    PSPs = hcat([(j, t -> m.w[i].*K.(m, t - j), i)
                 for i = 1:N
                 for j ∈ inp[i]])
    return PSPs
end

function (m::Tempotron)(inp::Array{Array{Tp1, 1}, 1};
                        t::Array{Tp2, 1} = nothing,
                        dt::Real = 1,
                        T_max::Real = 0) where {Tp1 <: Any,
                                                Tp2 <: Real}
    N, T = ValidateInput(m, inp, T_max)

    PSPs = GetPSPs(m, inp, T)
    PSP(t) = m.V₀ + sum(x -> x[2](t), PSPs)

    η(t) = t < 0 ? 0 : exp(-t/m.τₘ)

    if t ≡ nothing
        t = 0:dt:T
    end
    V = PSP.(t)
    for j = 1:length(t)
        if V[j] > m.θ && j < length(t)
            V[j] = m.θ + 0.3(m.θ - m.V₀)
            V -= (m.θ - m.V₀).*η.(t .- t[j + 1])
        end
    end
    return V
end
