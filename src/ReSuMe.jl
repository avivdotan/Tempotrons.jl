"""
    train_corr!(m::Tempotron, inp::SpikesInput, y₀::SpikesInput{<:Real,1}; aᵣ = 0, τᵣ = m.τₘ, f = t -> (t < 0 ? 0.0 : exp(-t / τᵣ)), optimizer = SGD(0.01))

Trains a neuron `m` to fire at specific times (according to y₀) in response to
an input vector of spike trains `inp`.

# Optional arguments

  - `aᵣ::Real = 0` a non-Hebbian term.
  - `τᵣ::Real = m.τₘ` the default kernel's time constant.
  - `fᵣ::Function = t -> (t < 0 ? 0.0 : exp(-t / τᵣ))` the kernel.
  - `optimizer::Optimizers.Optimizer = Optimizers.SGD(0.01)`: a gradient-based optimization method (see [`Optimizers`](@ref)).

# Learning rule

This method implements the ReSuMe learning rule from eq. 2.10 in [1].
Assuming SGD and the default exponential kernel, the update rule is (eq. 29 in [2]):

```math
\\Delta w_i = \\gamma\\left(
    \\sum_{\\tilde{t}^f} \\left(a_R + \\sum_{t_i^g<\\tilde{t}^f} \\exp{\\left(-\\frac{\\tilde{t}^f-t_i^g}{\\tau_R}\\right)}\\right)
    - \\sum_{t^f} \\left(a_R + \\sum_{t_i^g<t^f} \\exp{\\left(-\\frac{t^f-t_i^g}{\\tau_R}\\right)}\\right)
\\right)
```

# References

[1] [Ponulak F. and Kasiński A. (2010). Supervised Learning in Spiking Neural Networks with ReSuMe: Sequence Learning, Classification, and Spike Shifting. Neural Computation, 22(2), 467-510](https://www.mitpressjournals.org/doi/abs/10.1162/neco.2009.11-08-901)

[2] [Florian R.V. (2012) The Chronotron: A Neuron That Learns to Fire Temporally Precise Spike Patterns. PLOS ONE, 7(8), e40233.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0040233)
"""
function train_corr!(m::Tempotron{N}, inp::SpikesInput{T1,N},
                     y₀::SpikesInput{T2,1}; aᵣ::Real = 0.0, τᵣ::Real = m.τₘ,
                     fᵣ::Function = t::Real -> (t < 0 ? 0.0 : exp(-t / τᵣ)),
                     optimizer = SGD(0.01)) where {T1<:Real,T2<:Real,N}

    # Get the current spike times
    spk_c = m(inp).spikes

    # Get the target spike times
    spk_t = y₀[1]

    # Kernel integral
    function λ(t::Real, x::Array{T,1})::Real where {T<:Real}
        ξ = filter(j -> j < t, x)
        return (aᵣ + (isempty(ξ) ? 0.0 : sum(j -> fᵣ(t - j), ξ)))
    end

    # Update weights
    Δ = [(isempty(spk_t) ? 0.0 : sum(t -> λ(t, inp[i]), spk_t)) -
         (isempty(spk_c) ? 0.0 : sum(t -> λ(t, inp[i]), spk_c)) for i = 1:N]
    m.w .+= optimizer(-Δ)
    return

end
