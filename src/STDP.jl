"""
    stdp_update(m::Tempotron, inp::SpikesInput, spk; τ = m.τₘ, α = 1.05, 
        μ = 0.05, K = Δt -> exp(-|Δt|/τ), f₋ = w -> (1 - w)^μ, 
        f₊ = w -> (1 - w)^μ))

The unsupervised STDP rule, calculated for an output neurn neuron `m`, input
spike trains `inp` and an output spike train `spk`.

# Optional arguments

  - `τ::Real = m.τₘ` the default kernel's time constant.
  - `α::Real = 1.05` a anti-Hebbian term coefficient.
  - `μ`::Real = 0.05` a transition between an additive (μ = 0) and multiplicative (μ = 1) rules
  - `K::Function = Δt -> exp(-|Δt|/τ)` the temporal kernel.
  - `f₋::Function = w -> αw^μ` the anti-Hebbian term.
  - `f₊::Function = w -> (1 - w)^μ` the Hebbian term.

# Returns

  - The weights update vector (up to multiplication by the learning rate)

# Learning rule

This method implements the STDP learning rule.
Assuming the default kernels, the update rule is (eq. 1 in [1]):

```math
\\Delta w = \\cases{\\begin{array}{rl}
-\\lambda f_-\\left(w\\right)\\times K\\left(\\Delta t\\right) &, \\Delta t \\le 0 \\\\ 
\\lambda f_+\\left(w\\right)\\times K\\left(\\Delta t\\right) &, \\Delta t \\gt 0
\\end{array}}
```

# References

[1] [Gütig, R., Aharonov, R., Rotter, S., & Sompolinsky, H. (2003). Learning input correlations through nonlinear temporally asymmetric Hebbian plasticity. Journal of Neuroscience, 23(9), 3697-3714.](https://doi.org/10.1523/JNEUROSCI.23-09-03697.2003)
"""
function stdp_update(m::Tempotron{N},
                     inp::SpikesInput{T1,N},
                     spk::Array{T2,1};
                     τ::Real = m.τₘ,
                     α::Real = 1.05,
                     μ::Real = 0.02,
                     K::Function = Δt::Real -> Δt > 5τ ? 0.0 : exp(-abs(Δt) / τ),
                     f₋::Function = w::Real -> α * w^μ,
                     f₊::Function = w::Real -> (1 - w)^μ) where {N,T1<:Real,
                                                                 T2<:Real}
    if length(spk) == 0
        return zeros(size(m.w))
    end

    f₋_vec = f₋.(m.w)
    f₊_vec = f₊.(m.w)

    return map(1:N) do i
        if length(inp[i]) == 0
            return 0.0
        end

        # Define the weight kernel
        f(Δt::Real)::Real = Δt > 0 ? f₊_vec[i] : -f₋_vec[i]

        # Get time differences
        Δt = spk .- inp[i]'

        # Apply the time kernel
        Ks = K.(Δt)

        # Remove zero updates (where the time kernel values are negligible)
        ind = Ks .≉ 0.0
        Δt = Δt[ind]
        Ks = Ks[ind]

        # Apply the weight kernel
        fs = f.(Δt)

        # Get all updates (for the current synapse)
        return sum(fs .* Ks)
    end
end

"""
    train_corr!(m::Tempotron, inp::SpikesInput; τ = m.τₘ, α = 1.05, μ = 0.05, 
        K = Δt -> exp(-|Δt|/τ), f₋ = w -> (1 - w)^μ, f₊ = w -> (1 - w)^μ),
        w_lims = (0, 1), optimizer = SGD(0.01))

Trains a neuron `m` using the unsupervised STDP rule.

# Optional arguments

  - `τ::Real = m.τₘ` the default kernel's time constant.
  - `α::Real = 1.05` a anti-Hebbian term coefficient.
  - `μ`::Real = 0.05` a transition between an additive (μ = 0) and multiplicative (μ = 1) rules
  - `K::Function = Δt -> exp(-|Δt|/τ)` the temporal kernel.
  - `f₋::Function = w -> αw^μ` the anti-Hebbian term.
  - `f₊::Function = w -> (1 - w)^μ` the Hebbian term.
  - `w_lims::Tuple = (0, 1)` weights minimal and maximal values.
  - `optimizer::Optimizers.Optimizer = Optimizers.SGD(0.01)`: a gradient-based optimization method (see [`Optimizers`](@ref)).

# Learning rule

This method implements the STDP learning rule.
Assuming SGD and the default kernels, the update rule is (eq. 1 in [1]):

```math
\\Delta w = \\cases{\\begin{array}{rl}
-\\lambda f_-\\left(w\\right)\\times K\\left(\\Delta t\\right) &, \\Delta t \\le 0 \\\\ 
\\lambda f_+\\left(w\\right)\\times K\\left(\\Delta t\\right) &, \\Delta t \\gt 0
\\end{array}}
```

# References

[1] [Gütig, R., Aharonov, R., Rotter, S., & Sompolinsky, H. (2003). Learning input correlations through nonlinear temporally asymmetric Hebbian plasticity. Journal of Neuroscience, 23(9), 3697-3714.](https://doi.org/10.1523/JNEUROSCI.23-09-03697.2003)
"""
function train_corr!(m::Tempotron{N},
                     inp::SpikesInput{T1,N};
                     w_lims::Tuple{Real,Real} = (0, 1),
                     optimizer = SGD(0.001),
                     kwargs...) where {T1<:Real,N}

    # Get the current spike times
    spk = Array{Real,1}(m(inp).spikes)

    # Get the STDP updates
    Δ = stdp_update(m, inp, spk; kwargs...)

    # Update the weights
    m.w .+= optimizer(-Δ)
    clamp!(m.w, w_lims...)
    return
end
