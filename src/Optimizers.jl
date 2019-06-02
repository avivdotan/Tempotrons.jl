"""
Gradient-based optimizers for the Tempotrons.jl package.
"""
module Optimizers

export Optimizer, SGD, RMSprop, Adam, reset!

"""
A general gradient-based optimizer
All optimizers should implement a function call recieving a gradient and
returning a weight update, and a reset! function resetting the inner
aggregated variables.
See also: [`SGD`](@ref), [`RMSprop`](@ref), [`Adam`](@ref), [`reset!`](@ref)
"""
abstract type Optimizer
end

"""
Stochastic Gradient-Descent (with momentum)
"""
mutable struct SGD <: Optimizer
    η::Real
    α::Real
    ∇₋₁
end

"""
    SGD(lr, [momentum = 0])
Creata a Stochastic Gradient-Descent optimizer with learning rate `lr` and
momentum coefficent `momentum` (default `0`).
"""
function SGD(lr::Real; momentum::Real = 0)
    if lr ≤ 0
        error("Learning rate must be positive. ")
    end
    if momentum < 0 || momentum > 1
        error("Momentum coefficient must be in [0, 1]. ")
    end
    return SGD(lr, momentum, 0)
end

"""
    (opt::Optimizer)(∇)
Calculate the weight change using the current gradient `∇`.
"""
function (opt::SGD)(∇)
    if opt.α == 0
        Δ = -opt.η.*∇
    else
        opt.∇₋₁ = ∇ .+ opt.α.*opt.∇₋₁
        Δ = -opt.η.*opt.∇₋₁
    end
    return Δ
end

"""
    reset!(opt::Optimizer)
Resets the inner aggregated variables of the optimizer `opt`.
"""
function reset!(opt::SGD)
    opt.∇₋₁ = 0
end

"""
RMSprop
[rmsprop: Divide the gradient by a running average of its recent magnitude](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
"""
mutable struct RMSprop <: Optimizer
    η::Real
    ρ::Real
    ϵ::Real
    Σ∇²::Real
end

"""
    RMSprop(lr[, [ρ = 0.9][, ϵ = eps(Float32)])
Creata a RMSprop optimizer with learning rate `lr`.
[rmsprop: Divide the gradient by a running average of its recent magnitude](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
"""
function RMSprop(lr::Real; ρ::Real = 0.9, ϵ = eps(Float32))
    if lr ≤ 0
        error("Learning rate must be positive. ")
    end
    if ρ < 0 || ρ ≥ 1
        error("ρ must be in [0, 1). ")
    end
    if ϵ ≤ 0
        error("ϵ must be positive. ")
    end
    return RMSprop(lr, ρ, ϵ, 0)
end

"""
    (opt::Optimizer)(∇)
Calculate the weight change using the current gradient `∇`.
"""
function (opt::RMSprop)(∇)
    ∇ᵥ = ∇[:]
    opt.Σ∇² = (1 - opt.ρ)*(∇ᵥ'*∇ᵥ) + opt.ρ*opt.Σ∇²
    Δ = -opt.η.*∇./√(opt.Σ∇² + opt.ϵ)
    return Δ
end

"""
    reset!(opt::Optimizer)
Resets the inner aggregated variables of the optimizer `opt`.
"""
function reset!(opt::RMSprop)
    opt.Σ∇² = 0
end

"""
Adam
[Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
"""
mutable struct Adam <: Optimizer
    η::Real
    β₁::Real
    β₂::Real
    ϵ::Real
    m
    v::Real
    Zₘ::Real
    Zᵥ::Real
end

"""
    Adam(lr[, β₁ = 0.9][, β₂ = 0.999][, ϵ = eps(Float32)])
Creata an Adam optimizer with learning rate `lr`.
[Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
"""
function Adam(lr::Real; β₁::Real = 0.9, β₂::Real = 0.999, ϵ = eps(Float32))
    if lr ≤ 0
        error("Learning rate must be positive. ")
    end
    if β₁ < 0 || β₁ ≥ 1
        error("β₁ must be in [0, 1). ")
    end
    if β₂ < 0 || β₂ ≥ 1
        error("β₂ must be in [0, 1). ")
    end
    if ϵ ≤ 0
        error("ϵ must be positive. ")
    end
    return Adam(lr, β₁, β₂, ϵ, 0, 0, 1, 1)
end

"""
    (opt::Optimizer)(∇)
Calculate the weight change using the current gradient `∇`.
"""
function (opt::Adam)(∇)
    ∇ᵥ = ∇[:]
    opt.m = (1 - opt.β₁).*∇ .+ opt.β₁.*opt.m
    opt.v = (1 - opt.β₂)*(∇ᵥ'*∇ᵥ) + opt.β₂*opt.v
    opt.Zₘ .*= opt.β₁
    opt.Zᵥ .*= opt.β₂
    m̂ = opt.m./(1 - opt.Zₘ)
    v̂ = opt.v/(1 - opt.Zᵥ)
    Δ = -opt.η.*m̂./(√v̂ + opt.ϵ)
    return Δ
end

"""
    reset!(opt::Optimizer)
Resets the inner aggregated variables of the optimizer `opt`.
"""
function reset!(opt::Adam)
    opt.m = 0
    opt.v = 0
    opt.Zₘ = 1
    opt.Zᵥ = 1
end

end
