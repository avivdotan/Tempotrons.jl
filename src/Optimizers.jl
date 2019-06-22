"""
Gradient-based optimizers for the Tempotrons.jl package.
"""
module Optimizers

export Optimizer, reset!
export SGD, RMSprop, Adagrad, Adadelta, Adam, AdaMax, Nadam

"""
A general gradient-based optimizer
All optimizers should implement a function call recieving a gradient and
returning a weight update, and a reset! function resetting the inner
aggregated variables.
See also: [`SGD`](@ref), [`RMSprop`](@ref), [`Adam`](@ref), [`reset!`](@ref)
"""
abstract type Optimizer
end

#------------------------------------------------------------------------------#
#   SGD (+ momentum)
#------------------------------------------------------------------------------#

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
function (opt::SGD)(∇::Array{Tp, N})::Array{Tp, N} where {Tp <: Real, N}
    if opt.α == 0
        Δ = -opt.η.*∇
    else
        opt.∇₋₁ = @. (1 - opt.α)*∇ + opt.α*opt.∇₋₁
        Δ       = @. -opt.η*opt.∇₋₁
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

#------------------------------------------------------------------------------#
#   RMSprop
#------------------------------------------------------------------------------#

"""
RMSprop
[rmsprop: Divide the gradient by a running average of its recent magnitude](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
"""
mutable struct RMSprop <: Optimizer
    η::Real
    ρ::Real
    ϵ::Real
    Σ∇²
end

"""
    RMSprop(lr[, [ρ = 0.9][, ϵ = eps(Float32)])
Creata a RMSprop optimizer with learning rate `lr`.
[rmsprop: Divide the gradient by a running average of its recent magnitude](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
"""
function RMSprop(lr::Real; ρ::Real = 0.9, ϵ::Real = eps(Float32))
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

#=
    (opt::Optimizer)(∇)
Calculate the weight change using the current gradient `∇`.
=#
function (opt::RMSprop)(∇::Array{Tp, N})::Array{Tp, N} where {Tp <: Real, N}
    opt.Σ∇² = @. (1 - opt.ρ)*∇^2 + opt.ρ*opt.Σ∇²
    Δ       = @. -opt.η*∇/√(opt.Σ∇² + opt.ϵ)
    return Δ
end

"""
    reset!(opt::Optimizer)
Resets the inner aggregated variables of the optimizer `opt`.
"""
function reset!(opt::RMSprop)
    opt.Σ∇² = 0
end

#------------------------------------------------------------------------------#
#   Adagrad
#------------------------------------------------------------------------------#

"""
Adagrad
[Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
"""
mutable struct Adagrad <: Optimizer
    η::Real
    ϵ::Real
    Σ∇²
end

"""
    Adagrad(lr[, ϵ = eps(Float32)])
Creata an Adagrad optimizer with learning rate `lr`.
[Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
"""
function Adagrad(lr::Real; ϵ::Real = eps(Float32))
    if lr ≤ 0
        error("Learning rate must be positive. ")
    end
    if ϵ ≤ 0
        error("ϵ must be positive. ")
    end
    return Adagrad(lr, ϵ, 0)
end

#=
    (opt::Optimizer)(∇)
Calculate the weight change using the current gradient `∇`.
=#
function (opt::Adagrad)(∇::Array{Tp, N})::Array{Tp, N} where {Tp <: Real, N}
    opt.Σ∇² = @. ∇^2 + opt.Σ∇²
    Δ       = @. -opt.η*∇/√(opt.Σ∇² + opt.ϵ)
    return Δ
end

"""
    reset!(opt::Optimizer)
Resets the inner aggregated variables of the optimizer `opt`.
"""
function reset!(opt::Adagrad)
    opt.Σ∇² = 0
end

#------------------------------------------------------------------------------#
#   Adadelta
#------------------------------------------------------------------------------#

"""
Adadelta
[Adadelta - an adaptive learning rate method](https://arxiv.org/abs/1212.5701)
"""
mutable struct Adadelta <: Optimizer
    ρ::Real
    ϵ::Real
    Σ∇²
    ΣΔ²
end

"""
    Adadelta([[ρ = 0.95][, ϵ = eps(Float32)])
Creata an Adadelta optimizer.
[Adadelta - an adaptive learning rate method](https://arxiv.org/abs/1212.5701)
"""
function Adadelta(; ρ::Real = 0.95, ϵ::Real = eps(Float32))
    if ρ < 0 || ρ ≥ 1
        error("ρ must be in [0, 1). ")
    end
    if ϵ ≤ 0
        error("ϵ must be positive. ")
    end
    return Adadelta(ρ, ϵ, 0, 0)
end

#=
    (opt::Optimizer)(∇)
Calculate the weight change using the current gradient `∇`.
=#
function (opt::Adadelta)(∇::Array{Tp, N})::Array{Tp, N} where {Tp <: Real, N}
    opt.Σ∇² = @. (1 - opt.ρ)*∇^2 + opt.ρ*opt.Σ∇²
    Δ       = @. -(√(opt.ΣΔ² + opt.ϵ)/√(opt.Σ∇² + opt.ϵ))*∇
    opt.ΣΔ² = @. (1 - opt.ρ)*Δ^2 + opt.ρ*opt.ΣΔ²
    return Δ
end

"""
    reset!(opt::Optimizer)
Resets the inner aggregated variables of the optimizer `opt`.
"""
function reset!(opt::Adadelta)
    opt.Σ∇² = 0
    opt.ΣΔ² = 0
end

#------------------------------------------------------------------------------#
#   Adam
#------------------------------------------------------------------------------#

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
    v
    Zₘ::Real
    Zᵥ::Real
end

"""
    Adam(lr[, β₁ = 0.9][, β₂ = 0.999][, ϵ = eps(Float32)])
Creata an Adam optimizer with learning rate `lr`.
[Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
"""
function Adam(lr::Real; β₁::Real = 0.9, β₂::Real = 0.999, ϵ::Real = eps(Float32))
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

#=
    (opt::Optimizer)(∇)
Calculate the weight change using the current gradient `∇`.
=#
function (opt::Adam)(∇::Array{Tp, N})::Array{Tp, N} where {Tp <: Real, N}
    opt.m   = @. (1 - opt.β₁)*∇ .+ opt.β₁*opt.m
    opt.v   = @. (1 - opt.β₂)*∇^2 + opt.β₂*opt.v
    opt.Zₘ *= opt.β₁
    opt.Zᵥ *= opt.β₂
    m̂       = @. opt.m/(1 - opt.Zₘ)
    v̂       = @. opt.v/(1 - opt.Zᵥ)
    Δ       = @. -opt.η*m̂/(√v̂ + opt.ϵ)
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

#------------------------------------------------------------------------------#
#   AdaMax
#------------------------------------------------------------------------------#

"""
AdaMax
[Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
"""
mutable struct AdaMax <: Optimizer
    η::Real
    β₁::Real
    β₂::Real
    ϵ::Real
    m
    u
    Zₘ::Real
end

"""
    AdaMax(lr[, β₁ = 0.9][, β₂ = 0.999])
Creata an AdaMax optimizer with learning rate `lr`.
[Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
"""
function AdaMax(lr::Real; β₁::Real = 0.9, β₂::Real = 0.999, ϵ::Real = eps(Float32))
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
    return AdaMax(lr, β₁, β₂, ϵ, 0, 0, 1)
end

#=
    (opt::Optimizer)(∇)
Calculate the weight change using the current gradient `∇`.
=#
function (opt::AdaMax)(∇::Array{Tp, N})::Array{Tp, N} where {Tp <: Real, N}
    opt.m   = @. (1 - opt.β₁)*∇ .+ opt.β₁*opt.m
    opt.u   = @. max(opt.β₂*opt.u, abs(∇))
    opt.Zₘ *= opt.β₁
    m̂       = @. opt.m/(1 - opt.Zₘ)
    Δ       = @. -opt.η*m̂/(opt.u + opt.ϵ)
    return Δ
end

"""
    reset!(opt::Optimizer)
Resets the inner aggregated variables of the optimizer `opt`.
"""
function reset!(opt::AdaMax)
    opt.m = 0
    opt.u = 0
    opt.Zₘ = 1
end

#------------------------------------------------------------------------------#
#   Nadam
#------------------------------------------------------------------------------#

"""
Nadam
[Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
"""
mutable struct Nadam <: Optimizer
    η::Real
    β₁::Real
    β₂::Real
    ϵ::Real
    m
    v
    Zₘ::Real
    Zᵥ::Real
end

"""
    Nadam(lr[, β₁ = 0.9][, β₂ = 0.999][, ϵ = eps(Float32)])
Creata a Nadam optimizer with learning rate `lr`.
[Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
"""
function Nadam(lr::Real; β₁::Real = 0.9, β₂::Real = 0.999, ϵ::Real = eps(Float32))
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
    return Nadam(lr, β₁, β₂, ϵ, 0, 0, 1, 1)
end

#=
    (opt::Optimizer)(∇)
Calculate the weight change using the current gradient `∇`.
=#
function (opt::Nadam)(∇::Array{Tp, N})::Array{Tp, N} where {Tp <: Real, N}
    opt.m   = @. (1 - opt.β₁)*∇ .+ opt.β₁*opt.m
    opt.v   = @. (1 - opt.β₂)*∇^2 + opt.β₂*opt.v
    opt.Zₘ *= opt.β₁
    opt.Zᵥ *= opt.β₂
    m̂       = @. opt.m/(1 - opt.Zₘ)
    v̂       = @. opt.v/(1 - opt.Zᵥ)
    Δ       = @. -opt.η*(opt.β₁*m̂ + ((1 - opt.β₁)/(1 - opt.Zₘ))*∇)/(√v̂ + opt.ϵ)
    return Δ
end

"""
    reset!(opt::Optimizer)
Resets the inner aggregated variables of the optimizer `opt`.
"""
function reset!(opt::Nadam)
    opt.m = 0
    opt.v = 0
    opt.Zₘ = 1
    opt.Zᵥ = 1
end

end
