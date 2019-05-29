module Optimizers

export Optimizer, SGD, RMSprop, Adam, reset!

abstract type Optimizer
end

mutable struct SGD <: Optimizer
    η::Real
    α::Real
    ∇₋₁
end
function SGD(lr::Real; momentum::Real = 0)
    if lr ≤ 0
        error("Learning rate must be positive. ")
    end
    if momentum < 0 || momentum > 1
        error("Momentum coefficient must be in [0, 1]. ")
    end
    return SGD(lr, momentum, 0)
end
function (opt::SGD)(∇)
    if opt.α == 0
        Δ = -opt.η.*∇
    else
        opt.∇₋₁ = ∇ .+ opt.α.*opt.∇₋₁
        Δ = -opt.η.*opt.∇₋₁
    end
    return Δ
end
function reset!(opt::SGD)
    opt.∇₋₁ = 0
end

mutable struct RMSprop <: Optimizer
    η::Real
    ρ::Real
    ϵ::Real
    Σ∇²::Real
end
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
function (opt::RMSprop)(∇)
    ∇ᵥ = ∇[:]
    opt.Σ∇² = (1 - opt.ρ)*(∇ᵥ'*∇ᵥ) + opt.ρ*opt.Σ∇²
    Δ = -opt.η.*∇./√(opt.Σ∇² + opt.ϵ)
    return Δ
end
function reset!(opt::RMSprop)
    opt.Σ∇² = 0
end

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
function (opt::Adam)(∇)
    ∇ᵥ = ∇[:]
    opt.m = (1 - opt.β₁).*∇ .+ opt.β₁.*opt.m
    opt.v = (1 - opt.β₂)*(∇ᵥ'*∇ᵥ) + opt.β₂*opt.v
    opt.Zₘ *= opt.β₁
    opt.Zᵥ *= opt.β₂
    m̂ = opt.m./(1 - opt.Zₘ)
    v̂ = opt.v/(1 - opt.Zᵥ)
    Δ = -opt.η.*m̂./(√v̂ + opt.ϵ)
    return Δ
end
function reset!(opt::Adam)
    opt.m = 0
    opt.v = 0
    opt.Zₘ = 1
    opt.Zᵥ = 1
end

end
