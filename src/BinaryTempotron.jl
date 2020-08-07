"""
    GetBinaryTrainingPotential(m::Tempotron, inp)
Get the tempotron `m`'s unresetted voltage for an input vector of spike trains
`inp` up to the first PSP which elicits a spike (if there is any).

Returns a tuple containing:
- `t_max`: the time of the maximal PSP.
- `PSPs` the list of PSPs (up to the first spike).
- spk`: the tempotron's output (`true` if there is a spike).
"""
function GetBinaryTrainingPotential(m::Tempotron,
                                    inp::Array{Array{Tp, 1}, 1}) where Tp <: Real

    # The normalized weights
    W = m.w / m.K_norm

    # Get the ongoing sum of the unresetted voltage.
    PSPs = sort(GetPSPs(m, inp), by = x -> x.time)

    # A temporary voltage function
    function Vt(t::Real)::Real
        tt = t - ΔTϵ
        emt, est = exp(-tt/m.τₘ), exp(-tt/m.τₛ)
        return (emt*sum_m - est*sum_s)
    end

    # Numerical hack
    Nϵ, ΔTϵ = 0, 0

    sum_m, sum_s = 0, 0
    t_max, V_max = 0, -Inf
    spk = false
    for P = 1:length(PSPs)
        (j, ΔV, i) = PSPs[P]

        # Get the next PSP's time
        next = (P < length(PSPs) ? PSPs[P + 1].time : j + 3m.τₘ)

        # Numerical hack
        N_ϵ = floor(j/m.Tϵ)
        if N_ϵ > Nϵ
            ΔNϵ = N_ϵ - Nϵ
            ΔT_ϵ = ΔNϵ*m.Tϵ
            sum_m *= exp(-ΔT_ϵ/m.τₘ)
            sum_s *= exp(-ΔT_ϵ/m.τₛ)
            Nϵ = N_ϵ
            ΔTϵ = Nϵ*m.Tϵ
        end
        jt = j - ΔTϵ

        # Analitically find the next local extermum
        sum_m += W[i]*exp(jt/m.τₘ)
        sum_s += W[i]*exp(jt/m.τₛ)
        t_max_c = GetNextTmax(m, j, next, ΔTϵ, sum_m, sum_s)[1]
        V_max_c = Vt(t_max_c)

        # Save the maximal local extermum
        if V_max_c > V_max
            t_max, V_max = t_max_c, V_max_c
        end

        # If a spike has occured, stop searching
        spk = V_max_c > (m.θ - m.V₀)
        if spk
            break
        end
    end

    # Filter out PSPs after the spike (shunting)
    PSPs_max = filter(x -> x.time ≤ t_max, PSPs)

    return t_max, PSPs_max, spk
end

"""
    Train_∇!(m::Tempotron, inp, y₀::Bool; optimizer = Optimizers.SGD(0.01))
Train a tempotron `m` to fire or not (according to y₀) in response to an input
vector of spike trains `inp`.

# Optional arguments
- `optimizer::Optimizers.Optimizer = Optimizers.SGD(0.01)`: a gradient-based optimization method (see [`Optimizers`](@ref)).

# Learning rule
The gradient is (eq. 4 in [1]):
```math
-\\frac{\\mathrm{d}E{\\pm}}{\\mathrm{d} w_i} = \\pm\\sum_{t_i^j<t_{max}} K\\left(t_{max} - t_i^j\\right).
```
where the sign (±) of the update is determined by the teacher.

Assuming SGD, the update rule in case of an error is (eq. 2 in [1]):
```math
\\Delta w_i=-\\lambda\\frac{\\mathrm{d}E{\\pm}}{\\mathrm{d} w_i} = \\pm\\lambda\\sum_{t_i^j<t_{max}} K\\left(t_{max} - t_i^j\\right).
```

# References
[1] [Gütig, R., & Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing–based decisions. Nature neuroscience, 9(3), 420.](https://www.nature.com/articles/nn1643).
"""
function Train_∇!(m::Tempotron,
                  inp::Array{Array{Tp, 1}, 1},
                  y₀::Bool;
                  optimizer::Optimizer = SGD(0.01)) where Tp <: Real

    # Get the relevant PSPs, the maximal PSP and the current (boolean) output of
    # the tempotron
    t_max, PSPs, spk = GetBinaryTrainingPotential(m, inp)

    # If the tempotron's output equals the teacher, do not update the weights.
    ∇ = zeros(size(m.w))
    if spk == y₀
        optimizer(∇)
        return
    end

    # Update the weights
    for (j, ~, i) ∈ PSPs
        ∇[i] += m.K.(t_max - j)
    end
    m.w .+= (y₀ ? -1 : 1).*optimizer(∇)

end
