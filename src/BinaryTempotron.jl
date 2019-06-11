using ..Optimizers

"""
    GetBinaryTrainingPotential(m::Tempotron, inp[, T_max])
Get the tempotron `m`'s unresetted voltage for an input vector of spike trains
`inp` up to the first PSP ehich elicits a spike (if there is any). Return the
time of the maximal PSP `t_max`, the list of PSPs (up to the first spike) `PSPs`
and and `spk` indicating whether there was a spike.
"""
function GetBinaryTrainingPotential(m::Tempotron,
                                    inp::Array{Array{Tp, 1}, 1}) where Tp <: Any

    # # A small preturbation
    # ϵ = eps(Float64)

    # The normalized weights
    W = m.w / m.K_norm

    # Get the ongoing sum of the unresetted voltage.
    PSPs = sort(GetPSPs(m, inp), by = x -> x.time)
    # ΔVs = []
    # V(t) = sum(x -> x(t), ΔVs)
    # A temporary voltage function
    function Vt(t)
        emt, est = exp(-t/m.τₘ), exp(-t/m.τₛ)
        return (emt*sum_m - est*sum_s)
    end

    sum_m, sum_s = 0, 0
    t_max, V_max = 0, -Inf
    spk = false
    for P = 1:length(PSPs)
        (j, ΔV, i) = PSPs[P]

        # Get the next PSP's time
        next = (P < length(PSPs) ? PSPs[P + 1].time : j + 3m.τₘ)

        # # Update the voltage function
        # push!(ΔVs, ΔV)

        # Analitically find the next local extermum
        sum_m += W[i]*exp(j/m.τₘ)
        sum_s += W[i]*exp(j/m.τₛ)
        t_max_c, ~ = GetNextTmax(m, j, next, sum_m, sum_s)
        V_max_c = Vt(t_max_c)

        # Save the maximal local extermum
        if V_max_c > V_max
            t_max, V_max = t_max_c, V_max_c
        end

        # If a spike has occured, stop searching
        spk = V_max_c > m.θ
        if spk
            break
        end
    end

    # Filter out PSPs after the spike (shunting)
    PSPs_max = filter(x -> x.time ≤ t_max, PSPs)

    return t_max, PSPs_max, spk
end

"""
    Train!(m::Tempotron, inp, y₀::Bool[, optimizer = SGD(0.01)])
Train a tempotron `m` to fire or not (according to y₀) in response to an input
vector of spike trains `inp`. Optional parameters is the optimizer to be used
(default is `SGD` with learning rate `0.01`).
For further details see [Gütig, R., & Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing–based decisions. Nature neuroscience, 9(3), 420.](https://www.nature.com/articles/nn1643).
"""
function Train!(m::Tempotron,
                inp::Array{Array{Tp, 1}, 1},
                y₀::Bool;
                optimizer::Optimizer = SGD(0.01)) where Tp <: Any
    N, T = ValidateInput(m, inp, 0)

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
