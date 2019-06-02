using ..Optimizers

"""
    GetBinaryTrainingPotential(m::Tempotron, inp[, T_max])
Get the tempotron `m`'s unresetted voltage for an input vector of spike trains
`inp` up to the first PSP ehich elicits a spike (if there is any). Return the
time of the maximal PSP `t_max`, the list of PSPs (up to the first spike) `PSPs`
and and `spk` indicating whether there was a spike.
"""
function GetBinaryTrainingPotential(m::Tempotron,
                                    inp::Array{Array{Tp, 1}, 1},
                                    T_max::Real = 0) where Tp <: Any
    N, T = ValidateInput(m, inp, T_max)

    # Get the comulative sums of the rnresetted voltage.
    PSPs = GetPSPs(m, inp, T)
    PSPs = sort(PSPs[:], by = x -> x[1])
    cumPSPs = [(PSPs[k][1], PSPs[k][3], t -> sum(x -> x[2](t), PSPs[1:k]))
               for k = 1:length(PSPs)]

    sum_m = 0
    sum_s = 0
    t_max = 0
    K_max = -Inf
    spk = false
    for (j, i, V) ∈ cumPSPs

        # Analitically find the next local extermum
        if m.w[i] < 0
            t_max_c = j
        else
            sum_m += m.w[i]*exp(j/m.τₘ)
            sum_s += m.w[i]*exp(j/m.τₛ)
            rem = sum_m/sum_s
            if rem <= 0
                continue
            end
            t_max_c = m.A*(m.log_α - log(rem))
            t_max_c = clamp(t_max_c, 0, T)
        end
        K_max_c = V(t_max_c)

        # Save the maximal local extermum
        if K_max_c > K_max
            K_max = K_max_c
            t_max = t_max_c
        end

        # If a spike has occured, stop searching
        spk = K_max_c > m.θ
        if spk
            break
        end
    end

    # Filter out PSPs after the spike (shunting)
    filter!(x -> x[1] ≤ t_max, PSPs)

    return t_max, PSPs, spk
end

"""
    Train!(m::Tempotron, inp, y₀::Bool[, optimizer = SGD(0.01)][, T_max])
Train a tempotron `m` to fire or not (according to y₀) in response to an input
vector of spike trains `inp`. Optional parameters are the optimizer to be used
(default is `SGD` with learning rate `0.01`) and maximal time `T`.
For further details see [Gütig, R., & Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing–based decisions. Nature neuroscience, 9(3), 420.](https://www.nature.com/articles/nn1643).
"""
function Train!(m::Tempotron,
                inp::Array{Array{Tp, 1}, 1},
                y₀::Bool;
                optimizer::Optimizer = SGD(0.01),
                T_max::Real = 0) where Tp <: Any
    N, T = ValidateInput(m, inp, T_max)

    # Get the relevant PSPs, the maximal PSP and the current (boolean) output of
    # the tempotron
    t_max, PSPs, spk = GetBinaryTrainingPotential(m, inp, T)

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
