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

    # A small preturbation
    ϵ = eps(Float64)
    
    # The normalized weights
    W = m.w / m.K_norm

    # Get the ongoing sum of the unresetted voltage.
    PSPs = sort(GetPSPs(m, inp), by = x -> x.time)
    ΔVs = []
    V(t) = sum(x -> x(t), ΔVs)

    # Voltage derivative
    der(f) = x -> ForwardDiff.derivative(f, float(x))
    V̇ = der(V)

    sum_m = 0
    sum_s = 0
    t_max = 0
    V_max = -Inf
    spk = false
    for (j, ΔV, i) ∈ PSPs

        # Update the voltage function
        push!(ΔVs, ΔV)
        V̇ = der(V)

        # Analitically find the next local extermum
        sum_m += W[i]*exp(j/m.τₘ)
        sum_s += W[i]*exp(j/m.τₛ)
        ex_max = !(W[i] < 0 && V̇(j + ϵ) < 0)
        t_max_c = NextTmax(m, j, ex_max, sum_m, sum_s)
        if t_max_c ≡ nothing
            continue
        end
        V_max_c = V(t_max_c)

        # Save the maximal local extermum
        if V_max_c > V_max
            V_max = V_max_c
            t_max = t_max_c
        end

        # If a spike has occured, stop searching
        spk = V_max_c > m.θ
        if spk
            break
        end
    end

    # Filter out PSPs after the spike (shunting)
    filter!(x -> x.time ≤ t_max, PSPs)

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
