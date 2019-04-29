function GetBinaryTrainingPotential(m::Tempotron,
                                inp::Array{Array{Tp, 1}, 1},
                                T_max::Real = 0) where Tp <: Any
    N, T = ValidateInput(m, inp, T_max)

    A = m.τₘ * m.τₛ / (m.τₘ - m.τₛ)
    log_α = log(m.τₘ/m.τₛ)

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
        sum_m += m.w[i]*exp(j/m.τₘ)
        sum_s += m.w[i]*exp(j/m.τₛ)
        rem = sum_m/sum_s
        if rem <= 0
            continue
        end
        t_max_c = A*(log_α - log(rem))
        t_max_c = clamp(t_max_c, 0, T)
        K_max_c = V(t_max_c)
        if K_max_c > K_max
            K_max = K_max_c
            t_max = t_max_c
        end
        spk = K_max_c > m.θ
        if spk
            break
        end
    end
    return t_max, PSPs, spk
end

function Train!(m::Tempotron,
                inp::Array{Array{Tp, 1}, 1},
                y₀::Bool;
                T_max::Real = 0) where Tp <: Any
    N, T = ValidateInput(m, inp, T_max)

    λ = m.λ * (y₀ ? 1 : -1)
    t_max, PSPs, spk = GetBinaryTrainingPotential(m, inp, T)
    if spk == y₀
        return
    end

    for (j, ~, i) ∈ PSPs
        if j >= t_max
            break
        end
        m.w[i] += λ.*K.(m, t_max - j)
    end
end
