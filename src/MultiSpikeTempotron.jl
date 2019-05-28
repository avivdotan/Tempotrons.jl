using Roots
using ForwardDiff
using ..Optimizers


# TODO: Remove
using Plots
import Base: Filesystem


function GetSpikes(m::Tempotron,
                    t_vec::Array{T1},
                    PSP_t::Array{T2},
                    η,
                    θ::Real,
                    T_max::Real) where {T1 <: Real,
                                        T2 <: Real}
    V = PSP_t
    spikes = []
    for j = 1:length(t_vec)
        if V[j] > θ
            t_spk = t_vec[j] + (t_vec[j - 1] - t_vec[j])*(θ - V[j])/(V[j - 1] - V[j])
            V -= θ.*η.(t_vec .- t_spk)
            push!(spikes, t_spk)
        end
    end
    return spikes
end

function GetCriticalThreshold(m::Tempotron,
                                PSPs,
                                t_vec::Array{T1},
                                PSP_t::Array{T2},
                                η,
                                y₀::Integer,
                                T_max::Real,
                                tol::Real = 1e-13) where {T1 <: Real,
                                                            T2 <: Real}
    θ₁ = m.V₀
    k₁ = typemax(Int)
    θ₂ = 10m.θ
    k₂ = 0
    spikes = []
    while k₁ ≠ y₀ || k₂ ≠ (y₀ - 1) || (θ₂ - θ₁) > 1e-2
        θ = (θ₁ + θ₂)/2
        spk = GetSpikes(m, t_vec, PSP_t, η, θ, T_max)
        k = length(spk)
        if k < y₀
            θ₂ = θ
            k₂ = k
            spikes = spk
        else
            θ₁ = θ
            k₁ = k
        end
        println("[", k₂, ", ", k₁, "], [", θ₁, ", ", θ₂, "]")
    end

    Ps = [(j, i, m.w[i]/m.K_norm, ΔV) for (j, ΔV, i) ∈ PSPs]
    Ns = [(j, 0, -θ₂, t -> -θ₂*η.(t .- j)) for j ∈ spikes]
    Vs = vcat(Ps, Ns)
    Vs = sort(Vs[:], by = x -> x[1])
    # dVs = [(Vs[k][1], Vs[k][2], Vs[k][3], t -> sum(x -> x[4](t), Vs[1:k]))
    #             for k = 1:length(Vs)]
    V(t) = sum(x -> x[4](t), Vs)
    V̇ = x -> ForwardDiff.derivative(V, float(x))
    # V̈ = x -> ForwardDiff.derivative(V̇, float(x))

    sum_m_t = 0
    sum_s_t = 0
    sum_e_t = 0
    sum_m = 0
    sum_s = 0
    sum_e = 0
    t_max = 0
    t_max_ex = true
    V_max = -Inf
    M_t = 0
    M = 0
    j_max = 0
    t_max_hist = []
    for (j, i, w, ~) ∈ Vs
        if i == 0
            sum_e_t += exp(j/m.τₘ)
            M_t += 1
        else
            sum_m_t += w*exp(j/m.τₘ)
            sum_s_t += w*exp(j/m.τₛ)
        end
        if i > 0 && w < 0
            t_max_c = j
            t_max_ex_c = false
        else
            rem = (sum_m_t - θ₂*sum_e_t)/sum_s_t
            if rem ≤ 0
                continue
            end
            t_max_c = m.A*(m.log_α - log(rem))
            t_max_c = clamp(t_max_c, 0, T_max)
            t_max_ex_c = true
        end
        push!(t_max_hist, t_max_c)
        V_max_c = V(t_max_c)
        if V_max_c > V_max && (!t_max_ex_c || abs(V̇(t_max_c)) < 1e-3) #&& V̈(t_max_c) < 0
            V_max = V_max_c
            t_max = t_max_c
            t_max_ex = t_max_ex_c
        end
        if j < t_max
            sum_m = sum_m_t
            sum_s = sum_s_t
            M = M_t
        end
    end

    Ps_max = filter(x -> x[1] < t_max, Ps)
    Vs_psp(t) = sum(x -> x[4](t), Ps_max)
    function v_max(θ)
        spk = GetSpikes(m, t_vec, PSP_t, η, θ, T_max)[1:M]
        if t_max_ex
            sum_e = isempty(spk) ? 0 : sum(exp.(spk./m.τₘ))
            t_max_θ = m.A*(m.log_α - log((sum_m - θ*sum_e)/sum_s))
        else
            t_max_θ = t_max
        end
        Vs_spk(t) = isempty(spk) ? 0 : sum(x -> -θ*η.(t .- x), spk)
        V(t) = Vs_psp(t) + Vs_spk(t)
        return V(t_max_θ)
    end

    f(x) = x - v_max(x)
    println("M = ", M, ", tₘₐₓ = ", t_max, ", Vₘₐₓ = ", V_max)
    println("vₘₐₓ(θ⃰)∈[", v_max(θ₁), ", ", v_max(θ₂), "], ",
    "f(θ⃰)∈[", f(θ₁), ", ", f(θ₂), "]")

    θ⃰ = 0
    try
        # θ⃰ = find_zero(f, (θ₁, θ₂), Roots.A42(), xatol = tol)
        # θ⃰ = find_zero(f, (θ₁ + θ₂)/2, Order1(), xatol = tol)
        θ⃰ = Roots.secant_method(f, (θ₁, θ₂), atol = tol)
    catch ex
        # TODO: Remove
        println("catch")
        function v(tt, θ)
            spk = GetSpikes(m, t_vec, PSP_t, η, θ, T_max)
            println("spikes: ", spk)
            Vs_spk(t) = isempty(spk) ? 0 : sum(x -> -θ*η.(t .- x), spk)
            V(t) = PSP_t + Vs_spk(t)
            return V.(tt)
        end
        V1 = v(t_vec, θ₁)
        V2 = v(t_vec, θ₂)
        pyplot(size = (700, 350))
        p = plot(t_vec, V1, linecolor = :blue, label = "V(t;θ₁)")
        plot!(t_vec, V2, linecolor = :red, label = "V(t;θ₂)")
        plot!(t_vec, m.θ*ones(length(t_vec)), linecolor = :black,
            linestyle = :dash, label = "")
        plot!(t_vec, θ₁*ones(length(t_vec)), linecolor = :blue,
            linestyle = :dash, label = "")
        plot!(t_vec, θ₂*ones(length(t_vec)), linecolor = :red,
            linestyle = :dash, label = "")
        for tm ∈ t_max_hist
            plot!([tm, tm], [m.V₀, max(m.θ, θ₂)*1.05],
                linecolor = :gray, linestyle = :dash, label = "")
        end
        plot!([t_max, t_max], [m.V₀, max(m.θ, θ₂)*1.05],
            linecolor = :purple, label = "")
        plot!([t_max, t_max], [m.V₀, max(m.θ, θ₂)*1.05],
            linecolor = :green, linestyle = :dash, label = "")
        yticks!([m.V₀, m.θ, θ₁, θ₂], ["V₀", "θ", "θ₁", "θ₂"])
        xticks!([0, t_max, T_max], ["0", "tₘₐₓ", string.(T_max)])
        ylabel!("V[mV]")
        xlabel!("t[ms]")
        plot(p)
        savefig("debug.png")
        i = 0
        filename = "DebugPlots\\debug" * string(i) * ".png"
        while isfile(filename)
            i += 1
            filename = "DebugPlots\\debug" * string(i) * ".png"
        end
        Filesystem.cp("debug.png", filename)
        throw(ex)
    end


    # f(x) = x - V(A*(log_α - log((sum_m - x*sum_e)/sum_s)))
    # d(f) = x -> ForwardDiff.derivative(f, float(x))
    # θ⃰ = find_zero((f, d(f)), θ₂, Roots.Newton(), xatol = tol)
    # θ⃰ = find_zero(f, (θ₁, θ₂), Roots.A42(), xatol = tol)

    println("θ⃰ = ", θ⃰)
    t⃰ = m.A*(m.log_α + log(sum_s/(sum_m - θ⃰*sum_e)))

    return t⃰, θ⃰, M
end

function GetGradient(m::Tempotron,
                     inp::Array{Array{T1, 1}, 1},
                     spk::Array{T2},
                     PSP,
                     η) where {T1 <: Any,
                               T2 <: Any}

    function Ση(tₓ)
        pre = filter(y -> y < tₓ, spk)
        return isempty(pre) ? 0 : sum(x -> η(tₓ - x), pre)
    end
    Σe = Ση.(spk)
    C = 1 .+ Σe
    V₀ = PSP.(spk)
    function ΣK(tₓ)
        sum_K = zeros(length(inp))
        for i = 1:length(inp)
            pre = filter(x -> x < tₓ, inp[i])
            sum_K[i] = isempty(pre) ? 0 : sum(K.(m, tₓ .- pre))
        end
        return sum_K
    end
    ∂V∂w = ΣK.(spk)./C

    ∂V = zeros(length(spk), length(spk))
    for k = 1:length(spk)
        a = -V₀[k]/(m.τₘ*C[k]^2)
        for j = 1:(k - 1)
            ∂V[k, j] = -a*η(spk[k] - spk[j])
        end
    end

    d(f) = x -> ForwardDiff.derivative(f, float(x))
    dV₀ = d(PSP).(spk)
    V̇ = dV₀./C + V₀.*Σe./(m.τₘ.*C.^2)

    A = []
    B = []
    for k = 1:length(spk)
        Aₖ = isempty(A) ? 1 : (1 - sum(j -> (A[j]/V̇[j])*∂V[k, j], 1:(k-1)))
        Bₖ = isempty(B) ? 0 : -sum(j -> (B[j]/V̇[j])*∂V[k, j], 1:(k-1))
        Bₖ = Bₖ .- ∂V∂w[k]
        push!(A, Aₖ)
        push!(B, Bₖ)
    end
    ∂θ⃰ = -B[end]/A[end]
    return ∂θ⃰
end


function Train!(m::Tempotron,
                inp::Array{Array{Tp, 1}, 1},
                y₀::Integer;
                optimizer::Optimizer,
                T_max::Real = 0,
                dt::Real    = 0.1) where Tp <: Any
    N, T = ValidateInput(m, inp, T_max)

    PSPs = GetPSPs(m, inp, T_max)
    PSP(t) = sum(x -> x[2](t), PSPs)
    t_vec = collect(0:dt:T_max)
    PSP_t = PSP.(t_vec)

    η(t) = t < 0 ? 0 : exp(-t/m.τₘ)

    k = length(GetSpikes(m, t_vec, PSP_t, η, m.θ, T_max))
    println("y₀ = ", y₀, "; y = ", k)
    if k == y₀
        ∇ = zeros(size(m.w))
        optimizer(∇)
        return
    end

    # λ = m.λ * (y₀ > k ? 1 : -1)
    o = y₀ > k ? k + 1 : k
    t⃰, θ⃰, M = GetCriticalThreshold(m, PSPs, t_vec, PSP_t, η, o, T_max)
    spk = GetSpikes(m, t_vec, PSP_t, η, θ⃰, T_max)[1:M]
    push!(spk, t⃰)
    ∇θ⃰ = GetGradient(m, inp, spk, PSP, η)
    # m.w .+= λ.*∇θ⃰
    m.w += (y₀ > k ? -1 : 1).*optimizer(∇θ⃰)

end
