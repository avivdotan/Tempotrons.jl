using Roots
using ForwardDiff
using ..Optimizers


# TODO: Remove
using Plots
import Base: Filesystem


function GetSpikes(m::Tempotron,
                    PSPs,
                    PSP,
                    θ::Real,
                    T_max::Real) where {T1 <: Real,
                                        T2 <: Real}
    ϵ = eps(Float64)
    W = m.w / m.K_norm
    Ps = [(j, ΔV, i, j) for (j, ΔV, i) ∈ PSPs]
    for a = 1:length(Ps)
        if m.w[Ps[a][3]] < 0
            Ps[a] = (Ps[a][1:3]..., a == 1 ? 0 : Ps[a - 1][4])
        end
    end
    sum_m = 0
    sum_s = 0
    sum_e = 0
    last_spk = (0, )
    spikes = []
    Vspk(t) = (isempty(spikes) ? 0 : sum(x -> x[2](t), spikes))
    V(t) = PSP(t) + (isempty(spikes) ? 0 : sum(x -> x[2](t), spikes))
    der(f) = x -> ForwardDiff.derivative(f, float(x))
    dVpsp = der(PSP)
    dVspk = der(Vspk)
    V̇(t) = dVpsp(t) + dVspk(t)

    for P = 1:length(Ps)
        (j, ~, i, k) = Ps[P]
        # if j < last_spk[1]
        #     last_spk = (last_spk[1:3]..., k)
        #     spikes[end] = last_spk
        # end
        sum_m += W[i]*exp(j/m.τₘ)
        sum_s += W[i]*exp(j/m.τₛ)
        if m.w[i] < 0 && V̇(j + ϵ) < 0
            t_max_j = j
        else
            rem = (sum_m - θ*sum_e)/sum_s
            if rem ≤ 0
                continue
            end
            t_max_j = m.A*(m.log_α - log(rem))
            # t_max_j = clamp(t_max_j, 0, T_max)
        end

        s = k
        while V(t_max_j) > θ
            t_spk = 0
            try
                t_spk = find_zero(t -> V(t) - θ, (s, t_max_j), Roots.A42())
            catch ex
                Vs, V_t_max = V(s), V(t_max_j)
                spsps = [sign(m.w[x[3]])*x[1] for x ∈ PSPs]
                spks = [x[1] for x ∈ spikes]
                @debug ("θ = $θ: [$s, $t_max_j] -> [$Vs, $V_t_max]\n" *
                "PSPs: $spsps" *
                "spikes (partial): $spks")
                throw(ex)
            end
            ΔV(t) = -θ*m.η.(t .- t_spk)
            # last_spk = (t_spk, ΔV, 0, k)
            # push!(spikes, last_spk)
            Q = P + 1
            while Q ≤ length(Ps) && Ps[Q][1] < t_spk
                Q += 1
            end
            c = Ps[Q - 1][4]
            push!(spikes, (t_spk, ΔV, 0, c))
            sum_e += exp(t_spk/m.τₘ)
            dVspk = der(Vspk)

            if m.w[i] < 0 && V̇(j + ϵ) < 0
                t_max_j = j
            else
                rem = (sum_m - θ*sum_e)/sum_s
                if rem ≤ 0
                    break
                end
                t_max_j = m.A*(m.log_α - log(rem))
                # t_max_j = clamp(t_max_j, 0, T_max)
            end
            # if P < length(PSPs) && t_max_j > PSPs[P + 1][1]
            #     break
            # end
            s = t_spk + ϵ
        end
    end
    return spikes
end

function GetVmax(m::Tempotron,
                    PSPs,
                    spikes,
                    θ::Real,
                    T_max::Real) where {T1 <: Real,
                                        T2 <: Real}

    ϵ = eps(Float64)
    W = m.w / m.K_norm
    Ps = [(j, ΔV, i, j) for (j, ΔV, i) ∈ PSPs]
    for a = 1:length(Ps)
        if m.w[Ps[a][3]] < 0
            Ps[a] = (Ps[a][1:3]..., a == 1 ? 0 : Ps[a - 1][4])
        end
    end
    Vs = vcat(Ps, spikes)
    Vs = sort(Vs[:], by = x -> x[1])
    V(t) = sum(x -> x[2](t), Vs)
    V̇ = x -> ForwardDiff.derivative(V, float(x))
    sum_m = 0
    sum_s = 0
    sum_e = 0
    j_prev = 0
    v_max = (-Inf, )
    t_max_hist = []
    for (j, ~, i, k) ∈ Vs
        if i == 0
            sum_e += exp(j/m.τₘ)
        else
            sum_m += W[i]*exp(j/m.τₘ)
            sum_s += W[i]*exp(j/m.τₛ)
        end
        ex_max = !(i > 0 && m.w[i] < 0 && V̇(j + ϵ) < 0)
        if !ex_max
            t_max_j = j
        else
            rem = (sum_m - θ*sum_e)/sum_s
            if rem ≤ 0
                continue
            end
            t_max_j = m.A*(m.log_α - log(rem))
            # t_max_j = clamp(t_max_j, 0, T_max)
        end
        push!(t_max_hist, t_max_j)
        v_max_j = V(t_max_j)
        # println("[", k, ", ", t_max_j, ", ", v_max_j, ", ", v_max[1], "]")
        if v_max_j > v_max[1] &&
            (ex_max && abs(V̇(t_max_j)) < 1e-3 ||
            !ex_max && V̇(t_max_j + ϵ) < 0)
            v_max = (v_max_j, t_max_j, k, ex_max, sum_m, sum_s)
        end
    end
    return v_max, t_max_hist
end

function VmaxLinked2Spike(spikes1,
                            spikes2,
                            v_max2)
    spikes1_c = [x[4] for x ∈ spikes1 if x[4] ≤ v_max2[3]]
    spikes2_c = [x[4] for x ∈ spikes2 if x[4] ≤ v_max2[3]]
    # println(spikes1_c, spikes2_c, v_max2[3])
    push!(spikes2_c, v_max2[3])
    # TODO: Better identifiers
    # return (sum(abs.(sort(spikes1_c) .- sort(spikes2_c))) < 1e-7length(spikes1_c))
    return (sort(spikes1_c) == sort(spikes2_c))
end

function GetCriticalThreshold(m::Tempotron,
                                PSPs,
                                PSP,
                                y₀::Integer,
                                T_max::Real,
                                tol::Real = 1e-13) where {T1 <: Real,
                                                            T2 <: Real}

    θ₁ = m.V₀
    k₁ = typemax(Int)
    θ₂ = 10m.θ
    k₂ = 0
    spikes1 = []
    spikes2 = []
    v_max2 = (nothing, nothing, -Inf)
    t_max_hist = []
    while k₁ ≠ y₀ || k₂ ≠ (y₀ - 1) || !VmaxLinked2Spike(spikes1, spikes2, v_max2)
        θ = (θ₁ + θ₂)/2
        spk = GetSpikes(m, PSPs, PSP, θ, T_max)
        k = length(spk)
        if k < y₀
            θ₂ = θ
            k₂ = k
            spikes2 = spk
            v_max2, t_max_hist = GetVmax(m, PSPs, spikes2, θ₂, T_max)
        else
            θ₁ = θ
            k₁ = k
            spikes1 = spk
        end
        @debug "[$k₂, $k₁], [$θ₁, $θ₂]"
        # TODO: Remove
        if θ₂ - θ₁ < tol
            spk1 = [x[1] for x ∈ spikes1]
            spk2 = [x[1] for x ∈ spikes2]
            vm2 = v_max2[1:2]
            spsps = [sign(m.w[x[3]])*x[1] for x ∈ PSPs]
            spkj1 = [x[4] for x ∈ spikes1]
            spkj2 = [x[4] for x ∈ spikes2]
            vm2j = v_max2[3]
            @debug ("spikes1: $spk1\n" *
            "spikes2: $spk2\n" *
            "v_max2: $vm2\n" *
            "PSPs: $spsps\n" *
            "spikes1_j: $spkj1\n" *
            "spikes2_j: $spkj2\n" *
            "v_max2_j: $vm2j")


            t_max = v_max2[2]
            function v(tt, θ)
                spk = GetSpikes(m, PSPs, PSP, θ, T_max)
                Vs_spk(t) = isempty(spk) ? 0 : sum(x -> x[2](t), spk)
                V(t) = PSP(t) + Vs_spk(t)
                return V.(tt)
            end
            t_vec = 0:0.01:T_max
            V1 = v(t_vec, θ₁)
            V2 = v(t_vec, θ₂)
            c1 = ones(length(t_vec))
            pyplot(size = (700, 350))
            p = plot(t_vec, V1, linecolor = :blue, label = "V(t;θ₁)")
            plot!(t_vec, V2, linecolor = :red, label = "V(t;θ₂)")
            plot!(t_vec, m.θ*c1, linecolor = :black,
                linestyle = :dash, label = "")
            plot!(t_vec, θ₁*c1, linecolor = :blue,
                linestyle = :dash, label = "")
            plot!(t_vec, θ₂*c1, linecolor = :red,
                linestyle = :dash, label = "")
            for tm ∈ t_max_hist
                plot!([tm, tm], [m.V₀, max(m.θ, θ₂)*1.05],
                    linecolor = :gray, linestyle = :dash, label = "")
            end
            plot!([t_max, t_max], [m.V₀, max(m.θ, θ₂)*1.05],
                linecolor = :green, label = "")
            yticks!([m.V₀, m.θ, θ₁, θ₂], ["V₀", "θ", "θ₁", "θ₂"])
            xticks!([0, t_max, T_max], ["0", "tₘₐₓ", string.(T_max)])
            ylabel!("V[mV]")
            xlabel!("t[ms]")
            plot(p)
            savefig(p, "debug.png")


            error("Bracketing did not converge")
        end
    end
    spk1 = [x[1] for x ∈ spikes1]
    spk2 = [x[1] for x ∈ spikes2]
    vm2 = v_max2[1:2]
    @debug ("spikes1: $spk1\n" *
    "spikes2: $spk2\n" *
    "v_max2: $vm2")

    (v_max, t_max, v_max_j, ex_max, sum_m, sum_s) = v_max2
    PSPs_max = filter(x -> x[1] ≤ t_max, PSPs)
    V_psp(t) = sum(x -> x[2](t), PSPs_max)
    M = length(filter(x -> x[1] < t_max, spikes2))

    function v_max(θ)
        spk = GetSpikes(m, PSPs_max, V_psp, θ, T_max)[1:M]
        # filter!(x -> x[4] < v_max_j, spk)
        if ex_max
            sum_e = isempty(spk) ? 0 : sum(x -> exp.(x[1]./m.τₘ), spk)
            t_max_θ = m.A*(m.log_α - log((sum_m - θ*sum_e)/sum_s))
        else
            t_max_θ = t_max
        end
        V_spk(t) = isempty(spk) ? 0 : sum(x -> x[2](t), spk)
        V(t) = V_psp(t) + V_spk(t)
        return V(t_max_θ)
    end

    f(x) = x - v_max(x)
    vm1, vm2 = v_max(θ₁), v_max(θ₂)
    f1, f2 = f(θ₁), f(θ₂)
    @debug ("M = $M\n" *
    "vₘₐₓ(θ⃰)∈[$vm1, $vm2], f(θ⃰)∈[$f1, $f2]")

    θ⃰ = NaN
    try
        θ⃰ = find_zero(f, (θ₁, θ₂), Roots.A42(), xatol = tol)
        # θ⃰ = find_zero(f, (θ₁ + θ₂)/2, Order1(), xatol = tol)
        # θ⃰ = Roots.secant_method(f, (θ₁, θ₂), atol = tol)
        @assert !isnan(θ⃰)
    catch ex
        # TODO: Remove
        @debug "catch"
        function v(tt, θ)
            spk = GetSpikes(m, PSPs, PSP, θ, T_max)
            Vs_spk(t) = isempty(spk) ? 0 : sum(x -> x[2](t), spk)
            V(t) = V_psp(t) + Vs_spk(t)
            return V.(tt)
        end
        t_vec = 0:0.01:T_max
        V1 = v(t_vec, θ₁)
        V2 = v(t_vec, θ₂)
        c1 = ones(length(t_vec))
        pyplot(size = (700, 350))
        p = plot(t_vec, V1, linecolor = :blue, label = "V(t;θ₁)")
        plot!(t_vec, V2, linecolor = :red, label = "V(t;θ₂)")
        plot!(t_vec, m.θ*c1, linecolor = :black,
            linestyle = :dash, label = "")
        plot!(t_vec, θ₁*c1, linecolor = :blue,
            linestyle = :dash, label = "")
        plot!(t_vec, θ₂*c1, linecolor = :red,
            linestyle = :dash, label = "")
        plot!([t_max, t_max], [m.V₀, max(m.θ, θ₂)*1.05],
            linecolor = :green, label = "")
        yticks!([m.V₀, m.θ, θ₁, θ₂], ["V₀", "θ", "θ₁", "θ₂"])
        xticks!([0, t_max, T_max], ["0", "tₘₐₓ", string.(T_max)])
        ylabel!("V[mV]")
        xlabel!("t[ms]")
        plot(p)
        savefig(p, "debug.png")
        # i = 0
        # filename = "DebugPlots\\debug" * string(i) * ".png"
        # while isfile(filename)
        #     i += 1
        #     filename = "DebugPlots\\debug" * string(i) * ".png"
        # end
        # Filesystem.cp("debug.png", filename)
        throw(ex)
    end


    # f(x) = x - V(A*(log_α - log((sum_m - x*sum_e)/sum_s)))
    # d(f) = x -> ForwardDiff.derivative(f, float(x))
    # θ⃰ = find_zero((f, d(f)), θ₂, Roots.Newton(), xatol = tol)
    # θ⃰ = find_zero(f, (θ₁, θ₂), Roots.A42(), xatol = tol)

    @debug "θ⃰ = $θ⃰"
    spk = GetSpikes(m, PSPs, PSP, θ⃰, T_max)
    filter!(x -> x[4] < v_max_j, spk)
    M = length(spk)
    if ex_max
        sum_e = isempty(spk) ? 0 : sum(x -> exp.(x[1]./m.τₘ), spk)
        t⃰ = m.A*(m.log_α + log(sum_s/(sum_m - θ⃰*sum_e)))
    else
        t⃰ = t_max
    end

    return t⃰, θ⃰, M
end

function GetGradient(m::Tempotron,
                     inp::Array{Array{T1, 1}, 1},
                     spk::Array{T2},
                     PSP) where {T1 <: Any,
                               T2 <: Any}

    function Ση(tₓ)
        pre = filter(y -> y < tₓ, spk)
        return isempty(pre) ? 0 : sum(x -> m.η(tₓ - x), pre)
    end
    Σe = Ση.(spk)
    C = 1 .+ Σe
    V₀ = PSP.(spk)
    function ΣK(tₓ)
        sum_K = zeros(length(inp))
        for i = 1:length(inp)
            pre = filter(x -> x < tₓ, inp[i])
            sum_K[i] = isempty(pre) ? 0 : sum(m.K.(tₓ .- pre))
        end
        return sum_K
    end
    ∂V∂w = ΣK.(spk)./C

    ∂V = zeros(length(spk), length(spk))
    for k = 1:length(spk)
        a = -V₀[k]/(m.τₘ*C[k]^2)
        for j = 1:(k - 1)
            ∂V[k, j] = -a*m.η(spk[k] - spk[j])
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

"""
    Train!(m::Tempotron, inp, y₀::Integer[, optimizer = SGD(0.01)][, T_max])
Train a tempotron `m` to fire y₀ spikes in response to an input vector of spike
trains `inp`. Optional parameters are the optimizer to be used (default is `SGD`
 with learning rate `0.01`) and maximal time `T`.
For further details see [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113).
"""
function Train!(m::Tempotron,
                inp::Array{Array{Tp, 1}, 1},
                y₀::Integer;
                optimizer::Optimizer    = SGD(0.01),
                T_max::Real             = 0) where Tp <: Any
    N, T = ValidateInput(m, inp, T_max)

    PSPs = sort(GetPSPs(m, inp, T_max), by = x -> x[1], dims = 1)
    PSP(t) = sum(x -> x[2](t), PSPs)

    k = length(GetSpikes(m, PSPs, PSP, m.θ, T_max))
    @debug "y₀ = $y₀; y = $k"
    if k == y₀
        ∇ = zeros(size(m.w))
        optimizer(∇)
        return
    end

    o = y₀ > k ? k + 1 : k
    t⃰, θ⃰, M = GetCriticalThreshold(m, PSPs, PSP, o, T_max)
    spk = [x[1] for x ∈ GetSpikes(m, PSPs, PSP, θ⃰, T_max)][1:M]
    push!(spk, t⃰)
    ∇θ⃰ = GetGradient(m, inp, spk, PSP)
    m.w += (y₀ > k ? -1 : 1).*optimizer(∇θ⃰)

end
