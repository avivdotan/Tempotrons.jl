using Roots
using ForwardDiff
using ..Optimizers


# TODO: Remove
using Plots
import Base: Filesystem


function VmaxLinked2Spike(spikes1,
                            spikes2,
                            v_max2)
    spikes1_c = [x.psp for x ∈ spikes1 if x.psp ≤ v_max2.s]
    spikes2_c = [x.psp for x ∈ spikes2 if x.psp ≤ v_max2.s]
    push!(spikes2_c, v_max2.s)
    # TODO: Better identifiers
    return (sort(spikes1_c) == sort(spikes2_c))
end

function GetCriticalThreshold(m::Tempotron,
                                PSPs,
                                PSP,
                                y₀::Integer,
                                tol::Real = 1e-13) where {T1 <: Real,
                                                            T2 <: Real}

    θ₁ = m.V₀
    k₁ = typemax(Int)
    θ₂ = 10m.θ
    k₂ = 0
    spikes1 = []
    spikes2 = []
    v_max2 = (s = 0, e = 0, asc = true, l_max = false, spk = false, v_e = -Inf,
                sum_m = 0, sum_s = 0)
    t_max_hist = []
    while k₁ ≠ y₀ || k₂ ≠ (y₀ - 1) || !VmaxLinked2Spike(spikes1, spikes2, v_max2)
        θ = (θ₁ + θ₂)/2
        spk, v_max = GetSpikes(m, PSPs, PSP, θ, return_v_max = true)
        k = length(spk)
        if k < y₀
            θ₂ = θ
            k₂ = k
            spikes2 = spk
            # v_max2, t_max_hist = GetVmax(m, PSPs, spikes2, θ₂, T_max)
            v_max2 = v_max
        else
            θ₁ = θ
            k₁ = k
            spikes1 = spk
        end
        @debug "[$k₂, $k₁], [$θ₁, $θ₂]"
        # TODO: Remove once stable
        if θ₂ - θ₁ < tol
            spk1 = [x.time for x ∈ spikes1]
            spk2 = [x.time for x ∈ spikes2]
            vm2 = (v_max2.v_e, v_max2.e)
            spsps = [sign(m.w[x.neuron])*x.time for x ∈ PSPs]
            spkj1 = [x.psp for x ∈ spikes1]
            spkj2 = [x.psp for x ∈ spikes2]
            vm2j = v_max2.s
            @debug ("spikes1: $spk1\n" *
            "spikes2: $spk2\n" *
            "v_max2: $vm2\n" *
            "PSPs: $spsps\n" *
            "spikes1_j: $spkj1\n" *
            "spikes2_j: $spkj2\n" *
            "v_max2_j: $vm2j")


            t_max = v_max2.e
            function v(tt, θ)
                spk = GetSpikes(m, PSPs, PSP, θ)
                Vs_spk(t) = isempty(spk) ? 0 : sum(x -> x.ΔV(t), spk)
                V(t) = PSP(t) + Vs_spk(t)
                return V.(tt)
            end
            T_max = maximum([p.time for p ∈ PSPs]) + 3m.τₘ
            t_vec = 0:0.1:T_max
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
    spk1 = [x.time for x ∈ spikes1]
    spk2 = [x.time for x ∈ spikes2]
    vm2 = (v_max2.v_e, v_max2.e)
    spsps = [sign(m.w[x.neuron])*x.time for x ∈ PSPs]
    spkj1 = [x.psp for x ∈ spikes1]
    spkj2 = [x.psp for x ∈ spikes2]
    vm2j = v_max2.s
    @debug ("spikes1: $spk1\n" *
    "spikes2: $spk2\n" *
    "v_max2: $vm2\n" *
    "PSPs: $spsps\n" *
    "spikes1_j: $spkj1\n" *
    "spikes2_j: $spkj2\n" *
    "v_max2_j: $vm2j")

    (v_max_j, t_max, ~, l_max, ~, v_max, sum_m, sum_s) = v_max2
    # PSPs_max = filter(x -> x.time ≤ t_max, PSPs)
    PSPs_max = filter(x -> x.time ≤ v_max_j, PSPs)
    V_psp(t) = sum(x -> x.ΔV(t), PSPs_max)
    M = length(filter(x -> (x.time < t_max && x.psp ≤ v_max_j), spikes2))

    function v_max(θ)
        # spk = GetSpikes(m, PSPs_max, V_psp, θ)[1:M]
            spk = GetSpikes(m, PSPs_max, V_psp, θ, M)
        # TODO: NextTmax
        if l_max
            sum_e = isempty(spk) ? 0 : sum(x -> exp.(x.time./m.τₘ), spk)
            t_max_θ = m.A*(m.log_α - log((sum_m - θ*sum_e)/sum_s))
        else
            t_max_θ = t_max
        end
        V_spk(t) = isempty(spk) ? 0 : (-θ*sum(x -> m.η(t - x.time), spk))
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
        function v(tt, θ, MM = typemax(Int))
            spk = GetSpikes(m, PSPs, PSP, θ, MM)
            Vs_spk(t) = isempty(spk) ? 0 : (-θ*sum(x -> m.η(t - x.time), spk))
            V(t) = PSP(t) + Vs_spk(t)
            return V.(tt)
        end
        T_max = maximum([p.time for p ∈ PSPs]) + 3m.τₘ
        t_vec = 0:0.1:T_max
        V1 = v(t_vec, θ₁)
        V2 = v(t_vec, θ₂)
        V3 = v(t_vec, θ₁, M)
        c1 = ones(length(t_vec))
        pyplot(size = (700, 350))
        p = plot(t_vec, V1, linecolor = :blue, label = "V(t;θ₁)")
        plot!(t_vec, V2, linecolor = :red, label = "V(t;θ₂)")
        plot!(t_vec, V3, linecolor = :purple, label = "Vᵤᵣ(t;θ₁)")
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
    # spk = GetSpikes(m, PSPs_max, V_psp, θ⃰)[1:M]
    spk = GetSpikes(m, PSPs_max, V_psp, θ⃰, M)
    # TODO: NextTmax
    if l_max
        sum_e = isempty(spk) ? 0 : sum(x -> exp.(x.time./m.τₘ), spk)
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

    PSPs = sort(GetPSPs(m, inp), by = x -> x.time)
    PSP(t) = sum(x -> x.ΔV(t), PSPs)

    k = length(GetSpikes(m, PSPs, PSP, m.θ))
    @debug "y₀ = $y₀; y = $k"
    if k == y₀
        ∇ = zeros(size(m.w))
        optimizer(∇)
        return
    end

    o = y₀ > k ? k + 1 : k
    t⃰, θ⃰, M = GetCriticalThreshold(m, PSPs, PSP, o)
    spk = [x.time for x ∈ GetSpikes(m, PSPs, PSP, θ⃰)][1:M]
    push!(spk, t⃰)
    ∇θ⃰ = GetGradient(m, inp, spk, PSP)
    m.w .+= (y₀ > k ? -1 : 1).*optimizer(∇θ⃰)

end
