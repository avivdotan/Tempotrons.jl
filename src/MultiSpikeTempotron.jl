using Roots
using ..Optimizers

export GetSTS

"""
    GetCriticalThreshold(m::Tempotron, PSPs, PSP, y₀)
Get the critical threshold θ⃰ₖ where k = `y₀`. Receiving a Tempotron `m`, a
sorted list of PSPs (`PSPs`), the unresetted voltage `PSP`, the index of the
relevant critical threshold `y₀` and a numerical precision tolerance `tol`
(default is `1e-13`). Returns the time `t⃰` of the spike elicited exactly at the
critical threshold `θ⃰`, the critical threshold `θ⃰`, and a list of the first
spike times elicited using `θ⃰` and happenning before t⃰.
"""
function GetCriticalThreshold(m::Tempotron,
                                PSPs,
                                PSP,
                                y₀::Integer,
                                tol::Real = 1e-13)

    # Checks whether vₘₐₓ(θ₂) can be linked analitically to a spike generated
    # by using θ₁.
    function VmaxLinked2Spike(spikes1,
                                spikes2,
                                v_max2)
        spikes1_c = [x.psp for x ∈ spikes1 if x.psp ≤ v_max2.psp]
        spikes2_c = [x.psp for x ∈ spikes2 if x.psp ≤ v_max2.psp]
        push!(spikes2_c, v_max2.psp)
        # TODO: Better identifiers
        return (sort(spikes1_c) == sort(spikes2_c))
    end

    # Start searching for θ⃰ by bracketing, until θ₁ is eliciting exactly y₀
    # spikes, θ₂ elicits exactly (y₀ -1) spikes and vₘₐₓ(θ₂) can be linked to a
    # spike generated using θ₁.
    θ₁, k₁ = m.V₀, typemax(Int)
    θ₂, k₂ = 10m.θ, 0
    spikes1, spikes2 = [], []
    v_max2 = (psp       = 0,
             t_max      = 0,
             next_psp   = 0,
             v_max      = -Inf,
             sum_m      = 0,
             sum_s      = 0)
    while k₁ ≠ y₀ || k₂ ≠ (y₀ - 1) || !VmaxLinked2Spike(spikes1, spikes2, v_max2)
        θ = (θ₁ + θ₂)/2
        spk, v_max = GetSpikes(m, PSPs, PSP, θ, return_v_max = true)
        k = length(spk)
        if k < y₀
            θ₂, k₂, spikes2, v_max2 = θ, k, spk, v_max
        else
            θ₁, k₁, spikes1 = θ, k, spk
        end
    end

    # Get details of vₘₐₓ(θ₂)
    (v_max_j, t_max, next_psp, v_max, sum_m, sum_s) = v_max2

    # Filter only PSPs hapenning before vₘₐₓ(θ₂)
    PSPs_max = filter(x -> x.time ≤ v_max_j, PSPs)
    V_psp(t) = sum(x -> x.ΔV(t), PSPs_max)

    # Count the spikes hapenning before vₘₐₓ(θ₂)
    M = length(filter(x -> (x.time < t_max && x.psp ≤ v_max_j), spikes2))

    # Create vₘₐₓ(θ), assuming θ ∈ [θ₁, θ₂] and the local maximum is the same
    # one found by the bracketing.
    function v_max(θ)
        spk = GetSpikes(m, PSPs_max, V_psp, θ, M)
        sum_e = isempty(spk) ? 0 : sum(x -> exp.(x.time./m.τₘ), spk)
        t_max_θ, ~ = GetNextTmax(m, v_max_j, next_psp, sum_m, sum_s, sum_e, θ)
        V_spk(t) = isempty(spk) ? 0 : (-θ*sum(x -> m.η(t - x.time), spk))
        V(t) = V_psp(t) + V_spk(t)
        return V(t_max_θ)
    end

    # Numerically solve θ - vₘₐₓ(θ) = 0 to find θ⃰
    f(x) = x - v_max(x)
    θ⃰ = find_zero(f, (θ₁, θ₂), Roots.A42(), xatol = tol)

    # Get t⃰
    spk = GetSpikes(m, PSPs_max, V_psp, θ⃰, M)
    sum_e = isempty(spk) ? 0 : sum(x -> exp.(x.time./m.τₘ), spk)
    t⃰, ~ = GetNextTmax(m, v_max_j, next_psp, sum_m, sum_s, sum_e, θ⃰)

    # Get the first M spike times
    spikes = [s.time for s ∈ spk]

    return t⃰, θ⃰, spikes
end

"""
    GetGradient(m::Tempotron, inp, spk, PSP)
Get the gradient of θ⃰ w.r.t. a tempotron's weights w. Recieving a tempotron `m`,
an input spike train `inp`, a list of spikes (`spk`) elicited using θ⃰ prior to
t⃰ (inclusive) and the unresetted voltage function `PSP` and its time derivative
`dPSP`.

The implementation follows [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113).
"""
function GetGradient(m::Tempotron,
                     inp::Array{Array{T1, 1}, 1},
                     spk::Array{T2},
                     PSP,
                     dPSP) where {T1 <: Any,
                               T2 <: Any}

    # The implementation follows [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113).
    # The relevant equation numbers from the paper are referenced.
    # TODO: Performance

    # Get C(tₓ) (eq. 29)
    function Ση(tₓ)
        pre = filter(y -> y < tₓ, spk)
        return isempty(pre) ? 0 : sum(x -> m.η(tₓ - x), pre)
    end
    Σe = Ση.(spk)
    C = 1 .+ Σe

    # Get V₀(t) (eq. 13)
    V₀ = PSP.(spk)

    # Get ∂V(tₓ)/∂wᵢ (eq. 30)
    function ΣK(tₓ)
        sum_K = zeros(length(inp))
        for i = 1:length(inp)
            pre = filter(x -> x < tₓ, inp[i])
            sum_K[i] = isempty(pre) ? 0 : sum(m.K.(tₓ .- pre))
        end
        return sum_K
    end
    ∂V∂w = ΣK.(spk)./C

    # Get ∂V(tₓ)/∂tₛᵏ (eq. 31)
    ∂V = zeros(length(spk), length(spk))
    for k = 1:length(spk)
        a = -V₀[k]/(m.τₘ*C[k]^2)
        for j = 1:(k - 1)
            ∂V[k, j] = -a*m.η(spk[k] - spk[j])
        end
    end

    # Get V̇(tₓ) (eq. 32)
    dV₀ = dPSP.(spk)
    V̇ = dV₀./C + V₀.*Σe./(m.τₘ.*C.^2)

    # Get A⃰ and B⃰ recursively (eqs. 25-26)
    A = []
    B = []
    for k = 1:length(spk)

        # (eqs. 23, 25)
        Aₖ = isempty(A) ? 1 : (1 - sum(j -> (A[j]/V̇[j])*∂V[k, j], 1:(k-1)))

        # (eqs. 24, 26)
        Bₖ = isempty(B) ? 0 : -sum(j -> (B[j]/V̇[j])*∂V[k, j], 1:(k-1))
        Bₖ = Bₖ .- ∂V∂w[k]

        push!(A, Aₖ)
        push!(B, Bₖ)
    end

    # Get ∂θ⃰/∂wᵢ (eq. 27)
    ∂θ⃰ = -B[end]/A[end]

    return ∂θ⃰
end

"""
    Train!(m::Tempotron, inp, y₀::Integer[, optimizer = SGD(0.01)])
Train a tempotron `m` to fire y₀ spikes in response to an input vector of spike
trains `inp`. An optional parameter is the optimizer to be used (default is `SGD`
 with learning rate `0.01`).
For further details see [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113).
"""
function Train!(m::Tempotron,
                inp::Array{Array{Tp, 1}, 1},
                y₀::Integer;
                optimizer::Optimizer = SGD(0.01)) where Tp <: Any
    N, T = ValidateInput(m, inp, 0)

    # Get the PSPs
    PSPs = sort(GetPSPs(m, inp), by = x -> x.time)
    PSP(t) = sum(x -> x.ΔV(t), PSPs)
    dPSP(t) = sum(x -> m.w[x.neuron]*m.K̇(t - x.time), PSPs)

    # Get the current number of spikes
    k = length(GetSpikes(m, PSPs, PSP, m.θ))

    # If the tempotron's number of spikes matches the teacher, do not learn.
    if k == y₀
        ∇ = zeros(size(m.w))
        optimizer(∇)
        return
    end

    # Get the relevant critical threshold's index
    o = y₀ > k ? k + 1 : k

    # Get the spike times elicited by the critical threshold up to the new
    # elicited apike (inclusive)
    t⃰, ~, spk = GetCriticalThreshold(m, PSPs, PSP, o)
    push!(spk, t⃰)

    # Get the gradient of θ⃰ w.r.t. the tempotron's weights
    ∇θ⃰ = GetGradient(m, inp, spk, PSP, dPSP)

    # Move θ⃰ using gradient descent/ascent based optimization
    m.w .+= (y₀ > k ? -1 : 1).*optimizer(∇θ⃰)

end

"""
    GetSTS(m::Tempotron, inp[, k_max = 10])
Get the Spike-Threshold Surface of the tempotron `m` for a given input spike
train `inp`. `k_max` sets the maximal k for which to evaluate θ⃰ₖ (default
is `10`). returns a list of θ⃰ values.
"""
function GetSTS(m::Tempotron,
                inp::Array{Array{Tp, 1}, 1};
                k_max::Integer = 10) where Tp <: Any

    # Get the PSPs
    PSPs = sort(GetPSPs(m, inp), by = x -> x.time)
    PSP(t) = sum(x -> x.ΔV(t), PSPs)
    return [GetCriticalThreshold(m, PSPs, PSP, k)[2] for k = 1:k_max]

end
