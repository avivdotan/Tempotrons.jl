using Roots
using Distributions
using Statistics
using ..Optimizers
using ..InputGen


"""
    NoPositivePSPError
An error for an input yielding nonpositive voltage, which has no θ⃰.
"""
struct NonPositiveVoltageError <: Exception
end

"""
    GetCriticalThreshold(m::Tempotron, PSPs, PSP, y₀)
Get the critical threshold θₖ* where k = `y₀`. Receiving a Tempotron `m`, a
sorted list of PSPs (`PSPs`), the index of the relevant critical threshold `y₀`
and a numerical precision tolerance `tol` (default is `1e-13`).
Returns the time `t⃰` of the spike elicited exactly at the critical threshold
θ*, the critical threshold `θ⃰`, and a list of the first spike times elicited
using θ* and happenning before t*.
"""
function GetCriticalThreshold(m::Tempotron,
                                PSPs::Array{Tp, 1},
                                y₀::Integer,
                                tol::Real = 1e-13) where Tp <: NamedTuple

    # Checks whether vₘₐₓ(θ₂) can be linked analitically to a spike generated
    # by using θ₁.
    function VmaxLinked2Spike(spikes1::Array{Tp1, 1},
                                spikes2::Array{Tp1, 1},
                                v_max2::Tp2) where {Tp1 <: Any,
                                                    Tp2 <: NamedTuple}
        spikes1_c = [x.psp for x ∈ spikes1 if x.psp.time ≤ v_max2.psp.time]
        spikes2_c = [x.psp for x ∈ spikes2 if x.psp.time ≤ v_max2.psp.time]
        push!(spikes2_c, v_max2.psp)
        # TODO: Better comparison
        return (sort(spikes1_c) == sort(spikes2_c))
    end

    # Start searching for θ⃰ by bracketing, until θ₁ is eliciting exactly y₀
    # spikes, θ₂ elicits exactly (y₀ -1) spikes and vₘₐₓ(θ₂) can be linked to a
    # spike generated using θ₁.
    θ₁, k₁ = 0, typemax(Int)
    θ₂, k₂ = 10(m.θ - m.V₀), 0
    spikes1, spikes2 = [], []
    v_max2 = (psp       = (time = 0, neuron = 0),
             t_max      = 0,
             next_psp   = 0,
             v_max      = -Inf,
             sum_m      = 0,
             sum_s      = 0)
    while k₁ ≠ y₀ || k₂ ≠ (y₀ - 1) || !VmaxLinked2Spike(spikes1, spikes2, v_max2)
        θ = (θ₁ + θ₂)/2
        spk, v_max = GetSpikes(m, PSPs, θ, return_v_max = true)
        if v_max.v_max ≤ 0
            throw(NonPositiveVoltageError())
        end
        k = length(spk)
        if k < y₀
            θ₂, k₂, spikes2, v_max2 = θ, k, spk, v_max
        else
            θ₁, k₁, spikes1 = θ, k, spk
        end
    end

    # Get details of vₘₐₓ(θ₂)
    (v_max_psp, t_max, next_psp, v_max, sum_m, sum_s) = v_max2
    v_max_j = v_max_psp.time

    # Filter only PSPs hapenning before vₘₐₓ(θ₂)
    PSPs_max = filter(x -> x.time ≤ v_max_j, PSPs)

    # Count the spikes hapenning before vₘₐₓ(θ₂)
    M = length(filter(x -> (x.time < t_max && x.psp.time ≤ v_max_j), spikes2))

    # Create vₘₐₓ(θ), assuming θ ∈ [θ₁, θ₂] and the local maximum is the same
    # one found by the bracketing.
    function v_max(θ::Real)::Real
        spk = GetSpikes(m, PSPs_max, θ, M).spikes
        sum_e = isempty(spk) ? 0 : sum(x -> exp.(x.time./m.τₘ), spk)
        t_max_θ = GetNextTmax(m, v_max_j, next_psp, sum_m, sum_s, sum_e, θ)[1]
        emt, est = exp(-t_max_θ/m.τₘ), exp(-t_max_θ/m.τₛ)
        return (emt*sum_m - est*sum_s - θ*emt*sum_e)
    end

    # Numerically solve θ - vₘₐₓ(θ) = 0 to find θ⃰
    θ⃰ = find_zero(ϑ -> ϑ - v_max(ϑ), (θ₁, θ₂), Roots.A42(), xatol = tol)

    # Get t⃰
    spk = GetSpikes(m, PSPs_max, θ⃰, M).spikes
    sum_e = isempty(spk) ? 0 : sum(x -> exp.(x.time./m.τₘ), spk)
    t⃰ = GetNextTmax(m, v_max_j, next_psp, sum_m, sum_s, sum_e, θ⃰)[1]

    # Get the first M spike times
    spikes = Real[s.time for s ∈ spk]

    return t⃰, θ⃰, spikes
end

"""
    GetGradient(m::Tempotron, inp, spk, PSP)
Get the gradient of θ* w.r.t. a tempotron's weights w. Recieving a tempotron `m`,
an input spike train `inp`, a list of spikes (`spk`) elicited using θ* prior to
t* (inclusive) and the unresetted voltage function `PSP` and its time derivative
`dPSP`.

The implementation follows the notations from [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113).
"""
function GetGradient(m::Tempotron,
                     inp::Array{Array{T1, 1}, 1},
                     spk::Array{T2, 1},
                     PSP::Function,
                     dPSP::Function) where {T1 <: Real,
                                            T2 <: Real}

    # The implementation follows [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113).
    # The relevant equation numbers from the paper are referenced.
    # TODO: Review and performance

    # Get C(tₓ) (eq. 29)
    Ση = map(spk) do tₓ::Real
        pre = filter(y -> y < tₓ, spk)
        return isempty(pre) ? 0 : sum(x -> m.η(tₓ - x), pre)
    end
    C = 1 .+ Ση

    # Get V₀(t) (eq. 13)
    V₀ = PSP.(spk)

    # Get ∂V(tₓ)/∂wᵢ (eq. 30)
    ΣK = map(spk) do tₓ::Real
        sum_K = zeros(length(inp))
        for i = 1:length(inp)
            pre = filter(x -> x < tₓ, inp[i])
            sum_K[i] = isempty(pre) ? 0 : sum(m.K.(tₓ .- pre))
        end
        return sum_K
    end
    ∂V∂w = ΣK./C

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
    V̇ = dV₀./C + V₀.*Ση./(m.τₘ.*C.^2)

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
                optimizer::Optimizer = SGD(0.01)) where Tp <: Real
    valid, N = ValidateInput(m, inp)
    if !valid
        return
    end

    # Get the PSPs
    PSPs = sort(GetPSPs(m, inp), by = x -> x.time)

    # Get the current number of spikes
    k = length(GetSpikes(m, PSPs, (m.θ - m.V₀)).spikes)
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
    spk, t⃰ = [], 0
    try
        t⃰, ~, spk = GetCriticalThreshold(m, PSPs, o)
    catch ex
        if isa(ex, NonPositiveVoltageError)
            return
        else
            throw(ex)
        end
    end
    push!(spk, t⃰)

    # Get the gradient of θ⃰ w.r.t. the tempotron's weights
    # TODO: Performance?
    PSP(t::Real)::Real = sum(x -> x.ΔV(t), PSPs)
    dPSP(t::Real)::Real = sum(x -> m.w[x.neuron]*m.K̇(t - x.time), PSPs)
    ∇θ⃰ = GetGradient(m, inp, spk, PSP, dPSP)

    # Move θ⃰ using gradient descent/ascent based optimization
    m.w .+= (y₀ > k ? -1 : 1).*optimizer(∇θ⃰)

end

"""
    GetSTS(m::Tempotron, inp[, k_max = 10])
Get the Spike-Threshold Surface of the tempotron `m` for a given input spike
train `inp`. `k_max` sets the maximal k for which to evaluate θₖ* (default
is `10`). returns a list of θ* values.
"""
function GetSTS(m::Tempotron,
                inp::Array{Array{Tp, 1}, 1};
                k_max::Integer = 10) where Tp <: Real

    # Get the PSPs
    PSPs = sort(GetPSPs(m, inp), by = x -> x.time)
    return (m.V₀ .+ [GetCriticalThreshold(m, PSPs, k)[2] for k = 1:k_max])

end

"""
    Pretrain!(m::Tempotron, ν_in::Real = 5, ν_out::Real = ν_in; T::Real = 1000,
              σᵢ::Real = 0.01, block_size::Integer = 100,
              opt::Optimizer = SGD(1e-3))
Pretrain a Multi-spike Tempotron `m` to fire at a given frequency `ν_out` in
response to input background noise of frequency `ν_in`. The pretraining process
is initialized with normally distributed weights with s.t.d. `σᵢ`, then trained
using optimizer `opt` and sample blocks of size `block size`, where each sample
is of length `T` and the label is Poisson distributed.
For further detail, see the 'Initialization' subsection under
'Materials and methods' in [Gütig, R. (2016). Spiking neurons can discover
predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113).
"""
function Pretrain!(m::Tempotron,
                   ν_in::Real           = 5,
                   ν_out::Real          = ν_in;
                   T::Real              = 1000,
                   σᵢ::Real             = 0.01,
                   block_size::Integer  = 100,
                   opt::Optimizer       = SGD(1e-3))

    m.w .= σᵢ*randn(size(m.w))
    μ = 0
    μₜ = 0.001ν_out*T
    while μ < μₜ
        block = [(x = [PoissonProcess(ν = ν_in, T = T) for i = 1:length(m.w)],
                  y = rand(Poisson(0.001ν_out*T)))
                 for s = 1:block_size]
        for s ∈ block
            Train!(m, s.x, s.y, optimizer = opt)
        end
        μ = mean([length(m(s.x)[1]) for s ∈ block])
    end

end
