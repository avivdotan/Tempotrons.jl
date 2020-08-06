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
    θ₁, k₁ = 0.0, typemax(Int)
    θ₂, k₂ = 10(m.θ - m.V₀), 0
    spikes1, spikes2 = [], []
    v_max2 = (psp       = (time = 0.0, neuron = 0),
             t_max      = 0.0,
             next_psp   = 0.0,
             v_max      = -Inf,
             sum_m      = 0.0,
             sum_s      = 0.0)

    while k₁ ≠ y₀ || k₂ ≠ (y₀ - 1) || !VmaxLinked2Spike(spikes1, spikes2, v_max2)
        # if abs(θ₁ - θ₂) < eps() #TODO: delete
        #     spikes1_n = [x.psp.neuron for x ∈ spikes1]
        #     spikes2_n = [x.psp.neuron for x ∈ spikes2]
        #     spikes1_t = [x.psp.time for x ∈ spikes1]
        #     spikes2_t = [x.psp.time for x ∈ spikes2]
        #     v_max2_v = v_max2.v_max
        #     v_max2_n = v_max2.psp.neuron
        #     v_max2_t = v_max2.psp.time
        #     @info "[k₁,k₂] = [$k₁,$k₂]\n" *
        #           "[θ₁,θ₂] = [$θ₁,$θ₂]\n" *
        #           "v_max2 = $v_max2_v\n" *
        #           "spikes1_n = $spikes1_n\n" *
        #           "spikes2_n = $spikes2_n\n" *
        #           "v_max2_n = $v_max2_n\n" *
        #           "spikes1_t = $spikes1_t\n" *
        #           "spikes2_t = $spikes2_t\n" *
        #           "v_max2_t = $v_max2_t\n"
        #     error("Bisection method did not converge.")
        # end
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
    (v_max_psp, t_max, next_psp, v_max, sum_m, sum_s, ΔTϵ) = v_max2
    v_max_j = v_max_psp.time

    # Filter only PSPs hapenning before vₘₐₓ(θ₂)
    PSPs_max = filter(x -> x.time ≤ v_max_j, PSPs)

    # Count the spikes hapenning before vₘₐₓ(θ₂)
    M = length(filter(x -> (x.time < t_max && x.psp.time ≤ v_max_j), spikes2))

    # Create vₘₐₓ(θ), assuming θ ∈ [θ₁, θ₂] and the local maximum is the same
    # one found by the bracketing.
    function v_max(θ::Real)::Real
        spk = GetSpikes(m, PSPs_max, θ, M).spikes
        sum_e = isempty(spk) ? 0.0 : sum(x -> exp.((x.time - ΔTϵ)./m.τₘ), spk)
        t_max_θ = GetNextTmax(m, v_max_j, next_psp, ΔTϵ,
                              sum_m, sum_s, sum_e, θ)[1]
        emt, est = exp(-(t_max_θ - ΔTϵ)/m.τₘ), exp(-(t_max_θ - ΔTϵ)/m.τₛ)
        return (emt*sum_m - est*sum_s - θ*emt*sum_e)
    end

    # Numerically solve θ - vₘₐₓ(θ) = 0 to find θ⃰
    # if (θ₁ - v_max(θ₁))*(θ₂ - v_max(θ₂)) ≥ 0    #TODO: delete
    #     spk_1 = GetSpikes(m, PSPs, θ₁, M).spikes
    #     spk_1 = [(time = s.time, psp = s.psp) for s ∈ spk_1]
    #     sum_e_1 = isempty(spk_1) ? 0 : sum(x -> exp.((x.time - ΔTϵ)./m.τₘ), spk_1)
    #     t_max_1 = GetNextTmax(m, v_max_j, next_psp, ΔTϵ,
    #                           sum_m, sum_s, sum_e_1, θ₁)[1]
    #     spk_2 = GetSpikes(m, PSPs, θ₂, M).spikes
    #     spk_2 = [(time = s.time, psp = s.psp) for s ∈ spk_2]
    #     sum_e_2 = isempty(spk_2) ? 0 : sum(x -> exp.((x.time - ΔTϵ)./m.τₘ), spk_2)
    #     t_max_2 = GetNextTmax(m, v_max_j, next_psp, ΔTϵ,
    #                           sum_m, sum_s, sum_e_2, θ₂)[1]
    #     println("spk_1 = ", spk_1, "\n" *
    #             "spk_2 = ", spk_2, "\n" *
    #             "[Σₑ¹,Σₑ²] = [", sum_e_1, ",", sum_e_2, "]\n" *
    #             "[θ₁,θ₂] = [", θ₁, ",", θ₂, "]\n" *
    #             "[θ₁-Vₘ(θ₁),θ₂-Vₘ(θ₂)] = [", θ₁ - v_max(θ₁), ",", θ₂ - v_max(θ₂), "]\n" *
    #             "[tᵐᵃˣ₁,tᵐᵃˣ₂] = [", t_max_1, ",", t_max_2, "]\n" *
    #             "ΔTϵ = ", ΔTϵ, "\n")
    # end
    θ⃰ = find_zero(ϑ::Real -> ϑ - v_max(ϑ), (θ₁, θ₂), Roots.Brent(), xatol = tol)

    # Get t⃰
    spk = GetSpikes(m, PSPs_max, θ⃰, M).spikes
    sum_e = isempty(spk) ? 0 : sum(x -> exp.((x.time - ΔTϵ)./m.τₘ), spk)
    t⃰ = GetNextTmax(m, v_max_j, next_psp, ΔTϵ,
                     sum_m, sum_s, sum_e, θ⃰)[1]

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
    Train_∇θ⃰!(m::Tempotron, inp, y₀::Integer[, optimizer = SGD(0.01)])
Train a tempotron `m` to fire y₀ spikes in response to an input vector of spike
trains `inp`. An optional parameter is the optimizer to be used (default is `SGD`
 with learning rate `0.01`).
For further details see [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113).
"""
function Train_∇θ!(m::Tempotron,
                    inp::Array{Array{Tp, 1}, 1},
                    y₀::Integer;
                    optimizer::Optimizer = SGD(0.01)) where Tp <: Real

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
        # Ignore samples with non-positive voltage trace
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


function GetEligibilities(m::Tempotron,
                          inp::Array{Array{T1, 1}, 1},
                          PSPs::Array{T2, 1},
                          spikes::Array{T3, 1},
                          V) where {T1 <: Real,
                                    T2 <: NamedTuple,
                                    T3 <: Any}

    C₁ = (m.α - 1)/(2m.K_norm*(m.α + 1))
    C₂ = 1/(m.α + 1)
    W = m.w./m.K_norm
    function 𝒱(tᵢʲ::Real)::Real
        spikes_b = filter(x -> x.time < tᵢʲ, spikes)
        spikes_a = filter(x -> x.time ≥ tᵢʲ, spikes)
        Σ₁ = isempty(PSPs) ? 0.0 : sum(PSPs) do x
            absdiff = abs(x.time - tᵢʲ)
            return W[x.neuron]*(m.τₘ*exp(-absdiff/m.τₘ) -
                                m.τₛ*exp(-absdiff/m.τₛ))
        end
        Σ₂ = isempty(spikes_b) ? 0.0 : sum(x -> exp(-(tᵢʲ - x.time)/m.τₘ),
                                           spikes_b)
        Σ₃ = isempty(spikes_a) ? 0.0 : sum(spikes_a) do x
            absdiff = x.time - tᵢʲ
            return exp(-absdiff/m.τₘ)/2 - C₂*exp(-absdiff/m.τₛ)
        end
        return C₁*Σ₁ - (m.θ - m.V₀)*m.τₘ*(C₁*Σ₂ + Σ₃/m.K_norm)
    end
    return [isempty(x) ? 0.0 : sum(𝒱, x) for x ∈ inp]

end

"""
    Train_corr!(m::Tempotron, inp, y₀::Integer)
Train a tempotron `m` to fire y₀ spikes in response to an input vector of spike
trains `inp`.
For further details see [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113).
"""
function Train_corr!(m::Tempotron,
                    inp::Array{Array{Tp, 1}, 1},
                    y₀::Integer;
                    optimizer::Optimizer = SGD(1e-5),
                    top_elig_update::Real = 0.1) where Tp <: Real

    ∇ = zeros(size(m.w))

    # Get the PSPs
    PSPs = sort(GetPSPs(m, inp), by = x -> x.time)

    # Get the current number of spikes and voltage trace
    spikes, V = GetSpikes(m, PSPs, (m.θ - m.V₀), return_V = true)
    k = length(spikes)

    # If the tempotron's number of spikes matches the teacher, do not learn.
    if k == y₀
        optimizer(∇)
        return
    end

    ℰ = GetEligibilities(m, inp, PSPs, spikes, V)
    max_k = Int(round(top_elig_update*length(ℰ)))
    idx = partialsortperm(ℰ, 1:max_k, rev = true)

    # Get the weight changes
    ∇[idx] .= 1;

    # Change tempotron's weights
    m.w .+= (y₀ > k ? -1 : 1).*optimizer(∇)

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
                method::Symbol = :∇θ,
                optimizer::Optimizer = SGD(0.01),
                top_elig_update::Real = 0.1) where Tp <: Real

    if method == :∇θ
        Train_∇θ!(m, inp, y₀, optimizer = optimizer)
    elseif method == :corr
        Train_corr!(m, inp, y₀,
                    optimizer = optimizer, top_elig_update = top_elig_update)
    else
        error("Unknown training method $method. ")
    end

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
