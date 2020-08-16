function vp_distance(spk1::Array{T1,1}, spk2::Array{T2,1}; τ_q::Real,
                     σ::Function = x -> x)::Real where {T1,T2}

    @assert τ_q ≥ 0
    n1, n2 = length(spk1), length(spk2)
    if τ_q == 0
        return n1 + n2
    elseif τ_q == Inf
        return abs(n1 - n2)
    end

    s1, s2 = sort(spk1), sort(spk2)

    D_prev = collect(0.0:n1)
    D = fill(NaN, (n1 + 1))
    D2 = fill(NaN, (n1 + 1))
    D3 = fill(NaN, n1)
    for i = 1:n2
        D[1] = i
        @. D2 = D_prev + 1.0
        @. D3 = D_prev[1:(end - 1)] + σ(abs(s1 - s2[i]) / τ_q)
        for j = 2:(n1 + 1)
            D[j] = min(D[j - 1] + 1.0, D2[j], D3[j - 1])
        end
        D_prev, D = D, D_prev
    end
    return D_prev[end]

end

function spilt_spikes(student::Array{T1,1}, teacher::Array{T2,1}; τ_q::Real,
                      σ::Function = x -> x) where {T1,T2}

    @assert τ_q > 0
    ns, nt = length(student), length(teacher)
    ss, st = sort(student), sort(teacher)

    D_prev = collect(0.0:nt)
    Ss_prev = [[] for j = 1:(nt + 1)]
    St_prev = [[]]
    for j = 1:nt
        push!(St_prev, [St_prev[end]..., (j,)])
    end
    D = fill(NaN, (nt + 1))
    D2 = fill(NaN, (nt + 1))
    D3 = fill(NaN, nt)
    Ss = [[] for j = 1:(nt + 1)]
    St = [[] for j = 1:(nt + 1)]
    for i = 1:ns
        D[1] = i
        @. D2 = D_prev + 1.0
        @. D3 = D_prev[1:(end - 1)] + σ(abs(st - ss[i]) / τ_q)
        Ss[1] = [Ss_prev[1]..., (i,)]
        St[1] = []
        for j = 2:(nt + 1)
            d1 = D[j - 1] + 1.0
            d2 = D2[j]
            ς = D3[j - 1]
            if d2 ≤ d1 && d2 ≤ ς
                D[j] = d2
                Ss[j] = [Ss_prev[j]..., (i,)]
                St[j] = St_prev[j]
            elseif d1 ≤ ς
                D[j] = d1
                Ss[j] = Ss[j - 1]
                St[j] = [St[j - 1]..., (j - 1,)]
            else
                D[j] = ς
                Ss[j] = [Ss_prev[j - 1]..., (i, j - 1)]
                St[j] = [St_prev[j - 1]..., (j - 1, i)]
            end
        end
        D_prev, D = D, D_prev
        Ss_prev, Ss = Ss, Ss_prev
        St_prev, St = St, St_prev
    end
    return D_prev[end], Ss_prev[end], St_prev[end]

end

function train_∇!(m::Tempotron{N}, inp::SpikesInput{T1,N},
                  y₀::SpikesInput{T2,1}; τ_q::Real = m.τₘ, γᵣ = m.τₘ,
                  optimizer = SGD(0.01)) where {T1<:Real,T2<:Real,N}

    # Get the current spike times
    spk_c = m(inp).spikes

    # Get the target spike times
    spk_t = y₀[1]

    # split the spikes into categories
    ~, S_c, S_t = spilt_spikes(spk_c, spk_t, τ_q = τ_q, σ = x -> x^2 / 2)
    spk_add = [spk_t[sa[1]] for sa ∈ filter(st -> length(st) == 1, S_t)]
    spk_rm = [spk_c[sr[1]] for sr ∈ filter(sc -> length(sc) == 1, S_c)]
    spk_mv = [(spk_c[sm[1]], spk_t[sm[2]])
              for sm ∈ filter(sc -> length(sc) == 2, S_c)]

    λ = [t -> isempty(inp[i]) ? 0.0 : sum(j -> m.K(t - j), inp[i]) for i = 1:N]
    κ = γᵣ / (τ_q)^2
    ∇ = [(isempty(spk_add) ? 0.0 : sum(λ[i], spk_add)) +
         (isempty(spk_rm) ? 0.0 : -sum(λ[i], spk_rm)) +
         κ .* (isempty(spk_mv) ? 0.0 :
          sum(mv -> (mv[1] - mv[2]) * λ[i](mv[1]), spk_mv)) for i = 1:N]
    m.w .+= optimizer(-∇)

end

function train_corr!(m::Tempotron{N}, inp::SpikesInput{T1,N},
                     y₀::SpikesInput{T2,1}; τ_q::Real = m.τₘ, γᵣ = m.τₘ,
                     optimizer = SGD(0.01)) where {T1<:Real,T2<:Real,N}

    # Get the current spike times
    spk_c = m(inp).spikes

    # Get the target spike times
    spk_t = y₀[1]

    ΔI(t) = t < 0 ? 0.0 : exp(-t / m.τₛ)
    I = [t -> isempty(inp[i]) ? 0.0 : sum(j -> ΔI(t - j), inp[i])
         for i = 1:N]
    # λ = [t -> isempty(inp[i]) ? 0.0 : sum(j -> m.K(t - j), inp[i]) for i = 1:N]
    ∇ = [((isempty(spk_t) ? 0.0 : sum(I[i], spk_t)) +
          (isempty(spk_c) ? 0.0 : -sum(I[i], spk_c))) for i = 1:N]
    m.w .+= optimizer(-∇)

end
