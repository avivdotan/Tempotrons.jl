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

function spilt_spikes(source::Array{T1,1}, target::Array{T2,1}; τ_q::Real,
                      σ::Function = x -> x) where {T1,T2}

    @assert τ_q > 0
    ns, nt = length(source), length(target)
    ss, st = sort(source), sort(target)

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
    spk_add = [spk_t[i] for (i,) ∈ filter(st -> length(st) == 1, S_t)]
    spk_rm = [spk_c[i] for (i,) ∈ filter(sc -> length(sc) == 1, S_c)]
    spk_mv = [(s = spk_c[i], t = spk_t[j])
              for (i, j) ∈ filter(sc -> length(sc) == 2, S_c)]

    λ(t, x) = isempty(x) ? 0.0 : sum(j -> m.K(t - j), x)
    κ = γᵣ / τ_q^2
    ∇ = [(isempty(spk_add) ? 0.0 : sum(j -> λ(j, inp[i]), spk_add)) +
         (isempty(spk_rm) ? 0.0 : -sum(j -> λ(j, inp[i]), spk_rm)) +
         κ .* (isempty(spk_mv) ? 0.0 :
          sum(sm -> (sm.s - sm.t) * λ(sm.s, inp[i]), spk_mv)) for i = 1:N]
    m.w .+= optimizer(-∇)
    return

end
