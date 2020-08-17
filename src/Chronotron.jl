#-------------------------------------------------------------------------------
# Chronotron utils
#-------------------------------------------------------------------------------

"""
    vp_distance(spk1, spk2; τ_q[, σ])

Calculate the Victor-Purpura Distance [1] between two spike-trains `spk1` and
`spk2`. `τ_q` is the time constant governing the transition between spike
shifting to addition and deletion (``\\tau_q\\equiv \frac{1}{q}`` where ``q``
is defined as in [1]).
`σ` is a transfer function for calculating the shifting distances, see [2] for
further details. The default is the identity transformation.

# References

[1] [Victor J.D. and Purpura K.P. (1996). Nature and precision of temporal coding in visual cortex: a metric-space analysis. Journal of Neurophysiology, 76(2), 1310-1326](https://journals.physiology.org/doi/abs/10.1152/jn.1996.76.2.1310)

[2] [Florian R.V. (2012) The Chronotron: A Neuron That Learns to Fire Temporally Precise Spike Patterns. PLOS ONE, 7(8), e40233.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0040233)
"""
function vp_distance(spk1::Array{T1,1}, spk2::Array{T2,1}; τ_q::Real,
                     σ::Function = x -> x)::Real where {T1,T2}

    # Validation check
    @assert τ_q ≥ 0
    n1, n2 = length(spk1), length(spk2)

    # Handle trivial cases
    if τ_q == 0
        return n1 + n2
    elseif τ_q == Inf
        return abs(n1 - n2)
    end

    # Just to be sure   #TODO: remove?
    s1, s2 = sort(spk1), sort(spk2)

    D_prev = collect(0.0:n1)    # Previous row i-1
    D = fill(NaN, (n1 + 1))     # current row i
    D2 = fill(NaN, (n1 + 1))    # A placeholder for D_prev + 1
    D3 = fill(NaN, n1)          # A placeholder for D_prev + σ(|s1[j] - s2[i]|/τ_q)

    for i = 1:n2

        # Initialize current row
        D[1] = i

        # Preformance
        @. D2 = D_prev + 1.0
        @. D3 = D_prev[1:(end - 1)] + σ(abs(s1 - s2[i]) / τ_q)

        # Process current row
        for j = 2:(n1 + 1)
            D[j] = min(D[j - 1] + 1.0, D2[j], D3[j - 1])
        end

        # Move to the next row
        D_prev, D = D, D_prev

    end

    return D_prev[end]

end

#-------------------------------------------------------------------------------
# Chronotron implementation
#-------------------------------------------------------------------------------
"""
    spilt_spikes(source, target; τ_q[, σ])

Calculate the Victor-Purpura Distance [1] between two spike-trains `spk1` and
`spk2`, and return the sequences of taken opreations. See [2] for details.

# References

[1] [Victor J.D. and Purpura K.P. (1996). Nature and precision of temporal coding in visual cortex: a metric-space analysis. Journal of Neurophysiology, 76(2), 1310-1326](https://journals.physiology.org/doi/abs/10.1152/jn.1996.76.2.1310)

[2] [Florian R.V. (2012) The Chronotron: A Neuron That Learns to Fire Temporally Precise Spike Patterns. PLOS ONE, 7(8), e40233.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0040233)
"""
function spilt_spikes(source::Array{T1,1}, target::Array{T2,1}; τ_q::Real,
                      σ::Function = x -> x) where {T1,T2}

    # Validation check
    @assert τ_q > 0
    ns, nt = length(source), length(target)
    ss, st = sort(source), sort(target)

    D_prev = collect(0.0:nt)            # Distance: previous row i-1
    Ss_prev = [[] for j = 1:(nt + 1)]   # Source operations: previous row i-1
    St_prev = [[]]                      # Target operations: previous row i-1
    for j = 1:nt
        push!(St_prev, [St_prev[end]..., (j,)]) # Initialize target operations
    end
    D = fill(NaN, (nt + 1))             # Distance: current row i-1
    D2 = fill(NaN, (nt + 1))            # A placeholder for D_prev + 1
    D3 = fill(NaN, nt)                  # A placeholder for D_prev + σ(|s1[j] - s2[i]|/τ_q)
    Ss = [[] for j = 1:(nt + 1)]        # Source operations: previous row i-1
    St = [[] for j = 1:(nt + 1)]        # Target operations: current row i-1

    for i = 1:ns

        # Initialize current row
        D[1] = i
        Ss[1] = [Ss_prev[1]..., (i,)]
        St[1] = []

        # Preformance
        @. D2 = D_prev + 1.0
        @. D3 = D_prev[1:(end - 1)] + σ(abs(st - ss[i]) / τ_q)

        # Process current row
        for j = 2:(nt + 1)

            # Get operation distances
            d1 = D[j - 1] + 1.0
            d2 = D2[j]
            ς = D3[j - 1]

            # Deletion
            if d2 ≤ d1 && d2 ≤ ς
                D[j] = d2
                Ss[j] = [Ss_prev[j]..., (i,)]
                St[j] = St_prev[j]

                # Addition
            elseif d1 ≤ ς
                D[j] = d1
                Ss[j] = Ss[j - 1]
                St[j] = [St[j - 1]..., (j - 1,)]

                # Shifting
            else
                D[j] = ς
                Ss[j] = [Ss_prev[j - 1]..., (i, j - 1)]
                St[j] = [St_prev[j - 1]..., (j - 1, i)]
            end

        end

        # Move to the next row
        D_prev, D = D, D_prev
        Ss_prev, Ss = Ss, Ss_prev
        St_prev, St = St, St_prev

    end

    return D_prev[end], Ss_prev[end], St_prev[end]

end

"""
    train_∇!(m::Tempotron, inp::SpikesInput, y₀::SpikesInput{<:Real, 1}; optimizer = Optimizers.SGD(0.01))

Trains a neuron `m` to fire at specific times (according to y₀) in response to
an input vector of spike trains `inp`.

# Optional arguments

  - `τ_q::Real = m.τₘ` the Victor-Purpura distance time constant.
  - `γᵣ::Real = m.τₘ` the shifting cost weight.
  - `optimizer::Optimizers.Optimizer = Optimizers.SGD(0.01)`: a gradient-based optimization method (see [`Optimizers`](@ref)).

# Learning rule

This method implements the E-Learning rule from [1].
Assuming SGD, the update rule is (eq. 4 in [1]):

```math
\\Delta w_i=\\gamma\\left(\\sum_{\\tilde{t}^f} \\lambda_i\\left(\\tilde{t}^f\\right)
- \\sum_{t^f} \\lambda_i\\left(t^f\\right)
+ \\frac{\\gamma_r}{\\tau_q^2} \\sum_{t^f, \\tilde{t}^g} \\left(t^f - \\tilde{t}^g\\right) \\lambda_i\\left(t^f\\right)\\right).
```

# References

[1] [Florian R.V. (2012) The Chronotron: A Neuron That Learns to Fire Temporally Precise Spike Patterns. PLOS ONE, 7(8), e40233.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0040233)
"""
function train_∇!(m::Tempotron{N}, inp::SpikesInput{T1,N},
                  y₀::SpikesInput{T2,1}; τ_q::Real = m.τₘ, γᵣ::Real = m.τₘ,
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

    # Update the weights
    λ(t, x) = isempty(x) ? 0.0 : sum(j -> m.K(t - j), x)
    κ = γᵣ / τ_q^2
    ∇ = [(isempty(spk_add) ? 0.0 : sum(j -> λ(j, inp[i]), spk_add)) +
         (isempty(spk_rm) ? 0.0 : -sum(j -> λ(j, inp[i]), spk_rm)) +
         κ .* (isempty(spk_mv) ? 0.0 :
          sum(sm -> (sm.s - sm.t) * λ(sm.s, inp[i]), spk_mv)) for i = 1:N]
    m.w .+= optimizer(-∇)
    return

end
