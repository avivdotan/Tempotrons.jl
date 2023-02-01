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
function vp_distance(spk1::Array{T1,1},
                     spk2::Array{T2,1};
                     τ_q::Real,
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

    # Set a couple of rows (instead of a whole matrix)
    D = fill(NaN, (n1 + 1))     # current row i
    D_prev = collect(0.0:n1)    # Previous row i-1

    # Performance
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
    split_spikes(source, target; τ_q[, σ])

Calculate the Victor-Purpura Distance [1] between two spike-trains `spk1` and
`spk2`, and return the sequences of taken opreations.
See algorithm 1 in [2] for details.

# References

[1] [Victor J.D. and Purpura K.P. (1996). Nature and precision of temporal coding in visual cortex: a metric-space analysis. Journal of Neurophysiology, 76(2), 1310-1326](https://journals.physiology.org/doi/abs/10.1152/jn.1996.76.2.1310)

[2] [Florian R.V. (2012) The Chronotron: A Neuron That Learns to Fire Temporally Precise Spike Patterns. PLOS ONE, 7(8), e40233.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0040233)
"""
function split_spikes(source::Array{T1,1},
                      target::Array{T2,1};
                      τ_q::Real,
                      σ::Function = x -> x) where {T1,T2}

    # Validation check
    @assert τ_q > 0
    ns, nt = length(source), length(target)
    ss, st = sort(source), sort(target)

    # Set a couple of rows (instead of a whole matrix)
    D = fill(NaN, (nt + 1))                                                     # Distance: current row i
    D_prev = collect(0.0:nt)                                                    # Distance: previous row i-1
    S = [(add = [], rm = [], mv = []) for j = 1:(nt + 1)]                       # Operations: current row i
    S_prev = [(add = collect(1:(j - 1)), rm = [], mv = []) for j = 1:(nt + 1)]  # Operations: previous row i-1

    # Set placeholders
    D2 = fill(NaN, (nt + 1))    # A placeholder for D_prev + 1
    D3 = fill(NaN, nt)          # A placeholder for D_prev + σ(|s1[j] - s2[i]|/τ_q)

    # For each row
    for i = 1:ns

        # Initialize current row
        D[1] = i
        S[1] = (add = [], rm = [deepcopy(S_prev[1].rm)..., i], mv = [])

        # Preformance
        @. D2 = D_prev + 1.0
        @. D3 = D_prev[1:(end - 1)] + σ(abs(st - ss[i]) / τ_q)

        # Process current row, cell by cell
        for j = 2:(nt + 1)

            # Get operation distances
            d1 = D[j - 1] + 1.0
            d2 = D2[j]
            ς = D3[j - 1]

            if d2 ≤ d1 && d2 ≤ ς            # Deletion
                D[j] = d2
                S[j] = deepcopy(S_prev[j])
                push!(S[j].rm, i)

            elseif d1 ≤ ς                   # Addition
                D[j] = d1
                S[j] = deepcopy(S[j - 1])
                push!(S[j].add, j - 1)

            else                            # Shifting
                D[j] = ς
                S[j] = deepcopy(S_prev[j - 1])
                push!(S[j].mv, (i, j - 1))
            end
        end

        # Move to the next row
        D_prev, D = D, D_prev
        S_prev, S = S, S_prev
    end

    return D_prev[end], S_prev[end]
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
function train_∇!(m::Tempotron{N},
                  inp::SpikesInput{T1,N},
                  y₀::SpikesInput{T2,1};
                  τ_q::Real = m.τₘ,
                  γᵣ::Real = m.τₘ,
                  optimizer = SGD(0.01)) where {T1<:Real,T2<:Real,N}

    # Get the current spike times
    spk_c = m(inp).spikes

    # Get the target spike times
    spk_t = y₀[1]

    # split the spikes into categories
    ~, S = split_spikes(spk_c, spk_t; τ_q = τ_q, σ = x -> x^2 / 2)
    spk_add = spk_t[S.add]
    spk_rm = spk_c[S.rm]
    spk_mv = [(s = spk_c[i], t = spk_t[j]) for (i, j) ∈ S.mv]

    # Update the weights
    λ(t, x) = isempty(x) ? 0.0 : sum(j -> m.K(t - j), x)
    κ = γᵣ / τ_q^2
    ∇ = [(isempty(spk_add) ? 0.0 : sum(j -> λ(j, inp[i]), spk_add)) +
         (isempty(spk_rm) ? 0.0 : -sum(j -> λ(j, inp[i]), spk_rm)) +
         κ .* (isempty(spk_mv) ? 0.0 :
               sum(sm -> (sm.s - sm.t) * λ(sm.s, inp[i]), spk_mv)) for
         i = 1:N]
    m.w .+= optimizer(-∇)
    return
end
