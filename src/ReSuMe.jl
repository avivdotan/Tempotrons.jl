function train_corr!(m::Tempotron{N}, inp::SpikesInput{T1,N},
                     y₀::SpikesInput{T2,1}; aᵣ::Real = 0.0, τᵣ::Real = m.τₘ,
                     fᵣ::Function = t::Real -> (t < 0 ? 0.0 : exp(-t / τᵣ)),
                     optimizer = SGD(0.01)) where {T1<:Real,T2<:Real,N}

    # Get the current spike times
    spk_c = m(inp).spikes

    # Get the target spike times
    spk_t = y₀[1]

    function λ(t::Real, x::Array{T, 1})::Real where {T <: Real}
        ξ = filter(j -> j < t, x)
        return (aᵣ + (isempty(ξ) ? 0.0 : sum(j -> fᵣ(t - j), ξ)))
    end

    Δ = [(isempty(spk_t) ? 0.0 : sum(t -> λ(t, inp[i]), spk_t)) -
         (isempty(spk_c) ? 0.0 : sum(t -> λ(t, inp[i]), spk_c)) for i = 1:N]
    m.w .+= optimizer(-Δ)
    return

end
