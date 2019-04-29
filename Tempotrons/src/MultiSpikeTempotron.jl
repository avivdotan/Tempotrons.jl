using Roots
using ForwardDiff

function GetSpikes(m::Tempotron,
                    PSP,
                    η,
                    θ::Real,
                    T_max::Real,
                    dt::Real = 0.1)
   t = 0:dt:T_max
   V = PSP.(t)
   spikes = []
   for j = 1:length(t)
       if V[j] > θ && j < length(t)
           V -= θ.*η.(t .- t[j + 1])
           push!(spikes, t[j])
       end
   end
   return spikes
end

function GetCriticalThreshold(m::Tempotron,
                             PSPs,
                             PSP,
                             η,
                             y₀::Integer,
                             T_max::Real,
                             tol::Real = 1e-13)
   θ₁ = m.V₀
   k₁ = typemax(Int)
   θ₂ = 10m.θ
   k₂ = 0
   spikes = []
   while k₁ ≠ y₀ || k₂ ≠ (y₀ - 1) # || (θ₂ - θ₁) > tol
      θ = (θ₁ + θ₂)/2
      spk = GetSpikes(m, PSP, η, θ, T_max)
      k = length(spk)
      if k < y₀
         θ₂ = θ
         k₂ = k
         spikes = spk
      else
         θ₁ = θ
         k₁ = k
      end
      println("[", k₂, ", ", k₁, "], [", θ₁, ", ", θ₂, "]")
   end

   A = m.τₘ * m.τₛ / (m.τₘ - m.τₛ)
   α = m.τₘ/m.τₛ
   log_α = log(α)
   K_norm = α^(-1/(α - 1)) - α^(-α/(α - 1))

   Ps = [(j, i, m.w[i]/K_norm, ΔV) for (j, ΔV, i) ∈ PSPs]
   Ns = [(j, 0, -θ₂, t -> -θ₂*η.(t .- j)) for j ∈ spikes]
   Vs = vcat(Ps, Ns)
   Vs = sort(Vs[:], by = x -> x[1])
   # dVs = [(Vs[k][1], Vs[k][2], Vs[k][3], t -> sum(x -> x[4](t), Vs[1:k]))
   #             for k = 1:length(Vs)]
   V(t) = sum(x -> x[4](t), Vs)

   sum_m_t = 0
   sum_s_t = 0
   sum_e_t = 0
   sum_m = 0
   sum_s = 0
   sum_e = 0
   t_max = 0
   V_max = -Inf
   M_t = 0
   M = 0
   j_max = 0
   for (j, i, w, ~) ∈ Vs
      if i == 0
         sum_e_t += exp(j/m.τₘ)
         M_t += 1
         continue
      end
      sum_m_t += w*exp(j/m.τₘ)
      sum_s_t += w*exp(j/m.τₛ)
      rem = (sum_m_t - θ₂*sum_e_t)/sum_s_t
      if rem ≤ 0
         continue
      end
      t_max_c = A*(log_α - log(rem))
      t_max_c = clamp(t_max_c, 0, T_max)
      V_max_c = V(t_max_c)
      if V_max_c > V_max
         V_max = V_max_c
         t_max = t_max_c
      end
      if j < t_max
         sum_m = sum_m_t
         sum_s = sum_s_t
         M = M_t
      end
   end

   Ps_max = filter(x -> x[1] < t_max, Ps)
   Vs_psp(t) = sum(x -> x[4](t), Ps_max)
   function v_max(θ)
      spk = GetSpikes(m, PSP, η, θ, T_max)[1:M]
      sum_e = isempty(spk) ? 0 : sum(exp.(spk./m.τₘ))
      Vs_spk(t) = isempty(spk) ? 0 : sum(x -> -θ*η.(t .- x), spk)
      V(t) = Vs_psp(t) + Vs_spk(t)
      return V(A*(log_α - log((sum_m - θ*sum_e)/sum_s)))
   end

   f(x) = x - v_max(x)
   # f(x) = x - V(A*(log_α - log((sum_m - x*sum_e)/sum_s)))
   # d(f) = x -> ForwardDiff.derivative(f, float(x))
   # θ⃰ = find_zero((f, d(f)), θ₂, Roots.Newton(), xatol = tol)
   θ⃰ = find_zero(f, (θ₁, θ₂), Roots.A42(), xatol = tol)

   println("θ⃰ = ", θ⃰)
   t⃰ = A*(log_α + log(sum_s/(sum_m - θ⃰*sum_e)))

   return t⃰, θ⃰
end

function Train!(m::Tempotron,
                inp::Array{Array{Tp, 1}, 1},
                y₀::Integer;
                T_max::Real = 0) where Tp <: Any
    N, T = ValidateInput(m, inp, T_max)

    PSPs = GetPSPs(m, inp, T_max)
    PSP(t) = sum(x -> x[2](t), PSPs)

    η(t) = t < 0 ? 0 : exp(-t/m.τₘ)

    k = length(GetSpikes(m, PSP, η, m.θ, T_max))
    println("y₀ = ", y₀, "; y = ", k)
    if k == y₀
        return
    end

    λ = m.λ * (y₀ > k ? 1 : -1)
    o = y₀ > k ? k + 1 : k
    t⃰, θ⃰ = GetCriticalThreshold(m, PSPs, PSP, η, o, T_max)

end
