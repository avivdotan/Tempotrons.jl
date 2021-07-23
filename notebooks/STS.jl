### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ db4750d0-dcca-11ea-0fc4-71d7dffb5f8a
begin
	import Pkg
	Pkg.activate("..")
	Pkg.instantiate()
	using Tempotrons
	using Tempotrons.InputGen
	using PlutoUI
	using Plots
end;

# ╔═╡ a7473760-dd3c-11ea-3df0-d5b516288e3b
md"""
# Spike-Treshold Surface (STS)
"""

# ╔═╡ 7c114a20-dce9-11ea-1d69-6bd2ba1526d7
md"""
The ``\mathrm{STS}:\mathbb{R}^+\to\mathbb{N}_0`` function is a steps function, mapping a voltage threshold ``\vartheta\equiv\theta-V_0`` to the number of spikes elicited by this threshold ``\vartheta\mapsto\mathrm{STS}\left(\vartheta\right)``.

Each step is characterized by the critical voltage threshold at which the number of spikes goes from ``k-1`` to ``k`` (i.e. a new spike is added):

``
\vartheta^*_k\equiv\sup\left\{\vartheta\in\mathbb{R}^+:\mathrm{STS}\left(\vartheta\right)=k\right\},\quad k\in\mathbb{N}_0
``.

Now we can write the STS function in terms of ``\left\{\theta^*_k\right\}_{k=1}^\infty``:

``
\mathrm{STS}\left(\vartheta^*_{k+1}\lt\vartheta\lt\vartheta^*_k\right)=k,\quad\text{and}\quad\mathrm{STS}\left(\vartheta\gt V_{max}\right)=0
``.


In this notebook you can fiddle with ``\vartheta`` to better grasp the concept of STS. 
"""

# ╔═╡ 5f83b3c0-dccb-11ea-141a-1b0386451f58
begin
	# Set parameters
	N = 10
	T = 500
	dt = 1
	t = collect(0:dt:T)
	ν = 3
	n_samples = 3
	V₀ = -70
	θ = -55
	W = 1.5(12rand(Float64, N) .- 3).*(θ - V₀)./N
	k_max = 30
	tmp0 = Tempotron(N, V₀ = V₀, θ = θ, weights = W)
end;

# ╔═╡ 1abd2ac0-dcce-11ea-3154-6548006e8651
begin
	θ_min = V₀ + 0.5(θ - V₀)
	θ_max = V₀ + 2(θ - V₀)
	Δθ = 0.05(θ - V₀)
	md"""
	This slider controls the value of ``\vartheta``. Slide it!

	``\vartheta``: $(@bind ϑ Slider(θ_min:Δθ:θ_max, default = V₀ + 1.2(θ - V₀), show_value = true)) [mV]

	$(@bind redraw_poisson Button("Draw new random Poisson samples"))
	"""
end

# ╔═╡ becf4d60-dcf0-11ea-1c8e-0be70eede37d
begin
	redraw_poisson
	samples = [poisson_spikes_input(N, ν = ν, T = T)
	           for j = 1:n_samples]
	out_0 = [tmp0(s, t = t) for s ∈ samples]
	STSs = [GetSTS(tmp0, s, k_max = k_max) for s ∈ samples]
end;

# ╔═╡ 559e68c0-dcce-11ea-119e-4b386732168e
begin
	tmp  = Tempotron(N, V₀ = V₀, θ = ϑ, weights = W)
end;

# ╔═╡ 01cd1cc0-dccc-11ea-3956-cf9aaeb6982d
begin
	gr(size = (650, 400))
	auto_color = palette(:default)
	font_size = 11
	dn_spk = 0.004*font_size
	
	voltage_plot = plot()
	for k ∈ 1:length(samples)
	    plot!(tmp, samples[k], ylims = :none, color = auto_color[k])
	    plot!(tmp0, t, out_0[k].V, ylims = :none, color = auto_color[k], 
			  linestyle = :dash)
	end
	yticks!([V₀, ϑ, θ], ["V₀", "ϑ", "θ"])
	ylims!((-Inf, V₀ + 1.2(θ_max - V₀)))
	title!("Voltage traces")
	
	sts_plot = plot()
	for k ∈ 1:length(samples)
		plotsts!(tmp, STSs[k], color = auto_color[k])
	end
	xticks!([V₀, ϑ, θ], ["V₀", "ϑ", "θ"])
	yl = ylims()
	plot!(θ*[1, 1], [yl[1], yl[2]], color = :black, linestyle = :dash)
	xlims!((V₀, max(xlims()[1], V₀ + 1.2(θ_max - V₀))))
	title!("STSs")
	
	ann = plot(border = :none)
	annotate!(0.25, 0.5, text(" \nV₀ = $V₀ [mV]\nθ = $θ [mV]\nϑ = $ϑ [mV]", font_size))
	annotate!(0.75, 0.5, text("# of spikes:\n ", font_size))
	annotate!(0.75 - dn_spk*(n_samples)/2, 0.5, text(" \nθ:", font_size))
	annotate!(0.75 - dn_spk*(n_samples)/2, 0.5, text(" \n \n \nϑ:", font_size))
	for k ∈ 1:length(samples)
		n_spk = length(tmp(samples[k]).spikes)
		n_spk_k = length(out_0[k].spikes)
		annotate!(0.75 - dn_spk*(n_samples)/2 + k*dn_spk, 0.5, 
				  text(" \n$n_spk_k", color = auto_color[k], font_size))
		annotate!(0.75 - dn_spk*(n_samples)/2 + k*dn_spk, 0.5, 
				  text(" \n \n \n$n_spk", color = auto_color[k], font_size))
	end
	
	plot(ann, voltage_plot, sts_plot, 
		 layout = grid(3, 1, heights = [0.04, 0.48, 0.48]))
end

# ╔═╡ Cell order:
# ╟─a7473760-dd3c-11ea-3df0-d5b516288e3b
# ╟─db4750d0-dcca-11ea-0fc4-71d7dffb5f8a
# ╟─7c114a20-dce9-11ea-1d69-6bd2ba1526d7
# ╟─5f83b3c0-dccb-11ea-141a-1b0386451f58
# ╟─becf4d60-dcf0-11ea-1c8e-0be70eede37d
# ╟─1abd2ac0-dcce-11ea-3154-6548006e8651
# ╟─559e68c0-dcce-11ea-119e-4b386732168e
# ╠═01cd1cc0-dccc-11ea-3956-cf9aaeb6982d
