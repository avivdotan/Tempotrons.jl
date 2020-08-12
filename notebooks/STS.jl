### A Pluto.jl notebook ###
# v0.11.4

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

# ╔═╡ 5f83b3c0-dccb-11ea-141a-1b0386451f58
begin
	# Set parameters
	N = 10
	T = 500
	dt = 1
	t = collect(0:dt:T)
	ν = 3
	n_samples = 3
	V₀ = 0
	θ₀ = 1
	W = (12rand(Float64, N) .- 3).*(θ₀ - V₀)./N
end;

# ╔═╡ 1abd2ac0-dcce-11ea-3154-6548006e8651
begin
	θ_min = V₀ + 0.2(θ₀ - V₀)
	θ_max = V₀ + 2(θ₀ - V₀)
	Δθ = 0.1(θ₀ - V₀)
	@bind θ Slider(θ_min:Δθ:θ_max, default = θ₀ + Δθ)
end

# ╔═╡ 559e68c0-dcce-11ea-119e-4b386732168e
tmp = Tempotron(N, V₀ = V₀, θ = θ, weights = W);

# ╔═╡ cbf256b0-dccb-11ea-3bb9-db2c8b08344d
samples = [poisson_spikes_input(N, ν = ν, T = T)
           for j = 1:n_samples];

# ╔═╡ 01cd1cc0-dccc-11ea-3956-cf9aaeb6982d
begin
	θ₀_color = :gray
	auto_color = palette(:default)
	dn_spk = 0.06

	voltage_plot = plot()
	for k ∈ 1:length(samples)
	    plot!(tmp, samples[k], ylims = :none, color = auto_color[k])
	end
	yticks!([V₀, θ, θ₀], ["V₀", "θ", "θ₀"])
	xl = xlims()
	plot!([xl[1], xl[2]], θ₀*[1, 1], color = θ₀_color)
	
	sts_plot = plot()
	for k ∈ 1:length(samples)
		plotsts!(tmp, samples[k], color = auto_color[k])
	end
	xticks!([V₀, θ, θ₀], ["V₀", "θ", "θ₀"])
	yl = ylims()
	plot!(θ₀*[1, 1], [yl[1], yl[2]], color = θ₀_color)
	
	ann = plot(border = :none)
	annotate!(0.25, 0.5, "θ₀ = $θ₀\nθ = $θ")
	annotate!(0.75, 0.5, "# of spikes:\n ")
	for k ∈ 1:length(samples)
		n_spk = length(tmp(samples[k]).spikes)
		annotate!(0.75 - dn_spk*(n_samples - 1)/2 + (k - 1)*dn_spk, 0.5, 
				  text(" \n$n_spk", color = auto_color[k]))
	end
	
	plot(ann, voltage_plot, sts_plot, layout = (3, 1))
end

# ╔═╡ Cell order:
# ╟─db4750d0-dcca-11ea-0fc4-71d7dffb5f8a
# ╟─5f83b3c0-dccb-11ea-141a-1b0386451f58
# ╟─1abd2ac0-dcce-11ea-3154-6548006e8651
# ╟─559e68c0-dcce-11ea-119e-4b386732168e
# ╟─cbf256b0-dccb-11ea-3bb9-db2c8b08344d
# ╟─01cd1cc0-dccc-11ea-3956-cf9aaeb6982d
