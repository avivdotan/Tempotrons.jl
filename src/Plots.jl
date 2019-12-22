"""
Some helper plot functions for the Tempotrons.jl package.
"""
module Plots

using ..Tempotrons
using Plots

export PlotInputs, PlotPotential, PlotSTS

"""
    PlotInputs(inp[, T_max][, color])
Generate a raster plot of the given input `inp`. Optional parameters are the
maximal time `T_max` and the dots' color `color`.
"""
function PlotInputs(inp::Array{Array{T1, 1}, 1};
                    color = :black,
                    events = nothing) where T1 <: Real
    inp_x = inp[:]
    inp_y = (i -> i*ones(length(inp[i]))).(1:length(inp))
    p = plot()
    scatter!(inp_x, inp_y, label = "",
             markercolor = color, markerstrokecolor = color, markersize = 1)
    if events ≢ nothing
        for e ∈ events
            plot!([e.time, e.time + e.length], [0, 0],
                ribbon = ([0], [length(inp) + 0.5]), color = e.color,
                linealpha = 0.5, label = "")
        end
    end
    xlabel!("t [ms]")
    ylabel!("Neuron #")
    xlims!((0, Inf))
    yticks!([1, length(inp)])
    ylims!((0, length(inp) + 0.5))
    return p
end

"""
    PlotPotential(m::Tempotron, out_b, out_a, t = nothing, color)
Plot a comparison between a tempotron's two output voltages, `out_b` and `out_a`.
Optional parameters are the time grid `t` (defaults to `1:length(out_b)`) and
the line color `color`.
"""
function PlotPotential(m::Tempotron;
                        out::Array{T1, 1},
                        out_b::Union{Array{T1, 1}, Nothing} = nothing,
                        t::Array{T2, 1} = 1:length(out_b),
                        color = :black,
                        events = nothing) where {T1 <: Real,
                                                T2 <: Real}

    # Scaling parameters
    plot_M = max(m.θ, maximum(out))
    plot_m = min(m.V₀, minimum(out))
    if out_b ≢ nothing
        plot_M = max(plot_M, maximum(out_b))
        plot_m = min(plot_m, minimum(out_b))
    end
    V_scale = plot_M - plot_m

    if out_b ≡ nothing
        p = plot(t, out, linecolor = color, linewidth = 0.5, label = "")
    else
        p = plot(t, out_b, linecolor = color, linewidth = 0.5,
                linestyle = :dash, label = "")
        plot!(t, out, linecolor = color, linewidth = 0.5, label = "")
    end
    plot!(t, m.θ*ones(length(out)), linecolor = :black, linestyle = :dash,
        label = "")
    if events ≢ nothing
        for e ∈ events
            plot!([e.time, e.time + e.length], (plot_m - 0.1V_scale)*[1, 1],
                ribbon = ([0], [1.2V_scale]),
                color = e.color, linealpha = 0.3, label = "")
        end
    end
    xlabel!("t [ms]")
    ylabel!("V [mV]")
    yticks!([m.V₀, m.θ], ["V₀", "θ"])
    ylims!((plot_m - 0.1V_scale, plot_M + 0.1V_scale))
    return p
end

"""
    PlotSTS(m::Tempotron, θ⃰_b, θ⃰_a[, color])
Plot a comparison between a tempotron's STSs, given by `θ⃰_b` and `θ⃰_a`.
An optional parameter is the line color `color`.
"""
function PlotSTS(m::Tempotron;
                θ⃰::Array{Tp1, 1},
                θ⃰_b::Union{Array{Tp1, 1}, Nothing} = nothing,
                color = :black) where {Tp1 <: Real}

    if θ⃰_b ≢ nothing
        @assert length(θ⃰_b) == length(θ⃰)
        x_b = sort!([θ⃰_b..., θ⃰_b..., m.V₀ + 1.2maximum(θ⃰_b .- m.V₀)])
    end
    x_b = sort!([θ⃰_b..., θ⃰_b..., m.V₀ + 1.2maximum(θ⃰_b .- m.V₀)])
    x = sort!([θ⃰..., θ⃰..., m.V₀ + 1.2maximum(θ⃰ .- m.V₀)])
    y = collect(1:length(θ⃰))
    y = sort!([0, (y .- 1)..., y...], rev = true)
    if θ⃰_b ≡ nothing
        p = plot(x, y, linecolor = color, label = "")
    else
        p = plot(x_b, y, linecolor = color, linestyle = :dash, label = "")
        plot!(x, y, linecolor = color, label = "")
    end
    plot!([m.θ, m.θ], [0, length(θ⃰) + 1], linecolor = :black,
          linestyle = :dash, label = "")
    xlabel!("θ [mV]")
    ylabel!("# of spikes")
    xlims!((m.V₀, Inf))
    ylims!((0, maximum(y)))
    return p
end

end
