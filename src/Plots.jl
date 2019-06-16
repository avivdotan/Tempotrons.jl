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
function PlotInputs(inp::Array{Array{T, 1}, 1};
                    T_max::Real = maximum(abs, maximum.(abs, inp)),
                    color = :black) where T <: Any
    p = scatter()
    for i = 1:length(inp)
        if !isempty(inp[i])
            scatter!(inp[i], i*ones(length(inp[i])),
            markercolor = color, markerstrokecolor = color,
            markersize = 1.5, label = "")
        end
    end
    xlabel!("t [ms]")
    ylabel!("Neuron #")
    xlims!((0, T_max))
    yticks!([1, length(inp)])
    ylims!((0, 1.05length(inp)))
    return p
end

"""
    PlotPotential(m::Tempotron, out_b, out_a, t = nothing, color)
Plot a comparison between a tempotron's two output voltages, `out_b` and `out_a`.
Optional parameters are the time grid `t` (defaults to `1:length(out_b)`) and
the line color `color`.
"""
function PlotPotential(m::Tempotron;
                        out_b::Array{T1, 1},
                        out_a::Array{T1, 1},
                        t::Array{T2, 1} = 1:length(out_b),
                        color = :black) where {T1 <: Real,
                                                T2 <: Real}
    p = plot(t, out_b, linecolor = color, linestyle = :dash, label = "")
    plot!(t, out_a, linecolor = color, label = "")
    plot!(t, m.θ*ones(length(out_b)), linecolor = :black, linestyle = :dash,
        label = "")
    xlabel!("t [ms]")
    ylabel!("V [mV]")
    yticks!([m.V₀, m.θ], ["V₀", "θ"])
    plot_M = max(m.θ, maximum(out_b), maximum(out_a))
    plot_m = min(m.V₀, minimum(out_b), minimum(out_a))
    V_scale = plot_M - plot_m
    ylims!((plot_m - 0.1V_scale, plot_M + 0.1V_scale))
    return p
end

"""
    PlotSTS(m::Tempotron, θ⃰_b, θ⃰_a[, color])
Plot a comparison between a tempotron's STSs, given by `θ⃰_b` and `θ⃰_a`.
An optional parameter is the line color `color`.
"""
function PlotSTS(m::Tempotron;
                θ⃰_b::Array{Tp1, 1},
                θ⃰_a::Array{Tp2, 1},
                color = :black) where {Tp1 <: Real,
                                        Tp2 <: Real}
    @assert length(θ⃰_b) == length(θ⃰_a)
    x_b = sort!([θ⃰_b..., θ⃰_b..., 1.2maximum(θ⃰_b)])
    x_a = sort!([θ⃰_a..., θ⃰_a..., 1.2maximum(θ⃰_a)])
    y = [(k - 1) for k = length(θ⃰_b):-1:1]
    y = sort!([0, y..., (y .+ 1)...], rev = true)
    p = plot(x_b, y, linecolor = color, linestyle = :dash, label = "")
    plot!(x_a, y, linecolor = color, label = "")
    plot!([m.θ, m.θ], [0, length(θ⃰_b) + 1], linecolor = :black,
          linestyle = :dash, label = "")
    xlabel!("θ [mV]")
    ylabel!("# of spikes")
    return p
end

end
