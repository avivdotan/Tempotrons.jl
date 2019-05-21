module Plots

using ..Tempotrons
using Plots

export PlotInputs, PlotPotential

function PlotInputs(inp::Array{Array{T, 1}, 1};
                    T_max::Real,
                    color) where T <: Any
    p = scatter()
    for i = 1:length(inp)
        if !isempty(inp[i])
            scatter!(inp[i], i*ones(length(inp[i])),
            markercolor = color, markerstrokecolor = color,
            markersize = 1.5, label = "")
        end
    end
    # xlabel!("t [ms]")
    ylabel!("Neuron")
    xlims!((0, T_max))
    yticks!([1, length(inp)])
    ylims!((0, 1.05length(inp)))
    return p
end

function PlotPotential(m::Tempotron;
                        out_b::Array{T1, 1},
                        out_a::Array{T1, 1},
                        t::Array{T2, 1} = nothing,
                        color = :blue) where {T1 <: Real,
    T2 <: Real}
    if t ≡ nothing
        t = 1:length(out_b)
    end
    p = plot(t, out_b, linecolor = color, linestyle = :dash,
    label = "")
    plot!(t, out_a, linecolor = color, label = "")
    plot!(t, m.θ*ones(length(out_b)),
    linecolor = :black, linestyle = :dash, label = "")
    # xlabel!("t [ms]")
    ylabel!("V [mV]")
    yticks!([m.V₀, m.θ], ["V₀", "θ"])
    plot_M = max(m.θ, maximum(out_b), maximum(out_a))
    plot_m = min(m.V₀, minimum(out_b), minimum(out_a))
    V_scale = plot_M - plot_m
    ylims!((plot_m - 0.1V_scale, plot_M + 0.1V_scale))
    return p
end

end
