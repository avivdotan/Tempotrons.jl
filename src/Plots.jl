"""
Some helper plot functions for the Tempotrons.jl package.
"""
module Plots

using ..Tempotrons
using Plots

export ReduceAfferents, PlotInputs, PlotPotential, PlotSTS

"""
    ReduceAfferents(inp[, percent_keep] )
Randomly choose `percent_keep`∈(0,1] afferent neurons out of `inp` and return
their spike trains (for visualization purposes).
"""
function ReduceAfferents(inp::Array{Array{T1, 1}, 1},
                         percent_keep::Real = 0.1) where T1 <: Real

@assert percent_keep > 0
@assert percent_keep ≤ 1
N = length(inp)
return inp[rand(1:N, Int(ceil(percent_keep*N)))]

end

"""
    PlotInputs(inp[, T_max][, color])
Generate a raster plot of the given input `inp`. Optional parameters are the
maximal time `T_max` and the dots' color `color`.
"""
function PlotInputs(inp::Array{Array{T1, 1}, 1};
                    color = default(:fg),
                    events = nothing) where T1 <: Real

    clr = color != :auto ? color : :black
    inp_x = inp[:]
    inp_y = (i -> i*ones(length(inp[i]))).(1:length(inp))
    # p = plot()
    p = scatter(inp_x, inp_y, label = "",
                markercolor = clr, markerstrokecolor = clr, markersize = 1)
    if events ≢ nothing
        for e ∈ events
            plot!([e.time, e.time + e.length], [0, 0],
                ribbon = ([0], [length(inp) + 0.5]), color = e.color,
                linealpha = 0.5, label = "")
        end
    end
    xlabel!("t [ms]")
    y_label = "# of spikes"
    if backend_name() == :pgfplotsx
        y_label = replace(y_label, "#" => "\\#")
    end
    ylabel!(y_label)
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
                        N::Union{Integer, Nothing} = nothing,
                        N_b::Union{Integer, Nothing} = nothing,
                        N_t::Union{Integer, Nothing} = nothing,
                        color = default(:fg),
                        events = nothing) where {T1 <: Real,
                                                T2 <: Real}

    clr = color != :auto ? color : :black
    fg_clr = default(:fg) != :auto ? default(:fg) : :black

    # Scaling parameters
    plot_M = max(m.θ, maximum(out))
    plot_m = min(m.V₀, minimum(out))
    if out_b ≢ nothing
        plot_M = max(plot_M, maximum(out_b))
        plot_m = min(plot_m, minimum(out_b))
    end
    V_scale = plot_M - plot_m

    if out_b ≡ nothing
        p = plot(t, out, linecolor = clr, linewidth = 0.5, label = "")
    else
        p = plot(t, out_b, linecolor = clr, linewidth = 0.5,
                linestyle = :dash, label = "")
        plot!(t, out, linecolor = clr, linewidth = 0.5, label = "")
    end
    plot!(t, m.θ*ones(length(out)), color = fg_clr,
          linestyle = :dash, label = "")
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
    if N ≢ nothing
        N_text = "# of spikes: "
        if N_b ≢ nothing
            N_text *= "$N_b → "
        end
        N_text *= "$N"
        if N_t ≢ nothing
            N_text *= (N_t == N ? " = " : " ≠ ")
            N_text *= "$N_t"
        end
        if backend_name() == :pgfplotsx
            N_text = replace(N_text, "#" => "\\#")
        end
        x_lims, y_lims = xlims(), ylims()
        annotate!(x_lims[1], y_lims[2],
                  text(N_text, fg_clr, 10, :left, :bottom))
    end
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
                color = default(:fg)) where {Tp1 <: Real}

    clr = color != :auto ? color : :black
    fg_clr = default(:fg) != :auto ? default(:fg) : :black

    if θ⃰_b ≢ nothing
        @assert length(θ⃰_b) == length(θ⃰)
        x_b = sort!([θ⃰_b..., θ⃰_b..., m.V₀ + 1.2maximum(θ⃰_b .- m.V₀)])
    end
    x_b = sort!([θ⃰_b..., θ⃰_b..., m.V₀ + 1.2maximum(θ⃰_b .- m.V₀)])
    x = sort!([θ⃰..., θ⃰..., m.V₀ + 1.2maximum(θ⃰ .- m.V₀)])
    y = collect(1:length(θ⃰))
    y = sort!([0, (y .- 1)..., y...], rev = true)
    if θ⃰_b ≡ nothing
        p = plot(x, y, linecolor = clr, label = "")
    else
        p = plot(x_b, y, linecolor = clr, linestyle = :dash, label = "")
        plot!(x, y, linecolor = clr, label = "")
    end
    plot!([m.θ, m.θ], [0, length(θ⃰) + 1], linecolor = fg_clr,
          linestyle = :dash, label = "")
    xlabel!("θ [mV]")
    y_label = "# of spikes"
    if backend_name() == :pgfplotsx
        y_label = replace(y_label, "#" => "\\#")
    end
    ylabel!(y_label)
    xlims!((m.V₀, Inf))
    ylims!((0, maximum(y)))
    return p
end

end
