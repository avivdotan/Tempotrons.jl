# import Plots
using .Plots
using .Plots.RecipesBase

# Default plot foreground color
fg_color() = (Plots.default(:fg) != :auto ? Plots.default(:fg) : :black)

# Fix hashtags escaping for the pgfplotsx backend
function str_esc_hashtag(x::AbstractString)::AbstractString
    return Plots.backend_name() != :pgfplotsx ? x :
           replace(x, "#" => "\\#")
end

# Handle dynamic plot limits
get_plot_lims() = Plots.xlims(), Plots.ylims()

@recipe function f(::Type{T}, si::T;
                   reduce_afferents = 1.0) where {T<:SpikesInput}

    N = length(si)

    if (isa(reduce_afferents, AbstractFloat) && reduce_afferents == 1.0) ||
       (isa(reduce_afferents, Integer) && reduce_afferents == N)
        idx_red = 1:N
    elseif isa(reduce_afferents, AbstractFloat)
        @assert 0.0 < reduce_afferents ≤ 1.0
        idx_red = randsubseq(1:N, reduce_afferents)
    elseif isa(reduce_afferents, Integer)
        @assert 0 < reduce_afferents ≤ N
        idx_red = randperm(N)[1:reduce_afferents]
    else
        isa(reduce_afferents, AbstractVector)
        @assert all(x -> 0 < x ≤ N, reduce_afferents)
        @assert allunique(reduce_afferents)
        idx_red = reduce_afferents
    end

    legend --> false
    seriescolor --> fg_color()
    markerstrokecolor --> :auto
    xlims --> (si.duration.from, si.duration.to)
    ylims --> (0.5, N + 0.5)
    yticks --> [1, N]
    xguide --> "t [ms]"
    yguide --> str_esc_hashtag("Neuron #")

    seriestype := :scatter

    xs = vcat(si.input[idx_red]...)
    ys = vcat(((i -> i .* ones(length(si[i]))).(idx_red))...)
    collect(zip(xs, ys))

end

function voltage_plot_recipe!(plotattributes, series_list, m, t, V)

    V_M = max(m.θ, maximum(V))
    V_m = min(m.V₀, minimum(V))
    V_scale = V_M - V_m
    t_lims = (t[begin], t[end])

    get!(plotattributes, :legend, false)
    get!(plotattributes, :seriescolor, fg_color())
    get!(plotattributes, :xlims, t_lims)
    get!(plotattributes, :ylims, (V_m - 0.1V_scale, V_M + 0.1V_scale))
    get!(plotattributes, :yticks, [m.V₀, m.θ])
    get!(plotattributes, :yformatter,
         x -> (x == m.V₀ ? "V₀" : (x == m.θ ? "θ" : "")))
    get!(plotattributes, :xguide, "t [ms]")
    get!(plotattributes, :yguide, "V [mV]")

    plotattributes[:seriestype] = :path

    let plotattributes = copy(plotattributes)
        plotattributes[:seriescolor] = fg_color()
        plotattributes[:linestyle] = :dash
        plotattributes[:linewidth] = 1
        plotattributes[:seriestype] = :path
        push!(series_list,
              RecipesBase.RecipeData(plotattributes,
                                     RecipesBase.wrap_tuple(collect(zip([t_lims[1],
                                                                         t_lims[2]],
                                                                        m.θ *
                                                                        ones(2))))))
    end

    return collect(zip(t, V))

end

@recipe function f(m::Tempotron, t::Array{T1,1},
                   V::Array{T2,1}) where {T1<:Real,T2<:Real}

    voltage_plot_recipe!(plotattributes, series_list, m, t, V)

end

@recipe function f(m::Tempotron{N}, si::SpikesInput{T1,N}) where {T1<:Real,N}

    # Get tempotron's voltage trace
    t = collect((si.duration.from):(si.duration.to))
    V = m(si, t = t).V
    voltage_plot_recipe!(plotattributes, series_list, m, t, V)

end

@userplot PlotSTS

@recipe function f(h::PlotSTS; k_max = nothing)

    if !(length(h.args) == 2) ||
       !(typeof(h.args[1]) <: Tempotron) ||
       !(typeof(h.args[2]) <: SpikesInput ||
         typeof(h.args[2]) <: AbstractVector)
        error("plotsts should be given a ::Tempotron and a " *
              "::SpikesInput or ::AbstractVector.  Got: $(typeof(h.args))")
    end
    if k_max ≢ nothing && !isa(k_max, Integer)
        error("plotsts k_max must be an <:Integer.  Got: $(typeof(k_max))")
    end

    m::Tempotron = h.args[1]
    if isa(h.args[2], SpikesInput)
        si::SpikesInput = h.args[2]
        sts_args = (si = si,)
        sts_kwargs = NamedTuple()
        if k_max ≢ nothing
            sts_kwargs = merge(sts_kwargs, (k_max = k_max,))
        end
        θ⃰ = get_sts(m, sts_args...; sts_kwargs...)
    elseif isa(h.args[2], AbstractVector)
        θ⃰::Array{Real,1} = h.args[2]
        if k_max ≢ nothing
            θ⃰ = θ⃰[1:min(end, k_max)]
        end
    end

    xs = sort!([θ⃰..., θ⃰..., m.V₀ + 1.2maximum(θ⃰ .- m.V₀)])
    ys = collect(1:length(θ⃰))
    ys = sort!([0, (ys .- 1)..., ys...], rev = true)

    legend --> false
    seriescolor --> fg_color()
    xlims --> (m.V₀, Inf)
    ylims --> (0, maximum(ys))
    xticks --> [m.θ]
    xformatter --> (x -> (x == m.θ ? "θ" : ""))
    xguide --> "θ [mV]"
    yguide --> str_esc_hashtag("# of spikes")

    seriestype := :path

    @series begin
        seriescolor := fg_color()
        linestyle := :dash
        linewidth := 1
        collect(zip(m.θ .* ones(2), [0, length(θ⃰) + 1]))
    end

    collect(zip(xs, ys))

end

function get_progress_annotations(N::Real;
                                  N_b::Union{Real,Nothing} = nothing,
                                  N_t::Union{Real,Nothing} = nothing,
                                  desc::AbstractString = "# of spikes",
                                  digits::Integer = 3)

    fround(n) = isa(n, AbstractFloat) ? round(n, digits = digits) : n
    text_clr = fg_color()
    N_text = isempty(desc) ? "" : (desc * ": ")
    if N_b ≢ nothing
        R_b = fround(N_b)
        N_text *= "$R_b → "
    end
    R = fround(N)
    N_text *= "$R"
    if N_t ≢ nothing
        if N_t == N
            N_text *= " = "
            text_clr = :lightseagreen
        else
            N_text *= " ≠ "
            text_clr = :salmon
        end
        R_t = fround(N_t)
        N_text *= "$R_t"
    end
    N_text = str_esc_hashtag(N_text)

    return N_text, text_clr

end

@recipe function f(::Type{T}, ti::T;
                   reduce_afferents = 1.0) where {T<:TimeInterval}

    yl = get_plot_lims()[2]

    xs = [ti.from, ti.to]
    ys = [yl[1], yl[1]]

    legend --> false
    linealpha --> 0.3

    ribbon := ([0], [yl[2] - yl[1]])

    collect(zip(xs, ys))

end
