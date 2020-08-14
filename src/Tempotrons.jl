"""
An implementation of The Binary Tempotron [1] and the Multi-Spike Tempotron[2].

Use the [`Tempotron`](@ref) struct to create a tempotron, then train it using
the [`train!`](@ref) method.

# Examples:

```julia
using Tempotrons
input = [InputGen.PoissonProcess(ν = 5, T = 500) for i = 1:10]
tmp = Tempotron(N = 10)                             # Create a tempotron
train!(tmp, input, true)                            # Binary tempotron
train!(tmp, input, true, method = :corr)            # Binary correlation-based
train!(tmp, input, 3)                               # Multi-spike tempotron
train!(tmp, input, 5, method = :corr)               # Multi-spike correlation-based
output = tmp(input).spikes                          # Get output spikes
voltage_trace = tmp(input, t = collect(0:500)).V    # Get output voltage trace
```

# References
## Binary tempotron:
[1] [Gütig, R., & Sompolinsky, H. (2006). The tempotron: a neuron that learns spike timing–based decisions. Nature neuroscience, 9(3), 420.](https://www.nature.com/articles/nn1643)

## Multi-spike tempotron:
[2] [Gütig, R. (2016). Spiking neurons can discover predictive features by aggregate-label learning. Science, 351(6277), aab4113.](https://science.sciencemag.org/content/351/6277/aab4113)
"""
module Tempotrons

# Dependencies
using Roots
using Distributions
using Statistics
using Random
using RecipesBase

# Exports
export InputGen, Optimizers         # submodules
export Tempotron                    # main structure
export train!, get_sts, pretrain!    # methods
export SpikesInput, TimeInterval    # additional structures

# Optimizers submodule
include("Optimizers.jl")
using ..Optimizers

# Inputs submodule
include("Inputs.jl")
using ..Inputs

# Input generation submodule
include("InputGen.jl")
using ..InputGen

# Core
include("BaseTempotron.jl")
include("BinaryTempotron.jl")
include("MultiSpikeTempotron.jl")
include("CorrelationLearning.jl")

# Plots
include("TempotronsRecipes.jl")

# Run at using\import time
function __init__()

    # Set defaults for the plot package
    function module_fqn(pkg::Base.PkgId)
        if (pkg.name == "Plots" &&
            string(pkg.uuid) == "91a5bcdd-55d7-5caf-9e0b-520d859cae80")
            try
                # "import" Plots
                plots = Base.root_module(pkg)

                # Set default foreground color
                function plots_fg_color()
                    def_fg = plots.default(:fg)
                    return (def_fg != :auto ? def_fg : :black)
                end
                set_fg_color(plots_fg_color)

                # Fix hashtags escaping for the pgfplotsx backend
                function backend_fix_hashtags(x::AbstractString)::AbstractString
                    return plots.backend_name() != :pgfplotsx ? x :
                           replace(x, "#" => "\\#")
                end
                set_str_esc_hashtag(backend_fix_hashtags)

                # Dynamically get current plot limits
                syn_get_plot_lims() = plots.xlims(), plots.ylims()
                set_get_plot_lims(syn_get_plot_lims)
            catch
            end
        end
        return
    end

    # Set defaults for packages already loaded
    foreach(x -> x |> Base.PkgId |> module_fqn, Base.loaded_modules_array())

    # Set defaults for packages to be loaded (julia's experimental feature)
    push!(Base.package_callbacks, module_fqn)

    return

end

end
