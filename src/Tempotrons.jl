"""
An implementation of The Binary Tempotron [1] and the Multi-Spike Tempotron[2].

Use the [`Tempotron`](@ref) struct to create a tempotron, then train it using
the [`Train!`](@ref) method.

# Examples:

```julia
using Tempotrons
input = [InputGen.PoissonProcess(ν = 5, T = 500) for i = 1:10]
tmp = Tempotron(N = 10)                             # Create a tempotron
Train!(tmp, input, true)                            # Binary tempotron
Train!(tmp, input, true, method = :corr)            # Binary correlation-based
Train!(tmp, input, 3)                               # Multi-spike tempotron
Train!(tmp, input, 5, method = :corr)               # Multi-spike correlation-based
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

# Exports
export InputGen, Optimizers         # submodules
export Tempotron                    # main structure
export Train!, GetSTS, Pretrain!    # methods
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
# include("Plots.jl")

# Run at using\import time
function __init__()

    # Set defaults for the plot package
    function module_fqn(pkg::Base.PkgId)
        if pkg.name == "Plots"
            try
                plots = Base.root_module(pkg)
                def_fg = plots.default(:fg)
                set_fg_color(def_fg != :auto ? def_fg : :black)
                function f(x::AbstractString)::AbstractString
                    if plots.backend_name() == :pgfplotsx
                        return replace(x, "#" => "\\#")
                    else
                        return x
                    end
                end
                set_str_esc_hashtag(f)
            catch
            end
        end
        return
    end

    # Set defaults for packages already loaded
    foreach(x -> x |> Base.PkgId |> module_fqn, Base.loaded_modules_array())

    # Set defaults for packages to be loaded (julia's experimental feature)
    push!(Base.package_callbacks, module_fqn)

end

end
